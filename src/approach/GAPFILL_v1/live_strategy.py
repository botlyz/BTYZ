#!/root/Botlyz_TG/.venv/bin/python3
"""SIGMA GAP — bot live gap-fill (AAPL + AMD) sur Lighter, 1:1 backtest.

Cron 5min (one-shot, PID lock). Réutilise la plomberie LighterClient du template
SIGMA (auth, rate-limit, proxy, scaling, /candles). Source de vérité = GapfillKernel
(kernel.py, prouvé bit-identique au backtest). Ordres MARKET (IOC limit agressif).

Données 1:1 backtest (cf. engine.data_loader.load_gapfill) :
  - px        = close de la dernière barre 5m Lighter COMPLÉTÉE (sa timestamp 't' = réf)
  - fair_value= close de la dernière barre 1h yfinance dont (start+1h) <= réf  (shift +1h)
  - real_age_min = (réf - bar_end_yf) en minutes
État kernel persisté entre cycles dans state.json.
DRY_RUN=1 → calcule et log SANS envoyer d'ordre.
"""
import asyncio
import atexit
import fcntl
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import aiohttp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXCHANGE_DIR = os.path.dirname(SCRIPT_DIR)
secret_file = os.path.join(EXCHANGE_DIR, 'secret.json')
config_file = os.path.join(SCRIPT_DIR, 'config.json')
state_file = os.path.join(SCRIPT_DIR, 'state.json')
pid_file = os.path.join(SCRIPT_DIR, '.strategy_lighter.pid')
log_file = os.path.join(SCRIPT_DIR, 'strategy.log')

sys.path.insert(0, SCRIPT_DIR)  # pour importer kernel.py (copié à côté)
from kernel import GapfillKernel

DRY_RUN = os.environ.get('DRY_RUN', '0') == '1'

INTEGRATOR_ACCOUNT_INDEX = 712276
INTEGRATOR_TAKER_FEE = 300    # 3 bps (= coût du backtest → live colle au BT)
INTEGRATOR_MAKER_FEE = 300

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

# ---------------- PID lock ----------------
_pid_fd = None

def _release_lock():
    global _pid_fd
    if _pid_fd is not None:
        try:
            fcntl.flock(_pid_fd, fcntl.LOCK_UN); os.close(_pid_fd); os.remove(pid_file)
        except Exception:
            pass
        _pid_fd = None

def cleanup_and_exit(code=0):
    _release_lock()
    sys.exit(code)

def acquire_lock():
    global _pid_fd
    _pid_fd = os.open(pid_file, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(_pid_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(_pid_fd, 0); os.write(_pid_fd, str(os.getpid()).encode())
        atexit.register(_release_lock)  # release-only à l'exit (pas de sys.exit)
        return True
    except (IOError, OSError):
        return False

# ---------------- Rate limiter (comme SIGMA) ----------------
class _RateLimiter:
    def __init__(self, max_requests=55, window=60.0):
        self.max = max_requests; self.window = window; self.calls = []
    async def acquire(self):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.window]
        if len(self.calls) >= self.max:
            await asyncio.sleep(self.window - (now - self.calls[0]) + 0.1)
        self.calls.append(time.time())

rate_limiter = _RateLimiter()

_proxy_url = None
try:
    sys.path.insert(0, '/root/Botlyz_TG/Botlyz_client_clean')
    from utils.LighterProxy import get_aiohttp_proxy
    _proxy_url = get_aiohttp_proxy  # résolu après chargement du secret
except Exception:
    _proxy_url = None


class LighterClient:
    """Sous-ensemble de la plomberie SIGMA : auth, candles, positions, market order."""
    def __init__(self, config: dict):
        self.config = config
        self.base_url = config['base_url']
        self.account_index = config.get('account_index')  # FORCÉ depuis secret (sous-compte Forex)
        self.signer = None
        self.market_info = {}
        self._session = None
        self._proxy = None
        try:
            if callable(_proxy_url):
                self._proxy = _proxy_url(config.get('eth_address', ''))
        except Exception:
            self._proxy = None
        self._account_cache = None
        self._account_cache_ttl = 30

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _request(self, method, url, max_retries=3, **kwargs):
        session = await self._get_session()
        if self._proxy:
            kwargs.setdefault('proxy', self._proxy)
        data = {}
        for attempt in range(max_retries):
            await rate_limiter.acquire()
            async with session.request(method, url, **kwargs) as resp:
                if resp.status == 429:
                    await asyncio.sleep((2 ** attempt) * 5); continue
                data = await resp.json()
                if isinstance(data, dict) and data.get('code') == 23000:
                    await asyncio.sleep((2 ** attempt) * 5); continue
                return data
        return data

    async def initialize(self):
        import lighter
        # account_index : si absent du secret, fallback accountsByL1Address[0] (à éviter pour un sous-compte)
        if self.account_index is None:
            d = await self._request('GET', f"{self.base_url}/api/v1/accountsByL1Address",
                                    params={"l1_address": self.config["eth_address"]})
            self.account_index = d['sub_accounts'][0]['index']
            logging.warning(f"[LIGHTER] account_index non fourni -> fallback sub_accounts[0]={self.account_index}")
        self.signer = lighter.SignerClient(
            url=self.base_url, account_index=self.account_index,
            api_private_keys={self.config['api_key_index']: self.config['api_key_private_key']})
        err = self.signer.check_client()
        if err:
            raise Exception(f"Lighter connection failed: {err}")
        await self._load_market_info()
        logging.info(f"[LIGHTER] Connecté — account_index={self.account_index}")

    async def _load_market_info(self):
        d = await self._request('GET', f"{self.base_url}/api/v1/orderBookDetails")
        for m in d.get('order_book_details', []):
            self.market_info[m['market_id']] = {
                'symbol': m['symbol'], 'size_decimals': m['size_decimals'],
                'price_decimals': m['price_decimals'], 'min_base': float(m['min_base_amount']),
                'min_quote': float(m['min_quote_amount']),
            }

    def scale_size(self, mid, size):
        return int(size * (10 ** self.market_info[mid]['size_decimals']))

    def scale_price(self, mid, price):
        return int(price * (10 ** self.market_info[mid]['price_decimals']))

    async def _account(self):
        now = time.time()
        if self._account_cache and now - self._account_cache['ts'] < self._account_cache_ttl:
            return self._account_cache['data']
        d = await self._request('GET', f"{self.base_url}/api/v1/account",
                                params={"by": "index", "value": str(self.account_index)})
        self._account_cache = {'data': d, 'ts': now}
        return d

    async def fetch_position(self, market_id) -> float:
        """Position signée en base (négatif = short, 0 = flat)."""
        d = await self._account()
        for p in d['accounts'][0].get('positions', []):
            if p['market_id'] == market_id:
                size = float(p.get('position', 0)); sign = int(p.get('sign', 0))
                if size != 0 and sign != 0:
                    return abs(size) * (1 if sign > 0 else -1)
        return 0.0

    async def fetch_candles_5m(self, market_id, limit=50) -> List[List]:
        now = int(time.time() * 1000)
        start = now - limit * 300 * 1000
        d = await self._request('GET', f"{self.base_url}/api/v1/candles",
                                params={"market_id": market_id, "resolution": "5m",
                                        "start_timestamp": start, "end_timestamp": now, "count_back": limit})
        out = []
        for c in d.get('c', []):
            out.append([int(c.get('t', 0)), float(c.get('c', 0))])
        out.sort()
        return out

    async def set_leverage(self, market_id, leverage=1):
        try:
            await self.signer.update_leverage(market_index=market_id, margin_mode=0, leverage=leverage)
        except Exception as e:
            if 'not updated' not in str(e).lower() and 'already' not in str(e).lower():
                logging.warning(f"[LEVERAGE] {market_id}: {e}")

    async def market_order(self, market_id, side, amount, ref_price, cap_slip, reduce_only):
        """Ordre MARKET = IOC limit au prix agressif (slippage protection)."""
        import lighter
        price = ref_price * (1 + cap_slip) if side == 'buy' else ref_price * (1 - cap_slip)
        await rate_limiter.acquire()
        result = await self.signer.create_order(
            market_index=market_id, client_order_index=int(time.time() * 1000),
            base_amount=self.scale_size(market_id, amount), price=self.scale_price(market_id, price),
            is_ask=(side == 'sell'), order_type=lighter.SignerClient.ORDER_TYPE_LIMIT,
            time_in_force=lighter.SignerClient.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
            reduce_only=reduce_only, trigger_price=lighter.SignerClient.NIL_TRIGGER_PRICE,
            order_expiry=lighter.SignerClient.DEFAULT_IOC_EXPIRY,
            integrator_account_index=INTEGRATOR_ACCOUNT_INDEX,
            integrator_taker_fee=INTEGRATOR_TAKER_FEE, integrator_maker_fee=INTEGRATOR_MAKER_FEE)
        _, resp, err = result
        if err:
            raise Exception(f"market_order failed: {err}")
        return resp.to_dict() if resp else {}

    async def close(self):
        try:
            if self.signer is not None:
                res = self.signer.close()
                if asyncio.iscoroutine(res):
                    await res
        except Exception:
            pass
        if self._session and not self._session.closed:
            await self._session.close()
        await asyncio.sleep(0.25)  # laisse les connectors se fermer (évite warning)


# ---------------- yfinance fair_value (1:1 load_gapfill) + cache + timeout ----------------
# Pendant le marché fermé (= quand on trade) le fair_value est GELÉ → inutile de
# re-fetch yfinance à chaque cycle. On le cache (TTL) → cycle ~3-5s au lieu de ~30s.
# real_age_min est recalculé à chaque cycle (cheap) depuis le bar_end caché.
_YF_TTL = 600           # 10 min : fair_value gelé off-hours, lag open-transition << recent_min
_YF_TIMEOUT = 12        # s : un hang yfinance ne bloque jamais le cycle

def _yf_fetch_bar(yf_ticker: str):
    """Dernière barre 1h COMPLÉTÉE (bar_end<=now), shift +1h. Renvoie (bar_end_ms, close) ou None.
    Exécuté dans un thread avec timeout pour ne jamais bloquer."""
    res = {}
    def _w():
        try:
            import yfinance as yf
            import pandas as pd
            df = yf.Ticker(yf_ticker).history(period="7d", interval="1h", auto_adjust=False)
            if df is None or df.empty:
                return
            idx = pd.to_datetime(df.index, utc=True) + pd.Timedelta("1h")
            ends = (idx.view("int64") // 1_000_000)
            closes = df["Close"].to_numpy(dtype=float)
            now_ms = int(time.time() * 1000)
            valid = ends <= now_ms
            if not valid.any():
                return
            last = int(valid.nonzero()[0][-1])
            res["bar_end_ms"] = int(ends[last]); res["fair"] = float(closes[last])
        except Exception:
            pass
    t = threading.Thread(target=_w, daemon=True); t.start(); t.join(_YF_TIMEOUT)
    return (res["bar_end_ms"], res["fair"]) if "fair" in res else None


def fetch_fair_value(yf_ticker: str, ref_ts_ms: int):
    """(fair_value, real_age_min) — cache TTL + timeout. Shift +1h appliqué dans le fetch."""
    cache_fp = os.path.join(SCRIPT_DIR, f".yf_cache_{yf_ticker}.json")
    cached = None
    if os.path.exists(cache_fp):
        try:
            cached = json.load(open(cache_fp))
        except Exception:
            cached = None
    fresh = cached and (time.time() - cached.get("fetched_at", 0) < _YF_TTL)
    if not fresh:
        bar = _yf_fetch_bar(yf_ticker)
        if bar is not None:
            cached = {"fetched_at": time.time(), "bar_end_ms": bar[0], "fair_value": bar[1]}
            try:
                json.dump(cached, open(cache_fp, "w"))
            except Exception:
                pass
        # si le fetch échoue mais qu'on a un cache (même périmé) → on l'utilise (mieux que rien)
    if cached is None:
        return None, None
    fair = float(cached["fair_value"])
    age_min = (ref_ts_ms - int(cached["bar_end_ms"])) / 60000.0
    return fair, age_min


# ---------------- State ----------------
def load_state():
    if os.path.exists(state_file):
        try:
            with open(state_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_state(st):
    tmp = state_file + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, state_file)


# ---------------- Per-pair cycle ----------------
async def process_pair(client: LighterClient, sc: dict, state: dict):
    sym = sc['symbol']; mid = sc['market_id']; yf_ticker = sc['yf_ticker']
    notional = float(sc['notional_usd']); cap_slip = float(sc.get('cap_slip', 0.005))
    log = lambda m: logging.info(f"[{sym}] {m}")

    # 1) Lighter 5m : dernière barre COMPLÉTÉE
    candles = await client.fetch_candles_5m(mid, limit=50)
    if len(candles) < 2:
        log("pas assez de candles -> skip"); return
    now_ms = int(time.time() * 1000)
    completed = [c for c in candles if c[0] + 300_000 <= now_ms]
    if not completed:
        log("aucune barre complétée -> skip"); return
    ref_ts, px = completed[-1]

    pstate = state.setdefault(sym, {})
    if pstate.get('last_bar_ts') == ref_ts:
        log(f"barre {ref_ts} déjà traitée -> skip"); return

    # 2) fair_value yfinance (1h, shift +1h)
    fair, age = fetch_fair_value(yf_ticker, ref_ts)
    if fair is None:
        log("yfinance indispo/stale -> skip cycle (pas de trade à l'aveugle)"); return
    dev = (px - fair) / fair
    closed = age > float(sc['recent_min'])

    # 3) Kernel
    k = GapfillKernel(sc['entry_thresh'], sc['exit_thresh'], sc['recent_min'], sc['sl_pct'])
    if 'kernel' in pstate:
        k.from_dict(pstate['kernel'])
    target_s = k.step(px, fair, age)

    log(f"ts={datetime.fromtimestamp(ref_ts/1000, timezone.utc):%Y-%m-%d %H:%M} "
        f"px={px:.4f} fair={fair:.4f} dev={dev*100:+.3f}% age={age:.0f}min closed={closed} -> target_s={target_s}")

    # 4) Exécution (réconciliation position réelle)
    cur = await client.fetch_position(mid)
    desired = target_s * notional / px
    delta = desired - cur
    min_base = client.market_info[mid]['min_base']

    if abs(delta) < max(min_base, 1e-9):
        log(f"position OK (cur={cur:.6f} desired={desired:.6f}, delta<min) -> rien à faire")
    elif DRY_RUN:
        log(f"[DRY_RUN] ORDRE simulé: cur={cur:.6f} desired={desired:.6f} delta={delta:+.6f} "
            f"side={'buy' if delta>0 else 'sell'} reduce_only={target_s==0}")
    else:
        side = 'buy' if delta > 0 else 'sell'
        try:
            await client.market_order(mid, side, abs(delta), px, cap_slip, reduce_only=(target_s == 0))
            log(f"ORDRE envoyé: {side} {abs(delta):.6f} {sym} @~{px:.4f} (reduce_only={target_s==0})")
        except Exception as e:
            log(f"ERREUR ordre: {e}")
            return  # on ne persiste pas l'état si l'ordre a échoué

    # 5) Persister
    pstate['kernel'] = k.to_dict()
    pstate['last_bar_ts'] = ref_ts
    pstate['last_target_s'] = target_s
    pstate['last_px'] = px
    pstate['last_fair'] = fair
    pstate['last_dev'] = dev
    pstate['last_update_utc'] = datetime.now(timezone.utc).isoformat()


async def main():
    _t0 = time.time()
    with open(secret_file) as f:
        secret = json.load(f)
    with open(config_file) as f:
        config = json.load(f)
    lcfg = dict(secret['lighter'])
    if 'account_index' in secret:
        lcfg['account_index'] = secret['account_index']

    client = LighterClient(lcfg)
    await client.initialize()
    state = load_state()

    for sc in config['strategies']:
        try:
            await client.set_leverage(sc['market_id'], int(sc.get('leverage', 1)))
            await process_pair(client, sc, state)
        except Exception as e:
            logging.error(f"[{sc.get('symbol')}] cycle erreur: {e}", exc_info=True)

    save_state(state)
    await client.close()
    logging.info(f"[MAIN] cycle terminé en {time.time()-_t0:.1f}s (DRY_RUN={DRY_RUN})")


if __name__ == '__main__':
    if not acquire_lock():
        print("Autre instance en cours. Exit."); sys.exit(0)
    asyncio.run(main())
