"""telechargement de données OHLCV
- exchanges classiques (binance etc) : via VBT/CCXT (import lourd)
- lighter : API directe + 100 proxys async, worker pool (import léger)
"""
import os
import sys
import time
import datetime


#=== CONFIG ===
PROXY_URL = os.environ.get("PROXY_URL", "")
LIGHTER_API = "https://mainnet.zklighter.elliot.ai/api/v1"
LIGHTER_START = 1737100800  #17 jan 2025 en timestamp secondes
MAX_CANDLES = 500  #max par requete lighter
RATE_LIMIT = 1.0  #1 req/s par proxy


#================================================================
# LIGHTER - worker pool async (1 worker = 1 proxy = 1 req/s)
#================================================================

def load_proxies():
    """telecharge la liste de proxys depuis webshare"""
    import urllib.request
    resp = urllib.request.urlopen(PROXY_URL)
    lines = resp.read().decode().strip().split('\n')
    proxies = []
    for line in lines:
        ip, port, user, pw = line.strip().split(':')
        proxies.append(f'http://{user}:{pw}@{ip}:{port}')
    print(f'{len(proxies)} proxys chargés')
    return proxies


def tf_to_seconds(tf):
    """convertit un timeframe en secondes (1m=60, 5m=300, 1h=3600 etc)"""
    units = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
    return int(tf[:-1]) * units[tf[-1]]


def _print_progress(progress, msg=''):
    """affiche la barre de progression globale"""
    done = progress['done']
    total = progress['total']
    pct = done / total * 100 if total > 0 else 0
    elapsed = time.time() - progress['start']

    if done > 0 and done < total:
        eta = elapsed / done * (total - done)
        eta_str = f'ETA {int(eta)}s'
    elif done >= total:
        eta_str = f'total {int(elapsed)}s'
    else:
        eta_str = '...'

    bar_len = 30
    filled = int(bar_len * done / total) if total > 0 else 0
    bar = '█' * filled + '░' * (bar_len - filled)

    print(f'\r  [{bar}] {done}/{total} ({pct:.0f}%) {eta_str} | {msg:<50}', end='', flush=True)
    if done >= total:
        print()


async def download_lighter_all(tf='1h', symbols=None):
    """telecharge TOUTES les paires futures de lighter en async
    architecture : worker pool, chaque worker a son proxy et fait 1 req/s
    une queue globale de jobs (market_id, chunk_start, chunk_end) est consommée par les workers"""
    import asyncio
    import aiohttp
    import pandas as pd

    #recup la liste des marchés perp
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{LIGHTER_API}/orderBooks') as resp:
            data = await resp.json()

    markets = [(ob['market_id'], ob['symbol'])
               for ob in data['order_books']
               if ob.get('market_type') == 'perp'
               and (symbols is None or ob['symbol'] in symbols)]
    print(f'{len(markets)} marchés perp trouvés sur Lighter')

    output_dir = f'data/raw/lighter/{tf}'
    os.makedirs(output_dir, exist_ok=True)
    proxies = load_proxies()

    tf_secs = tf_to_seconds(tf)
    end_ts = int(time.time())
    chunk_size = MAX_CANDLES * tf_secs

    #preparer tous les jobs : (market_id, symbol, chunk_idx, start, end)
    #et les structures pour stocker les resultats
    market_chunks = {}  #symbol -> {total_chunks, existing_df, candles: [None]*n}
    jobs = []

    for market_id, symbol in markets:
        csv_path = f'{output_dir}/{symbol}.csv'
        start_ts = LIGHTER_START
        existing_df = None

        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                existing_df = pd.read_csv(csv_path)
                if len(existing_df) > 0:
                    last_ts = int(existing_df['date'].iloc[-1]) // 1000
                    start_ts = last_ts + tf_secs
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                existing_df = None

        if start_ts >= end_ts:
            market_chunks[symbol] = {'skip': True}
            continue

        chunks = []
        t = start_ts
        while t < end_ts:
            chunk_end = min(t + chunk_size, end_ts)
            chunks.append((t, chunk_end))
            t = chunk_end

        market_chunks[symbol] = {
            'skip': False,
            'existing_df': existing_df,
            'candles': [None] * len(chunks),
            'total': len(chunks),
        }

        for idx, (cs, ce) in enumerate(chunks):
            jobs.append((market_id, symbol, idx, cs, ce))

    n_skipped = sum(1 for v in market_chunks.values() if v.get('skip'))
    n_to_dl = len(markets) - n_skipped
    print(f'{n_to_dl} paires a télécharger, {n_skipped} deja a jour, {len(jobs)} requetes a faire')

    if not jobs:
        print('Rien a faire.')
        return

    #progress : on track les paires terminées (pas les requetes)
    progress = {
        'done': 0, 'total': n_to_dl,
        'start': time.time(),
    }
    #compteur de chunks done par symbol pour savoir quand sauvegarder
    chunks_done = {s: 0 for s in market_chunks if not market_chunks[s].get('skip')}
    total_candles = 0
    lock = asyncio.Lock()

    def _save_symbol(symbol):
        """assemble et sauvegarde un symbol dès que tous ses chunks sont finis"""
        nonlocal total_candles
        info = market_chunks[symbol]
        rows = []
        for chunk_candles in info['candles']:
            if chunk_candles:
                for c in chunk_candles:
                    rows.append({
                        'date': c['t'],
                        'open': c['o'],
                        'high': c['h'],
                        'low': c['l'],
                        'close': c['c'],
                        'volume': c['v'],
                    })

        if not rows:
            return

        new_df = pd.DataFrame(rows)
        new_df = new_df.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)

        existing_df = info.get('existing_df')
        if existing_df is not None and len(existing_df) > 0:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
        else:
            combined = new_df

        csv_path = f'{output_dir}/{symbol}.csv'
        combined.to_csv(csv_path, index=False)
        total_candles += len(new_df)

    #queue de jobs
    queue = asyncio.Queue()
    for job in jobs:
        await queue.put(job)

    async def worker(proxy, session):
        """1 worker = 1 proxy, consomme la queue, 1 req/s"""
        while True:
            try:
                market_id, symbol, chunk_idx, start, end = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            url = f'{LIGHTER_API}/candles'
            params = {
                'market_id': market_id,
                'resolution': tf,
                'start_timestamp': start,
                'end_timestamp': end,
                'count_back': MAX_CANDLES,
            }

            candles = []
            for attempt in range(5):
                try:
                    async with session.get(url, params=params, proxy=proxy,
                                           timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 429:
                            await asyncio.sleep(2 + attempt * 2)
                            continue
                        data = await resp.json()
                        if data.get('code') == 200:
                            candles = data.get('c', [])
                        break
                except Exception:
                    await asyncio.sleep(1 + attempt)

            market_chunks[symbol]['candles'][chunk_idx] = candles

            #checker si tous les chunks de ce symbol sont finis
            async with lock:
                chunks_done[symbol] += 1
                if chunks_done[symbol] >= market_chunks[symbol]['total']:
                    #tous les chunks sont la, sauvegarder immediatement
                    _save_symbol(symbol)
                    progress['done'] += 1
                    _print_progress(progress, f'{symbol} sauvegardé')

            await asyncio.sleep(RATE_LIMIT)

    #lancer les workers (1 par proxy)
    connector = aiohttp.TCPConnector(limit=len(proxies), ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [worker(proxy, session) for proxy in proxies]
        await asyncio.gather(*workers)

    elapsed = time.time() - progress['start']
    print(f'\nTerminé : {n_to_dl} paires, {total_candles} candles, {n_skipped} deja a jour ({int(elapsed)}s)')


#================================================================
# LIGHTER FUNDING - même architecture worker pool / proxys
#================================================================

async def download_lighter_funding_all(symbols=None):
    """telecharge l'historique funding 1h pour TOUTES les paires perp Lighter.
    Architecture identique au download candles : pool de workers, 1 par proxy.
    Pagination dynamique : page N+1 enfilée par le worker si page N revient pleine,
    stop quand page courte (< COUNT_BACK) ou vide reçue. Cap safety à MAX_PAGES.
    """
    import asyncio
    import aiohttp
    import pandas as pd

    #recup la liste des marchés perp
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{LIGHTER_API}/orderBooks') as resp:
            data = await resp.json()

    markets = [(ob['market_id'], ob['symbol'])
               for ob in data['order_books']
               if ob.get('market_type') == 'perp'
               and (symbols is None or ob['symbol'] in symbols)]
    print(f'{len(markets)} marchés perp trouvés sur Lighter')

    output_dir = 'data/raw/lighter/funding'
    os.makedirs(output_dir, exist_ok=True)
    proxies = load_proxies()

    #pagination jusqu'a epuisement (cap safety MAX_PAGES = ~2.5 ans)
    COUNT_BACK = 750
    MAX_PAGES = 30

    #stockage : symbol -> {existing_df, pages: dict{idx: rows}, is_complete, enqueued, received}
    market_state = {}
    jobs = []

    for market_id, symbol in markets:
        csv_path = f'{output_dir}/{symbol}.csv'
        existing_df = None
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                existing_df = pd.read_csv(csv_path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                existing_df = None

        market_state[symbol] = {
            'existing_df': existing_df,
            'pages': {},          # {page_idx: list_of_funding_rows}
            'is_complete': False, # True quand page courte/vide recue
            'enqueued': 1,        # nb de pages enfilees (page 0 a la creation)
            'received': 0,        # nb de pages traitees
            'saved': False,       # eviter double-save
            'market_id': market_id,
        }
        #page 0 enfilee d'abord ; pages suivantes enfilees par le worker
        jobs.append((market_id, symbol, 0, int(time.time())))

    total_pairs = len(markets)
    print(f'{total_pairs} paires à télécharger (funding 1h, jusqu\'à {MAX_PAGES} pages = ~2.5 ans)')

    progress = {'done': 0, 'total': total_pairs, 'start': time.time()}
    total_rows = 0
    lock = asyncio.Lock()

    def _save_symbol(symbol):
        """assemble + sauvegarde (append-only)"""
        nonlocal total_rows
        info = market_state[symbol]
        if info['saved']:
            return
        info['saved'] = True
        rows = []
        for page_idx in sorted(info['pages'].keys()):
            rows.extend(info['pages'][page_idx])
        if not rows:
            return
        new_df = pd.DataFrame(rows)
        new_df['timestamp'] = new_df['timestamp'].astype(int)
        new_df['rate'] = pd.to_numeric(new_df['rate'], errors='coerce')
        new_df['value'] = pd.to_numeric(new_df['value'], errors='coerce')
        new_df = new_df.dropna(subset=['rate'])
        new_df['signed_rate'] = new_df['rate'] * new_df['direction'].map({'long': +1, 'short': -1})
        new_df['apr'] = new_df['signed_rate'] * 365 * 24 * 100
        new_df = new_df[['timestamp', 'rate', 'value', 'direction', 'signed_rate', 'apr']]
        new_df = new_df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

        existing_df = info.get('existing_df')
        if existing_df is not None and len(existing_df) > 0:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
            n_new = len(combined) - len(existing_df)
        else:
            combined = new_df
            n_new = len(combined)

        csv_path = f'{output_dir}/{symbol}.csv'
        combined.to_csv(csv_path, index=False)
        total_rows += n_new

    queue = asyncio.Queue()
    for job in jobs:
        await queue.put(job)

    async def worker(proxy, session):
        while True:
            #pagination dynamique : on peut avoir la queue temporairement vide
            #alors que des workers traitent des pages dont les suivantes arriveront bientot.
            #stop quand TOUTES les paires sont marquees done (progress['done'] == total_pairs).
            try:
                market_id, symbol, page_idx, end_ts = queue.get_nowait()
            except asyncio.QueueEmpty:
                if progress['done'] >= total_pairs:
                    break
                await asyncio.sleep(0.2)
                continue

            url = f'{LIGHTER_API}/fundings'
            params = {
                'market_id': market_id,
                'resolution': '1h',
                'start_timestamp': 0,
                'end_timestamp': end_ts,
                'count_back': COUNT_BACK,
            }

            fundings = []
            for attempt in range(5):
                try:
                    async with session.get(url, params=params, proxy=proxy,
                                           timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 429:
                            await asyncio.sleep(2 + attempt * 2)
                            continue
                        data = await resp.json()
                        if isinstance(data, dict) and data.get('code') == 23000:
                            await asyncio.sleep(2 + attempt * 2)
                            continue
                        if isinstance(data, dict):
                            fundings = data.get('fundings', [])
                        break
                except Exception:
                    await asyncio.sleep(1 + attempt)

            async with lock:
                state = market_state[symbol]
                state['pages'][page_idx] = fundings
                state['received'] += 1

                #page pleine ET pas encore au cap ET pas deja flagge complete : enfile la suivante
                #NB: l'API renvoie souvent 749 quand on demande 750 (off-by-one), seuil large pour ne pas
                #s'arreter par erreur. Une vraie fin d'historique renvoie << 700 (BTC: 371 a la limite).
                page_full = len(fundings) >= COUNT_BACK - 100  # 650+ = considere comme plein
                can_continue = (page_idx + 1 < MAX_PAGES) and not state['is_complete']
                if page_full and can_continue:
                    oldest = min(int(f['timestamp']) for f in fundings)
                    next_end = oldest - 1
                    state['enqueued'] += 1
                    await queue.put((market_id, symbol, page_idx + 1, next_end))
                else:
                    #page courte/vide ou cap atteint : fin d'historique pour cette paire
                    state['is_complete'] = True

                #save quand toutes les pages enfilees ont ete recues ET on est en mode complete
                if state['is_complete'] and state['received'] >= state['enqueued']:
                    _save_symbol(symbol)
                    if not state.get('counted'):
                        progress['done'] += 1
                        state['counted'] = True
                        _print_progress(progress, f'{symbol} sauvegardé ({state["enqueued"]} pages)')

            await asyncio.sleep(RATE_LIMIT)

    connector = aiohttp.TCPConnector(limit=len(proxies), ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [worker(proxy, session) for proxy in proxies]
        await asyncio.gather(*workers)

    elapsed = time.time() - progress['start']
    print(f'\nTerminé : {progress["done"]}/{total_pairs} paires, {total_rows} nouvelles lignes ({int(elapsed)}s)')
    print(f'Output : {output_dir}/')


#================================================================
# HYPERLIQUID FUNDING - POST API, séquentiel par coin
# Doc: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals
# fundingHistory: signed rate per hour, frac (e.g. 0.0000125 = 0.00125%/h)
#================================================================

HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"


async def download_hyperliquid_funding_all(symbols=None):
    """telecharge l'historique funding 1h pour TOUS les perps Hyperliquid.

    Architecture : worker pool aiohttp, 1 worker par proxy (parallèle).
    Pagination en arrière de 20j par call (~480 records / call, HL cap autour de 500).

    Convention CSV (alignée Lighter `data/raw/lighter/funding/`) :
      timestamp(sec), rate(%/h × 100 pour matcher Lighter scale), premium,
      signed_rate(= rate déjà signé sur HL), apr(= signed_rate × 24 × 365 × 100)
    Le ×100 sur rate est volontaire : HL expose fundingRate en frac/h, Lighter
    en %/h (= frac/h × 100). On normalise sur convention Lighter pour que le
    z-score consensus cross-exchange soit calculable directement.
    """
    import asyncio
    import aiohttp
    import pandas as pd

    output_dir = 'data/raw/hyperliquid/funding'
    os.makedirs(output_dir, exist_ok=True)

    # 1) Récup univers via meta (1 call, no proxy needed)
    async with aiohttp.ClientSession() as session:
        async with session.post(HYPERLIQUID_API, json={"type": "meta"}) as resp:
            meta = await resp.json()
    universe = meta.get('universe', [])
    coins = [u['name'] for u in universe if not u.get('isDelisted', False)]
    if symbols:
        coins = [c for c in coins if c in symbols]
    print(f'{len(coins)} coins HL trouvés (perp, non-delisted)')

    # 2) Load proxys (fallback séquentiel si indispo)
    try:
        proxies = load_proxies()
    except Exception as e:
        print(f'Proxys indisponibles ({type(e).__name__}: {e}), fallback séquentiel')
        proxies = [None]

    PAGE_DAYS = 20      # ~480 records 1h par call (HL cap empirique ~500)
    MAX_PAGES = 60      # ~3.3 ans cap safety
    PAGE_SLEEP = 0.1    # entre calls par worker

    # 3) Queue de coins + worker pool
    queue = asyncio.Queue()
    for coin in coins:
        await queue.put(coin)

    progress = {'done': 0, 'total': len(coins), 'rows': 0, 'start': time.time()}
    lock = asyncio.Lock()

    async def worker(proxy, session):
        while True:
            try:
                coin = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            csv_path = f'{output_dir}/{coin}.csv'
            existing_df = None
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                try:
                    existing_df = pd.read_csv(csv_path)
                except (pd.errors.EmptyDataError, pd.errors.ParserError):
                    existing_df = None

            end_ts_ms = int(time.time() * 1000)
            all_rows = []

            for page in range(MAX_PAGES):
                start_ts_ms = end_ts_ms - PAGE_DAYS * 86400 * 1000
                payload = {
                    "type": "fundingHistory",
                    "coin": coin,
                    "startTime": start_ts_ms,
                    "endTime": end_ts_ms,
                }
                data = None
                for attempt in range(5):
                    try:
                        kwargs = {'timeout': aiohttp.ClientTimeout(total=30)}
                        if proxy:
                            kwargs['proxy'] = proxy
                        async with session.post(HYPERLIQUID_API, json=payload, **kwargs) as resp:
                            if resp.status == 429:
                                await asyncio.sleep(2 + attempt * 2)
                                continue
                            data = await resp.json()
                            break
                    except Exception:
                        await asyncio.sleep(1 + attempt)

                if data is None or not isinstance(data, list) or not data:
                    break
                all_rows.extend(data)
                if len(data) < PAGE_DAYS * 24 - 100:
                    break
                end_ts_ms = min(int(r['time']) for r in data) - 1
                await asyncio.sleep(PAGE_SLEEP)

            if not all_rows:
                async with lock:
                    progress['done'] += 1
                continue

            new_df = pd.DataFrame(all_rows)
            new_df['timestamp'] = (new_df['time'].astype('int64') // 1000).astype(int)
            new_df['rate'] = pd.to_numeric(new_df['fundingRate'], errors='coerce') * 100.0
            new_df['premium'] = pd.to_numeric(new_df['premium'], errors='coerce')
            new_df = new_df.dropna(subset=['rate'])
            new_df['signed_rate'] = new_df['rate']
            new_df['apr'] = new_df['signed_rate'] * 365 * 24 * 100
            new_df = new_df[['timestamp', 'rate', 'premium', 'signed_rate', 'apr']]
            new_df = new_df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

            if existing_df is not None and len(existing_df) > 0:
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
                n_new = len(combined) - len(existing_df)
            else:
                combined = new_df
                n_new = len(combined)

            combined.to_csv(csv_path, index=False)

            async with lock:
                progress['done'] += 1
                progress['rows'] += n_new
                elapsed = int(time.time() - progress['start'])
                eta = elapsed / progress['done'] * (progress['total'] - progress['done']) if progress['done'] > 0 else 0
                print(f"  [{progress['done']}/{progress['total']}] {coin}: +{n_new} rows "
                      f"(total {len(combined)}) | {elapsed}s elapsed, eta {int(eta)}s")

    # 4) Lancement workers (1 par proxy)
    n_workers = len(proxies)
    connector = aiohttp.TCPConnector(limit=n_workers, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        workers_tasks = [worker(p, session) for p in proxies]
        await asyncio.gather(*workers_tasks)

    elapsed = int(time.time() - progress['start'])
    print(f'\nTerminé HL : {progress["done"]}/{len(coins)} coins, '
          f'{progress["rows"]} nouvelles lignes ({elapsed}s, {n_workers} workers)')
    print(f'Output : {output_dir}/')


#================================================================
# HYPERLIQUID OHLCV — POST API, candleSnapshot, paginé par chunks de 5000
# Doc: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint
# Format réponse: [{"t":ms, "T":ms, "s":coin, "i":tf, "o":, "c":, "h":, "l":, "v":, "n":}, ...]
# Rate limit: weight = 20 + items/60 ≈ 103/call à 5000 candles → ~11 calls/min/IP
# Avec 100 proxys : ~1100 calls/min → 200 coins × 3 pages 1h = 600 calls = 30-60s
#================================================================


async def download_hyperliquid_candles_all(tf: str = '1h', symbols=None):
    """telecharge l'historique OHLCV pour TOUS les perps Hyperliquid en async.

    Architecture : worker pool aiohttp, 1 worker par proxy.
    Pagination en arrière par chunks de PAGE_SIZE candles (HL cap empirique = 5000).
    Stop quand page courte (< 80% PAGE_SIZE) ou vide reçue.

    LIMITATION HL : l'API ne sert QUE les ~5000 candles les plus récentes,
    quel que soit le startTime demandé. Plages plus anciennes → réponse vide.
    Couverture effective par timeframe (5000 candles) :
      1m → 3.5 jours      5m → 17 jours       15m → 52 jours
      1h → 208 jours      4h → 833 jours      1d → 13.7 ans

    Convention CSV (alignée Lighter `data/raw/lighter/<tf>/`) :
      date(ms epoch), open, high, low, close, volume
    """
    import asyncio
    import aiohttp
    import pandas as pd

    tf_seconds = tf_to_seconds(tf)
    PAGE_SIZE = 5000        # HL cap empirique
    PAGE_SLEEP = 0.05       # entre calls par worker (rate-limit weight)
    MAX_PAGES = 30          # ~3 ans cap safety pour 1h

    output_dir = f'data/raw/hyperliquid/{tf}'
    os.makedirs(output_dir, exist_ok=True)

    # 1) Univers via meta (1 call sans proxy)
    async with aiohttp.ClientSession() as session:
        async with session.post(HYPERLIQUID_API, json={"type": "meta"}) as resp:
            meta = await resp.json()
    universe = meta.get('universe', [])
    coins = [u['name'] for u in universe if not u.get('isDelisted', False)]
    if symbols:
        coins = [c for c in coins if c in symbols]
    print(f'{len(coins)} coins HL trouvés (perp, non-delisted)')

    # 2) Proxys (fallback séquentiel si indispo)
    try:
        proxies = load_proxies()
    except Exception as e:
        print(f'Proxys indisponibles ({type(e).__name__}: {e}) — fallback séquentiel')
        proxies = [None]

    # 3) Queue de coins + worker pool
    queue = asyncio.Queue()
    for coin in coins:
        await queue.put(coin)

    progress = {'done': 0, 'total': len(coins), 'rows': 0, 'start': time.time()}
    lock = asyncio.Lock()

    async def worker(proxy, session):
        while True:
            try:
                coin = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            csv_path = f'{output_dir}/{coin}.csv'
            existing_df = None
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                try:
                    existing_df = pd.read_csv(csv_path)
                except (pd.errors.EmptyDataError, pd.errors.ParserError):
                    existing_df = None

            end_ts_ms = int(time.time() * 1000)
            # Si data existante, on stop dès qu'on dépasse le dernier timestamp connu
            existing_last_ms = None
            if existing_df is not None and len(existing_df) > 0 and 'date' in existing_df.columns:
                try:
                    existing_last_ms = int(existing_df['date'].max())
                except Exception:
                    existing_last_ms = None

            all_rows = []
            for page in range(MAX_PAGES):
                # Demande une fenêtre large : HL renverra max PAGE_SIZE candles depuis endTime
                start_ts_ms = max(0, end_ts_ms - PAGE_SIZE * tf_seconds * 1000)
                payload = {
                    "type": "candleSnapshot",
                    "req": {
                        "coin": coin,
                        "interval": tf,
                        "startTime": start_ts_ms,
                        "endTime": end_ts_ms,
                    }
                }
                data = None
                for attempt in range(5):
                    try:
                        kwargs = {'timeout': aiohttp.ClientTimeout(total=30)}
                        if proxy:
                            kwargs['proxy'] = proxy
                        async with session.post(HYPERLIQUID_API, json=payload, **kwargs) as resp:
                            if resp.status == 429:
                                await asyncio.sleep(2 + attempt * 2)
                                continue
                            data = await resp.json()
                            break
                    except Exception:
                        await asyncio.sleep(1 + attempt)

                if data is None or not isinstance(data, list) or not data:
                    break
                all_rows.extend(data)
                # Fin d'historique : page < 80% du cap
                if len(data) < int(PAGE_SIZE * 0.8):
                    break
                oldest = min(int(r['t']) for r in data)
                # Si on a déjà tout cet historique en CSV, stop early
                if existing_last_ms is not None and oldest <= existing_last_ms:
                    break
                end_ts_ms = oldest - 1
                await asyncio.sleep(PAGE_SLEEP)

            if not all_rows:
                async with lock:
                    progress['done'] += 1
                continue

            new_df = pd.DataFrame(all_rows)
            new_df['date'] = new_df['t'].astype('int64')
            for c in ('o', 'h', 'l', 'c', 'v'):
                new_df[c] = pd.to_numeric(new_df[c], errors='coerce')
            new_df = new_df.rename(columns={
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume',
            })
            new_df = new_df.dropna(subset=['open', 'high', 'low', 'close'])
            new_df = new_df[['date', 'open', 'high', 'low', 'close', 'volume']]
            new_df = new_df.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)

            if existing_df is not None and len(existing_df) > 0:
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
                n_new = len(combined) - len(existing_df)
            else:
                combined = new_df
                n_new = len(combined)

            combined.to_csv(csv_path, index=False)

            async with lock:
                progress['done'] += 1
                progress['rows'] += n_new
                elapsed = int(time.time() - progress['start'])
                eta = elapsed / progress['done'] * (progress['total'] - progress['done']) if progress['done'] > 0 else 0
                print(f"  [{progress['done']}/{progress['total']}] {coin}: +{n_new} rows "
                      f"(total {len(combined)}) | {elapsed}s elapsed, eta {int(eta)}s")

    # 4) Lancement workers (1 par proxy)
    n_workers = len(proxies)
    connector = aiohttp.TCPConnector(limit=n_workers, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        workers_tasks = [worker(p, session) for p in proxies]
        await asyncio.gather(*workers_tasks)

    elapsed = int(time.time() - progress['start'])
    print(f'\nTerminé HL candles {tf} : {progress["done"]}/{len(coins)} coins, '
          f'{progress["rows"]} nouvelles lignes ({elapsed}s, {n_workers} workers)')
    print(f'Output : {output_dir}/')


#================================================================
# SNAPSHOT BID-ASK SPREAD — Lighter + Hyperliquid
# N rounds espacés d'interval_s sec, sampling top-of-book parallèle sur les 2
# venues. Output : stats agrégées (median/p25/p75/n) par paire × venue en bps.
#================================================================

async def snapshot_spreads_all(n_rounds: int = 10, interval_s: int = 30,
                                common_pairs_path: str = 'data/raw/common_pairs.json'):
    """Snapshot du spread bid-ask top-of-book sur Lighter et Hyperliquid.

    Architecture :
      - N rounds espacés d'interval_s sec (défaut 10 × 30s = 5min)
      - Par round : query parallèle de toutes les paires Lighter ∩ HL sur les 2 venues
      - Extract bid/ask top-of-book → half-spread bps = (ask-bid)/2/mid * 1e4
      - Agrégat final : mean/median/p25/p75 par paire × venue

    Output : data/raw/spread_snapshot_<UTC_YYYY-MM-DD_HH-MM>.json
    """
    import asyncio
    import aiohttp
    import json
    import numpy as np
    from collections import defaultdict
    from datetime import datetime

    with open(common_pairs_path) as f:
        common = json.load(f)

    # Paires Lighter ∩ HL avec market_id valide
    pairs = {k: v for k, v in common.items()
             if v.get('hyperliquid') and v.get('lighter_market_id') is not None}
    print(f'{len(pairs)} paires Lighter ∩ HL à snapshoter')
    print(f'Plan : {n_rounds} rounds × {interval_s}s = {n_rounds * interval_s}s total')

    samples = defaultdict(lambda: {'lighter': [], 'hyperliquid': []})

    # Charge la pool de proxys (1 IP par paire pour bypass rate-limit Lighter).
    # HL ne rate-limite pas → connexion directe est OK.
    try:
        proxies = load_proxies()
    except Exception as e:
        print(f'Proxys indisponibles ({type(e).__name__}: {e}) — fallback no-proxy (rate-limit attendu)')
        proxies = [None] * 100

    # Assignation déterministe sym → proxy (idx mod len), même proxy à chaque round
    sym_list = list(pairs.keys())
    sym_proxy = {sym: proxies[i % len(proxies)] for i, sym in enumerate(sym_list)}

    async def snap_lighter(session, mid, sym, proxy):
        url = f'{LIGHTER_API}/orderBookOrders'
        params = {'market_id': mid, 'limit': 1}
        kwargs = {'timeout': aiohttp.ClientTimeout(total=10)}
        if proxy:
            kwargs['proxy'] = proxy
        try:
            async with session.get(url, params=params, **kwargs) as resp:
                data = await resp.json()
                if data.get('total_bids', 0) > 0 and data.get('total_asks', 0) > 0:
                    bid = float(data['bids'][0]['price'])
                    ask = float(data['asks'][0]['price'])
                    mid_px = (bid + ask) / 2.0
                    if mid_px > 0 and ask >= bid:
                        return sym, (ask - bid) / 2.0 / mid_px * 1e4
        except Exception:
            pass
        return sym, None

    async def snap_hl(session, coin, sym):
        try:
            async with session.post(HYPERLIQUID_API,
                                    json={"type": "l2Book", "coin": coin},
                                    timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                lvls = data.get('levels') if isinstance(data, dict) else None
                if lvls and len(lvls) == 2 and lvls[0] and lvls[1]:
                    bid = float(lvls[0][0]['px'])
                    ask = float(lvls[1][0]['px'])
                    mid_px = (bid + ask) / 2.0
                    if mid_px > 0 and ask >= bid:
                        return sym, (ask - bid) / 2.0 / mid_px * 1e4
        except Exception:
            pass
        return sym, None

    connector = aiohttp.TCPConnector(limit=300, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        for r in range(n_rounds):
            t0 = time.time()

            tasks_l = [snap_lighter(session, pairs[sym]['lighter_market_id'], sym, sym_proxy[sym])
                       for sym in sym_list]
            tasks_h = [snap_hl(session, pairs[sym]['hyperliquid'], sym)
                       for sym in sym_list]

            res_l, res_h = await asyncio.gather(
                asyncio.gather(*tasks_l),
                asyncio.gather(*tasks_h),
            )

            for sym, bps in res_l:
                if bps is not None:
                    samples[sym]['lighter'].append(bps)
            for sym, bps in res_h:
                if bps is not None:
                    samples[sym]['hyperliquid'].append(bps)

            ok_l = sum(1 for _, bps in res_l if bps is not None)
            ok_h = sum(1 for _, bps in res_h if bps is not None)
            elapsed = time.time() - t0
            print(f'Round {r+1}/{n_rounds}: Lighter {ok_l}/{len(tasks_l)} ok, '
                  f'HL {ok_h}/{len(tasks_h)} ok ({elapsed:.1f}s)')

            if r < n_rounds - 1:
                sleep_left = max(0.0, interval_s - elapsed)
                if sleep_left > 0:
                    await asyncio.sleep(sleep_left)

    # Agrégat
    out = {}
    for sym in sorted(pairs.keys()):
        info = pairs[sym]
        entry = {
            'lighter_symbol': info['lighter'],
            'lighter_market_id': info['lighter_market_id'],
            'hyperliquid_symbol': info['hyperliquid'],
        }
        for venue in ('lighter', 'hyperliquid'):
            arr = np.array(samples[sym][venue], dtype=float)
            if len(arr) > 0:
                entry[f'{venue}_half_spread_bps'] = {
                    'n': int(len(arr)),
                    'mean': float(arr.mean()),
                    'median': float(np.median(arr)),
                    'p25': float(np.percentile(arr, 25)),
                    'p75': float(np.percentile(arr, 75)),
                    'min': float(arr.min()),
                    'max': float(arr.max()),
                }
            else:
                entry[f'{venue}_half_spread_bps'] = None
        out[sym] = entry

    stamp = datetime.utcnow().strftime('%Y-%m-%d_%H-%M')
    out_path = f'data/raw/spread_snapshot_{stamp}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'snapshot_meta': {
                'n_rounds': n_rounds,
                'interval_s': interval_s,
                'n_pairs': len(pairs),
                'utc_start': stamp,
            },
            'pairs': out,
        }, f, indent=2)
    print(f'\nSaved: {out_path}')

    # Tableau récap (median half-spread, trié croissant côté Lighter)
    print('\n=== Median half-spread (bps) — top 30 plus serrés Lighter ===')
    print(f'{"Symbol":<14} {"Lighter":>10} {"HL":>10}  ratio L/HL')
    lines = []
    for sym, entry in out.items():
        l = entry.get('lighter_half_spread_bps')
        h = entry.get('hyperliquid_half_spread_bps')
        if l and h:
            ratio = l['median'] / h['median'] if h['median'] > 0 else float('nan')
            lines.append((sym, l['median'], h['median'], ratio))
    lines.sort(key=lambda x: x[1])
    for sym, l_med, h_med, ratio in lines[:30]:
        print(f'{sym:<14} {l_med:>10.2f} {h_med:>10.2f}  {ratio:>5.2f}x')
    print(f'... ({len(lines)} paires totales avec données sur les 2 venues)')

    return out


#================================================================
# CROSS-EXCHANGE PAIR MAPPING
# Identifie les paires communes Lighter ∩ HL ∩ Binance ∩ Bybit.
# Gère les conventions de naming :
#   - Lighter "BTC", HL "BTC", Binance "BTCUSDT", Bybit "BTCUSDT"
#   - Lighter "1000PEPE", HL "kPEPE", Binance "1000PEPEUSDT", Bybit "1000PEPEUSDT"
#   - Lighter "HYPE" (natif), HL "HYPE" (natif), pas sur Binance/Bybit
#================================================================

def _normalize_symbol(sym: str) -> str:
    """Normalise un symbole vers une clé commune.

    Règles :
      - strip USDT/USDC/USD suffix (Binance/Bybit perp)
      - HL prefix 'k' (kilo, ex kPEPE) → '1000' prefix (Lighter/Binance convention)
      - uppercase
    """
    s = sym.upper()
    # Strip perp suffix
    for suf in ('USDT', 'USDC', 'USD', 'PERP'):
        if s.endswith(suf) and len(s) > len(suf):
            s = s[:-len(suf)]
            break
    # HL k-prefix → 1000-prefix (kPEPE → 1000PEPE)
    # k is lowercase in HL original; after .upper() it becomes K. Detect by original sym.
    if sym.startswith('k') and len(sym) > 1 and sym[1].isupper():
        s = '1000' + sym[1:].upper()
    return s


def find_common_pairs(save: bool = True, verbose: bool = True):
    """Identifie les paires communes entre Lighter, Hyperliquid, Binance UM, Bybit linear.

    Returns dict {normalized_symbol: {lighter, hyperliquid, binance, bybit, market_ids...}}
    Sauvegarde dans data/raw/common_pairs.json par défaut.
    """
    import json
    import urllib.request
    import urllib.error

    def _get_json(url, post_body=None):
        try:
            if post_body is not None:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(post_body).encode(),
                    headers={'Content-Type': 'application/json'},
                    method='POST',
                )
            else:
                req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as e:
            print(f'  [WARN] {url}: {type(e).__name__}: {e}')
            return None

    # 1) Lighter perp universe
    lighter_data = _get_json(f"{LIGHTER_API}/orderBooks")
    lighter_pairs = {}
    if lighter_data:
        for ob in lighter_data.get('order_books', []):
            if ob.get('market_type') == 'perp':
                lighter_pairs[ob['symbol']] = {'market_id': ob['market_id']}
    if verbose:
        print(f'Lighter perps     : {len(lighter_pairs)}')

    # 2) Hyperliquid universe
    hl_data = _get_json(HYPERLIQUID_API, post_body={"type": "meta"})
    hl_pairs = {}
    if hl_data:
        for u in hl_data.get('universe', []):
            if not u.get('isDelisted', False):
                hl_pairs[u['name']] = u
    if verbose:
        print(f'Hyperliquid perps : {len(hl_pairs)}')

    # 3) Binance UM (perpetual) — exchangeInfo, gratuit, no auth
    binance_data = _get_json("https://fapi.binance.com/fapi/v1/exchangeInfo")
    binance_pairs = {}
    if binance_data:
        for s in binance_data.get('symbols', []):
            if s.get('contractType') == 'PERPETUAL' and s.get('status') == 'TRADING':
                binance_pairs[s['symbol']] = s
    if verbose:
        print(f'Binance UM perps  : {len(binance_pairs)}')

    # 4) Bybit linear (USDT perp) — instruments-info
    bybit_data = _get_json("https://api.bybit.com/v5/market/instruments-info?category=linear")
    bybit_pairs = {}
    if bybit_data and bybit_data.get('retCode') == 0:
        for s in bybit_data.get('result', {}).get('list', []):
            if s.get('status') == 'Trading' and s.get('contractType') == 'LinearPerpetual':
                bybit_pairs[s['symbol']] = s
    if verbose:
        print(f'Bybit linear perps: {len(bybit_pairs)}')

    # 5) Build normalized maps
    def _idx(pairs):
        out = {}
        for sym in pairs:
            key = _normalize_symbol(sym)
            # En cas de collision (rare), garde la 1ère
            out.setdefault(key, sym)
        return out

    lighter_idx = _idx(lighter_pairs)
    hl_idx = _idx(hl_pairs)
    binance_idx = _idx(binance_pairs)
    bybit_idx = _idx(bybit_pairs)

    # 6) Intersection : on garde toutes les clés qui sont sur Lighter ET (HL ou Binance ou Bybit)
    common = {}
    for key, lighter_sym in sorted(lighter_idx.items()):
        hl_sym = hl_idx.get(key)
        binance_sym = binance_idx.get(key)
        bybit_sym = bybit_idx.get(key)
        if not (hl_sym or binance_sym or bybit_sym):
            continue
        common[key] = {
            'lighter': lighter_sym,
            'lighter_market_id': lighter_pairs.get(lighter_sym, {}).get('market_id'),
            'hyperliquid': hl_sym,
            'binance': binance_sym,
            'bybit': bybit_sym,
        }

    n_all4   = sum(1 for v in common.values() if v['hyperliquid'] and v['binance'] and v['bybit'])
    n_hl     = sum(1 for v in common.values() if v['hyperliquid'])
    n_bin    = sum(1 for v in common.values() if v['binance'])
    n_byb    = sum(1 for v in common.values() if v['bybit'])
    n_lhl    = sum(1 for v in common.values() if v['hyperliquid'] and not v['binance'] and not v['bybit'])
    n_lonly  = len(lighter_idx) - len(common)
    if verbose:
        print(f'\n=== Paires communes ===')
        print(f'  Total Lighter ∩ (HL ∪ Binance ∪ Bybit): {len(common)}')
        print(f'  Sur les 4 venues simultanément        : {n_all4}')
        print(f'  Lighter ∩ HL                          : {n_hl}')
        print(f'  Lighter ∩ Binance                     : {n_bin}')
        print(f'  Lighter ∩ Bybit                       : {n_byb}')
        print(f'  Lighter ∩ HL seul (pas Binance/Bybit) : {n_lhl}')
        print(f'  Lighter uniquement (pas d\'équivalent) : {n_lonly}')

    if save:
        out = 'data/raw/common_pairs.json'
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            json.dump(common, f, indent=2)
        if verbose:
            print(f'\nSaved: {out}')

    return common


#================================================================
# CCXT/VBT - pour les exchanges classiques (binance, bybit etc)
#================================================================

def _save_symbol_vbt(data, symbol, output_dir):
    """sauvegarde un symbol vbt en pickle + csv"""
    from vectorbtpro import vbt

    csv_path = f'{output_dir}/{symbol}.csv'
    pickle_path = f'{output_dir}/{symbol}.pickle'

    sym_data = data.select(symbol) if len(data.symbols) > 1 else data
    vbt.save(sym_data, pickle_path)

    df = sym_data.get()
    df.columns = [c.lower() for c in df.columns]
    df.index.name = 'date'
    df.index = df.index.astype('int64') // 10**6
    df.to_csv(csv_path)
    print(f'  {symbol} -> {csv_path} ({len(df)} lignes)')


def download_symbols(symbols, start='2017-01-01', end=None,
                     timeframe='1h', exchange='binance'):
    """telecharge via VBT/CCXT (binance, bybit etc)"""
    from vectorbtpro import vbt

    if end is None:
        end = datetime.datetime.now().strftime('%Y-%m-%d')

    output_dir = f'data/raw/{exchange}/{timeframe}'
    os.makedirs(output_dir, exist_ok=True)

    to_update = []
    to_download = []
    for sym in symbols:
        pickle_path = f'{output_dir}/{sym}.pickle'
        if os.path.exists(pickle_path):
            to_update.append(sym)
        else:
            to_download.append(sym)

    for sym in to_update:
        pickle_path = f'{output_dir}/{sym}.pickle'
        print(f'Update {sym}...')
        data = vbt.load(pickle_path)
        data = data.update()
        _save_symbol_vbt(data, sym, output_dir)

    if to_download:
        print(f'Download {to_download} sur {exchange} ({start} -> {end}, {timeframe})...')
        data = vbt.CCXTData.pull(
            to_download,
            start=start, end=end,
            timeframe=timeframe, exchange=exchange,
            execute_kwargs=dict(engine='threadpool'),
        )
        for sym in to_download:
            _save_symbol_vbt(data, sym, output_dir)

    print('Done.')


#================================================================
# CLI
#================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage :")
        print("  python src/download.py --exchange lighter --tf 1h")
        print("  python src/download.py --exchange lighter --funding")
        print("  python src/download.py --exchange hyperliquid --funding [BTC ETH SOL ...]")
        print("  python src/download.py --exchange hyperliquid --tf 1h [BTC ETH SOL ...]")
        print("       (OHLCV HL ; sans symbols = toutes les paires perp)")
        print("  python src/download.py --exchange binance --tf 1h BTCUSDT ETHUSDT")
        print("  python src/download.py --tf 5m BTCUSDT  (binance par defaut)")
        print("  python src/download.py --common-pairs")
        print("       (identifie paires communes Lighter/HL/Binance/Bybit, sauve common_pairs.json)")
        print("  python src/download.py --spread-snapshot [--rounds N] [--interval S]")
        print("       (snapshot bid-ask Lighter+HL pour paires communes, sauve spread_snapshot_<ts>.json)")
        sys.exit(1)

    exchange = 'binance'
    timeframe = '1h'
    symbols = []
    mode = 'candles'
    snapshot_rounds = 10
    snapshot_interval = 30

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--exchange':
            exchange = args[i + 1]
            i += 2
        elif args[i] == '--tf':
            timeframe = args[i + 1]
            i += 2
        elif args[i] == '--funding':
            mode = 'funding'
            i += 1
        elif args[i] == '--common-pairs':
            mode = 'common-pairs'
            i += 1
        elif args[i] == '--spread-snapshot':
            mode = 'spread-snapshot'
            i += 1
        elif args[i] == '--rounds':
            snapshot_rounds = int(args[i + 1])
            i += 2
        elif args[i] == '--interval':
            snapshot_interval = int(args[i + 1])
            i += 2
        else:
            symbols.append(args[i])
            i += 1

    # Mode standalone : list common pairs (avant le dispatch par exchange)
    if mode == 'common-pairs':
        find_common_pairs(save=True, verbose=True)
        sys.exit(0)

    if mode == 'spread-snapshot':
        import asyncio
        print(f'Mode spread-snapshot : {snapshot_rounds} rounds × {snapshot_interval}s')
        asyncio.run(snapshot_spreads_all(
            n_rounds=snapshot_rounds,
            interval_s=snapshot_interval,
        ))
        sys.exit(0)

    print(f"Exchange : {exchange}")
    print(f"Mode     : {mode}")
    if mode == 'candles':
        print(f"Timeframe : {timeframe}")

    if exchange == 'lighter':
        import asyncio
        if mode == 'funding':
            print("Mode Lighter funding : download async avec proxys")
            asyncio.run(download_lighter_funding_all(symbols=symbols if symbols else None))
        else:
            print("Mode Lighter candles : download async avec proxys")
            asyncio.run(download_lighter_all(tf=timeframe, symbols=symbols if symbols else None))
    elif exchange == 'hyperliquid':
        import asyncio
        if mode == 'funding':
            print("Mode Hyperliquid funding : POST /info {type:fundingHistory}, async proxys")
            asyncio.run(download_hyperliquid_funding_all(symbols=symbols if symbols else None))
        else:
            print(f"Mode Hyperliquid candles {timeframe} : POST /info {{type:candleSnapshot}}, async proxys")
            asyncio.run(download_hyperliquid_candles_all(tf=timeframe,
                                                        symbols=symbols if symbols else None))
    else:
        if mode == 'funding':
            print("Erreur : --funding n'est supporté que pour Lighter et Hyperliquid")
            sys.exit(1)
        if not symbols:
            print("Erreur : il faut au moins 1 symbol pour les exchanges classiques")
            sys.exit(1)
        print(f"Paires : {symbols}")
        download_symbols(symbols, exchange=exchange, timeframe=timeframe)
