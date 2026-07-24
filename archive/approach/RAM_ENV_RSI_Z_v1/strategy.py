"""RAM_ENV_RSI_Z_v1 — variante builder Sofien (enveloppe + RSI[38-62] + Zscore + filtre SMA).

Réplique EXACTEMENT la logique du builder (validée 1:1 vs vbt sur HYPE) :
  LONG  : close croise ↓ ENV.lower ET RSI(80)∈[38,62] ET ENV.middle>=SMA(sma_w) ET Z(z_w)<=-z_th
  SHORT : close croise ↑ ENV.upper ET RSI∈[38,62] ET ENV.middle<=SMA(sma_w) ET Z>=+z_th
  Sortie long : close croise ↑ middle ; sortie short : close croise ↓ middle. SL hard %.
  Exécution = MARKET au close de la barre de signal (from_signals).

OPTIMISÉ (réponse à "régler l'enveloppe par paire") :
  ma_window  [20..200]   : période enveloppe (le déclencheur, scale-dépendant)
  env_pct    choices      : largeur bandes
  z_th       [1.5..3.0]   : seuil Zscore
  sl_pct     [0.03..0.12] : stop-loss
FIXE (valeurs builder) : RSI période 80 / bande [38,62], Zscore période 50, SMA tendance 2000.
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np, pandas as pd
from numba import njit
from vectorbtpro import vbt
from engine.strategy_interface import BaseStrategy

_target_fees: float = 0.0003       # 3 bps (overridé par --bps)
_target_slippage: float = 0.0002   # 2 bps (défaut builder)
_target_freq: str = "5min"
_init_cash: float = 10_000.0

ENV_PCT_CHOICES = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08]
RSI_LOW, RSI_HIGH, RSI_P, Z_P, SMA_TREND = 38.0, 62.0, 80, 50, 2000


def _xdn(s, lv): return (s.shift(1) >= lv.shift(1)) & (s < lv)
def _xup(s, lv): return (s.shift(1) <= lv.shift(1)) & (s > lv)


@njit(cache=True)
def _envz_nb(le, lx, se, sx, close, low, high, mid, upper, sl):
    """Kernel à état avec RÉINTÉGRATION (anti SL-on-SL), 1:1 builder.
    Après un SL long -> bloque la ré-entrée long jusqu'à close croise ↑ middle.
    Après un SL short -> bloque la ré-entrée short jusqu'à close croise ↑ upper.
    Retourne (target_size ±1/0, exec_price) pour from_orders TargetPercent."""
    n = close.shape[0]
    ts = np.zeros(n); ep = np.full(n, np.nan)
    pos = 0; entry = 0.0
    block_long = False; block_short = False
    for i in range(n):
        px = close[i]
        # libération des blocages de réintégration
        if block_long and i > 0 and close[i - 1] <= mid[i - 1] and px > mid[i]:
            block_long = False
        if block_short and i > 0 and close[i - 1] <= upper[i - 1] and px > upper[i]:
            block_short = False
        # gestion position
        if pos == 1:
            sl_lvl = entry * (1.0 - sl)
            if low[i] <= sl_lvl:
                pos = 0; ts[i] = 0.0; ep[i] = sl_lvl; block_long = True
            elif lx[i]:
                pos = 0; ts[i] = 0.0; ep[i] = px
            else:
                ts[i] = 1.0; ep[i] = px; continue
        elif pos == -1:
            sl_lvl = entry * (1.0 + sl)
            if high[i] >= sl_lvl:
                pos = 0; ts[i] = 0.0; ep[i] = sl_lvl; block_short = True
            elif sx[i]:
                pos = 0; ts[i] = 0.0; ep[i] = px
            else:
                ts[i] = -1.0; ep[i] = px; continue
        # entrées (si flat), gate réintégration
        if pos == 0:
            if le[i] and not block_long:
                pos = 1; entry = px; ts[i] = 1.0; ep[i] = px
            elif se[i] and not block_short:
                pos = -1; entry = px; ts[i] = -1.0; ep[i] = px
            else:
                ts[i] = 0.0; ep[i] = px
    return ts, ep


class Strategy(BaseStrategy):
    def score(self, metrics):
        from engine.scoring import score_robust
        return score_robust(metrics)

    def param_space(self, trial) -> Dict[str, Any]:
        return {
            "ma_window": trial.suggest_int("ma_window", 20, 200, step=10),
            "env_pct": ENV_PCT_CHOICES[trial.suggest_int("env_idx", 0, len(ENV_PCT_CHOICES) - 1)],
            "z_th": round(trial.suggest_float("z_th", 1.5, 3.0, step=0.25), 2),
            "sl_pct": round(trial.suggest_float("sl_pct", 0.03, 0.12, step=0.01), 4),
        }

    def _indicators(self, data, params):
        c = data["close"]; mw = int(params["ma_window"])
        ep = float(params["env_pct"]); zth = float(params["z_th"])
        mid = c.rolling(mw).mean(); lower = mid * (1 - ep); upper = mid * (1 + ep)
        smat = c.rolling(SMA_TREND).mean()
        d = c.diff()
        up = d.clip(lower=0).ewm(alpha=1 / RSI_P, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / RSI_P, adjust=False).mean()
        rsi = 100 - 100 / (1 + up / dn)
        z = (c - c.rolling(Z_P).mean()) / c.rolling(Z_P).std()
        rok = (rsi >= RSI_LOW) & (rsi <= RSI_HIGH)
        le = (_xdn(c, lower) & rok & (mid >= smat) & (z <= -zth)).fillna(False)
        lx = _xup(c, mid).fillna(False)
        se = (_xup(c, upper) & rok & (mid <= smat) & (z >= zth)).fillna(False)
        sx = _xdn(c, mid).fillna(False)
        return le, lx, se, sx, mid, upper

    def run_backtest(self, data, params):
        """Backtest réel = from_signals (event-based, propre, source du verdict WFA)."""
        need = max(params["ma_window"], SMA_TREND, Z_P, RSI_P) + 10
        if data is None or len(data) < need:
            return None
        le, lx, se, sx, _mid, _up = self._indicators(data, params)
        return vbt.Portfolio.from_signals(
            open=data["open"], high=data["high"], low=data["low"], close=data["close"],
            entries=le, exits=lx, short_entries=se, short_exits=sx,
            sl_stop=float(params["sl_pct"]),
            fees=_target_fees, slippage=_target_slippage, init_cash=_init_cash, freq=_target_freq)

    def compute_target_arrays(self, data, params):
        """Pour la §4 replay : size SPARSE (ordre uniquement sur changement de position,
        NaN sinon) -> pas de rebalancing TargetPercent à chaque barre. Simule signaux + SL."""
        need = max(params["ma_window"], SMA_TREND, Z_P, RSI_P) + 10
        if data is None or len(data) < need:
            return None, None
        le, lx, se, sx, _mid, _up = self._indicators(data, params)
        le, lx, se, sx = (s.to_numpy() for s in (le, lx, se, sx))
        c = data["close"].to_numpy(); lo = data["low"].to_numpy(); hi = data["high"].to_numpy()
        sl = float(params["sl_pct"]); n = len(c)
        pos = np.zeros(n); p = 0; entry = 0.0
        for i in range(n):
            if p == 1 and (lx[i] or lo[i] <= entry * (1 - sl)):
                p = 0
            elif p == -1 and (sx[i] or hi[i] >= entry * (1 + sl)):
                p = 0
            if p == 0:
                if le[i]:
                    p = 1; entry = c[i]
                elif se[i]:
                    p = -1; entry = c[i]
            pos[i] = p
        # SPARSE : ne garder un ordre que sur les transitions de position
        size = np.full(n, np.nan)
        chg = np.r_[True, pos[1:] != pos[:-1]]
        size[chg] = pos[chg]
        return pd.Series(size, index=data.index), pd.Series(c, index=data.index)
