"""RAM_ENV_SMA_v1 — Mean-reversion 1 enveloppe + filtre de tendance SMA (exécution LIMIT).

Port de RAM_DCA_v1 (kernel single-envelope, fills LIMIT à la bande) auquel on AJOUTE
un filtre de tendance par grande SMA, comme dans le builder BTYZ de Sofien :

  - LONG  : touche la bande basse `MA(ma_window)·(1-env_pct)`  ET  close >= SMA(sma_trend)
            (on ne fade le creux que dans une tendance haussière)
  - SHORT : touche la bande haute `MA(ma_window)·(1+env_pct)`  ET  close <= SMA(sma_trend)
            (on ne fade le rip que dans une tendance baissière)
  - Sortie : retour à la MA (TP). SL hard %. Cooldown post-SL (retour MA). Fills à la bande.

Exécution LIMIT native : le kernel remplit à `lower_env[i-1]` / `upper_env[i-1]` quand le low/high
de la barre touche la bande (modèle optimiste « rempli si touché » — voir caveat adverse selection).

⚠️ La SMA de tendance a besoin de `sma_trend` barres de warmup : lancer la WFA avec
`--warmup >= max(sma_trend testé)` (ex. `--warmup 5000`) sinon la SMA est NaN en début d'OOS.

Params optimisés :
- ma_window  : int [20, 200] step 10   (période enveloppe)
- env_idx    : choice ENV_PCT_CHOICES   (0.25% .. 10%)
- sma_trend  : int [10, 5000] step 10   (filtre de tendance — demande de Sofien)
- sl_pct     : float [0.01, 0.10] step 0.005
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from numba import njit
from vectorbtpro import vbt

from engine.strategy_interface import BaseStrategy

_target_fees: float = 0.0001
_target_slippage: float = 0.0002    # 2 bps default (limit ≈ maker, override via --bps)
_target_freq: str = "5min"
_init_cash: float = 10_000.0

ENV_PCT_CHOICES = [0.0025, 0.005, 0.01, 0.02, 0.03, 0.035, 0.04, 0.05,
                   0.06, 0.07, 0.08, 0.09, 0.10]

ALLOC_FIXED = 0.10  # 10% du capital par ordre (TargetPercent), fixe


@njit
def _ram_env_sma_nb(high, low, close, ma, upper_env, lower_env, sma_trend, sl_pct, alloc):
    """Single-envelope mean-rev + filtre tendance SMA. LIMIT fills à la bande.

    Identique à _ram_single_env_nb (RAM_DCA_v1) + gate de tendance sur les ENTRÉES :
      long autorisé seulement si close[i-1] >= sma_trend[i-1] ;
      short autorisé seulement si close[i-1] <= sma_trend[i-1].
    SL > TP priority. pending_sl pour same-bar entry+SL. Cooldown post-SL = retour MA.
    """
    n = len(close)
    target_size = np.full(n, np.nan)
    exec_price  = np.full(n, np.nan)

    pos_dir       = 0
    avg_entry     = 0.0
    sl_cooldown   = False
    pending_sl    = False
    pending_sl_px = 0.0

    for i in range(1, n):
        if pending_sl:
            target_size[i] = 0.0
            exec_price[i]  = pending_sl_px
            pos_dir     = 0
            avg_entry   = 0.0
            sl_cooldown = True
            pending_sl  = False
            continue

        ma_prev = ma[i - 1]
        if np.isnan(ma_prev):
            continue

        if sl_cooldown:
            touched = low[i] <= ma_prev <= high[i]
            crossed = min(close[i - 1], close[i]) <= ma_prev <= max(close[i - 1], close[i])
            if touched or crossed:
                sl_cooldown = False
            else:
                continue

        # SL prioritaire sur TP (position existante)
        if pos_dir != 0:
            if pos_dir == 1:
                sl_level_cur = avg_entry * (1.0 - sl_pct)
                if low[i] <= sl_level_cur:
                    target_size[i] = 0.0
                    exec_price[i]  = sl_level_cur
                    pos_dir = 0; avg_entry = 0.0; sl_cooldown = True
                    continue
            else:
                sl_level_cur = avg_entry * (1.0 + sl_pct)
                if high[i] >= sl_level_cur:
                    target_size[i] = 0.0
                    exec_price[i]  = sl_level_cur
                    pos_dir = 0; avg_entry = 0.0; sl_cooldown = True
                    continue

            exit_hit = (pos_dir == 1 and high[i] >= ma_prev) or \
                       (pos_dir == -1 and low[i] <= ma_prev)
            if exit_hit:
                target_size[i] = 0.0
                exec_price[i]  = ma_prev
                pos_dir = 0; avg_entry = 0.0
                continue

            continue

        # --- Détection entrée + FILTRE TENDANCE SMA ---
        st_prev = sma_trend[i - 1]
        if np.isnan(st_prev):
            continue
        c_prev = close[i - 1]
        lim_lo = lower_env[i - 1]
        lim_hi = upper_env[i - 1]
        new_dir = 0
        entry_px = 0.0

        if low[i] <= lim_lo and c_prev >= st_prev:        # bande basse en tendance haussière
            new_dir = 1
            entry_px = lim_lo
        elif high[i] >= lim_hi and c_prev <= st_prev:     # bande haute en tendance baissière
            new_dir = -1
            entry_px = lim_hi
        else:
            continue

        if new_dir == 1:
            sl_level = entry_px * (1.0 - sl_pct)
            sl_hit = low[i] <= sl_level
        else:
            sl_level = entry_px * (1.0 + sl_pct)
            sl_hit = high[i] >= sl_level

        target_size[i] = alloc if new_dir == 1 else -alloc
        exec_price[i]  = entry_px
        pos_dir   = new_dir
        avg_entry = entry_px
        if sl_hit:
            pending_sl    = True
            pending_sl_px = sl_level

    return target_size, exec_price


class Strategy(BaseStrategy):
    """RAM single-envelope mean-rev + filtre tendance SMA (limit)."""

    SIZE_PCT = ALLOC_FIXED

    def score(self, metrics):
        from engine.scoring import score_robust
        return score_robust(metrics)

    def param_space(self, trial) -> Dict[str, Any]:
        ma_window = trial.suggest_int("ma_window", 20, 200, step=10)
        env_idx   = trial.suggest_int("env_idx", 0, len(ENV_PCT_CHOICES) - 1)
        env_pct   = ENV_PCT_CHOICES[env_idx]
        sma_trend = trial.suggest_int("sma_trend", 10, 5000, step=10)
        sl_pct    = round(trial.suggest_float("sl_pct", 0.01, 0.10, step=0.005), 4)

        return {
            "ma_window": ma_window,
            "env_pct":   env_pct,
            "sma_trend": sma_trend,
            "sl_pct":    sl_pct,
        }

    def compute_target_arrays(self, data, params):
        need = max(int(params["ma_window"]), int(params["sma_trend"])) + 10
        if data is None or len(data) < need:
            return None, None

        src = data["close"]
        ma = vbt.MA.run(src, window=params["ma_window"]).ma
        sma_t = vbt.MA.run(src, window=params["sma_trend"]).ma
        env_pct = float(params["env_pct"])
        upper = (ma * (1.0 + env_pct)).values
        lower = (ma * (1.0 - env_pct)).values

        ts, px = _ram_env_sma_nb(
            data["high"].values, data["low"].values, data["close"].values,
            ma.values, upper, lower, sma_t.values,
            float(params["sl_pct"]), 1.0,
        )
        return pd.Series(ts, index=data.index), pd.Series(px, index=data.index)

    def run_backtest(self, data, params):
        size_s, price_s = self.compute_target_arrays(data, params)
        if size_s is None:
            return None
        size_s = size_s * self.SIZE_PCT
        pf = vbt.Portfolio.from_orders(
            close=data["close"],
            size=size_s,
            price=price_s,
            size_type="TargetPercent",
            init_cash=_init_cash,
            leverage=1.0,
            fees=_target_fees,
            slippage=_target_slippage,
            freq=_target_freq,
        )
        return pf
