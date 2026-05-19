"""RAM_DCA_v1 (single envelope) — Mean-reversion 1 bande, version de comparaison directe avec ATR_ENV_v1.

Différence vs **RAM_DCA_v1_MULTI** :
  - 1 SEULE enveloppe (pas de DCA multi-niveaux)
  - Alloc fixe 10% (TargetPercent), pas de % par bande
  - Reste identique : TP au retour MA, SL hard %, cooldown post-SL, pending_sl

Différence vs **ATR_ENV_v1** :
  - Enveloppe = `MA ± env_pct × MA` (% MA fixe)  vs  ATR_ENV `MA ± atr_mult × ATR` (volatilité)
  - SL = `avg_entry × (1 ± sl_pct)` (% absolu)   vs  ATR_ENV `entry ± sl_mult × ATR`

Fix 2026-05-17 inclus : SL prioritaire sur TP (pas de bug "TP first").

Différence vs **RAM_DCA_RSI_v1** :
  - Pas de filtre RSI (entries permises tout le temps)

Params optimisés:
- ma_window         : int [20, 200] step 10
- env_pct           : choice ENV_PCT_CHOICES (0.25%, 0.5%, 1%, …, 10%)
- sl_pct            : float [0.01, 0.10] step 0.005
- ohlc4             : False (fixe)
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from numba import njit
from vectorbtpro import vbt

from engine.strategy_interface import BaseStrategy

_target_fees: float = 0.0001
_target_slippage: float = 0.0002    # 2 bps default
_target_freq: str = "5min"
_init_cash: float = 10_000.0

# Choix discrets cohérents avec RAM_DCA_RSI_v1
ENV_PCT_CHOICES = [0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
                   0.06, 0.07, 0.08, 0.09, 0.10]

ALLOC_FIXED = 0.10  # 10% du capital par ordre (TargetPercent), fixe


@njit
def _ram_single_env_nb(high, low, close, ma, upper_env, lower_env, sl_pct, alloc):
    """Single-envelope mean-rev kernel (sans filtre RSI).

    SL > TP priority (fix 2026-05-17). pending_sl pour same-bar entry+SL.
    Cooldown post-SL : attend retour MA (touch ou cross).
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
        # SL différé du bar précédent
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

        # Cooldown : attend retour MA avant nouvelle entrée
        if sl_cooldown:
            touched = low[i] <= ma_prev <= high[i]
            crossed = min(close[i - 1], close[i]) <= ma_prev <= max(close[i - 1], close[i])
            if touched or crossed:
                sl_cooldown = False
            else:
                continue

        # SL prioritaire sur TP (existing position)
        if pos_dir != 0:
            if pos_dir == 1:
                sl_level_cur = avg_entry * (1.0 - sl_pct)
                if low[i] <= sl_level_cur:
                    target_size[i] = 0.0
                    exec_price[i]  = sl_level_cur
                    pos_dir     = 0
                    avg_entry   = 0.0
                    sl_cooldown = True
                    continue
            else:
                sl_level_cur = avg_entry * (1.0 + sl_pct)
                if high[i] >= sl_level_cur:
                    target_size[i] = 0.0
                    exec_price[i]  = sl_level_cur
                    pos_dir     = 0
                    avg_entry   = 0.0
                    sl_cooldown = True
                    continue

            # TP après SL : retour MA
            exit_hit = (pos_dir == 1 and high[i] >= ma_prev) or \
                       (pos_dir == -1 and low[i] <= ma_prev)
            if exit_hit:
                target_size[i] = 0.0
                exec_price[i]  = ma_prev
                pos_dir   = 0
                avg_entry = 0.0
                continue

            continue  # en position, ni SL ni TP → attend

        # Détection entrée — long si touche bande basse, short si touche bande haute
        lim_lo = lower_env[i - 1]
        lim_hi = upper_env[i - 1]
        new_dir = 0
        entry_px = 0.0

        if low[i] <= lim_lo:
            new_dir = 1
            entry_px = lim_lo
        elif high[i] >= lim_hi:
            new_dir = -1
            entry_px = lim_hi
        else:
            continue

        # Check SL immédiat (gap intra-bar)
        if new_dir == 1:
            sl_level = entry_px * (1.0 - sl_pct)
            sl_hit = low[i] <= sl_level
        else:
            sl_level = entry_px * (1.0 + sl_pct)
            sl_hit = high[i] >= sl_level

        if sl_hit:
            target_size[i] = alloc if new_dir == 1 else -alloc
            exec_price[i]  = entry_px
            pos_dir   = new_dir
            avg_entry = entry_px
            pending_sl    = True
            pending_sl_px = sl_level
        else:
            target_size[i] = alloc if new_dir == 1 else -alloc
            exec_price[i]  = entry_px
            pos_dir   = new_dir
            avg_entry = entry_px

    return target_size, exec_price


class Strategy(BaseStrategy):
    """RAM_DCA single-envelope mean-rev (sans filtre RSI)."""

    SIZE_PCT = ALLOC_FIXED

    def score(self, metrics):
        from engine.scoring import score_robust
        return score_robust(metrics)

    def param_space(self, trial) -> Dict[str, Any]:
        ma_window = trial.suggest_int("ma_window", 20, 200, step=10)
        env_idx   = trial.suggest_int("env_idx", 0, len(ENV_PCT_CHOICES) - 1)
        env_pct   = ENV_PCT_CHOICES[env_idx]
        sl_pct    = round(trial.suggest_float("sl_pct", 0.01, 0.10, step=0.005), 4)

        return {
            "ma_window": ma_window,
            "env_pct":   env_pct,
            "sl_pct":    sl_pct,
            "ohlc4":     False,
        }

    def compute_target_arrays(self, data, params):
        if data is None or len(data) < params["ma_window"] + 10:
            return None, None

        src = data["close"] if not params.get("ohlc4") else \
              (data["open"] + data["high"] + data["low"] + data["close"]) / 4
        ma = vbt.MA.run(src, window=params["ma_window"]).ma
        env_pct = float(params["env_pct"])
        upper = (ma * (1.0 + env_pct)).values
        lower = (ma * (1.0 - env_pct)).values

        ts, px = _ram_single_env_nb(
            data["high"].values, data["low"].values, data["close"].values,
            ma.values, upper, lower,
            float(params["sl_pct"]), 1.0,   # kernel retourne ±1, scale dans run_backtest
        )
        return pd.Series(ts, index=data.index), pd.Series(px, index=data.index)

    def run_backtest(self, data, params):
        size_s, price_s = self.compute_target_arrays(data, params)
        if size_s is None:
            return None
        size_s = size_s * self.SIZE_PCT      # ±1 → ±0.10
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
