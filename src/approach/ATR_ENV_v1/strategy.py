"""ATR Envelope mean-reversion — Strategy plugin for BTYZ engine.

Mean-reversion sur enveloppe SMA ± atr_mult × ATR avec :
- SL dynamique (sl_mult × ATR à l'entrée)
- Cooldown post-SL (attend retour à la MA)
- TP = retour à la MA

Plugged via engine.approach_loader.instantiate_strategy("ATR_ENV_v1").
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from numba import njit
from vectorbtpro import vbt

from engine.strategy_interface import BaseStrategy

# Runtime hints set by the engine before each backtest:
_target_fees: float = 0.0001    # bps × 1e-4 — overwritten by engine
_target_freq: str = "3min"      # overwritten by engine
_init_cash: float = 10_000.0


@njit(cache=True)
def _atr_envelope_nb(high, low, close, ma, upper, lower, sl_mult, atr_vals):
    """Single-level enveloppe ATR avec SL dynamique + cooldown."""
    n = len(close)
    target_size = np.full(n, np.nan)
    exec_price  = np.full(n, np.nan)

    pos_dir       = 0
    avg_entry     = 0.0
    sl_cooldown   = False
    pending_sl    = False
    pending_sl_px = 0.0
    entry_atr     = 0.0

    for i in range(1, n):
        if pending_sl:
            target_size[i] = 0.0
            exec_price[i]  = pending_sl_px
            pos_dir = 0
            avg_entry = 0.0
            sl_cooldown = True
            pending_sl = False
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

        # TP : retour MA
        if pos_dir != 0:
            exit_hit = (pos_dir == 1 and high[i] >= ma_prev) or \
                       (pos_dir == -1 and low[i] <= ma_prev)
            if exit_hit:
                target_size[i] = 0.0
                exec_price[i]  = ma_prev
                pos_dir = 0
                avg_entry = 0.0
                continue

        # Entrée
        if pos_dir == 0:
            lo_lim = lower[i - 1]
            hi_lim = upper[i - 1]
            if not np.isnan(lo_lim) and low[i] <= lo_lim:
                target_size[i] = 1.0
                exec_price[i]  = lo_lim
                pos_dir = 1
                avg_entry = lo_lim
                entry_atr = atr_vals[i - 1] if not np.isnan(atr_vals[i - 1]) else 0.0
            elif not np.isnan(hi_lim) and high[i] >= hi_lim:
                target_size[i] = -1.0
                exec_price[i]  = hi_lim
                pos_dir = -1
                avg_entry = hi_lim
                entry_atr = atr_vals[i - 1] if not np.isnan(atr_vals[i - 1]) else 0.0

        # SL dynamique (post entrée même barre)
        if pos_dir != 0:
            _atr = entry_atr if entry_atr > 0 else (atr_vals[i - 1] if not np.isnan(atr_vals[i - 1]) else 0.0)
            sl_dist = sl_mult * _atr
            if pos_dir == 1:
                sl_level = avg_entry - sl_dist
                if low[i] <= sl_level:
                    target_size[i] = 0.0
                    exec_price[i]  = sl_level
                    pos_dir = 0
                    avg_entry = 0.0
                    sl_cooldown = True
            else:
                sl_level = avg_entry + sl_dist
                if high[i] >= sl_level:
                    target_size[i] = 0.0
                    exec_price[i]  = sl_level
                    pos_dir = 0
                    avg_entry = 0.0
                    sl_cooldown = True

    return target_size, exec_price


class Strategy(BaseStrategy):
    """ATR Envelope mean-reversion (single-level)."""

    def param_space(self, trial) -> Dict[str, Any]:
        return {
            "ma_window":  trial.suggest_int("ma_window",  20, 200),
            "atr_window": trial.suggest_int("atr_window", 10, 60),
            "atr_mult":   round(trial.suggest_float("atr_mult", 1.0, 4.0, step=0.1), 2),
            "sl_mult":    round(trial.suggest_float("sl_mult", 1.0, 5.0, step=0.1), 2),
            "ohlc4":      trial.suggest_categorical("ohlc4", [False, True]),
        }

    def run_backtest(self, data, params):
        if data is None or len(data) < params["ma_window"] + params["atr_window"] + 10:
            return None

        if params["ohlc4"]:
            src = (data["open"] + data["high"] + data["low"] + data["close"]) / 4
        else:
            src = data["close"]

        ma = vbt.MA.run(src, window=params["ma_window"]).ma
        atr = vbt.ATR.run(data["high"], data["low"], data["close"], window=params["atr_window"]).atr

        upper = (ma + params["atr_mult"] * atr).values
        lower = (ma - params["atr_mult"] * atr).values

        ts, px = _atr_envelope_nb(
            data["high"].values, data["low"].values, data["close"].values,
            ma.values, upper, lower,
            float(params["sl_mult"]), atr.values,
        )
        size_s = pd.Series(ts, index=data.index)
        price_s = pd.Series(px, index=data.index)

        pf = vbt.Portfolio.from_orders(
            close=data["close"],
            size=size_s,
            price=price_s,
            size_type="TargetPercent",
            init_cash=_init_cash,
            leverage=1.0,
            fees=_target_fees,
            slippage=0.0,
            freq=_target_freq,
        )
        return pf
