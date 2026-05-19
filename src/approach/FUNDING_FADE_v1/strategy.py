"""FUNDING_FADE_v1 — Mean reversion sur le prix après funding extrême.

Hypothèse :
  Quand le funding rate Lighter atteint un z-score extrême (positions crowded),
  le PRIX mean-revert à court terme parce que les longs/shorts forcés capitulent
  sous le coût du carry. On FADE la direction crowded.

Décisions (no lookahead) :
  - z[t-1] > +z_threshold → longs crowded → on SHORT au close[t]
  - z[t-1] < -z_threshold → shorts crowded → on LONG au close[t]
  - Exit : timeout (hold_max_hours), SL (% du prix), OR z revient vers 0

Convention z :
  z[t] = (signed_rate[t-1] - mean(signed_rate[t-N..t-1])) / std(signed_rate[t-N..t-1])
  shift(1) sur tout → no lookahead.

Data attendue :
  DataFrame indexé 1h, colonnes : open, high, low, close, volume, signed_rate.
  Le z-score est calculé INSIDE compute_target_arrays (pas dépendant du loader).

Diff vs FUNDING_HARVEST_v1 :
  - HARVEST : long edge structurel (collecter le carry). Microscopique mais
    persistant. Short-only quand funding très positif.
  - FADE    : capture le mean-rev du PRIX après funding extrême. Bidirectionnel.
    Edge plus gros en absolu (le prix bouge bcp + que le funding).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from numba import njit
from vectorbtpro import vbt

from engine.strategy_interface import BaseStrategy

# Runtime hints
_target_fees: float = 0.0002      # 2bps integrator
_target_slippage: float = 0.0002  # 2bps slippage
_target_freq: str = "1h"
_init_cash: float = 10_000.0

# Sizing fixe (1% par paire en live multi-pair, override slider notebook).
ALLOC_FIXED = 0.01


@njit(cache=True)
def _funding_fade_nb(
    high, low, close,
    z_signed,                 # z-score causal du signed_rate
    z_threshold: float,       # entry si |z| > threshold
    hold_max_hours: int,      # timeout en heures (bars)
    sl_pct: float,            # stop-loss en % du prix d'entrée
    exit_z: float,            # exit si |z| < exit_z (funding revient à la médiane)
):
    """Kernel funding fade — bidirectionnel, single position.

    Returns (target_size [-1,0,+1] × ALLOC sized outside, exec_price).
    Same-bar SL: pending_sl mechanism → preserve entry trade record.
    """
    n = len(close)
    target_size = np.full(n, np.nan)
    exec_price  = np.full(n, np.nan)

    pos_dir       = 0
    avg_entry     = 0.0
    hours_held    = 0
    pending_sl    = False
    pending_sl_px = 0.0

    for i in range(1, n):
        # 1) Same-bar SL deferred
        if pending_sl:
            target_size[i] = 0.0
            exec_price[i]  = pending_sl_px
            pos_dir = 0; avg_entry = 0.0; hours_held = 0
            pending_sl = False
            continue

        # 2) Exits si en position
        if pos_dir != 0:
            hours_held += 1

            # 2a) SL intra-bar (prioritaire)
            if pos_dir == 1:
                sl_level = avg_entry * (1.0 - sl_pct)
                if low[i] <= sl_level:
                    target_size[i] = 0.0
                    exec_price[i]  = sl_level
                    pos_dir = 0; avg_entry = 0.0; hours_held = 0
                    continue
            else:  # short
                sl_level = avg_entry * (1.0 + sl_pct)
                if high[i] >= sl_level:
                    target_size[i] = 0.0
                    exec_price[i]  = sl_level
                    pos_dir = 0; avg_entry = 0.0; hours_held = 0
                    continue

            # 2b) Exit z normalisé (funding revenu à la médiane)
            z = z_signed[i-1]
            if not np.isnan(z) and abs(z) < exit_z:
                target_size[i] = 0.0
                exec_price[i]  = close[i]
                pos_dir = 0; avg_entry = 0.0; hours_held = 0
                continue

            # 2c) Timeout
            if hours_held >= hold_max_hours:
                target_size[i] = 0.0
                exec_price[i]  = close[i]
                pos_dir = 0; avg_entry = 0.0; hours_held = 0
                continue

        # 3) Entry si flat
        entered_this_bar = False
        if pos_dir == 0:
            z = z_signed[i-1]
            if not np.isnan(z):
                if z > z_threshold:
                    # Long crowded → FADE = SHORT
                    target_size[i] = -1.0
                    exec_price[i]  = close[i]
                    pos_dir = -1
                    avg_entry = close[i]
                    hours_held = 0
                    entered_this_bar = True
                elif z < -z_threshold:
                    # Short crowded → FADE = LONG
                    target_size[i] = 1.0
                    exec_price[i]  = close[i]
                    pos_dir = 1
                    avg_entry = close[i]
                    hours_held = 0
                    entered_this_bar = True

        # 4) Same-bar SL check : si entry hit son SL dans la même bar
        if entered_this_bar:
            if pos_dir == 1:
                sl_level = avg_entry * (1.0 - sl_pct)
                if low[i] <= sl_level:
                    pending_sl = True
                    pending_sl_px = sl_level
            elif pos_dir == -1:
                sl_level = avg_entry * (1.0 + sl_pct)
                if high[i] >= sl_level:
                    pending_sl = True
                    pending_sl_px = sl_level

    return target_size, exec_price


def _causal_zscore(s: pd.Series, window: int) -> pd.Series:
    """Z-score rolling causal : z[t] = (s[t] - mean[t-N..t-1]) / std[t-N..t-1].

    On `shift(1)` les rolling stats pour ne PAS inclure s[t] dans la baseline.
    """
    rolled = s.rolling(window=window, min_periods=window)
    mean = rolled.mean().shift(1)
    std = rolled.std().shift(1)
    z = (s - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


class Strategy(BaseStrategy):
    """Funding-driven mean reversion sur le prix (bidirectionnel)."""

    SIZE_PCT = ALLOC_FIXED
    MIN_BARS_NEEDED = 350  # ~14j + buffer pour z-score 336

    def score(self, metrics):
        from engine.scoring import score_robust
        return score_robust(metrics)

    def param_space(self, trial) -> Dict[str, Any]:
        return {
            "z_window":       trial.suggest_int("z_window",      168, 504, step=24),    # 7-21j
            "z_threshold":    round(trial.suggest_float("z_threshold", 1.5, 4.0, step=0.25), 2),
            "exit_z":         round(trial.suggest_float("exit_z",      0.0, 1.5, step=0.25), 2),
            "hold_max_hours": trial.suggest_int("hold_max_hours", 1, 48, step=1),
            "sl_pct":         round(trial.suggest_float("sl_pct", 0.005, 0.10, step=0.005), 4),
        }

    def compute_target_arrays(self, data: pd.DataFrame, params: Dict[str, Any]):
        required = {"open", "high", "low", "close", "signed_rate"}
        if data is None or not required.issubset(set(data.columns)):
            return None, None
        if len(data) < self.MIN_BARS_NEEDED + 2:
            return None, None

        z_window = int(params.get("z_window", 336))
        z_signed = _causal_zscore(data["signed_rate"], window=z_window).values.astype(np.float64)

        ts, px = _funding_fade_nb(
            data["high"].values.astype(np.float64),
            data["low"].values.astype(np.float64),
            data["close"].values.astype(np.float64),
            z_signed,
            float(params["z_threshold"]),
            int(params["hold_max_hours"]),
            float(params["sl_pct"]),
            float(params.get("exit_z", 0.5)),
        )
        return pd.Series(ts, index=data.index), pd.Series(px, index=data.index)

    def compute_cash_dividends(self, data: pd.DataFrame,
                               params: Dict[str, Any] = None) -> pd.Series:
        """Funding payments encaissés/payés pendant qu'on tient la position.

        Effet secondaire du fade (signal principal = mean rev sur prix).
        Fix 2026-05-16 : /100 car signed_rate est en %/h.
        """
        if "signed_rate" not in data.columns:
            return None
        return -data["signed_rate"].fillna(0) / 100.0 * data["close"]

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]):
        size_s, price_s = self.compute_target_arrays(data, params)
        if size_s is None:
            return None

        size_scaled = size_s * self.SIZE_PCT
        cd = self.compute_cash_dividends(data, params)

        pf = vbt.Portfolio.from_orders(
            close=data["close"],
            size=size_scaled,
            price=price_s,
            size_type="TargetPercent",
            init_cash=_init_cash,
            cash_dividends=cd,
            leverage=1.0,
            fees=_target_fees,
            slippage=_target_slippage,
            freq=_target_freq,
        )
        return pf
