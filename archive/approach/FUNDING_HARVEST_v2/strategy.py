"""FUNDING_HARVEST_v2 — Funding harvest with **adaptive APR thresholds** (rolling quantiles).

Différence vs v1 :
  - v1 : entry si `apr[i-1] > threshold_apr` (seuil fixe)
  - v2 : entry si `apr[i-1] > rolling_quantile(apr, window, q_entry)` (top decile/quintile
         dynamique). S'adapte au régime de funding de chaque paire et chaque période.

Motivation :
  - Le funding rate distribution varie énormément par paire (HYPE ≠ BTC) et au cours du
    temps (compression observée -60 à -80% sur 14 mois). Un seuil fixe rate certains
    régimes (folds à 0 trades dans v1 WFA).
  - Le seuil adaptatif détecte les extrêmes RELATIFS, peu importe le niveau absolu.

Exit aussi adaptatif :
  - Sortie quand `apr` redescend sous `rolling_quantile(apr, window, q_exit)` (typiquement
    P50 = médiane). Évite les sorties trop tardives en régime calme.

Convention identique à v1 :
  - SHORT only (collecte funding quand apr > 0)
  - Same-bar SL via pending_sl
  - cash_dividends = -signed_rate/100 × close (vbt scale par position auto)
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from numba import njit
from vectorbtpro import vbt

from engine.strategy_interface import BaseStrategy

_target_fees: float = 0.0001
_target_slippage: float = 0.0002
_target_freq: str = "1h"
_init_cash: float = 10_000.0

ALLOC_FIXED = 0.01


@njit(cache=True)
def _funding_harvest_v2_nb(
    high, low, close,
    signed_rate, apr,
    apr_q_entry,            # rolling quantile entry pré-computé (par bar)
    apr_q_exit,             # rolling quantile exit pré-computé (par bar)
    hold_max_hours: int,
    sl_pct: float,
):
    """Kernel adaptatif. Returns target_size (±1, scalé externe par ALLOC), exec_price.
    Same-bar SL via pending_sl.
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
        # 1) Sortie différée
        if pending_sl:
            target_size[i] = 0.0
            exec_price[i]  = pending_sl_px
            pos_dir = 0; avg_entry = 0.0; hours_held = 0
            pending_sl = False
            continue

        # 2) En position : checks exit
        if pos_dir == -1:
            hours_held += 1

            # 2a) SL intra-bar (prioritaire)
            sl_level = avg_entry * (1.0 + sl_pct)
            if high[i] >= sl_level:
                target_size[i] = 0.0
                exec_price[i]  = sl_level
                pos_dir = 0; avg_entry = 0.0; hours_held = 0
                continue

            # 2b) Exit funding adaptatif : apr < rolling P_exit
            ap = apr[i-1]
            q_exit = apr_q_exit[i-1]
            if not np.isnan(ap) and not np.isnan(q_exit) and ap < q_exit:
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

        # 3) Flat : check entry adaptatif
        entered_this_bar = False
        if pos_dir == 0:
            ap = apr[i-1]
            q_entry = apr_q_entry[i-1]
            if not np.isnan(ap) and not np.isnan(q_entry) and ap > q_entry:
                target_size[i] = -1.0
                exec_price[i]  = close[i]
                pos_dir = -1
                avg_entry = close[i]
                hours_held = 0
                entered_this_bar = True

        # 4) Same-bar SL : defer au bar suivant
        if entered_this_bar and pos_dir == -1:
            sl_level = avg_entry * (1.0 + sl_pct)
            if high[i] >= sl_level:
                pending_sl = True
                pending_sl_px = sl_level

    return target_size, exec_price


def _causal_rolling_quantile(s: pd.Series, window: int, q: float) -> pd.Series:
    """Rolling quantile causal : utilise les `window` dernières valeurs UNCLUS de t.
    shift(1) pour exclure t-courant → no lookahead, valeur dispo au close[t-1].
    """
    return s.rolling(window=window, min_periods=window).quantile(q).shift(1)


class Strategy(BaseStrategy):
    """Funding harvest adaptatif : seuils rolling quantile par paire/régime."""

    SIZE_PCT = ALLOC_FIXED
    MIN_BARS_NEEDED = 24

    def score(self, metrics):
        from engine.scoring import score_robust
        return score_robust(metrics)

    def param_space(self, trial) -> Dict[str, Any]:
        return {
            # Window de la rolling quantile (heures). 168-1440 = 7-60 jours
            "window":         trial.suggest_int("window", 168, 1440, step=168),  # multiples 7j
            # Quantile entry : top 5-30% (0.70 = P70, 0.95 = P95)
            "q_entry":        round(trial.suggest_float("q_entry", 0.70, 0.95, step=0.05), 2),
            # Quantile exit : sortie quand apr revient sous (médiane à P70 = sortie progressive)
            "q_exit":         round(trial.suggest_float("q_exit",  0.30, 0.70, step=0.05), 2),
            "hold_max_hours": trial.suggest_int("hold_max_hours", 4, 72, step=4),
            "sl_pct":         round(trial.suggest_float("sl_pct", 0.01, 0.10, step=0.005), 4),
        }

    def compute_target_arrays(self, data: pd.DataFrame, params: Dict[str, Any]):
        required = {"open", "high", "low", "close", "signed_rate", "apr"}
        if data is None or not required.issubset(set(data.columns)):
            return None, None
        if len(data) < params.get("window", 720) + 50:
            return None, None

        window = int(params["window"])
        q_entry = float(params["q_entry"])
        q_exit  = float(params["q_exit"])

        # Pré-compute rolling quantiles causaux
        apr_s = data["apr"]
        q_entry_arr = _causal_rolling_quantile(apr_s, window, q_entry).values.astype(np.float64)
        q_exit_arr  = _causal_rolling_quantile(apr_s, window, q_exit).values.astype(np.float64)

        ts, px = _funding_harvest_v2_nb(
            data["high"].values.astype(np.float64),
            data["low"].values.astype(np.float64),
            data["close"].values.astype(np.float64),
            data["signed_rate"].values.astype(np.float64),
            data["apr"].values.astype(np.float64),
            q_entry_arr,
            q_exit_arr,
            int(params["hold_max_hours"]),
            float(params["sl_pct"]),
        )
        return pd.Series(ts, index=data.index), pd.Series(px, index=data.index)

    def compute_cash_dividends(self, data: pd.DataFrame,
                               params: Dict[str, Any] = None) -> pd.Series:
        """Funding per-share (identique à v1) : -signed_rate/100 × close, vbt scale par position."""
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
