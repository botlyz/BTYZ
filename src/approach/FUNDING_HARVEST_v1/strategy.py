"""FUNDING_HARVEST_v1 — Short perp Lighter quand funding extrême positif, collecter le funding.

Hypothèse :
  Sur Lighter, certaines paires ont un funding rate persistanly positif (longs paient shorts)
  largement au-dessus du marché global. Shorter le perp et tenir collecte ce funding.
  Le risque = mouvement directionnel contre nous (prix qui rallye).

Décisions (au close 1h de chaque bar, no lookahead) :
  - ENTRY (passer SHORT) si pas en pos ET apr[i-1] > threshold_apr
  - EXIT (sortir SHORT) si :
      a) high[i] >= avg_entry × (1 + sl_pct)               → stop loss (intra-bar, prioritaire)
      b) apr[i-1] < exit_apr                                 → funding normalisé
      c) hours_in_position >= hold_max_hours                 → timeout

Conventions :
  - signed_rate > 0 = longs paient shorts (perp > spot) ; short COLLECTE.
  - On utilise apr[i-1] et signed_rate[i-1] pour la décision au bar i (no lookahead).
  - SL même bar que l'entry → pending_sl, exit reporté au bar i+1 (preserve l'entry signal).

Data attendue par run_backtest :
  DataFrame indexé timeframe, colonnes : open, high, low, close, volume, signed_rate, apr
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from numba import njit
from vectorbtpro import vbt

from engine.strategy_interface import BaseStrategy

# Runtime hints injectés par le moteur
_target_fees: float = 0.0001     # 1 bp integrator fee (override par --bps)
_target_slippage: float = 0.0002 # 2 bps slippage (engine ne l'override pas)
_target_freq: str = "1h"
_init_cash: float = 10_000.0

# Sizing fixe : 1% du capital par position short (live = 11 paires en parallele).
# Expose au notebook analyse (§4 utilise `getattr(module, "ALLOC_FIXED", None)` pour
# scaler via slider).
ALLOC_FIXED = 0.01


@njit(cache=True)
def _funding_harvest_nb(
    high, low, close,
    signed_rate, apr,
    threshold_apr: float, exit_apr: float,
    hold_max_hours: int, sl_pct: float,
):
    """Kernel funding harvest — short-only, single position.

    Returns target_size (vbt TargetPercent: ±1 puis scaled outside), exec_price.
    Same-bar SL: pending_sl mechanism → trade record préservé.
    """
    n = len(close)
    target_size = np.full(n, np.nan)
    exec_price  = np.full(n, np.nan)

    pos_dir       = 0      # 0 = flat, -1 = short
    avg_entry     = 0.0
    hours_held    = 0
    pending_sl    = False
    pending_sl_px = 0.0

    for i in range(1, n):
        # 1) Sortie différée (same-bar SL du bar précédent)
        if pending_sl:
            target_size[i] = 0.0
            exec_price[i]  = pending_sl_px
            pos_dir = 0
            avg_entry = 0.0
            hours_held = 0
            pending_sl = False
            continue

        # 2) Si en position, check exits AVANT entry
        if pos_dir == -1:
            hours_held += 1

            # 2a) SL intra-bar (prioritaire) — short SL au-dessus entry
            sl_level = avg_entry * (1.0 + sl_pct)
            if high[i] >= sl_level:
                target_size[i] = 0.0
                exec_price[i]  = sl_level
                pos_dir = 0
                avg_entry = 0.0
                hours_held = 0
                continue

            # 2b) Exit funding normalisé (signal au close bar i-1)
            ap = apr[i-1]
            if not np.isnan(ap) and ap < exit_apr:
                target_size[i] = 0.0
                exec_price[i]  = close[i]
                pos_dir = 0
                avg_entry = 0.0
                hours_held = 0
                continue

            # 2c) Timeout
            if hours_held >= hold_max_hours:
                target_size[i] = 0.0
                exec_price[i]  = close[i]
                pos_dir = 0
                avg_entry = 0.0
                hours_held = 0
                continue

        # 3) Si flat, check entry
        entered_this_bar = False
        if pos_dir == 0:
            ap = apr[i-1]
            if not np.isnan(ap) and ap > threshold_apr:
                target_size[i] = -1.0      # short, sized externally
                exec_price[i]  = close[i]  # fill au close de la bar de décision
                pos_dir = -1
                avg_entry = close[i]
                hours_held = 0
                entered_this_bar = True

        # 4) Same-bar SL : si on vient d'entrer ET high[i] >= sl_level,
        #    defer le SL au bar i+1 via pending_sl (sinon entry est silently dropped).
        if entered_this_bar and pos_dir == -1:
            sl_level = avg_entry * (1.0 + sl_pct)
            if high[i] >= sl_level:
                pending_sl = True
                pending_sl_px = sl_level
                # NB: target_size[i] = -1.0 (entry) reste, pos_dir reste -1.
                # Le bar i+1 fermera la position via le bloc pending_sl.

    return target_size, exec_price


class Strategy(BaseStrategy):
    """Funding harvest mean-revert (short-only, single position)."""

    SIZE_PCT = ALLOC_FIXED  # source de verite = module-level ALLOC_FIXED
    MIN_BARS_NEEDED = 24  # warmup minimum

    def score(self, metrics):
        from engine.scoring import score_robust
        return score_robust(metrics)

    def param_space(self, trial) -> Dict[str, Any]:
        return {
            "threshold_apr":  trial.suggest_int("threshold_apr", 100, 2000, step=50),
            "exit_apr":       trial.suggest_int("exit_apr",     -200, 500,  step=50),
            "hold_max_hours": trial.suggest_int("hold_max_hours",  4,  72,  step=4),
            "sl_pct":         round(trial.suggest_float("sl_pct", 0.01, 0.10, step=0.005), 4),
        }

    def compute_target_arrays(self, data: pd.DataFrame, params: Dict[str, Any]):
        required = {"open", "high", "low", "close", "signed_rate", "apr"}
        if data is None or not required.issubset(set(data.columns)):
            return None, None
        if len(data) < self.MIN_BARS_NEEDED + 2:
            return None, None

        ts, px = _funding_harvest_nb(
            data["high"].values.astype(np.float64),
            data["low"].values.astype(np.float64),
            data["close"].values.astype(np.float64),
            data["signed_rate"].values.astype(np.float64),
            data["apr"].values.astype(np.float64),
            float(params["threshold_apr"]),
            float(params["exit_apr"]),
            int(params["hold_max_hours"]),
            float(params["sl_pct"]),
        )
        return pd.Series(ts, index=data.index), pd.Series(px, index=data.index)

    def compute_cash_dividends(self, data: pd.DataFrame,
                               params: Dict[str, Any] = None) -> pd.Series:
        """Per-share cash flow du funding (vbt scale auto par la position courante).

        `cash_dividends` est un revenu PAR SHARE que vbt multiplie automatiquement
        par `assets` (le nombre de shares detenues, negatif pour short).
        Avantage cle vs `cash_earnings` (notional fixe) : suit le compound ET les
        changements de taille (slider size_pct / leverage) sans hack.

        Convention :
          signed_rate > 0 = longs payent shorts.
          Si short (assets < 0) avec signed_rate > 0, on collecte (cash positif).
          → cash_dividends = -signed_rate × close   (negatif)
          → × assets (negatif si short) = positif (revenu)

        Unités (CRITIQUE — fix 2026-05-16) : `signed_rate` est en **%/h** (pas frac/h)
        car Lighter expose `rate` directement en pourcentage. Cross-check avec le
        champ `value` Lighter (= rate × price / 100). Division /100 ici pour avoir
        le vrai $ funding par share, matche l'annualized affiché Lighter (ex BNB 10.5%).
        """
        if "signed_rate" not in data.columns:
            return None
        # Per-share funding revenue paid TO holder. /100 car signed_rate est en %/h.
        # Negate so that short pos collects.
        return -data["signed_rate"].fillna(0) / 100.0 * data["close"]

    def compute_cash_earnings(self, data: pd.DataFrame, size_series: pd.Series,
                              params: Dict[str, Any] = None) -> pd.Series:
        """DEPRECATED — laissee pour compat. Utiliser compute_cash_dividends qui
        scale natif avec la position courante (vbt-native, sans approximation)."""
        return None

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]):
        """Build vbt.Portfolio AND add the collected funding to PnL via cash_dividends.

        VBT seul ne sait pas que pendant qu'on tient une short, on collecte signed_rate
        chaque heure. On passe `cash_dividends = -signed_rate × close` et vbt scale
        automatiquement par la position courante (compound-correct).
        """
        size_s, price_s = self.compute_target_arrays(data, params)
        if size_s is None:
            return None

        # Scale size to SIZE_PCT (kernel emits ±1.0)
        size_scaled = size_s * self.SIZE_PCT

        # Funding per-share, vbt multiplie par position assets automatiquement
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
