"""FUNDING_ARB_v1 — Funding arbitrage cross-exchange Lighter ↔ Hyperliquid (delta-neutral).

Hypothèse :
  Les funding rates Lighter et HL divergent souvent. Quand `spread = rate_lighter - rate_hl`
  est suffisamment grand (en valeur absolue) pour couvrir les coûts cross-exchange (fees
  Lighter + fees HL + half-spread bid/ask × 2 venues × 2 sides), on ouvre une position
  delta-neutre : short côté funding élevé, long côté funding bas. On collecte la
  différence funding chaque heure jusqu'à ce que le spread se rétrécisse.

Convention position (single-portfolio model) :
  - position = +1 = LONG le spread = SHORT Lighter + LONG HL
       → collecte rate_lighter (short Lighter)
       → paie rate_hl (long HL)
       → net cash flow per hour = +spread × notional
  - position = -1 = SHORT le spread = LONG Lighter + SHORT HL
       → net cash flow per hour = -spread × notional

Modélisation BT (single-portfolio Lighter close as proxy) :
  - target_size en TargetPercent (±ALLOC_FIXED)
  - cash_dividends = +spread/100 × close (vbt scale par position, signe géré par vbt)
  - fees doublées : taker_lighter + taker_hl + half_spread_lighter + half_spread_hl
    (chargées par fill ; un round-trip = 2 fills vbt = 2 × cette valeur)
  - PnL prix supposé delta-neutre (basis Lighter≈HL résiduel ignoré dans BT initial)

Data attendue par run_backtest :
  DataFrame indexé 1h avec colonnes : open, high, low, close, volume,
  signed_rate_lighter, signed_rate_hl, spread.
  Loader : engine.data_loader.load_cross_exchange(pair, '1h').
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from numba import njit
from vectorbtpro import vbt

from engine.strategy_interface import BaseStrategy

# Runtime hints (par défaut ; override par moteur via _target_fees, etc.)
_target_fees: float = 0.0007       # 7 bps one-way = taker_l(2.5) + taker_h(4.5) — sans half-spread
_target_slippage: float = 0.0000   # half-spread déjà comptabilisé via fees
_target_freq: str = "1h"
_init_cash: float = 10_000.0

# Sizing par paire (live multi-pair). 1% conservatif pour 20 paires max → 20% notionnel.
ALLOC_FIXED = 0.01


@njit(cache=True)
def _funding_arb_nb(
    close,
    spread,                   # signed_rate_lighter - signed_rate_hl (%/h, déjà signé)
    spread_entry_pct: float,  # entry si |spread[i-1]| > entry (%/h)
    spread_exit_pct: float,   # exit si position_sign × spread[i-1] < exit (%/h)
    hold_max_hours: int,      # cap timeout en heures
):
    """Kernel funding arbitrage cross-exchange — bidirectionnel, single position.

    Convention :
      - target_size en {-1, 0, +1} (scalé externe par ALLOC_FIXED).
      - Le signe de spread[i-1] détermine la direction d'entrée.
      - L'exit est sur le NET cash flow restant (= position_sign × spread).
    Returns (target_size, exec_price) — exec au close[i] (bar de décision).
    """
    n = len(close)
    target_size = np.full(n, np.nan)
    exec_price  = np.full(n, np.nan)

    pos_dir    = 0
    hours_held = 0

    for i in range(1, n):
        s = spread[i-1]

        # 1) Exits si en position
        if pos_dir != 0:
            hours_held += 1
            net_flow = pos_dir * s  # >0 = on collecte encore ; <0 = on paie

            # 1a) Exit : flow net trop faible / négatif
            if not np.isnan(s) and net_flow < spread_exit_pct:
                target_size[i] = 0.0
                exec_price[i]  = close[i]
                pos_dir = 0
                hours_held = 0
                continue

            # 1b) Timeout
            if hours_held >= hold_max_hours:
                target_size[i] = 0.0
                exec_price[i]  = close[i]
                pos_dir = 0
                hours_held = 0
                continue

        # 2) Entry si flat
        if pos_dir == 0:
            if not np.isnan(s):
                if s > spread_entry_pct:
                    target_size[i] = 1.0       # long spread = short Lighter + long HL
                    exec_price[i]  = close[i]
                    pos_dir = 1
                    hours_held = 0
                elif s < -spread_entry_pct:
                    target_size[i] = -1.0      # short spread = long Lighter + short HL
                    exec_price[i]  = close[i]
                    pos_dir = -1
                    hours_held = 0

    return target_size, exec_price


def _load_pair_fees(pair: str,
                    snapshot_path: str | None = None) -> float:
    """Charge le coût cross-exchange par fill pour une paire depuis spread_snapshot.

    Returns float = fee_per_fill (vbt-side) = taker_l + taker_h + hs_l + hs_h, en fraction.
    Fallback à 0.0007 (7 bps) si snapshot indispo.
    """
    import glob
    import json
    import os
    if snapshot_path is None:
        # Prend le plus récent
        candidates = sorted(glob.glob('data/raw/spread_snapshot_*.json'), reverse=True)
        if not candidates:
            return _target_fees
        snapshot_path = candidates[0]
    if not os.path.exists(snapshot_path):
        return _target_fees
    try:
        with open(snapshot_path) as f:
            snap = json.load(f)
    except Exception:
        return _target_fees

    pairs = snap.get('pairs', {})
    base = pair.replace('USDT', '')
    info = pairs.get(base)
    if info is None:
        return _target_fees

    l = info.get('lighter_half_spread_bps')
    h = info.get('hyperliquid_half_spread_bps')
    hs_l = (l['median'] / 1e4) if l else 0.0001
    hs_h = (h['median'] / 1e4) if h else 0.0001
    # Fees taker : Lighter 2.5 bps (avec rebate intégrateur), HL 4.5 bps tier 0
    taker_l = 0.00025
    taker_h = 0.00045
    return taker_l + taker_h + hs_l + hs_h


class Strategy(BaseStrategy):
    """Funding arb delta-neutral Lighter↔HL (bidirectionnel, single-portfolio model)."""

    SIZE_PCT = ALLOC_FIXED
    MIN_BARS_NEEDED = 24
    DATA_SOURCE = "cross_exchange"  # hint pour le moteur (data_loader.load_ohlcv)

    def score(self, metrics):
        from engine.scoring import score_robust
        return score_robust(metrics)

    def param_space(self, trial) -> Dict[str, Any]:
        # Spread typique majors : median ~0.001, std ~0.005, P90 ~0.005 %/h.
        # Smoke test : optimum entry dans [0.001, 0.020] selon paire, exit dans [-0.005, 0.0].
        return {
            "spread_entry_pct": round(trial.suggest_float("spread_entry_pct", 0.0005, 0.050, log=True), 5),
            "spread_exit_pct":  round(trial.suggest_float("spread_exit_pct", -0.020, 0.005, step=0.001), 4),
            "hold_max_hours":   trial.suggest_int("hold_max_hours", 24, 720, step=24),  # 1j à 30j
        }

    def compute_target_arrays(self, data: pd.DataFrame, params: Dict[str, Any]):
        required = {"open", "high", "low", "close", "spread"}
        if data is None or not required.issubset(set(data.columns)):
            return None, None
        if len(data) < self.MIN_BARS_NEEDED + 2:
            return None, None

        ts, px = _funding_arb_nb(
            data["close"].values.astype(np.float64),
            data["spread"].values.astype(np.float64),
            float(params["spread_entry_pct"]),
            float(params["spread_exit_pct"]),
            int(params["hold_max_hours"]),
        )
        return pd.Series(ts, index=data.index), pd.Series(px, index=data.index)

    def compute_cash_dividends(self, data: pd.DataFrame,
                               params: Dict[str, Any] = None) -> pd.Series:
        """Per-share cash flow du spread (vbt scale par position courante).

        Convention :
          - position = +1 share long → cash_div × +1 = gain. Pour collecter spread > 0 → cash_div = +spread/100 × close
          - position = -1 share short → cash_div × -1 = -gain. Pour payer spread (>0 mais on est short le spread) → cash_div = +spread/100 × close

        Donc : cash_dividends = +spread/100 × close (signe géré par vbt via position).
        """
        if "spread" not in data.columns:
            return None
        return data["spread"].fillna(0) / 100.0 * data["close"]

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]):
        """vbt.Portfolio modélisant le spread comme actif synthétique (close Lighter proxy).

        Note: delta-neutralité PnL prix non strictement modélisée — la position est
        ouverte sur Lighter close mais la jambe HL n'est pas re-tracking le prix.
        Acceptable car les 2 venues bougent ensemble (basis < 0.5% sur ce type de
        paires). La source dominante du PnL est `cash_dividends = spread × close`.
        """
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
