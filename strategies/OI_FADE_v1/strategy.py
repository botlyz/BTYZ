"""OI_FADE_v1 — fade du pump surpeuplé (OI Binance croissant + move intense).

Migration 1:1 de archive/approach/OI_FADE_v1/strategy.py vers le contrat
quantlab (signals -> le moteur backteste). Voir RATIONALE.md pour la thèse.

Signal (1h, source "oi" = OHLCV + colonne `oi` Binance) :
  doi   = variation OI sur doi_w barres ; r = ret prix sur doi_w barres
  z-scores ROLLING past-only (fenêtre ROLL=240 barres) ->
      inten = max(z_doi, 0) * |z_r|
  entrée SHORT quand inten > quantile rolling(ROLL, pctl) ET r > 0 ET doi > 0.
  Sortie : horizon fixe `hold` barres (td_stop) + SL dur sl_pct.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from quantlab.contract import BaseStrategy, Signals


class Strategy(BaseStrategy):
    FAMILY = "OI_FADE"
    VERSION = 1

    DATA_SOURCE = "oi"          # OHLCV Lighter + open interest Binance (colonne `oi`)
    TF = "1h"
    SCREEN_TFS = ["1h"]         # OI Binance en granularité 5min -> pas sous 15m
    WARMUP_BARS = 260           # ROLL + doi_w max (240 + 8 + marge)

    # médianes des runs WFA existants (results/OI_FADE_v1/full, 597 folds)
    DEFAULT_PARAMS = {"doi_w": 6, "pctl": 0.93, "hold": 36, "sl_pct": 0.08}

    # structurel : le scoring par fold se fait sur des tranches de 30j (720 barres
    # 1h) -> une fenêtre rolling 30j y serait NaN partout ; 10j max structurel.
    FIXED = {"roll": 240}

    def signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> Signals:
        p = self.full_params(params)
        roll = int(p["roll"])
        w = int(p["doi_w"])

        c, oi = data["close"], data["oi"]
        doi = oi.pct_change(w)
        r = c.pct_change(w)
        zd = (doi - doi.rolling(roll).mean()) / doi.rolling(roll).std()
        zr = (r - r.rolling(roll).mean()) / r.rolling(roll).std()
        inten = zd.clip(lower=0) * zr.abs()
        thr = inten.rolling(roll).quantile(float(p["pctl"]))
        se = ((inten > thr) & (r > 0) & (doi > 0)).fillna(False)

        return Signals(
            short_entries=se,
            td_stop=int(p["hold"]),          # barres 1h == heures
            sl_stop=float(p["sl_pct"]),
        )

    def param_space(self, trial) -> Dict[str, Any]:
        return {
            "doi_w": trial.suggest_int("doi_w", 2, 8, step=2),
            "pctl": round(trial.suggest_float("pctl", 0.85, 0.99, step=0.02), 2),
            "hold": trial.suggest_int("hold", 12, 60, step=12),
            "sl_pct": round(trial.suggest_float("sl_pct", 0.04, 0.16, step=0.04), 2),
        }
