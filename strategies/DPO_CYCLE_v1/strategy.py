"""DPO_CYCLE_v1 — cycle détendancé (Detrended Price Oscillator) normalisé ±1,5σ.

Thèse : le DPO retire la tendance pour isoler le CYCLE. Normalisé par son
écart-type, ses creux/sommets deviennent comparables d'une paire à l'autre.
On achète les creux cycliques à la remontée, on sort au sommet cyclique.

Indicateur (causal) :
  k    = period//2 + 1                       (décalage classique du DPO)
  dpo  = close(t-k) - SMA(period)(t)         (prix passé - tendance courante)
  z    = dpo / rolling_std(norm)(dpo)        (normalisation par l'écart-type)

Signal (long + short, sortie sur l'extrême cyclique opposé, sans stop) :
  LONG  entrée : z croise ↑ -thr   (remontée depuis survente)
  LONG  sortie : z croise ↑ +thr   (sommet cyclique)
  SHORT entrée : z croise ↓ +thr   (repli depuis surachat)
  SHORT sortie : z croise ↓ -thr   (creux cyclique)

Grille d'optimisation (3 params libres, 27 combinaisons) :
  period ∈ {10, 20, 40} · norm ∈ {50, 100, 200} · thr ∈ {1.0, 1.5, 2.0}
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from quantlab.contract import BaseStrategy, Signals


class Strategy(BaseStrategy):
    FAMILY = "DPO_CYCLE"
    VERSION = 1

    DATA_SOURCE = "lighter"      # OHLCV simple
    TF = "4h"                    # native : le builder tourne en 4h (cycle lent)
    SCREEN_TFS = ["1m", "3m", "5m", "15m", "1h", "4h"]   # on cherche la bonne TF
    WARMUP_BARS = 300            # max(period)+max(norm)+décalage + marge

    # preset exact du builder (period 20, norm 100, seuil 1,5σ).
    DEFAULT_PARAMS = {"period": 20, "norm": 100, "thr": 1.5}

    # pas de stop / TP : sortie sur l'extrême cyclique opposé (fidèle au builder).
    FIXED: Dict[str, Any] = {}

    def _zscore(self, close: pd.Series, period: int, norm: int) -> pd.Series:
        k = period // 2 + 1
        dpo = close.shift(k) - close.rolling(period).mean()   # causal
        std = dpo.rolling(norm).std()
        return dpo / std.replace(0.0, pd.NA)

    def signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> Signals:
        p = self.full_params(params)
        thr = float(p["thr"])
        z = self._zscore(data["close"], int(p["period"]), int(p["norm"]))
        zp = z.shift(1)

        def cross_up(level):
            return ((zp < level) & (z >= level)).fillna(False)

        def cross_dn(level):
            return ((zp > level) & (z <= level)).fillna(False)

        return Signals(
            long_entries=cross_up(-thr),
            long_exits=cross_up(thr),
            short_entries=cross_dn(thr),
            short_exits=cross_dn(-thr),
        )

    def param_space(self, trial) -> Dict[str, Any]:
        return {
            "period": trial.suggest_categorical("period", [10, 20, 40]),
            "norm": trial.suggest_categorical("norm", [50, 100, 200]),
            "thr": trial.suggest_categorical("thr", [1.0, 1.5, 2.0]),
        }
