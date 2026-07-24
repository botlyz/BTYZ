"""RAM_ENV_v1 — retour à la moyenne par enveloppe + filtre de tendance.

Thèse : sur un horizon court, un écart marqué du prix à sa moyenne mobile est
en partie du bruit qui se résorbe. On fade l'écart et on sort au retour à la
moyenne — MAIS seulement dans le sens de la tendance de fond (grande SMA), pour
ne pas se coucher devant un train en tendance forte. Voir RATIONALE.md.

Signal (source "lighter" = OHLCV simple), STRICTEMENT causal (rolling past-only) :
  ma    = SMA(ma_window) du close
  trend = SMA(sma_trend) du close    (filtre de tendance, grande fenêtre)
  bande basse = ma·(1 - env_pct) , bande haute = ma·(1 + env_pct)
  LONG  quand close < bande basse  ET  close >= trend  (fade le creux en uptrend)
  SHORT quand close > bande haute  ET  close <= trend  (fade le rip en downtrend)
  Sortie : retour à la moyenne (close recroise `ma`). SL dur `sl_pct` (FIXED).

Grille d'optimisation (3 params libres, 12 combinaisons) :
  ma_window ∈ {50, 200} · env_pct ∈ {3 %, 5 %, 8 %} · sma_trend ∈ {500, 2000}
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from quantlab.contract import BaseStrategy, Signals


class Strategy(BaseStrategy):
    FAMILY = "RAM_ENV"
    VERSION = 1

    DATA_SOURCE = "lighter"      # OHLCV simple
    TF = "3m"                    # timeframe native : la micro-réversion vit sur du rapide
    SCREEN_TFS = ["1m", "3m", "5m"]
    WARMUP_BARS = 2100           # max(sma_trend) + marge

    # point unique du screening (mêmes valeurs sur toutes les paires) :
    # MA courte + enveloppe serrée + filtre de tendance permissif -> assez de
    # trades pour une p-value fiable. Reste un point de la grille déclarée.
    DEFAULT_PARAMS = {"ma_window": 50, "env_pct": 0.03, "sma_trend": 500}

    # stop de protection structurel — jamais optimisé.
    FIXED = {"sl_pct": 0.15}

    def signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> Signals:
        p = self.full_params(params)
        w = int(p["ma_window"])
        env = float(p["env_pct"])
        st = int(p["sma_trend"])

        close = data["close"]
        ma = close.rolling(w).mean()          # causal (past-only)
        trend = close.rolling(st).mean()      # filtre de tendance, causal
        lower = ma * (1.0 - env)
        upper = ma * (1.0 + env)

        # filtre de RÉGIME : la MA rapide au-dessus/dessous de la grande SMA
        # (ne pas utiliser close vs trend -> le creux d'entrée annulerait le
        # signal, vidant la stratégie sur les majors peu volatiles).
        uptrend = ma >= trend
        downtrend = ma <= trend

        long_entries = ((close < lower) & uptrend).fillna(False)
        long_exits = (close >= ma).fillna(False)
        short_entries = ((close > upper) & downtrend).fillna(False)
        short_exits = (close <= ma).fillna(False)

        return Signals(
            long_entries=long_entries,
            long_exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            sl_stop=float(p["sl_pct"]),
        )

    def param_space(self, trial) -> Dict[str, Any]:
        return {
            "ma_window": trial.suggest_categorical("ma_window", [50, 200]),
            "env_pct": trial.suggest_categorical("env_pct", [0.03, 0.05, 0.08]),
            "sma_trend": trial.suggest_categorical("sma_trend", [500, 2000]),
        }
