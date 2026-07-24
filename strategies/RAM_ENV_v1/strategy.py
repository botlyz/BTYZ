"""RAM_ENV_v1 — retour à la moyenne par enveloppe sur moyenne mobile.

Thèse : sur un horizon court, un écart marqué du prix à sa moyenne mobile est
en partie du bruit qui se résorbe (sur-réaction, liquidations, mèches). On fade
l'écart et on sort au retour à la moyenne. Voir RATIONALE.md.

Signal (source "lighter" = OHLCV simple), STRICTEMENT causal (rolling past-only) :
  ma    = SMA(ma_window) du close
  bande basse = ma·(1 - env_pct) , bande haute = ma·(1 + env_pct)
  LONG  quand close < bande basse   -> sortie quand close >= ma (retour moyenne)
  SHORT quand close > bande haute   -> sortie quand close <= ma
  SL dur `sl_pct` (FIXED, structurel — un mean-rev sans stop peut saigner en
  tendance forte ; ce n'est pas un paramètre de recherche).

Grille d'optimisation (2 params libres, 6 combinaisons) :
  ma_window ∈ {50, 200}   ·   env_pct ∈ {3 %, 5 %, 8 %}
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from quantlab.contract import BaseStrategy, Signals


class Strategy(BaseStrategy):
    FAMILY = "RAM_ENV"
    VERSION = 1

    DATA_SOURCE = "lighter"      # OHLCV simple
    TF = "1h"                    # timeframe native (référence)
    # l'utilisateur veut screener toutes les timeframes :
    SCREEN_TFS = ["1m", "3m", "5m", "15m", "1h", "4h"]
    WARMUP_BARS = 260            # max(ma_window) + marge

    # point unique du screening (mêmes valeurs sur toutes les paires) :
    # centre de la grille, robuste par défaut.
    DEFAULT_PARAMS = {"ma_window": 200, "env_pct": 0.05}

    # stop de protection structurel — jamais optimisé.
    FIXED = {"sl_pct": 0.15}

    def signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> Signals:
        p = self.full_params(params)
        w = int(p["ma_window"])
        env = float(p["env_pct"])

        close = data["close"]
        ma = close.rolling(w).mean()          # causal (past-only)
        lower = ma * (1.0 - env)
        upper = ma * (1.0 + env)

        long_entries = (close < lower).fillna(False)
        long_exits = (close >= ma).fillna(False)
        short_entries = (close > upper).fillna(False)
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
        }
