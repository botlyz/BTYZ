"""RAM_DCA_v1_MULTI_FULL_SCOREV2 — fork MULTI_FULL, score_robust_v2.

Identique à RAM_DCA_v1_MULTI_FULL pour kernel + param_space + run_backtest.
Seule différence : `score()` utilise `score_robust_v2` (cf. engine.scoring) qui
rend le trade count dominant dans le score → force optuna à choisir des params
qui tradent ≥ 30/fold au lieu des 5-15/fold habituels.
"""
from __future__ import annotations

from approach.RAM_DCA_v1_MULTI_FULL.strategy import Strategy as _MultiFullStrategy


class Strategy(_MultiFullStrategy):
    def score(self, metrics):
        from engine.scoring import score_robust_v2
        return score_robust_v2(metrics)
