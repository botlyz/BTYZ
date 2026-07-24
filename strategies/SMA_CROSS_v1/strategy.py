"""SMA_CROSS_v1 — croisement SMA fast/slow, long only.

Stratégie fixture : trivialement causale et rapide, sert de test au pipeline
quantlab (contract, lookahead, screening...). Aucune prétention d'edge.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from quantlab.contract import BaseStrategy, Signals


class Strategy(BaseStrategy):
    FAMILY = "SMA_CROSS"
    VERSION = 1

    DATA_SOURCE = "lighter"
    TF = "1h"
    WARMUP_BARS = 210           # slow max (200) + marge

    DEFAULT_PARAMS = {"fast": 20, "slow": 100}

    def signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> Signals:
        p = self.full_params(params)
        fast = data["close"].rolling(int(p["fast"])).mean()
        slow = data["close"].rolling(int(p["slow"])).mean()
        above = (fast > slow) & fast.notna() & slow.notna()
        prev = above.shift(1, fill_value=False)
        return Signals(
            long_entries=above & ~prev,      # cross up
            long_exits=~above & prev,        # cross down
        )

    def param_space(self, trial) -> Dict[str, Any]:
        return {
            "fast": trial.suggest_int("fast", 5, 50, step=5),
            "slow": trial.suggest_int("slow", 60, 200, step=20),
        }
