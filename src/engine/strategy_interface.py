"""Strategy contract: subclass BaseStrategy and export Strategy from approach/<ID>/strategy.py."""
from abc import ABC, abstractmethod
from typing import Any, Dict

import optuna


class BaseStrategy(ABC):
    """Minimal contract for a BTYZ strategy plugged into the engine."""

    @abstractmethod
    def param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define the Optuna search space, return dict {param_name: value}."""
        ...

    @abstractmethod
    def run_backtest(self, data, params: Dict[str, Any]):
        """Execute backtest. Return a vbt.Portfolio (preferred) or a metrics dict."""
        ...

    def score(self, metrics: Dict[str, Any]) -> float:
        """Composite score for Optuna. Override for custom scoring."""
        from .scoring import score_default
        return score_default(metrics)
