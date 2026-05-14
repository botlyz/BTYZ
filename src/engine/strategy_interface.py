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

    def compute_target_arrays(self, data, params: Dict[str, Any]):
        """OPTIONAL — returns (size_series, price_series) for `data` with `params`.

        Used by the analyse notebook to stitch a global vbt.Portfolio across
        all walk-forward folds (instead of concatenating per-fold equities).

        Must be a pure transformation: same call as inside run_backtest but
        without the Portfolio construction. Default returns (None, None) and
        the notebook falls back to per-fold equity concat.
        """
        return None, None
