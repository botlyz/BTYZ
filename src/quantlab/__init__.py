"""quantlab — pipeline de validation de stratégies quant.

Package léger : aucun import lourd au niveau module. Importer les sous-modules
explicitement (`from quantlab import ledger`, `from quantlab.backtest import ...`).
"""
__all__ = [
    "config", "contract", "ledger", "registry",
    "lookahead", "backtest", "progress",
]
