"""Contrat de stratégie quantlab.

Une stratégie est un dossier dans BTYZ/strategies/<FAMILY>_v<N>/ contenant :
  - strategy.py     : une classe `Strategy(BaseStrategy)`
  - RATIONALE.md    : la thèse économique (obligatoire)
  - manifest.yaml   : métadonnées optionnelles (tfs à screener, paires exclues...)

Le moteur backteste lui-même via vbt.Portfolio.from_signals — la stratégie ne
produit QUE des signaux. C'est ce qui rend le test anti-look-ahead générique
et le pooling multi-paires possible.

Règle causale : signals(data.iloc[:i]) doit produire les mêmes valeurs sur
[0, i) que signals(data) — vérifié automatiquement par quantlab.lookahead.
"""
from __future__ import annotations

import hashlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Signals:
    """Sortie de Strategy.signals(). Toutes les séries sont indexées comme `data`.

    Les booléens manquants valent False (pas de trade). sl/tp en fraction
    (0.05 = 5 %), td_stop en nombre de barres.
    """
    long_entries: Optional[pd.Series] = None
    long_exits: Optional[pd.Series] = None
    short_entries: Optional[pd.Series] = None
    short_exits: Optional[pd.Series] = None
    sl_stop: Optional[float] = None
    tp_stop: Optional[float] = None
    td_stop: Optional[int] = None          # time-decay stop, en barres

    def any_entries(self) -> bool:
        for s in (self.long_entries, self.short_entries):
            if s is not None and bool(np.asarray(s).any()):
                return True
        return False

    def align(self, index: pd.Index) -> "Signals":
        """Réaligne (reindex, fill False) toutes les séries sur `index`."""
        def _fix(s):
            if s is None:
                return None
            return s.reindex(index, fill_value=False).astype(bool)
        return Signals(
            long_entries=_fix(self.long_entries),
            long_exits=_fix(self.long_exits),
            short_entries=_fix(self.short_entries),
            short_exits=_fix(self.short_exits),
            sl_stop=self.sl_stop, tp_stop=self.tp_stop, td_stop=self.td_stop,
        )


class BaseStrategy(ABC):
    """Contrat minimal. Sous-classe obligatoire nommée `Strategy`."""

    # --- identité (ledger / research_debt) ---
    FAMILY: str = ""            # ex. "OI_FADE" — clé du research_debt, versions confondues
    VERSION: int = 1

    # --- données ---
    DATA_SOURCE: str = "lighter"   # source engine.data_loader.load_ohlcv
    TF: str = "1h"                 # timeframe native
    SCREEN_TFS: Optional[List[str]] = None   # None -> [TF] ; sinon sous-ensemble de config.TIMEFRAMES
    WARMUP_BARS: int = 300         # barres consommées par les indicateurs

    # --- paramètres ---
    DEFAULT_PARAMS: Dict[str, Any] = {}   # utilisés au screening (identiques sur toutes les paires)
    FIXED: Dict[str, Any] = {}            # fixés par logique économique, jamais optimisés

    @abstractmethod
    def signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> Signals:
        """Calcule les signaux. DOIT être causal (aucune info du futur)."""

    @abstractmethod
    def param_space(self, trial) -> Dict[str, Any]:
        """Espace Optuna. <= config.MAX_FREE_PARAMS dimensions (refus sinon)."""

    # ------------------------------------------------------------------ infra
    @classmethod
    def strategy_id(cls) -> str:
        return f"{cls.FAMILY}_v{cls.VERSION}"

    @classmethod
    def code_hash(cls) -> str:
        """Hash du code source du module de la stratégie (fichier complet)."""
        src_file = inspect.getsourcefile(cls)
        with open(src_file, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]

    def n_free_params(self) -> int:
        """Nombre de dimensions de param_space, compté via un trial figé."""
        import optuna
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        study.optimize(lambda t: (self.param_space(t), 0.0)[1], n_trials=1,
                       catch=(Exception,))
        return len(study.trials[0].params)

    def full_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {**self.FIXED, **self.DEFAULT_PARAMS, **params}


def validate_strategy(cls) -> List[str]:
    """Validation statique du contrat. Retourne la liste des erreurs (vide = OK)."""
    errors: List[str] = []
    if not getattr(cls, "FAMILY", ""):
        errors.append("FAMILY manquant")
    if not isinstance(getattr(cls, "VERSION", None), int):
        errors.append("VERSION doit être un int")
    if not getattr(cls, "DEFAULT_PARAMS", None):
        errors.append("DEFAULT_PARAMS vide — requis pour le screening")
    from quantlab.config import TIMEFRAMES
    if getattr(cls, "TF", None) not in TIMEFRAMES:
        errors.append(f"TF invalide (choix: {TIMEFRAMES})")
    for tf in (getattr(cls, "SCREEN_TFS", None) or []):
        if tf not in TIMEFRAMES:
            errors.append(f"SCREEN_TFS: tf inconnue {tf}")
    return errors
