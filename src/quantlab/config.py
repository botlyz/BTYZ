"""Configuration centrale du pipeline quantlab.

Tous les seuils viennent du document de référence (docs/QUANTLAB.md §tableau).
Ne pas les modifier pour faire passer une stratégie.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- chemins
SRC_ROOT = Path(__file__).resolve().parent.parent          # BTYZ/src
PROJECT_ROOT = SRC_ROOT.parent                             # BTYZ
DATA_ROOT = PROJECT_ROOT / "data"
STORE_ROOT = DATA_ROOT / "quantlab_store"                  # parquet canonique
RESULTS_ROOT = PROJECT_ROOT / "results" / "quantlab"
STRATEGIES_ROOT = PROJECT_ROOT / "strategies"
LEDGER_DB = PROJECT_ROOT / "ledger.db"
LIQUIDITY_JSON = PROJECT_ROOT / "liquidity.json"

# ---------------------------------------------------------------- univers
SOURCES = ["lighter", "binance"]
TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h", "4h"]
# alias tf -> freq pandas (resample / vbt)
FREQ_MAP = {"1m": "1min", "3m": "3min", "5m": "5min",
            "15m": "15min", "1h": "1h", "4h": "4h"}
MIN_BARS = 1000            # série plus courte -> paire exclue pour cette tf

# ---------------------------------------------------------------- coûts
DEFAULT_INIT_CASH = 10_000
DEFAULT_SLIPPAGE = 0.0002
DEFAULT_FEES_BPS = 3.0      # taker Lighter ~ 2-3 bps ; override par run
ILLIQUID_FEES_BPS = 10.0    # pénalité screening pour paires hors liquidity.json

# ---------------------------------------------------------------- holdout (0.2)
HOLDOUT_FRACTION = 0.20     # 20 % de l'historique...
HOLDOUT_MAX_DAYS = 183      # ...plafonné à 6 mois (décision utilisateur)
HOLDOUT_MIN_DAYS = 30       # paire trop jeune -> exclue de l'univers

# ---------------------------------------------------------------- look-ahead (0.3)
LOOKAHEAD_N_PREFIXES = 20
LOOKAHEAD_MIN_PREFIX = 500  # barres minimum du premier préfixe

# ---------------------------------------------------------------- étape 1 screening
FDR_Q = 0.10                     # Benjamini-Hochberg
MIN_POSITIVE_PAIR_FRAC = 0.45    # généralisation du "5 paires sur 11"
NULL_STRATEGIES_PER_CELL = 50    # stratégies aléatoires appariées / cellule
NULL_QUANTILE = 0.95             # le candidat doit battre q95 de la nulle
SCREEN_BOOTSTRAP = 1000          # bootstrap stationnaire pour la p-value du Sharpe

# ---------------------------------------------------------------- étape 2 optuna WF
MAX_FREE_PARAMS = 4              # refus au-delà
MIN_TRADES_PER_PARAM = 20        # par fold de validation
DEFAULT_K_FOLDS = 5
DEFAULT_EMBARGO_BARS = None      # None -> auto: max(hold, indicateurs) estimé
DEFAULT_TRIALS = 300             # budget fixé AVANT le run, loggé au ledger
DEFAULT_ANCHORED = False         # folds glissants par défaut

# ---------------------------------------------------------------- étape 3 plateau
PERTURBATION_PCT = 0.20          # ±20 % sur chaque paramètre
MAX_SHARPE_DROP = 0.40           # chute > 40 % -> pic étroit -> REJET

# ---------------------------------------------------------------- étape 4 stats
PBO_BLOCKS = 16                  # S blocs pour le CSCV
PBO_REJECT = 0.30                # PBO > 30 % -> rejet
PBO_PASS = 0.15                  # < 15 % -> pass
DSR_CONFIDENCE = 0.95            # DSR > 0 à 95 %
PERMUTATION_RUNS = 200           # relances pipeline sur données permutées
PERMUTATION_PVALUE = 0.05
PERMUTATION_BLOCK_BARS = 24     # block bootstrap: taille de bloc (barres)

# ---------------------------------------------------------------- étape 5 déploiement
WFE_REJECT = 0.30                # < 0.3 -> rejet (l'adaptatif ne sauve PAS)
WFE_FIXED = 0.50                 # > 0.5 + optima stables -> déploiement fixe

# ---------------------------------------------------------------- pipeline
STAGES = [
    "contract",      # 0.3 look-ahead + validation du contrat
    "screening",     # étape 1
    "optimize",      # étape 2
    "plateau",       # étape 3
    "stats",         # étape 4 (PBO + DSR [+ permutation])
    "deploy_rule",   # étape 5
    "holdout",       # étape 6 (gate humain, one-shot)
    "incubation",    # étape 7 (gate humain, live)
    "production",
]
