"""Engine global config: paths, WFA defaults, scoring thresholds."""
from pathlib import Path

# Project paths (BTYZ/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
RESULTS_ROOT = PROJECT_ROOT / "results"
LOGS_ROOT = PROJECT_ROOT / "logs"
APPROACH_ROOT = PROJECT_ROOT / "src" / "approach"

# Approach folder layout: src/approach/<APPROACH_ID>/strategy.py exporting `Strategy`
APPROACH_STRATEGY_FILE = "strategy.py"

# Trading defaults
# Note: actual fees come from CLI `--bps` flag (run_grid converts bps × 1e-4).
# DEFAULT_SLIPPAGE is the strategy-side default applied when run_backtest is
# called outside the engine context.
DEFAULT_SLIPPAGE = 0.0002   # 2 bps
INIT_CASH = 10_000
MIN_TRADES = 30

# Walk-forward defaults
DEFAULT_TRAIN_DAYS = 90
DEFAULT_TEST_DAYS = 21
DEFAULT_STEP_DAYS = 21
DEFAULT_TRIALS_PER_FOLD = 500
DEFAULT_MIN_TRADES_PER_FOLD = 30
DEFAULT_FOLD_WORKERS = 2
DEFAULT_PAIR_WORKERS = 6
DEFAULT_WARMUP_BARS = 300

# Timeframe → minutes
BAR_MINUTES = {
    "1min": 1, "3min": 3, "5min": 5, "15min": 15, "30min": 30,
    "1h": 60, "2h": 120, "4h": 240, "1d": 1440, "1D": 1440,
}

# Short TF alias → vbt freq string
TF_ALIAS = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "30m": "30min",
    "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1D",
}

FREQ_MAP = {**{k: k for k in BAR_MINUTES}, **TF_ALIAS}

# Robustness classification thresholds
VIABLE_WFE_PCT = 50.0
VIABLE_POS_FOLDS_PCT = 70.0
MARGINAL_WFE_PCT = 15.0
