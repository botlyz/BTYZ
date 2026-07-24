# BTYZ engine — moteur générique Optuna + Walk-Forward

Plug-and-play : dépose une stratégie sous `src/approach/<APPROACH_ID>/strategy.py`
implémentant `engine.strategy_interface.BaseStrategy`, et lance une opti complète
sur grid (tf × bps × paires) × folds.

## Structure

```
src/
├── engine/                       # le moteur générique
│   ├── strategy_interface.py     # contrat BaseStrategy (param_space, run_backtest, score)
│   ├── config.py                 # paths + defaults
│   ├── metrics.py                # extract_from_portfolio() — port.stats()
│   ├── scoring.py                # score_default + score_high_frequency + score_trend
│   ├── tpe_search.py             # Optuna TPE par fold
│   ├── walk_forward.py           # compute_folds rolling
│   ├── data_loader.py            # load_ohlcv (Lighter 1m / Binance)
│   ├── trades.py                 # extract_trades_df depuis pf.trades
│   ├── approach_loader.py        # discovery + import dynamique
│   ├── wfa_runner.py             # orchestrateur grid × pairs × folds
│   ├── mccv_runner.py            # random-date OOS (MCCV)
│   └── cli.py                    # `python -m engine.cli ...`
└── approach/
    └── <APPROACH_ID>/
        ├── __init__.py
        └── strategy.py           # ta stratégie
```

## Strategy contract

```python
from engine.strategy_interface import BaseStrategy

class Strategy(BaseStrategy):
    def param_space(self, trial):
        return {"window": trial.suggest_int("window", 10, 200), ...}

    def run_backtest(self, data, params):
        # data: DataFrame OHLCV
        # returns: vbt.Portfolio (preferred) or metrics dict
        ...
```

Module-level optionnels `_target_fees`, `_target_freq` sont injectés par le moteur
avant chaque backtest.

## CLI

Toujours lancer depuis `BTYZ/src/`:

```bash
cd /home/devbox/BTYZ/src

# Lister les approches détectées
python -m engine.cli list

# Walk-Forward complet (grid)
python -m engine.cli wfa \
  --approach ATR_ENV_v1 \
  --tf 3m 5m 15m \
  --bps 0 1 2 \
  --pairs BTC ETH HYPE \
  --trials 500 \
  --workers 2 --pair-workers 6

# MCCV (random-date OOS)
python -m engine.cli mccv \
  --approach ATR_ENV_v1 \
  --pairs BTC ETH \
  --tf 3m \
  --bps 1 \
  --n-targets 12 \
  --trials 300
```

## Output

```
results/<APPROACH_ID>/
├── full/
│   └── <tf>_<bps>bps/
│       └── <PAIR>/
│           ├── summary.json     # n_folds, folds[{fold, params, train_metrics, test_metrics}]
│           ├── all_folds.csv    # vue plate
│           └── trades/
│               └── fold_<i>.parquet
└── mccv/
    └── <PAIR>_<tf>_<bps>bps.json   # rows[{target, train_sh, oos_sh, oos_ret, oos_dd, ...}]
```

## Analyse

```bash
marimo edit /home/devbox/BTYZ/notebooks/analyse_engine.py
```

Le notebook détecte automatiquement les approches dans `results/`, puis :
- §1 table des folds avec params + metrics
- §2 cross-run fee sensitivity (par pair)
- §3 stabilité des params par fold
- §4 WFE check (train sharpe vs test sharpe scatter + corr)
- §5 walk-forward equity reconstruite depuis les trades parquet
- §6 MCCV (si disponible) — box plot des OOS sharpes

## Exemple : approche `ATR_ENV_v1`

Mean-reversion sur enveloppe SMA ± atr_mult×ATR avec SL dynamique ATR + cooldown.
- Params optimisés : `ma_window`, `atr_window`, `atr_mult`, `sl_mult`, `ohlc4`
- Backtest : `vbt.Portfolio.from_orders(size_type='TargetPercent')`

Smoke test (15min × 1bps × BTC, 30 trials, 7 folds) → ~50s, 5/7 folds profitables.
