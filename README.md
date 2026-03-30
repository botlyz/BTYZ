# BTYZ

Backtesting, optimisation de stratégies mean reversion crypto, et pipeline ML order flow — BTC/Binance.

## Structure

```
BTYZ/
├── config.py
├── data/raw/                         # OHLCV CSV (perp 1m, spot 1m, metrics 5m)
├── src/
│   ├── download.py                   # téléchargement via CCXT + proxies
│   ├── opti.py                       # optimisation walk-forward multiprocess (VBT)
│   ├── strategies/
│   │   ├── keltner.py                # Keltner Channel (EMA + ATR)
│   │   ├── atr_envelope.py           # Enveloppe ATR (SMA + ATR)
│   │   └── ram_dca.py                # RAM DCA (mean reversion + DCA adaptatif)
│   └── ml/
│       ├── data.py                   # chargement + normalisation rolling z-score
│       ├── labels.py                 # Triple Barrier + Max Gain/Loss (2 outputs)
│       ├── features.py               # features brutes + momentum CVD/prix
│       ├── train_xgb.py              # XGBoost walk-forward (classification)
│       ├── train_xgb_rr.py           # XGBoost walk-forward (régression R/R dynamique)
│       └── train_lstm.py             # LSTM walk-forward (classification)
├── notebooks/
│   ├── analyse_full.py               # analyse marimo multi-stratégies
│   └── research/                     # notebooks de recherche
├── cache/
│   ├── *.pickle                      # résultats opti VBT
│   └── ml/                           # résultats ML (pkl par config)
└── logs/                             # logs horodatés des runs ML
```

## Partie 1 — Mean Reversion (VBT)

Stratégies d'enveloppe testées sur toutes les paires :

- **Enveloppe ATR** : SMA ± mult × ATR
- **Keltner Channel** : EMA ± mult × ATR + stop loss optionnel
- **RAM DCA** : mean reversion avec entrées progressives (DCA adaptatif)

### Workflow

```
1. src/download.py      → télécharge les données OHLCV
2. src/opti.py          → grid search walk-forward multiprocess, sauvegarde pickle
3. notebooks/analyse_full.py → charge le pickle, analyse robustesse, backtest final
```

### Optimisation

- `vbt.Splitter` — walk-forward 10 fenêtres (70% train / 30% test)
- `vbt.parameterized()` + `engine='pathos'` — multiprocess
- Métriques : sharpe, drawdown, win rate, profit factor, calmar, sortino
- Filtre robustesse : sharpe > 0 sur 50%+ des fenêtres, DD max < 30%

## Partie 2 — ML Order Flow (branche feature/ml-signal)

Pipeline ML pour trouver de l'alpha dans les déséquilibres d'order flow.

### Données

- BTC/USDT perp 1m + spot 1m + metrics 5m (Binance)
- 2.9M bougies M1, 2020-09 → 2026-03
- Features : CVD perp/spot, OI, OI delta, ls_ratio, taker_ls_vol, momentum prix
- Normalisation : rolling z-score (1440 bars, sans lookahead)

### Phase 1 — XGBoost (classification Triple Barrier)

**Label** : Triple Barrier (Lopez de Prado)
- +1 = prix touche TP avant SL → trade gagnant
- -1 = prix touche SL avant TP → trade perdant
- 0  = ni l'un ni l'autre (neutre, exclu)

**Résultats** (swing_30min, tp=1.5%, sl=0.9%) :
- Lift moyen OOS : **+0.102** (lift = precision - tp_rate, > 0 = vrai alpha)
- Folds positifs : 76.8% sur 5 ans
- ~761 signaux/mois OOS
- Features #1 : price_d30, price_d60, cvd_spot, ls_ratio, oi

### Phase 1b — XGBoost (régression R/R dynamique)

**Label** : 2 outputs sur horizon 90 bougies (1h30)
- `max_gain_90m` = meilleur move accessible dans les 90 prochaines bougies
- `max_loss_90m` = pire move accessible dans les 90 prochaines bougies

**Signal** : entrer quand R/R prédit = max_gain / abs(max_loss) > seuil
→ TP et SL dynamiques par bougie, pas de seuil fixé à l'avance

### Phase 2 — LSTM (classification Triple Barrier)

Même label que XGBoost mais le modèle voit une **séquence de 60 bougies**.
Théoriquement capable de détecter des patterns temporels (divergence CVD qui se forme).

**Résultat** : LSTM underperform XGBoost sur toutes les configs.
Cause identifiée : features uniquement flow sans prix → prix ajouté (return_1m, close_z) pour relance.

### Lancement ML

```bash
# Phase 1 — XGBoost classification
.venv/bin/python src/ml/train_xgb.py

# Phase 1b — XGBoost R/R dynamique
.venv/bin/python src/ml/train_xgb_rr.py
.venv/bin/python src/ml/train_xgb_rr.py --horizon 60
.venv/bin/python src/ml/train_xgb_rr.py --horizon 120

# Phase 2 — LSTM
.venv/bin/python src/ml/train_lstm.py --config swing_30min
```

## Stack

- Python 3.13
- VectorBT PRO — backtesting + indicateurs
- XGBoost 3.2 + CUDA — GPU training
- PyTorch 2.6 + CUDA — LSTM
- Numba — kernels labels (Triple Barrier, Max Gain/Loss)
- Optuna — optimisation TPE (stratégies)
- Plotly / Marimo — visualisation
