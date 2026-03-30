# BTYZ

Backtesting et optimisation de stratégies mean reversion crypto avec VectorBT PRO.

## Structure

```
BTYZ/
├── config.py                    # constantes globales (fees, seuils, etc.)
├── data/raw/                    # données brutes OHLCV (csv)
├── src/
│   ├── download.py              # telechargement données via CCXT
│   ├── opti.py                  # optimisation multiprocess (VBT natif)
│   └── strategies/              # stratégies clean
│       ├── keltner.py           # Keltner Channel (EMA + ATR)
│       └── atr_envelope.py      # Enveloppe ATR (SMA + ATR)
├── notebooks/
│   ├── analyse.ipynb            # analyse des resultats d'opti
│   └── research/                # notebooks de recherche (brouillons)
├── cache/                       # resultats d'opti (.pickle)
└── requirements.txt
```

## Workflow

```
1. notebooks/research/  → prototypage & recherche de strats
2. src/strategies/      → version clean de la strat (.py)
3. src/opti.py          → lance l'opti multiprocess + sauvegarde pickle
4. notebooks/analyse.ipynb → charge le pickle et analyse les resultats
```

## Stratégies

Deux variantes d'enveloppe testées en mean reversion :

- **Enveloppe ATR** : SMA ± multiplicateur × ATR
- **Keltner Channel** : EMA ± multiplicateur × ATR (+ stop loss optionnel)

Logique : on achète quand le prix touche la bande inf (survente), on vend quand il revient à la moyenne. Pareil en short sur la bande sup. Prix d'exécution simulés aux niveaux des bandes (limit orders).

## Optimisation

Walk-forward avec VBT natif :
- `vbt.Splitter` pour créer les fenêtres glissantes (70% train / 30% test)
- `vbt.parameterized()` + `vbt.split()` pour le grid search
- Multiprocess avec `engine='pathos'`
- Resultats sauvegardés en .pickle dans `cache/`

Params optimisés : `ma_window`, `atr_window`, `atr_mult`, `sl_stop`

## Analyse

Le notebook `analyse.ipynb` charge le pickle et fait :
- Correlation train/test (overfitting check)
- Heatmaps sharpe median + delta test-train
- Recherche de combos robustes (sharpe > 0 sur 50%+ des fenetres)
- Backtest final avec plot des bandes
- Top 10 des meilleures combos

## Stack

- Python 3.13
- VectorBT PRO (backtesting + indicateurs)
- Plotly (visualisation)
- Pandas / NumPy / Numba
