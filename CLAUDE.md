# Règles Claude

- Ne JAMAIS modifier le code, commit ou push sans que Sofien le demande explicitement. Quand on discute d'un sujet, c'est de la discussion — pas un ordre de modifier.
- Ne jamais mentionner Claude/IA dans les commits, PR, code ou commentaires.

# Contexte projet BTYZ

## Qui suis-je
Sofien, 22 ans, dev Python autodidacte, entrepreneur (Botlyz — SaaS trading crypto). En cours d'obtention d'une alternance chez Trade n Change, prop shop crypto fondée par Gautier Pialat. Basé à Villeneuve-d'Ascq (Lille).

## La boîte — Trade n Change
- Prop shop algo trading crypto fondée en 2020 par Gautier Pialat (Dijon)
- SAS, SIREN 889914263, nom commercial: Notonlygeek
- Stack: Python, TypeScript, Docker, architecture micro-services
- Strats: mean reversion, market making, arbitrage inter-exchange
- Full remote, 1 call/semaine, très autonome
- Gautier: ingénieur EISTI, ex Scrum Master/Agile Coach, contributeur Freqtrade
- Hugo (Shaft): alternant années 4-5, développeur/quant
- 3ème associé: pas encore rencontré, a son mot à dire

## Mon alternance
- École: ESGI Lille, bachelor 3 ans (+1, +2, +3), puis master (+4, +5)
- Début: 1er septembre 2025
- Rythme: 8 semaines cours / 8 semaines entreprise au début, puis 2j cours / 3j entreprise
- 3ème année: 1 semaine cours / 3 semaines entreprise
- Spécialisation au +3: big data/IA ou blockchain/Solidity (à décider selon les besoins du projet)
- Mini projet à faire mardi 10 mars 17h-18h (call avec Gautier)

## Ce que Gautier attend
- "Hacker d'alpha" — pas le 100 000ème arbitrage Bybit/Binance
- Posture naïve dans la recherche, pas d'ego
- Rigueur + excellence technique + créativité
- Transformer une idée vague en quelque chose de concret et original
- Penser détail mais comprendre l'environnement macro
- 30% de probabilité d'association (parts) si ça se passe bien

## Stack technique du projet
- Python 3.13
- VectorBT PRO (backtesting + indicateurs)
- Optuna (optimisation TPE)
- Plotly (visualisation)
- Pandas / NumPy / Numba

## Structure du projet

### Workflow
```
1. notebooks/research/  → prototypage & recherche de stratégies
2. src/strategies/      → version clean de la strat (.py)
3. src/opti.py          → lance l'opti multiprocess + sauvegarde pickle
4. notebooks/analyse.ipynb → charge le pickle et analyse les résultats
```

### Arbre
```
BTYZ/
├── config.py                    # constantes globales (fees, seuils, etc.)
├── data/
│   └── raw/                     # données brutes (OHLCV CSV)
├── src/
│   ├── download.py              # téléchargement données via CCXT
│   ├── opti.py                  # optimisation multiprocess (VBT natif)
│   └── strategies/              # stratégies clean
│       ├── keltner.py           # Keltner Channel (EMA + ATR)
│       └── atr_envelope.py      # Enveloppe ATR (SMA + ATR)
├── notebooks/
│   ├── analyse.ipynb            # analyse universelle des résultats d'opti
│   └── research/                # notebooks de recherche (brouillons)
│       ├── keltner_channel_walkforward.ipynb
│       ├── keltner_channel_walkforward_wFSL.ipynb
│       └── atr_envelope_walkforward.ipynb
├── cache/                       # résultats d'opti (.pickle)
├── requirements.txt
└── README.md
```

## Stratégies développées

### 1. Enveloppe ATR (SMA + ATR)
- Formule: SMA ± multiplicateur × ATR
- Mean reversion: achète quand prix touche bande inférieure, vend au retour à la moyenne
- Params: ma_window, atr_window, atr_mult
- Code: src/strategies/atr_envelope.py
- Research: notebooks/research/atr_envelope_walkforward.ipynb

### 2. Keltner Channel (EMA + ATR)
- Formule: EMA ± multiplicateur × ATR
- Même logique que enveloppe ATR mais avec EMA (réagit plus vite)
- Params: ma_window, atr_window, atr_mult, sl_stop
- Code: src/strategies/keltner.py
- Research: notebooks/research/keltner_channel_walkforward.ipynb

### 3. WorldQuant Alpha #12
- Formule: sign(delta(volume, d)) * (-1 * delta(close, d))
- Divergence volume/prix — mean reversion
- Normalisé en z-score (rolling) pour comparabilité cross-périodes
- Params: lookback, norm_window, threshold, sl_stop, tp_stop

## Workflow de recherche

### Optimisation (opti.py — grid search walk-forward + multiprocess)
1. `vbt.Splitter.from_n_rolling()` pour créer les fenêtres walk-forward (10 fenêtres, 70/30 train/test)
2. `vbt.parameterized()` + `vbt.split()` pour le grid search exhaustif (toutes les combinaisons)
3. `kc_objective` retourne `pf.stats()` → toutes les métriques VBT par combo × fenêtre
4. Multiprocess avec `engine='pathos'`
5. Cache des résultats en .pickle dans `cache/`
6. Mode `full` : scanne tous les CSV dans `data/raw/<exchange>/<tf>/`, un pickle par paire dans `cache/full_<exchange>_<tf>/`

### Analyse des résultats (notebooks/analyse.ipynb)
1. Charger le pickle
2. Séparer train/test
3. Corrélation train/test (overfitting check)
4. Heatmaps sharpe médian + delta test-train
5. Recherche de combos robustes (sharpe > 0 sur 50%+ des fenêtres)
6. Performance out-of-sample des meilleurs params
7. Backtest final avec plot des bandes
8. Top 10 des meilleures combos

### Contraintes de validation
- Minimum 30 trades par fenêtre
- Drawdown max 30%
- Score objectif: pf.stats() retourne toutes les métriques VBT (sharpe, drawdown, win rate, profit factor, sortino, calmar, etc.) — le tri/filtrage se fait dans l'analyse

## Limitations connues
- Le backtest final sur tout le dataset est in-sample (les paramètres viennent des mêmes données)
- Survivorship bias sur les altcoins (tester sur BTC/ETH d'abord)
- Les alphas du paper WorldQuant (2015) sont connues et probablement arbitrées

## Prochaines étapes
- Mini projet de Gautier (mardi 10 mars 17h)
- Rencontre avec le 3ème associé
- Si validation → alternance septembre 2025
- Explorer d'autres alphas originales (pas les 101 connues)
- Implémenter le Deflated Sharpe Ratio et le Probabilistic Sharpe Ratio
- Tester CPCV (Combinatorial Purged Cross-Validation)

## Commandes utiles
```bash
# Lancer l'optimisation multiprocess
python src/opti.py

# Installer les dépendances
pip install -r requirements.txt --break-system-packages
```

## Notes VBT PRO
- EMA: `vbt.MA.run(close, window=X, wtype='Exp').ma`
- SMA: `vbt.MA.run(close, window=X).ma`
- ATR: `vbt.ATR.run(high, low, close, window=X).atr`
- Param grid: `vbt.Param(np.arange(start, stop, step))`
- Splitter: `vbt.Splitter.from_n_rolling(index, n=10, length='optimize', split=0.7)`
- Attention: `splitter.plots()` crash avec WebGL, utiliser `vbt.settings.plotting.use_webgl = False`
- Renderer PNG pour VSCode: `pio.renderers.default = "png"`
- Multiprocessing: `engine='pathos'` marche dans script .py mais pas dans notebook
