# BTYZ

Backtesting et optimisation de stratégies mean reversion sur HYPE/USDT (5min) avec VectorBT PRO et Optuna.

## Données

- **HYPEUSDT.csv** : ~416 jours de données OHLCV 5 minutes (dec 2024 → jan 2026)

## Stratégies

Deux variantes d'enveloppe testées :

- **Enveloppe ATR** : SMA ± multiplicateur × ATR
- **Keltner Channel** : EMA ± multiplicateur × ATR

Logique identique pour les deux : on achète quand le prix touche la bande inférieure, on vend quand il revient à la moyenne. Pareil en short sur la bande supérieure. Les prix d'exécution sont simulés aux niveaux des bandes (limit orders).

## Workflow

### 1. Backtest simple

On teste la strat avec des paramètres arbitraires sur une période courte (2 mois) juste pour valider que la logique fonctionne. Résultat pas concluant → normal, les params sont choisis au pif.

### 2. Walk-forward optimization

On découpe tout le dataset en 10 fenêtres glissantes (90 jours de train, 30 jours de test, step de 30 jours). Pour chaque fenêtre, Optuna (TPE) cherche les meilleurs paramètres sur le train (200 trials), puis on valide le meilleur sur le test.

Paramètres optimisés :
- `ma_window` : période de la moyenne mobile (10-500)
- `atr_window` : période de l'ATR (5-100)
- `atr_mult` : multiplicateur des bandes (0.5-10.0)

Contraintes : minimum 30 trades par fenêtre, drawdown max 30%.

Score : `sharpe * 0.7 + total_return * 0.3 * (1 - max_drawdown)`

### 3. Analyse de clusters cross-fenêtres

Le problème du walk-forward classique : chaque fenêtre peut trouver des paramètres complètement différents. Pour trouver des paramètres robustes, on analyse la stabilité cross-fenêtres :

1. Pour chaque fenêtre, on garde le top 30% des trials (par score sur le train)
2. On normalise les paramètres entre 0 et 1 (sinon `ma_window` 10-500 écraserait `atr_mult` 0.5-10)
3. Pour chaque bon trial, on calcule la distance euclidienne avec les bons trials des autres fenêtres
4. Si un trial a des voisins proches (distance < 30%) dans 70%+ des autres fenêtres → zone verte (robuste). 50-70% → orange.
5. Paramètres finaux = médiane du meilleur cluster

### 4. Backtest final

On lance le backtest sur tout le dataset avec les paramètres trouvés par l'analyse de clusters (100% du capital cette fois).

## Stack

- Python 3
- VectorBT PRO (backtesting + indicateurs)
- Optuna (optimisation TPE)
- Plotly (visualisation)
- Pandas / NumPy