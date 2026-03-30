# BTYZ — Research Log ML

Journal de recherche de la branche `feature/ml-signal`.
Ordre chronologique, ce qui a marché / pas marché, et les prochaines étapes.

---

## Contexte

Objectif : trouver de l'alpha dans les déséquilibres d'order flow sur BTC/USDT Binance M1,
en utilisant CVD (Cumulative Volume Delta), OI (Open Interest), ls_ratio et taker flow.
Pas de TA classique — laisser le modèle trouver les patterns seul.

---

## Étape 1 — Architecture des données (`src/ml/data.py`)

### Ce qu'on a fait
- Chargement perp 1m + spot 1m + metrics 5m (OI, ls_ratio, taker_ls_vol)
- CVD calculé explicitement : `cumsum(taker_buy - taker_sell)`
- OI en 5m → ffill sur 1m (dernière valeur connue, pas d'interpolation = honnête)
- `oi_delta` = diff de l'OI (variation, pas le niveau absolu)
- Normalisation rolling z-score (fenêtre 1440 bars = 1 jour M1) → sans lookahead bias
- `return_1m` ajouté plus tard (pct_change du close, clippé à ±5%)

### Résultat
- 2,918,880 bougies M1, 2020-09-01 → 2026-03-20
- 908k NaN en warm-up (attendu, 1440 bars de chauffe par feature)

### Notes
- OI constant pendant 5 min (ffill) : pas un problème pour XGBoost qui voit 1 bougie
  à la fois ; légèrement trompeur pour LSTM mais préférable à exclure l'OI
- Le CVD cumulatif sur toute la période crée une tendance longue terme →
  pour LSTM (séquences courtes), le z-score rolling corrige ça

---

## Étape 2 — Labels (`src/ml/labels.py`)

### Triple Barrier (Lopez de Prado)

Label de classification pour chaque bougie :
- `+1` : prix touche +tp% avant -sl% → trade long gagnant
- `-1` : prix touche -sl% avant +tp% → trade long perdant
- `0`  : ni l'un ni l'autre dans max_bars → neutre (ignoré en training)

Kernel numba pour performance sur 2.9M bougies.

Configs testées (`BARRIER_CONFIGS`) :
| Nom | TP | SL | Max bars |
|-----|----|----|----------|
| scalp_15min | 1.0% | 0.6% | 15 |
| swing_30min | 1.5% | 0.9% | 30 |
| move_1h | 3.0% | 1.5% | 60 |
| move_2h | 6.0% | 3.0% | 120 |

Distribution swing_30min : +1=1.2%, -1=4.6%, 0=94.2% → très déséquilibré.
Solution : `scale_pos_weight` dans XGBoost, `BCEWithLogitsLoss(pos_weight=...)` dans LSTM.

### Max Gain / Max Loss (nouveau — Phase 1b)

Label de régression 2 outputs sur horizon N bougies :
- `max_gain` = max(close[t+1:t+N]) / close[t] - 1 → meilleur move possible
- `max_loss` = min(close[t+1:t+N]) / close[t] - 1 → pire move possible (négatif)
- `rr` = max_gain / abs(max_loss) → R/R potentiel du trade

Avantage : pas de seuil TP/SL fixé à l'avance. Chaque bougie a son propre R/R.
Signal : entrer quand R/R prédit > seuil (1.5, 2.0, 2.5...).
TP dynamique = max_gain prédit. SL dynamique = max_loss prédit.

---

## Étape 3 — XGBoost Classification (`src/ml/train_xgb.py`)

### Setup
- Walk-forward : 6 mois train / 1 mois test, avance d'1 mois → 60 folds
- Features : colonnes brutes + momentum (cvd_perp_d5/15/30/60, price_d5/15/30/60,
  divergence CVD/prix, taker_ratio)
- Metric principale : **lift = precision - tp_rate** (> 0 = vrai alpha au-dessus du hasard)
  Pourquoi pas accuracy : avec tp_rate=0.12, prédire tout "sl" donne 88% d'accuracy → trompeur

### Problème rencontré
Premiers runs montraient accuracy=100% sur certains folds → cause : déséquilibre de classe.
Fix : basculer sur lift comme métrique principale.

### Résultats

| Config | Lift moy | Lift>0 | Signaux/fold |
|--------|----------|--------|-------------|
| swing_30min | **+0.102** | 76.8% | 761 |
| scalp_15min | +0.074 | 82.1% | 780 |
| move_2h | +0.091 | 24.5% | 77 |
| move_1h | +0.057 | 33.9% | 216 |

**Alpha confirmé sur swing_30min.** 76.8% des mois OOS sont positifs sur 5 ans.

Feature importance XGBoost swing_30min :
1. price_d30 (0.073) ← momentum prix 30 bougies
2. price_d60 (0.069)
3. cvd_spot (0.061)
4. ls_ratio (0.060)
5. oi (0.059)
6. cvd_perp (0.058)

Surprise : le momentum prix domine, pas seulement le CVD.
CVD reste dans le top 6 mais c'est la combinaison prix + flow qui prédit.

---

## Étape 4 — LSTM Classification (`src/ml/train_lstm.py`)

### Hypothèse
LSTM voit 60 bougies de contexte → devrait détecter des patterns temporels
(divergence CVD qui se forme sur 1h) mieux que XGBoost qui voit 1 bougie à la fois.

### Problème critique — V1 (sans prix)
Features initiales : uniquement colonnes _z (volume, CVD, OI).
**Le prix n'était pas dans les features.**
Le modèle essayait de prédire l'évolution du prix sans voir le prix.
Résultat : lift moyen ~+0.04 sur toutes configs, moins bon que XGBoost.

### Fix — V2 (avec prix)
Ajout de `return_1m` (pct_change close) et `close_z` dans les features LSTM.
Autres corrections : EPOCHS 30→60, PATIENCE 3→5, log automatique horodaté,
skip des pkl existants, barre de progression globale tqdm, flag --config.

### Résultats V2 (partiel, run interrompu)
swing_30min LSTM avec prix — résultats très volatils :
- Bons folds : +0.218, +0.276, +0.207
- Mauvais folds : -0.160, -0.169, -0.117
- ~36% folds positifs vs 76.8% pour XGBoost

**Verdict : LSTM underperform XGBoost même avec le prix.**
Hypothèse : LSTM sensible aux régimes (bear/bull/range) — 6 mois de train pas assez
pour s'adapter à tous les régimes.

### Décision
Mettre LSTM en pause. Passer à l'approche R/R dynamique (Phase 1b) qui est
conceptuellement plus solide et ne nécessite pas de définir TP/SL à l'avance.

---

## Étape 5 — XGBoost R/R Dynamique (`src/ml/train_xgb_rr.py`)

### Concept
Au lieu de "est-ce que ça va faire TP ou SL ?", le modèle prédit :
- `max_gain_90m` : jusqu'où le prix peut monter dans les 90 prochaines bougies
- `max_loss_90m` : jusqu'où le prix peut descendre dans les 90 prochaines bougies

Signal : entrer quand R/R prédit > seuil → TP et SL définis par la prédiction elle-même.

Avantages vs Triple Barrier :
- Pas de seuil fixé à l'avance (1.5%/0.9% pour tous les trades)
- Chaque bougie a son propre TP/SL optimal selon le contexte
- Modèle peut détecter "ce moment = squeeze probable, potentiel +4% en 1h30"

### Métriques
- `corr_rr` : corrélation R/R prédit vs R/R réel → si > 0, le modèle est utile
- `lift_rr` à chaque seuil : est-ce que les signaux filtrés ont un meilleur R/R réel ?

### Résultats (en cours)
Horizon 90 bougies, premiers folds :
- Fold 1 : corr_gain=+0.012, corr_loss=+0.128, corr_rr=+0.039, lift@1.5=+0.190
- Fold 2 : corr_gain=+0.191, corr_loss=+0.092, corr_rr=+0.001, lift@1.5=+0.095
- Fold 3 : corr_gain=+0.344, lift@1.5=+0.032
- Fold 5 : lift@1.5=**-0.363** ← fold négatif
- Fold 6 : corr_gain=+0.150, lift@1.5=+0.127

Corrélations faibles (~0.01-0.34) → le modèle prédit imparfaitement les niveaux exacts,
mais le lift positif sur la majorité des folds suggère un signal utile.

---

## Prochaines étapes

### Court terme
1. **evaluate.py** — backtest VBT sur les signaux OOS XGBoost swing_30min
   - Optimiser le seuil de probabilité (0.3 → 0.9)
   - Courbe d'équité, sharpe, drawdown avec frais réels
   - Comparer long-only vs long+short
   - Vérifier que lift=+0.102 se traduit en PnL positif après frais

2. **Analyser les résultats xgb_rr** une fois terminés
   - Si lift_rr > 0 stable → intégrer dans evaluate.py avec TP/SL dynamiques
   - Comparer horizon 60 vs 90 vs 120 bougies

### Moyen terme
3. **LSTM avec prix** — relancer V2 une fois evaluate.py validé
   - Tester uniquement swing_30min pour aller vite
   - Comparer lift LSTM vs XGBoost sur les mêmes folds

4. **TP/SL dynamique en production**
   - Si xgb_rr valide : en live, poser TP=max_gain_prédit, SL=max_loss_prédit par trade
   - Plus flexible que des niveaux fixes

5. **Améliorations potentielles**
   - Données de liquidations (non encore intégrées)
   - Transformer/Attention à la place du LSTM
   - Meta-labeling : modèle 2 qui prédit si le modèle 1 va avoir raison
   - Horizon variable : laisser le modèle prédire le meilleur moment de sortie

### Points de vigilance
- **Survivorship bias** : testé uniquement sur BTC. Paires altcoins = résultats différents.
- **Régime de marché** : alpha peut disparaître dans certains régimes (fold 5, fold 20).
- **Frais** : lift=+0.102 ne garantit pas de PnL positif. evaluate.py est critique.
- **Capacité** : ~761 signaux/mois OOS = ~25/jour. Viable pour du trading manuel ou automatisé.
- **Le label max_gain/max_loss suppose qu'on peut toujours sortir au prix prédit.**
  En pratique : slippage, liquidité à considérer.
