# RAM DCA — Logique de trading

## Concept

Mean reversion multi-niveaux avec DCA (Dollar Cost Averaging). Le prix oscille autour d'une moyenne mobile ; on entre progressivement quand il s'en éloigne, et on sort quand il y revient.

## Indicateurs

- **MA** : SMA (Simple Moving Average) sur `ma_window` périodes, calculée sur le close (ou OHLC4 en option)
- **Bandes d'enveloppe** : `MA * (1 ± env_pct)` — enveloppes symétriques au-dessus et en dessous de la MA
- Multi-niveaux possible : plusieurs bandes à distances croissantes (ex: 1%, 2%, 3%)

## Paramètres

| Paramètre | Description | Exemple |
|-----------|-------------|---------|
| `ma_window` | Période de la SMA | 20 |
| `envelope_levels` | Liste de % pour les bandes (distances à la MA) | [0.02] |
| `allocations` | Fraction du capital allouée par niveau | [1.0] |
| `sl_pct` | Stop loss en % du prix moyen d'entrée | 0.05 (5%) |

En mode opti grid search, la config est simplifiée à 1 seul niveau : `env_pct` (= `envelope_levels[0]`) et `sl_pct`.

## Signaux d'entrée

### Long
Le prix (low de la bougie) touche ou passe sous la bande inférieure `MA * (1 - env_pct)` de la bougie **précédente**.

- Exécution au prix de la bande (limit order simulé)
- Si multi-niveaux : chaque bande touchée ajoute une tranche (DCA), le prix moyen d'entrée est recalculé

### Short
Le prix (high de la bougie) touche ou dépasse la bande supérieure `MA * (1 + env_pct)` de la bougie **précédente**.

- Même logique inverse
- Un short ne peut pas se déclencher si un long a déjà été ouvert sur la même bougie (et inversement)

## Signaux de sortie

### Take Profit (retour à la MA)
- **Long** : le high de la bougie atteint ou dépasse la MA précédente → clôture à la MA
- **Short** : le low de la bougie atteint ou passe sous la MA précédente → clôture à la MA
- Exécution au prix exact de la MA (limit order simulé)

### Stop Loss
- **Long** : `avg_entry * (1 - sl_pct)` — si le low touche ce niveau → clôture
- **Short** : `avg_entry * (1 + sl_pct)` — si le high touche ce niveau → clôture
- Le SL est basé sur le **prix moyen d'entrée** (pondéré par les allocations en cas de DCA)
- Après un SL, le système entre en **cooldown**

## Mécanismes spéciaux

### Cooldown post-SL
Après un stop loss, toutes les entrées sont bloquées jusqu'à ce que le prix **touche ou traverse la MA**. Cela évite de re-rentrer immédiatement dans un mouvement directionnel fort.

Condition de fin de cooldown :
- Le range [low, high] de la bougie contient la MA, **ou**
- Le close a traversé la MA entre la bougie précédente et la courante

### SL différé (entrée + SL sur la même bougie)
Si une entrée et un SL se produisent sur la même bougie (le prix touche la bande puis continue jusqu'au SL) :
1. L'entrée est enregistrée normalement sur cette bougie
2. Le SL est **reporté à la bougie suivante**

Sans ce mécanisme, VBT ignorerait le trade (taille 0 immédiatement après l'entrée = pas de trade).

## Exécution dans VBT

Le noyau Numba (`ram_dca_nb`) produit deux arrays :
- `target_size[i]` : taille cible en % du capital (+1 = 100% long, -1 = 100% short, 0 = flat, NaN = pas d'action)
- `exec_price[i]` : prix d'exécution (prix de la bande, de la MA, ou du SL)

Ces arrays sont passés à `vbt.Portfolio.from_orders()` avec :
- `size_type='TargetPercent'` — la taille est un % du capital
- `fees` et `slippage` appliqués à chaque trade

## Coûts de trading (config.py)

| Paramètre | Valeur | Raison |
|-----------|--------|--------|
| `FEES` | 0.0 (0 bps) | Lighter : maker/taker = 0% |
| `SLIPPAGE` | 0.0002 (2 bps) | Spread bid-ask estimé sur paires liquides Lighter |
| `INIT_CASH` | 10 000 $ | Capital initial de backtest |

## Flux d'exécution par bougie

```
Bougie i :
│
├─ SL différé en attente ? → exécuter le SL, passer en cooldown, next
│
├─ MA précédente = NaN ? → skip (pas assez de données)
│
├─ En cooldown ? → vérifier si prix touche/traverse la MA
│   ├─ Oui → fin du cooldown
│   └─ Non → skip
│
├─ En position ? → vérifier take profit (retour à la MA)
│   └─ TP touché → clôturer, next
│
├─ Vérifier entrées DCA (bandes inférieures si flat/long, supérieures si flat/short)
│   └─ Bande touchée → ajouter la tranche, recalculer avg_entry
│
├─ Vérifier SL sur le nouvel état
│   ├─ SL + entrée même bougie (flat→position→SL) → enregistrer entrée, différer SL
│   ├─ SL sur position existante → clôturer, cooldown
│   └─ Pas de SL → enregistrer l'entrée si elle a eu lieu
│
└─ Prochain i
```

## Résumé de la thèse

La stratégie exploite le retour à la moyenne : quand le prix s'éloigne "trop" de sa moyenne mobile, il a tendance à y revenir. Le DCA multi-niveaux améliore le prix moyen d'entrée en cas de mouvement prolongé. Le SL protège contre les cas où le mean reversion ne fonctionne pas (breakout directionnel). Le cooldown évite de se faire découper en re-rentrant trop tôt après un SL.
