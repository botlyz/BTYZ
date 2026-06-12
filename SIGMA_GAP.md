# SIGMA GAP · gap-fill

### Fader le bruit pendant que le marché dort.

> Suivi live, en conditions réelles : **[botlyz.com/live2](https://botlyz.com/live2)**

---

## Le problème des stratégies classiques

La **mean reversion** classique part d'une idée simple : un prix qui s'éloigne trop vite de sa moyenne finit par y revenir. On achète quand ça a trop baissé, on vend quand ça a trop monté.

Le souci, c'est que cette idée est **connue de tout le monde**. Sur des marchés ouverts 24/7 (crypto) ou aux heures de bourse, chaque écart est immédiatement arbitré par des milliers d'acteurs. L'edge a fondu. Une stratégie qui marchait sur 2024 peut s'écrouler sur 2026, simplement parce que tout le monde joue le même retour à la moyenne, au même moment, sur les mêmes signaux.

Résultat : on se bat pour des miettes sur un terrain saturé.

---

## L'angle de SIGMA GAP : un terrain que personne ne regarde

SIGMA GAP ne joue pas le retour à la moyenne sur un marché ouvert. Elle exploite une **fenêtre structurelle** propre à un actif d'un genre nouveau : les **actions tokenisées** (type AAPL, AMD) tradées en perpétuel sur des DEX comme **Lighter**.

La particularité de ces actifs :

- Le **perp tokenisé** se trade **24/7**, sans interruption.
- Mais le **vrai sous-jacent** (l'action en bourse) **ferme** — la nuit, les week-ends, les jours fériés.

Pendant ces heures de fermeture, il n'y a plus de "prix de référence" réel. Le perp continue de vivre tout seul, animé uniquement par le flux des traders crypto. Il **dérive** du dernier cours réel connu.

C'est dans cette fenêtre — **quand le marché réel dort** — que SIGMA GAP travaille. Pendant que les actions sont fermées, le bruit s'installe, et c'est ce bruit qu'on vient fader.

---

## Pourquoi c'est une logique nouvelle

| | Mean reversion classique | SIGMA GAP · gap-fill |
|---|---|---|
| **Marché** | ouvert, liquide, saturé | fenêtre de fermeture, peu exploitée |
| **Concurrence** | tout le monde | très peu d'acteurs |
| **Déclencheur** | écart vs moyenne | dérive off-hours du perp vs cours réel figé |
| **Quand ça trade** | en continu | uniquement quand le sous-jacent est fermé |
| **Nature de l'edge** | arbitré depuis longtemps | structurel, lié à un actif récent |

Ce n'est pas le 100 000ᵉ arbitrage entre deux exchanges. C'est l'exploitation d'une **inefficience de structure** née de l'arrivée des actifs TradFi tokenisés : un perp qui tourne quand son référent est éteint. Tant que ces produits sont jeunes, cette fenêtre reste peu disputée.

---

## La rigueur derrière

Une idée originale ne suffit pas — encore faut-il qu'elle **tienne hors échantillon**, pas seulement sur les données où on l'a calibrée.

SIGMA GAP a été soumise à un protocole strict avant tout déploiement :

- **Walk-forward** : la stratégie est ré-évaluée fenêtre après fenêtre, toujours testée sur des périodes qu'elle n'a jamais vues à l'entraînement.
- **Validation croisée sur dates aléatoires (MCCV)** : on tire au sort des centaines de périodes de test pour s'assurer que la performance ne tient pas à un coup de chance de calendrier.
- **Coûts réels** : frais, slippage et liquidité de fermeture intégrés dès la simulation — pas de performance "papier" qui s'évapore en réel.
- **Sélection sévère** : seuls les actifs qui **survivent** à tous ces tests sont déployés. La plupart sont écartés.

Posture assumée : on cherche l'alpha avec naïveté et rigueur, pas avec de l'ego. Une stratégie qu'on ne sait pas casser, on ne la déploie pas.

---

## La preuve par l'exécution

Le plus dur, en trading, ce n'est pas de montrer une belle courbe de backtest — c'est de prouver qu'elle **se reproduit en réel**.

C'est tout l'objet de **[botlyz.com/live2](https://botlyz.com/live2)** :

- Capital réel engagé, exécution on-chain sur Lighter.
- Le **backtest est rejoué en parallèle du live**, avec exactement la même logique, aligné sur les mêmes entrées.
- Chaque divergence — slippage, latence d'ordre, fill manqué — est **tracée en clair**.

Si le live tient ses promesses du backtest, ça se voit. S'il dévie, ça se voit aussi. Tout est transparent, tout est vérifiable, en continu.

---

### En résumé

SIGMA GAP n'est pas une énième mean reversion sur un marché bondé. C'est une logique **neuve**, qui exploite une fenêtre que la tokenisation des actions vient tout juste d'ouvrir : **le perp qui dérive pendant que la bourse est fermée.**

**→ Suivez-la en direct : [botlyz.com/live2](https://botlyz.com/live2)**
