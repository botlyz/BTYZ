# RAM_ENV_v1 — retour à la moyenne par enveloppe

## Thèse économique

Sur un horizon court, le prix d'un perp crypto oscille autour de sa moyenne
mobile. Une partie des écarts marqués à cette moyenne n'est pas de l'information
mais du **bruit de microstructure** : sur-réaction à une nouvelle, cascade de
liquidations, mèches sur carnet fin, exécution agressive d'un gros ordre. Ce
bruit se résorbe statistiquement — le prix « revient à la moyenne ».

On fade donc l'écart, **mais uniquement dans le sens de la tendance de fond**
(filtre par une grande SMA `sma_trend`) :
- prix **sous** la bande basse `MA·(1 − env)` **ET** en uptrend (close ≥ SMA_trend)
  → on achète (le creux est excessif, la tendance nous porte),
- prix **au-dessus** de la bande haute `MA·(1 + env)` **ET** en downtrend
  (close ≤ SMA_trend) → on vend (le pic est excessif),
- on **sort au retour à la moyenne** (close qui recroise la MA).

Le filtre de tendance est le cœur de cette version : fader à nu un rip dans un
marché qui monte saigne. En ne fadant que dans le sens de la tendance, on
concentre les trades sur les setups à plus fort edge — l'objectif est de faire
passer le gain par trade au-dessus des coûts, là où la version nue échouait.

## Pourquoi ça peut marcher — et quand ça casse

L'edge existe si la **réversion domine le momentum** à l'échelle de la fenêtre.
C'est vrai sur beaucoup de paires en régime de range, faux en tendance forte :
en trending, fader la bande revient à se coucher devant un train (d'où le stop
de protection `sl_pct` structurel). Le pipeline tranche ça objectivement — si la
réversion ne paie pas net de frais, le screening puis le PBO l'éjectent.

Le choix `env` (largeur) arbitre fréquence vs qualité : bande étroite (3 %) =
beaucoup de trades, écarts faibles, sensible aux frais ; bande large (8 %) = peu
de trades, écarts francs, plus rare mais plus net. Le choix `ma_window` (50 vs
200) fixe l'horizon de « moyenne » qu'on suppose juste.

## Causalité

Tout est en `rolling(...).mean()` past-only ; aucun z-score/quantile global,
aucun `shift(-1)`. Le test de préfixes du pipeline le vérifie automatiquement.

## Coûts

Exécution supposée au marché (taker). Sur bande étroite et TF basse, les frais
mangent vite l'edge — c'est précisément ce que l'étape de screening (frais
réels par paire) et l'incubation live doivent valider.
