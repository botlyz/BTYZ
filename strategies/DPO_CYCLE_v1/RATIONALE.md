# DPO_CYCLE_v1 — cycle détendancé (DPO ±1,5σ)

## Thèse économique

Le prix d'un actif mélange **tendance** et **cycle**. Le Detrended Price
Oscillator (DPO) retire la composante de tendance (via une moyenne mobile
décalée) pour ne garder que l'oscillation autour de cette tendance — le cycle.

Une fois le cycle isolé, on le **normalise par son écart-type** : un creux à
−1,5σ sur BTC devient comparable à un creux à −1,5σ sur DOGE, quelle que soit
la volatilité propre de chacun. Ça rend un seuil unique (±1,5σ) applicable à
tout l'univers — exactement ce que le screening multi-paires exige.

La thèse de trading est un retour à la moyenne cyclique :
- le cycle touche le bas (−1,5σ) puis **remonte** → on entre long (le creux se
  résorbe), on sort quand le cycle atteint son **sommet** (+1,5σ) ;
- symétriquement en short depuis le surachat.

C'est un mean-reversion sur le cycle détendancé, pas sur le prix brut : en
retirant la tendance, on évite (en partie) de fader un mouvement directionnel.

## Causalité — le point critique

Le DPO classique est souvent **affiché** décalé vers l'avant pour aligner
visuellement l'oscillateur sur le prix — ce décalage vers le futur serait une
fuite. Ici le DPO est calculé en version **strictement causale** :
`dpo(t) = close(t − k) − SMA_period(t)` avec `k = period/2 + 1`, où les deux
termes n'utilisent que des données ≤ t. La normalisation est un écart-type
`rolling(norm)` past-only, et les seuils sont des croisements sur `z` et
`z.shift(1)`. Le test de préfixes du pipeline le vérifie automatiquement.

## Coûts et limites

Sans stop ni TP (fidèle au builder) : une position n'est fermée qu'au passage
de l'extrême cyclique opposé, ce qui peut immobiliser du capital longtemps si
le cycle ne se referme pas. À évaluer par le pipeline (fréquence de trades,
tenue en tendance forte). Testé à coût conservateur (10 bps).
