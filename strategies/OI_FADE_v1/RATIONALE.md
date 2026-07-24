# OI_FADE_v1 — fade du pump surpeuplé

## Thèse économique

Un mouvement de prix violent **accompagné d'une expansion d'open interest** signale
du levier neuf qui entre tard : les retardataires (FOMO) s'empilent au sommet du
pump. Ces positions sont fragiles — mal placées, sur-leveragées — et se font
secouer dans les 24-48h qui suivent (liquidations en cascade, prise de profit des
early longs). Le payeur de l'edge est le FOMO tardif ; nous prenons l'autre côté.

Le filtre OI est ce qui distingue un vrai repricing (move sans expansion d'OI,
qu'il ne faut PAS fader) d'un pump de crowding (move + OI en forte hausse).
Validé au screening historique : t-stats 2.3-3.0 après déduplication des
événements. Le côté long symétrique (fade du dump surpeuplé) est désactivé :
96 % des événements détectés sont des pumps et le long n'a rien montré.

## Construction du signal

Sur 1h : doi = ΔOI sur `doi_w` barres, r = rendement prix sur `doi_w` barres.
Z-scores rolling **past-only** (fenêtre ROLL=240 barres, fixe) ; intensité
= max(z_doi, 0) × |z_r|. Entrée short quand l'intensité dépasse son propre
quantile rolling(240, `pctl`) ET r > 0 ET doi > 0. Sortie : horizon fixe
`hold` barres (td_stop) + stop-loss dur `sl_pct`.

## Causalité

Aucune statistique full-série : z-scores, quantile de seuil — tout est rolling
sur les 240 dernières barres uniquement (le seuil inclut la barre courante dans
sa fenêtre, ce qui reste strictement causal). Le `fillna(False)` ne touche que
les NaN de warm-up. Logique identique à l'implémentation historique
(archive/approach/OI_FADE_v1), aucune fuite n'a dû être corrigée à la migration.
