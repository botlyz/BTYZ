# SMA_CROSS_v1 — fixture de test du pipeline

Croisement de moyennes mobiles simples (fast/slow), long only : entrée quand la
SMA rapide passe au-dessus de la lente, sortie au croisement inverse.

Ce n'est PAS une thèse d'edge — le suivi de tendance par SMA cross est l'exemple
canonique de stratégie sur-documentée et arbitragée. Cette stratégie existe
uniquement comme **fixture** : trivialement causale (rolling means + shift(1)),
rapide à calculer, comportement prévisible. Elle sert à tester le pipeline
quantlab (validation du contrat, test look-ahead, screening, backtest moteur)
et devrait normalement être REJETÉE aux étapes statistiques — c'est aussi un
bon test négatif du pipeline.
