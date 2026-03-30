#constantes globales du projet, importées un peu partout

FEES = 0.001 #frais de trading (0.1% par trade, standard binance)
INIT_CASH = 10000 #capital de depart pour les bt
MIN_TRADES = 30 #en dessous de 30 trades c'est pas fiable statistiquement
MAX_DRAWDOWN = 0.30 #dd max 30%, au dela c'est trop risqué

#poids du score composite (pour optuna dans les notebooks de research)
SHARPE_WEIGHT = 0.7 #on privilegie le sharpe
RETURN_WEIGHT = 0.3 #le return compte un peu mais penalisé par le dd

#seuils de distance pour l'analyse de clusters
#plus y'a de dimensions plus le seuil doit etre large
#car la distance euclidienne augmente avec le nb de dimensions
CLUSTER_DISTANCE_THRESHOLDS = {
    2: 0.15,
    3: 0.30,
    4: 0.35,
    5: 0.40,
}

#walk-forward
DEFAULT_N_WINDOWS = 10 #10 fenetres glissantes
DEFAULT_SPLIT_RATIO = 0.7 #70% train, 30% test

#optuna
DEFAULT_N_TRIALS = 200 #200 trials par fenetre, bon compromis vitesse/qualité

#analyse de clusters
TOP_QUANTILE = 0.70 #garder le top 30% des essais par fenetre (quantile 0.70)
ROBUST_RATIO_GREEN = 0.70 #vert = 70%+ des autres fenetres ont un voisin proche
ROBUST_RATIO_ORANGE = 0.50 #orange = 50-70%
