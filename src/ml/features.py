"""
Sélection et construction des features pour XGBoost et LSTM.

XGBoost  → colonnes brutes (il normalise seul via les splits)
LSTM     → colonnes _z (z-score rolling, sans lookahead)
"""

import pandas as pd
import numpy as np


# Configs Triple Barrier à tester
BARRIER_CONFIGS = [
    # (tp,    sl,    max_bars, nom)
    (0.010, 0.006,  15, 'scalp_15min'),   # 1.0% en 15min
    (0.015, 0.009,  30, 'swing_30min'),   # 1.5% en 30min
    (0.030, 0.015,  60, 'move_1h'),       # 3.0% en 1h
    (0.060, 0.030, 120, 'move_2h'),       # 6.0% en 2h
]

# Features brutes pour XGBoost
FEATURES_RAW = [
    'volume',
    'taker_buy_volume',
    'taker_sell_volume',
    'cvd_perp',
    'cvd_spot',
    'spot_taker_buy',
    'oi',
    'oi_delta',
    'ls_ratio',
    'taker_ls_vol',
]

# Features normalisées pour LSTM
FEATURES_NORM = [f'{c}_z' for c in FEATURES_RAW]


def get_X(df: pd.DataFrame, normalized: bool = False) -> pd.DataFrame:
    """
    Retourne la matrice de features.

    normalized=False → brutes pour XGBoost
    normalized=True  → _z pour LSTM
    """
    cols = FEATURES_NORM if normalized else FEATURES_RAW
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def get_Xy(
    df:         pd.DataFrame,
    label_col:  str,
    normalized: bool = False,
    dropna:     bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Retourne (X, y) prêts pour l'entraînement.

    Paramètres
    ----------
    label_col : str
        Nom de la colonne label dans df (ex: 'tb_tp0.01_sl0.005_t60').
    normalized : bool
        True pour LSTM, False pour XGBoost.
    dropna : bool
        Supprime les lignes avec NaN dans X ou y.
    """
    X = get_X(df, normalized=normalized)
    y = df[label_col]

    if dropna:
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

    return X, y


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features de momentum CVD — aide XGBoost à voir
    les divergences sans fenêtre temporelle.

    Ces features sont des différences sur N bougies :
    cvd_delta_5  = CVD maintenant vs il y a 5min
    cvd_delta_15 = CVD maintenant vs il y a 15min
    price_delta_5, price_delta_15 = pareil pour le prix
    divergence_5  = cvd monte mais prix descend (ou inverse)
    """
    df = df.copy()

    for lag in [5, 15, 30, 60]:
        # Delta CVD
        df[f'cvd_perp_d{lag}'] = df['cvd_perp'].diff(lag)
        df[f'cvd_spot_d{lag}']  = df['cvd_spot'].diff(lag)
        # Delta prix
        df[f'price_d{lag}']     = df['close'].diff(lag)
        # Divergence CVD/prix (signe opposé = divergence)
        df[f'div_perp_{lag}']   = (
            np.sign(df[f'cvd_perp_d{lag}']) * -np.sign(df[f'price_d{lag}'])
        )  # +1 = divergence, -1 = confirmation, 0 = neutre

    # Taker ratio (pression acheteuse relative)
    df['taker_ratio'] = (
        df['taker_buy_volume'] / (df['volume'] + 1e-9)
    )

    return df
