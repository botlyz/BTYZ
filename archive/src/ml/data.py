"""
Chargement et préparation des données pour le pipeline ML.

Sources :
  - data/raw/binance/um/1m/BTCUSDT.csv   → klines perp 1m (OHLCV + taker_buy)
  - data/raw/binance/spot/1m/BTCUSDT.csv → klines spot 1m (OHLCV + taker_buy)
  - data/raw/binance/um/metrics/BTCUSDT.csv → OI, ls_ratio, taker_ls_vol (5m → ffill 1m)

Output : DataFrame 1m avec colonnes :
  close, volume, taker_buy_perp, taker_sell_perp,
  cvd_perp, cvd_spot,
  oi, oi_delta, ls_ratio, taker_ls_vol
  (toutes normalisées rolling pour éviter le lookahead)
"""

import os
import pandas as pd
import numpy as np

# ── Chemins par défaut ────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'binance')

PERP_PATH    = os.path.join(_BASE, 'um',   '1m',      'BTCUSDT.csv')
SPOT_PATH    = os.path.join(_BASE, 'spot', '1m',      'BTCUSDT.csv')
METRICS_PATH = os.path.join(_BASE, 'um',   'metrics', 'BTCUSDT.csv')


def _load_klines(path: str) -> pd.DataFrame:
    """Charge un CSV klines Binance (date en ms)."""
    df = pd.read_csv(path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], unit='ms', utc=True)
    df = df.set_index('date').sort_index()
    df = df[['open', 'high', 'low', 'close', 'volume', 'taker_buy_volume']]
    return df.apply(pd.to_numeric, errors='coerce')


def _load_metrics(path: str) -> pd.DataFrame:
    """Charge le CSV metrics (OI, ls_ratio, taker_ls_vol) en 5m."""
    df = pd.read_csv(path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], unit='ms', utc=True)
    df = df.set_index('date').sort_index()
    cols = [c for c in ['oi', 'oi_value', 'top_trader_ls', 'ls_ratio', 'taker_ls_vol'] if c in df.columns]
    df = df[cols].apply(pd.to_numeric, errors='coerce')
    return df


def load_dataset(
    perp_path:    str = PERP_PATH,
    spot_path:    str = SPOT_PATH,
    metrics_path: str = METRICS_PATH,
    norm_window:  int = 1440,   # rolling normalization sur 1440 bougies (1 jour M1)
    start:        str = '2020-09-01',
    end:          str = None,
) -> pd.DataFrame:
    """
    Charge et assemble le dataset ML complet en 1m.

    Paramètres
    ----------
    norm_window : int
        Fenêtre rolling pour la normalisation (z-score).
        1440 = 1 jour M1. Evite le lookahead bias.
    start : str
        Date de début (2020-09-01 = début des metrics OI).
    end : str
        Date de fin (None = jusqu'à la dernière bougie disponible).

    Retourne
    --------
    DataFrame avec colonnes brutes + normalisées (_z suffix).
    """
    print("Chargement klines perp 1m...")
    perp = _load_klines(perp_path)

    print("Chargement klines spot 1m...")
    spot = _load_klines(spot_path)

    print("Chargement metrics 5m...")
    metrics = _load_metrics(metrics_path)

    # ── CVD perp (cumulatif sur toute la période) ─────────────────────────────
    perp['taker_sell_volume'] = perp['volume'] - perp['taker_buy_volume']
    perp['cvd_perp'] = (perp['taker_buy_volume'] - perp['taker_sell_volume']).cumsum()

    # ── CVD spot ──────────────────────────────────────────────────────────────
    spot['taker_sell_volume'] = spot['volume'] - spot['taker_buy_volume']
    spot['cvd_spot'] = (spot['taker_buy_volume'] - spot['taker_sell_volume']).cumsum()

    # ── Merge perp + spot ─────────────────────────────────────────────────────
    df = perp[['open', 'high', 'low', 'close', 'volume',
               'taker_buy_volume', 'taker_sell_volume', 'cvd_perp']].copy()
    df = df.join(
        spot[['taker_buy_volume', 'cvd_spot']].rename(columns={
            'taker_buy_volume': 'spot_taker_buy',
        }),
        how='left',
    )

    # ── Merge metrics (5m → ffill sur 1m) ────────────────────────────────────
    # Resample metrics sur index 1m puis forward fill (dernière valeur connue)
    metrics_1m = metrics.reindex(df.index, method='ffill')
    df = df.join(metrics_1m, how='left')

    # ── OI delta (variation par rapport à la mise à jour précédente) ──────────
    if 'oi' in df.columns:
        df['oi_delta'] = df['oi'].diff().fillna(0)

    # ── Return 1m (momentum prix immédiat) ───────────────────────────────────
    df['return_1m'] = df['close'].pct_change().fillna(0).clip(-0.05, 0.05)

    # ── Filtre temporel ───────────────────────────────────────────────────────
    df = df.loc[start:]
    if end:
        df = df.loc[:end]

    # ── Supprime les lignes avec trop de NaN ─────────────────────────────────
    df = df.dropna(subset=['close', 'cvd_perp'])

    # ── Normalisation rolling z-score (sans lookahead) ───────────────────────
    # Chaque valeur est normalisée par rapport aux `norm_window` bougies précédentes
    # → le modèle voit des signaux relatifs, pas des valeurs absolues
    _cols_to_norm = [
        'close', 'volume', 'taker_buy_volume', 'taker_sell_volume',
        'cvd_perp', 'cvd_spot', 'spot_taker_buy',
        'oi', 'oi_delta', 'ls_ratio', 'taker_ls_vol',
    ]
    for col in _cols_to_norm:
        if col not in df.columns:
            continue
        roll = df[col].rolling(norm_window, min_periods=norm_window // 2)
        mu   = roll.mean()
        sig  = roll.std().replace(0, 1)
        df[f'{col}_z'] = ((df[col] - mu) / sig).clip(-5, 5)

    n_rows = len(df)
    n_days = n_rows / 1440
    print(f"\nDataset prêt : {n_rows:,} bougies M1 ({n_days:.0f} jours)")
    print(f"Période      : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Colonnes     : {list(df.columns)}")
    print(f"NaN restants : {df.isnull().sum().sum()}")

    return df


def get_feature_cols(df: pd.DataFrame, normalized: bool = True) -> list:
    """
    Retourne les colonnes features à donner au modèle.

    normalized=True  → colonnes _z (recommandé pour LSTM)
    normalized=False → colonnes brutes (pour XGBoost, qui normalise seul)
    """
    if normalized:
        return [c for c in df.columns if c.endswith('_z')]
    else:
        return [
            c for c in [
                'volume', 'taker_buy_volume', 'taker_sell_volume',
                'cvd_perp', 'cvd_spot', 'spot_taker_buy',
                'oi', 'oi_delta', 'ls_ratio', 'taker_ls_vol',
            ]
            if c in df.columns
        ]


if __name__ == '__main__':
    df = load_dataset()
    print("\nAperçu :")
    print(df[get_feature_cols(df, normalized=False)].tail(3).to_string())
