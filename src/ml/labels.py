"""
Génération des labels pour le pipeline ML.

Trois méthodes disponibles :

1. Return brut (régression)
   label = close.shift(-horizon) / close - 1

2. Triple Barrier (Lopez de Prado)
   Laquelle de ces 3 barrières est touchée en premier ?
   ├── Take profit  (+tp%)  → label = +1
   ├── Stop loss    (-sl%)  → label = -1
   └── Temps max    (N min) → label =  0 (neutre)

3. Max Gain / Max Loss (régression 2 outputs) ← nouveau
   max_gain = meilleur move possible dans les N prochaines bougies
   max_loss = pire move possible dans les N prochaines bougies
   → R/R dynamique par bougie, TP/SL définis par la prédiction elle-même
"""

import numpy as np
import pandas as pd
from numba import njit


# ── 1. Return brut ────────────────────────────────────────────────────────────

def make_return_label(close: pd.Series, horizon: int = 30) -> pd.Series:
    """
    Return brut à `horizon` bougies dans le futur.

    Le modèle prédit le move exact (régression).
    Tu ne choisis que l'horizon, pas le seuil.

    Paramètres
    ----------
    horizon : int
        Nombre de bougies M1 dans le futur (30 = 30 minutes).
    """
    label = close.shift(-horizon) / close - 1
    label.name = f'return_{horizon}m'
    return label


# ── 2. Triple Barrier ─────────────────────────────────────────────────────────

@njit
def _triple_barrier_nb(close_arr, tp, sl, max_bars):
    """
    Kernel numba — calcule le label triple barrier pour chaque bougie.

    Retourne :
      +1 si tp touché en premier
      -1 si sl touché en premier
       0 si temps max écoulé sans toucher ni tp ni sl
    """
    n = len(close_arr)
    labels = np.zeros(n, dtype=np.int8)

    for i in range(n - 1):
        entry = close_arr[i]
        upper = entry * (1 + tp)
        lower = entry * (1 - sl)
        end   = min(i + max_bars, n - 1)

        result = 0
        for j in range(i + 1, end + 1):
            price = close_arr[j]
            if price >= upper:
                result = 1
                break
            if price <= lower:
                result = -1
                break
        labels[i] = result

    return labels


def make_triple_barrier_label(
    close:     pd.Series,
    tp:        float = 0.010,   # take profit  +1.0%
    sl:        float = 0.005,   # stop loss    -0.5%
    max_bars:  int   = 60,      # temps max    60 bougies M1
) -> pd.Series:
    """
    Triple Barrier — Lopez de Prado.

    Pour chaque bougie, regarde les `max_bars` suivantes :
      - prix monte > tp%  → label +1  (bon move)
      - prix descend > sl% → label -1  (mauvais move)
      - ni l'un ni l'autre → label  0  (neutre)

    Paramètres
    ----------
    tp : float
        Take profit en fraction (0.01 = 1%).
    sl : float
        Stop loss en fraction (0.005 = 0.5%).
    max_bars : int
        Temps maximum en bougies (60 M1 = 1 heure).
    """
    arr    = close.values.astype(np.float64)
    labels = _triple_barrier_nb(arr, tp, sl, max_bars)
    return pd.Series(labels, index=close.index, name=f'tb_tp{tp}_sl{sl}_t{max_bars}')


# ── 3. Max Gain / Max Loss (2 outputs) ───────────────────────────────────────

@njit
def _max_gain_loss_nb(close_arr, horizon):
    """
    Kernel numba — pour chaque bougie i, regarde les `horizon` bougies suivantes :
      max_gain[i] = max(close[i+1:i+horizon]) / close[i] - 1  (meilleur cas)
      max_loss[i] = min(close[i+1:i+horizon]) / close[i] - 1  (pire cas, négatif)
    """
    n = len(close_arr)
    max_gain = np.full(n, np.nan)
    max_loss = np.full(n, np.nan)

    for i in range(n - horizon):
        entry  = close_arr[i]
        future = close_arr[i + 1 : i + horizon + 1]
        max_gain[i] = (np.max(future) - entry) / entry
        max_loss[i] = (np.min(future) - entry) / entry   # valeur négative

    return max_gain, max_loss


def make_max_gain_loss_label(
    close:   pd.Series,
    horizon: int = 90,
) -> pd.DataFrame:
    """
    Labels de régression 2 outputs pour l'horizon donné.

    Paramètres
    ----------
    horizon : int
        Nombre de bougies M1 dans le futur (90 = 1h30).

    Retourne
    --------
    DataFrame avec colonnes ['max_gain', 'max_loss'] :
      max_gain  → meilleur move dans les `horizon` bougies suivantes (positif)
      max_loss  → pire move dans les `horizon` bougies suivantes (négatif)
      rr        → ratio max_gain / abs(max_loss) → R/R potentiel du trade
    """
    arr             = close.values.astype(np.float64)
    gain_arr, loss_arr = _max_gain_loss_nb(arr, horizon)

    df = pd.DataFrame({
        'max_gain': gain_arr,
        'max_loss': loss_arr,
    }, index=close.index)

    # R/R potentiel (clippé pour éviter les valeurs aberrantes)
    df['rr'] = (df['max_gain'] / df['max_loss'].abs().replace(0, np.nan)).clip(0, 20)

    return df


# ── 4. ATR Barrier (TP = rr×ATR, SL = 1×ATR) ────────────────────────────────

@njit
def _atr_barrier_nb(close_arr, atr_pct_arr, rr, max_bars, is_long=True):
    """
    Kernel numba — Triple Barrier avec niveaux ATR dynamiques.
    
    is_long=True  : TP = entry + rr*ATR, SL = entry - 1*ATR
    is_long=False : TP = entry - rr*ATR, SL = entry + 1*ATR
    """
    n = len(close_arr)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n - 1):
        if np.isnan(atr_pct_arr[i]) or atr_pct_arr[i] <= 0:
            continue
        entry = close_arr[i]
        
        if is_long:
            upper = entry * (1.0 + rr * atr_pct_arr[i])
            lower = entry * (1.0 - 1.0 * atr_pct_arr[i])
        else:
            upper = entry * (1.0 + 1.0 * atr_pct_arr[i]) # SL
            lower = entry * (1.0 - rr * atr_pct_arr[i])  # TP
            
        end   = min(i + max_bars, n - 1)
        for j in range(i + 1, end + 1):
            price = close_arr[j]
            if is_long:
                if price >= upper:
                    labels[i] = 1
                    break
                elif price <= lower:
                    labels[i] = -1
                    break
            else: # short
                if price <= lower: # TP pour un short
                    labels[i] = 1
                    break
                elif price >= upper: # SL pour un short
                    labels[i] = -1
                    break
    return labels


def make_atr_barrier_label(
    close:      pd.Series,
    high:       pd.Series,
    low:        pd.Series,
    rr:         float = 2.0,
    atr_period: int   = 14,
    max_bars:   int   = 60,
    side:       str   = 'long',
) -> pd.Series:
    """
    Triple Barrier avec TP/SL définis par l'ATR courant.

    TP = rr × ATR  (break-even = 1/(rr+1))
    SL =  1 × ATR

    Paramètres
    ----------
    rr : float
        Ratio TP/SL. rr=3 → TP=3×ATR, SL=1×ATR, break-even=25%.
    atr_period : int
        Période ATR (14 par défaut).
    max_bars : int
        Temps maximum en bougies M1.
    side : str
        'long' ou 'short'.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr     = tr.rolling(atr_period).mean()
    atr_pct = (atr / close).values.astype(np.float64)

    labels = _atr_barrier_nb(
        close.values.astype(np.float64),
        atr_pct,
        float(rr),
        int(max_bars),
        is_long=(side.lower() == 'long')
    )
    be = 1.0 / (rr + 1.0)
    return pd.Series(
        labels, index=close.index,
        name=f'atr_{side}_rr{rr}_t{max_bars}_be{be:.0%}',
    )


# ── 5. ATR 3 classes : +1=long / -1=short / 0=neutre ────────────────────────

@njit
def _atr_3class_nb(close_arr, atr_pct_arr, rr, max_bars):
    """
    Pour chaque bougie, regarde les `max_bars` suivantes :
      +1 : prix monte rr×ATR avant de descendre 1×ATR  → bon long
      -1 : prix descend rr×ATR avant de monter 1×ATR   → bon short
       0 : ni l'un ni l'autre                           → neutre
    """
    n = len(close_arr)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n - 1):
        if np.isnan(atr_pct_arr[i]) or atr_pct_arr[i] <= 0:
            continue
        entry    = close_arr[i]
        tp_long  = entry * (1.0 + rr  * atr_pct_arr[i])
        sl_long  = entry * (1.0 - 1.0 * atr_pct_arr[i])
        tp_short = entry * (1.0 - rr  * atr_pct_arr[i])
        sl_short = entry * (1.0 + 1.0 * atr_pct_arr[i])
        end = min(i + max_bars, n - 1)
        for j in range(i + 1, end + 1):
            price = close_arr[j]
            if price >= tp_long:
                labels[i] = 1
                break
            elif price <= tp_short:
                labels[i] = -1
                break
            elif price <= sl_long or price >= sl_short:
                # SL commun des deux côtés → neutre
                break
    return labels


def make_atr_3class_label(
    close:      pd.Series,
    high:       pd.Series,
    low:        pd.Series,
    rr:         float = 2.0,
    atr_period: int   = 14,
    max_bars:   int   = 60,
) -> pd.Series:
    """
    Label 3 classes symétrique pour un modèle long/short/neutre unifié.

      +1 : bon long  (monte rr×ATR avant de descendre 1×ATR)
      -1 : bon short (descend rr×ATR avant de monter 1×ATR)
       0 : neutre    (ni l'un ni l'autre dans max_bars)

    Break-even identique des deux côtés : 1/(rr+1)
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr     = tr.rolling(atr_period).mean()
    atr_pct = (atr / close).values.astype(np.float64)

    labels = _atr_3class_nb(
        close.values.astype(np.float64),
        atr_pct,
        float(rr),
        int(max_bars),
    )
    be = 1.0 / (rr + 1.0)
    return pd.Series(
        labels, index=close.index,
        name=f'atr3c_rr{rr}_t{max_bars}_be{be:.0%}',
    )


# ── Utilitaire : stats des labels ─────────────────────────────────────────────

def label_stats(label: pd.Series) -> None:
    """Affiche la distribution des labels — utile pour vérifier l'équilibre."""
    label = label.dropna()
    n = len(label)
    print(f"\nLabel : {label.name}")
    print(f"Total : {n:,} bougies")

    if label.dtype == float:
        print(f"  mean   : {label.mean():.4%}")
        print(f"  std    : {label.std():.4%}")
        print(f"  > 0    : {(label > 0).mean():.1%}")
        print(f"  > 0.5% : {(label > 0.005).mean():.1%}")
        print(f"  > 1%   : {(label > 0.01).mean():.1%}")
    else:
        for v, name in [(-1, 'sl touché'), (0, 'neutre'), (1, 'tp touché')]:
            pct = (label == v).sum() / n
            print(f"  {v:+d} ({name:12s}) : {pct:.1%}")


# ── Test standalone ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from src.ml.data import load_dataset

    print("Chargement données...")
    df = load_dataset()
    close = df['close'].dropna()

    # Test return brut
    ret = make_return_label(close, horizon=30)
    label_stats(ret)

    # Test triple barrier
    print("\nCalcul Triple Barrier (peut prendre 1-2 min sur 2.9M bougies)...")
    tb = make_triple_barrier_label(close, tp=0.010, sl=0.005, max_bars=60)
    label_stats(tb)
