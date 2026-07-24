"""
Step 1 — Détecteur de mouvement (volatilité)
XGBoost qui prédit : "est-ce qu'un move > seuil va arriver dans les N prochaines bougies ?"
Walk-forward : 6 mois train, 1 mois test, rolling mensuel.
Timeframe : 5min. Features : OI, CVD, LS ratio, funding, volume, etc.
"""
import os
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['OPENBLAS_NUM_THREADS'] = '24'
os.environ['MKL_NUM_THREADS'] = '24'

import sys, time, warnings, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

print("=" * 60)
print("  MOVE DETECTOR — XGBoost Walk-Forward")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════
print("\nChargement...", flush=True)

# Futures 1min → 5min
data = pd.read_csv('data/raw/binance/um/1m/BTCUSDT.csv')
data['date'] = pd.to_datetime(data['date'], unit='ms')
data = data.set_index('date').sort_index()
data = data[~data.index.duplicated(keep='first')]
for c in data.columns:
    data[c] = pd.to_numeric(data[c], errors='coerce')

d5 = data.resample('5min').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'taker_buy_volume': 'sum'
}).dropna()
d5['taker_sell_vol'] = d5['volume'] - d5['taker_buy_volume']
d5['delta'] = d5['taker_buy_volume'] - d5['taker_sell_vol']
d5['cvd'] = d5['delta'].cumsum()

# Spot 1min → 5min
spot = pd.read_csv('data/raw/binance/spot/1m/BTCUSDT.csv')
spot['date'] = pd.to_datetime(spot['date'], unit='ms')
spot = spot.set_index('date').sort_index()
spot = spot[~spot.index.duplicated(keep='first')]
for c in spot.columns:
    spot[c] = pd.to_numeric(spot[c], errors='coerce')
s5 = spot.resample('5min').agg({
    'volume': 'sum', 'taker_buy_volume': 'sum', 'close': 'last'
}).dropna()
s5['spot_delta'] = s5['taker_buy_volume'] - (s5['volume'] - s5['taker_buy_volume'])
s5['spot_cvd'] = s5['spot_delta'].cumsum()

# Metrics 5min
metrics = pd.read_csv('data/raw/binance/um/metrics/BTCUSDT.csv')
metrics['date'] = pd.to_datetime(metrics['date'], unit='ms')
metrics = metrics.set_index('date').sort_index()
metrics = metrics[~metrics.index.duplicated(keep='first')]
for c in metrics.columns:
    metrics[c] = pd.to_numeric(metrics[c], errors='coerce')

# Funding
funding = pd.read_csv('data/raw/binance/um/funding/BTCUSDT.csv')
funding['date'] = pd.to_datetime(funding['date'], unit='ms')
funding = funding.set_index('date').sort_index()
funding['funding_rate'] = pd.to_numeric(funding['funding_rate'], errors='coerce')

# Merge
df = d5[['open', 'high', 'low', 'close', 'volume', 'delta', 'cvd']].copy()
df = df.join(s5[['spot_delta', 'spot_cvd', 'close']].rename(columns={'close': 'spot_close'}), how='left')
df = df.join(metrics[['oi', 'oi_value', 'top_trader_ls', 'ls_ratio', 'taker_ls_vol']], how='left')
df = df.join(funding[['funding_rate']], how='left')
df['funding_rate'] = df['funding_rate'].ffill()
df['basis'] = (df['close'] - df['spot_close']) / df['spot_close']
df = df.ffill().dropna(subset=['close', 'oi_value'])

print(f"Données: {len(df)} candles 5min, {df.index[0]} → {df.index[-1]}")

# ═══════════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════════
print("Features...", flush=True)

close = df['close'].values
N = len(df)

def pct_change_vec(arr, p):
    """Vectorisé avec pandas."""
    s = pd.Series(arr)
    shifted = s.shift(p)
    return ((s - shifted) / shifted.abs().replace(0, np.nan)).values

def zscore_vec(arr, w):
    """Vectorisé avec pandas rolling."""
    s = pd.Series(arr)
    m = s.rolling(w, min_periods=w).mean()
    std = s.rolling(w, min_periods=w).std()
    return ((s - m) / std.replace(0, np.nan)).values

def abs_change_norm_vec(arr, p, std_w=36):
    """Changement absolu normalisé — vectorisé."""
    s = pd.Series(arr)
    delta = (s - s.shift(p)).abs()
    std = s.rolling(std_w, min_periods=std_w).std()
    return (delta / std.replace(0, np.nan)).values

from tqdm import tqdm

feat = pd.DataFrame(index=df.index)

def _add_features():
    """Calcul vectorisé de toutes les features avec barre de progression."""
    tasks = []

    # OI
    oi_s = pd.Series(df['oi_value'].values, index=df.index)
    for lb in [3, 6, 12, 36, 72]:
        tasks.append((f'oi_pct_{lb}', pct_change_vec(oi_s.values, lb)))
        tasks.append((f'oi_norm_{lb}', abs_change_norm_vec(oi_s.values, lb)))
    for w in [36, 72, 144, 288]:
        tasks.append((f'oi_z_{w}', zscore_vec(oi_s.values, w)))

    # CVD futures
    cvd_v = df['cvd'].values
    delta_v = df['delta'].values
    for lb in [3, 6, 12, 36, 72]:
        tasks.append((f'cvd_norm_{lb}', abs_change_norm_vec(cvd_v, lb)))
    for w in [36, 72, 144]:
        tasks.append((f'delta_z_{w}', zscore_vec(delta_v, w)))

    # CVD spot
    spot_cvd_v = df['spot_cvd'].values
    for lb in [3, 6, 12, 36]:
        tasks.append((f'spot_cvd_norm_{lb}', abs_change_norm_vec(spot_cvd_v, lb)))

    # Volume
    vol_v = df['volume'].values
    for lb in [3, 6, 12, 36]:
        tasks.append((f'vol_roc_{lb}', pct_change_vec(vol_v, lb)))
    for w in [36, 72, 144]:
        tasks.append((f'vol_z_{w}', zscore_vec(vol_v, w)))

    # LS ratio
    ls_v = df['ls_ratio'].values
    tasks.append(('ls_ratio', ls_v))
    for w in [36, 72, 144, 288]:
        tasks.append((f'ls_z_{w}', zscore_vec(ls_v, w)))

    # Top trader LS
    top_v = df['top_trader_ls'].values
    for w in [36, 72, 144]:
        tasks.append((f'top_ls_z_{w}', zscore_vec(top_v, w)))

    # Taker LS vol
    taker_v = df['taker_ls_vol'].values
    for w in [36, 72, 144]:
        tasks.append((f'taker_z_{w}', zscore_vec(taker_v, w)))

    # Funding
    fr_v = df['funding_rate'].values
    tasks.append(('funding', fr_v))
    for w in [288, 576]:
        tasks.append((f'funding_z_{w}', zscore_vec(fr_v, w)))

    # Basis
    basis_v = df['basis'].values
    tasks.append(('basis', basis_v))
    for w in [36, 72, 288]:
        tasks.append((f'basis_z_{w}', zscore_vec(basis_v, w)))

    # Prix — volatilité récente
    hl_v = (df['high'].values - df['low'].values) / df['close'].values
    for lb in [6, 12, 36]:
        tasks.append((f'hl_range_z_{lb}', zscore_vec(hl_v, lb)))

    # Return récent
    for lb in [3, 6, 12, 36]:
        tasks.append((f'ret_{lb}', pct_change_vec(close, lb)))

    return tasks

print("  Calcul vectorisé...", flush=True)
_tasks = _add_features()
for name, values in tqdm(_tasks, desc="Features"):
    feat[name] = values

feature_cols = list(feat.columns)
print(f"{len(feature_cols)} features")

# ═══════════════════════════════════════════════════════════
# LABELS — move > seuil dans les N prochaines bougies
# ═══════════════════════════════════════════════════════════
print("Labels...", flush=True)

# Horizons à tester
HORIZON = 12  # 1h (12 bougies de 5min)

# Max move futur — vectorisé avec stride tricks
_high_v = df['high'].values
_low_v = df['low'].values
max_move = np.full(N, np.nan)
print("  Calcul max move futur...", flush=True)
for i in tqdm(range(0, N - HORIZON, 1000), desc="Labels"):
    end = min(i + 1000, N - HORIZON)
    for j in range(i, end):
        fh = np.max(_high_v[j+1:j+1+HORIZON])
        fl = np.min(_low_v[j+1:j+1+HORIZON])
        max_move[j] = (fh - fl) / close[j] * 100

df['max_move_1h'] = max_move

# Seuils : top 20% de volatilité = "gros move"
valid_mask = ~np.isnan(max_move)
THRESHOLDS = {
    'top20': np.percentile(max_move[valid_mask], 80),
    'top10': np.percentile(max_move[valid_mask], 90),
    'top5': np.percentile(max_move[valid_mask], 95),
}
print(f"Seuils de move 1h:")
for k, v in THRESHOLDS.items():
    n_pos = (max_move[valid_mask] >= v).sum()
    print(f"  {k}: {v:.3f}% ({n_pos:,} positifs = {n_pos/valid_mask.sum()*100:.1f}%)")

# ═══════════════════════════════════════════════════════════
# WALK-FORWARD
# ═══════════════════════════════════════════════════════════
print(f"\nWalk-Forward (6 mois train / 1 mois test)...", flush=True)

# Préparer X, y
X = feat.values
y_dict = {k: (max_move >= v).astype(int) for k, v in THRESHOLDS.items()}

# Créer les folds mensuels
months = pd.Series(df.index).dt.to_period('M').unique()
train_months = 6
# Pré-calculer le masque de lignes valides (pas de NaN dans les features)
_valid_rows = np.all(np.isfinite(X), axis=1)
print(f"Lignes valides: {_valid_rows.sum():,}/{N:,} ({_valid_rows.sum()/N*100:.1f}%)")

results = {k: [] for k in THRESHOLDS}
all_preds = {k: [] for k in THRESHOLDS}
last_model = None

from tqdm import tqdm

t0 = time.time()
total_folds = len(months) - train_months - 1

for i in tqdm(range(train_months, len(months) - 1), total=total_folds, desc="Walk-Forward"):
    test_month = months[i]
    train_start = months[i - train_months]

    train_mask = (df.index >= train_start.start_time) & (df.index < test_month.start_time)
    test_mask = (df.index >= test_month.start_time) & (df.index < (test_month + 1).start_time)

    # Filtrer les NaN
    train_valid = train_mask & ~np.isnan(max_move) & _valid_rows
    test_valid = test_mask & ~np.isnan(max_move) & _valid_rows

    n_train = train_valid.sum()
    n_test = test_valid.sum()
    if n_train < 1000 or n_test < 100:
        if i == train_months:  # Premier fold, debug
            print(f"  DEBUG: train_mask={train_mask.sum()}, test_mask={test_mask.sum()}, "
                  f"~nan_move={((~np.isnan(max_move)) & train_mask).sum()}, "
                  f"feat_notna={feat.notna().all(axis=1).sum()}/{len(feat)}, "
                  f"train_valid={n_train}, test_valid={n_test}", flush=True)
        continue

    X_train = X[train_valid]
    X_test = X[test_valid]

    for th_name, threshold in THRESHOLDS.items():
        y = y_dict[th_name]
        y_train = y[train_valid]
        y_test = y[test_valid]

        # Balance des classes
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale = n_neg / n_pos if n_pos > 0 else 1

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            nthread=24,
            n_jobs=24,
            tree_method='hist',
        )
        model.fit(X_train, y_train)
        last_model = model

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba > 0.5).astype(int)

        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        baseline = y_test.mean()
        lift = prec - baseline

        results[th_name].append({
            'fold': str(test_month),
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'baseline': baseline,
            'lift': lift,
            'n_pred': pred.sum(),
            'n_test': len(y_test),
        })

        # Sauver les prédictions OOS
        test_dates = df.index[test_valid]
        for j in range(len(test_dates)):
            all_preds[th_name].append({
                'date': test_dates[j],
                'proba': proba[j],
                'label': y_test[j],
                'max_move': max_move[test_valid][j],
            })

    elapsed = time.time() - t0
    fold_n = i - train_months + 1
    total_folds = len(months) - train_months - 1
    print(f"  Fold {fold_n}/{total_folds} | {test_month} | "
          f"top20: prec={results['top20'][-1]['precision']:.3f} lift={results['top20'][-1]['lift']:+.3f} | "
          f"top10: prec={results['top10'][-1]['precision']:.3f} lift={results['top10'][-1]['lift']:+.3f} | "
          f"{elapsed:.0f}s", flush=True)

# ═══════════════════════════════════════════════════════════
# RÉSULTATS
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"RÉSULTATS — Horizon 1h ({HORIZON} bougies 5min)")
print(f"{'='*60}")

for th_name, th_val in THRESHOLDS.items():
    res = results[th_name]
    if not res:
        continue
    df_res = pd.DataFrame(res)
    avg_prec = df_res['precision'].mean()
    avg_rec = df_res['recall'].mean()
    avg_f1 = df_res['f1'].mean()
    avg_base = df_res['baseline'].mean()
    avg_lift = df_res['lift'].mean()
    folds_positive = (df_res['lift'] > 0).sum()

    print(f"\n  {th_name} (seuil={th_val:.3f}%):")
    print(f"    Precision OOS:  {avg_prec:.3f}  (baseline={avg_base:.3f})")
    print(f"    Lift:           {avg_lift:+.3f}")
    print(f"    Recall:         {avg_rec:.3f}")
    print(f"    F1:             {avg_f1:.3f}")
    print(f"    Folds positifs: {folds_positive}/{len(res)}")

# Feature importance (dernier modèle)
print(f"\nTop 15 features (dernier fold):")
if last_model is None:
    print("  Aucun modèle entraîné!")
    sys.exit(1)
imp = last_model.feature_importances_
top_idx = np.argsort(imp)[::-1][:15]
for idx in top_idx:
    print(f"  {feature_cols[idx]:<25} {imp[idx]:.4f}")

# Sauver
save_data = {
    'name': 'move_detector_1h',
    'horizon': HORIZON,
    'thresholds': THRESHOLDS,
    'results': results,
    'all_preds': {k: pd.DataFrame(v) for k, v in all_preds.items()},
    'feature_cols': feature_cols,
}
os.makedirs('cache/ml', exist_ok=True)
with open('cache/ml/move_detector_1h.pkl', 'wb') as f:
    pickle.dump(save_data, f)
print(f"\nSauvé → cache/ml/move_detector_1h.pkl")
