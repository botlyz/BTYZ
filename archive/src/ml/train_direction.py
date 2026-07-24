"""
Step 2 — Prédicteur de direction
XGBoost qui prédit long/short UNIQUEMENT quand le step 1 détecte un gros move.
Reçoit la proba du step 1 comme feature.
Walk-forward : 6 mois train, 1 mois test, rolling mensuel.
Timeframe : 5min BTC.
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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

print("=" * 60)
print("  DIRECTION PREDICTOR — XGBoost Walk-Forward")
print("  (utilise les probas du move detector comme feature)")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
# CHARGER LE MOVE DETECTOR
# ═══════════════════════════════════════════════════════════
print("\nChargement move detector...", flush=True)
with open('cache/ml/move_detector_1h.pkl', 'rb') as f:
    move_model = pickle.load(f)

move_preds = move_model['all_preds']['top20']  # utiliser top20 (le plus fiable)
move_preds = move_preds.set_index('date') if 'date' in move_preds.columns else move_preds
print(f"  Prédictions move: {len(move_preds)} lignes")

# ═══════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════
print("Chargement données...", flush=True)

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

# Spot
spot = pd.read_csv('data/raw/binance/spot/1m/BTCUSDT.csv')
spot['date'] = pd.to_datetime(spot['date'], unit='ms')
spot = spot.set_index('date').sort_index()
spot = spot[~spot.index.duplicated(keep='first')]
for c in spot.columns: spot[c] = pd.to_numeric(spot[c], errors='coerce')
s5 = spot.resample('5min').agg({
    'volume': 'sum', 'taker_buy_volume': 'sum', 'close': 'last'
}).dropna()
s5['spot_delta'] = s5['taker_buy_volume'] - (s5['volume'] - s5['taker_buy_volume'])
s5['spot_cvd'] = s5['spot_delta'].cumsum()

# Metrics
metrics = pd.read_csv('data/raw/binance/um/metrics/BTCUSDT.csv')
metrics['date'] = pd.to_datetime(metrics['date'], unit='ms')
metrics = metrics.set_index('date').sort_index()
metrics = metrics[~metrics.index.duplicated(keep='first')]
for c in metrics.columns: metrics[c] = pd.to_numeric(metrics[c], errors='coerce')

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

N = len(df)
close = df['close'].values
print(f"  {N} candles 5min")

# ═══════════════════════════════════════════════════════════
# FEATURES (mêmes que step 1 + probas move detector)
# ═══════════════════════════════════════════════════════════
print("Features...", flush=True)

def pct_change_vec(arr, p):
    s = pd.Series(arr)
    shifted = s.shift(p)
    return ((s - shifted) / shifted.abs().replace(0, np.nan)).values

def zscore_vec(arr, w):
    s = pd.Series(arr)
    m = s.rolling(w, min_periods=w).mean()
    std = s.rolling(w, min_periods=w).std()
    return ((s - m) / std.replace(0, np.nan)).values

def abs_change_norm_vec(arr, p, std_w=36):
    s = pd.Series(arr)
    delta = (s - s.shift(p)).abs()
    std = s.rolling(std_w, min_periods=std_w).std()
    return (delta / std.replace(0, np.nan)).values

def change_vec(arr, p):
    """Changement signé (pas absolu) — important pour la direction."""
    s = pd.Series(arr)
    shifted = s.shift(p)
    return ((s - shifted) / shifted.abs().replace(0, np.nan)).values

feat = pd.DataFrame(index=df.index)
_tasks = []

# OI — direction + magnitude
oi_v = df['oi_value'].values
for lb in [3, 6, 12, 36, 72]:
    _tasks.append((f'oi_pct_{lb}', pct_change_vec(oi_v, lb)))
    _tasks.append((f'oi_norm_{lb}', abs_change_norm_vec(oi_v, lb)))
for w in [36, 72, 144, 288]:
    _tasks.append((f'oi_z_{w}', zscore_vec(oi_v, w)))

# CVD futures — direction (signé, crucial pour long/short)
cvd_v = df['cvd'].values
delta_v = df['delta'].values
for lb in [3, 6, 12, 36, 72]:
    _tasks.append((f'cvd_change_{lb}', change_vec(cvd_v, lb)))
    _tasks.append((f'cvd_norm_{lb}', abs_change_norm_vec(cvd_v, lb)))
for w in [36, 72, 144]:
    _tasks.append((f'delta_z_{w}', zscore_vec(delta_v, w)))

# CVD spot — direction
spot_cvd_v = df['spot_cvd'].values
spot_delta_v = df['spot_delta'].values
for lb in [3, 6, 12, 36]:
    _tasks.append((f'spot_cvd_change_{lb}', change_vec(spot_cvd_v, lb)))
for w in [36, 72]:
    _tasks.append((f'spot_delta_z_{w}', zscore_vec(spot_delta_v, w)))

# Divergence CVD/prix (signé) — clé pour la direction
for lb in [6, 12, 36, 72]:
    price_chg = change_vec(close, lb)
    cvd_chg = change_vec(cvd_v, lb)
    # Divergence = CVD monte mais prix baisse → bullish
    div = np.where(np.isfinite(price_chg) & np.isfinite(cvd_chg),
                   np.sign(cvd_chg) - np.sign(price_chg), np.nan)
    _tasks.append((f'div_cvd_price_{lb}', div))

# Volume
vol_v = df['volume'].values
for lb in [3, 6, 12, 36]:
    _tasks.append((f'vol_roc_{lb}', pct_change_vec(vol_v, lb)))
for w in [36, 72, 144]:
    _tasks.append((f'vol_z_{w}', zscore_vec(vol_v, w)))

# LS ratio (directionnel — clé)
ls_v = df['ls_ratio'].values
_tasks.append(('ls_ratio', ls_v))
for w in [36, 72, 144, 288]:
    _tasks.append((f'ls_z_{w}', zscore_vec(ls_v, w)))

# Top trader LS
top_v = df['top_trader_ls'].values
_tasks.append(('top_ls', top_v))
for w in [36, 72, 144]:
    _tasks.append((f'top_ls_z_{w}', zscore_vec(top_v, w)))

# Taker LS vol (directionnel)
taker_v = df['taker_ls_vol'].values
_tasks.append(('taker_ls', taker_v))
for w in [36, 72, 144]:
    _tasks.append((f'taker_z_{w}', zscore_vec(taker_v, w)))

# Funding (directionnel)
fr_v = df['funding_rate'].values
_tasks.append(('funding', fr_v))
for w in [288, 576]:
    _tasks.append((f'funding_z_{w}', zscore_vec(fr_v, w)))

# Basis (directionnel)
basis_v = df['basis'].values
_tasks.append(('basis', basis_v))
for w in [36, 72, 288]:
    _tasks.append((f'basis_z_{w}', zscore_vec(basis_v, w)))

# Return récent (momentum)
for lb in [1, 3, 6, 12, 36, 72]:
    _tasks.append((f'ret_{lb}', pct_change_vec(close, lb)))

# RSI
def rsi_vec(close, period):
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values

for p in [14, 60]:
    _tasks.append((f'rsi_{p}', rsi_vec(close, p)))

# Volatilité récente (ATR-like)
hl_v = (df['high'].values - df['low'].values) / df['close'].values
for lb in [6, 12, 36]:
    _tasks.append((f'hl_range_z_{lb}', zscore_vec(hl_v, lb)))

for name, values in tqdm(_tasks, desc="Features"):
    feat[name] = values

# Ajouter les probas du move detector comme feature
move_proba = pd.Series(np.nan, index=df.index)
common_idx = df.index.intersection(move_preds.index)
move_proba.loc[common_idx] = move_preds.loc[common_idx, 'proba'].values
feat['move_proba'] = move_proba.values

feature_cols = list(feat.columns)
print(f"  {len(feature_cols)} features (dont move_proba)")

# ═══════════════════════════════════════════════════════════
# LABELS — direction du move sur 1h
# ═══════════════════════════════════════════════════════════
print("Labels...", flush=True)

HORIZON = 12  # 1h

# Direction = signe du return sur l'horizon
future_ret = np.full(N, np.nan)
for i in range(N - HORIZON):
    future_ret[i] = (close[i + HORIZON] - close[i]) / close[i]

# Label : 1 = long (prix monte), 0 = short (prix baisse)
direction = (future_ret > 0).astype(float)
direction[np.isnan(future_ret)] = np.nan

# Magnitude du move
future_abs_ret = np.abs(future_ret)

X = feat.values
_valid_rows = np.all(np.isfinite(X), axis=1) & ~np.isnan(direction)
print(f"  Lignes valides: {_valid_rows.sum():,}/{N:,} ({_valid_rows.sum()/N*100:.1f}%)")

# ═══════════════════════════════════════════════════════════
# WALK-FORWARD
# ═══════════════════════════════════════════════════════════
print(f"\nWalk-Forward...", flush=True)

months = pd.Series(df.index).dt.to_period('M').unique()
train_months = 6

results_all = []
results_filtered = []  # Seulement quand move_proba > seuil

# Seuils de filtrage move detector
MOVE_THRESHOLDS = [0.0, 0.3, 0.5, 0.7]

t0 = time.time()

for i in tqdm(range(train_months, len(months) - 1), desc="Walk-Forward"):
    test_month = months[i]
    train_start = months[i - train_months]

    train_mask = (df.index >= train_start.start_time) & (df.index < test_month.start_time)
    test_mask = (df.index >= test_month.start_time) & (df.index < (test_month + 1).start_time)

    train_valid = train_mask & _valid_rows
    test_valid = test_mask & _valid_rows

    if train_valid.sum() < 1000 or test_valid.sum() < 100:
        continue

    X_train = X[train_valid]
    X_test = X[test_valid]
    y_train = direction[train_valid]
    y_test = direction[test_valid]
    test_move_proba = feat['move_proba'].values[test_valid]
    test_abs_ret = future_abs_ret[test_valid]

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
        nthread=24,
        n_jobs=24,
        tree_method='hist',
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]  # proba de long
    pred = (proba > 0.5).astype(int)

    # Résultats globaux (tous les moments)
    acc = accuracy_score(y_test, pred)
    results_all.append({
        'fold': str(test_month),
        'accuracy': acc,
        'baseline': 0.5,
        'lift': acc - 0.5,
    })

    # Résultats filtrés par move detector
    for move_th in MOVE_THRESHOLDS:
        if move_th == 0:
            filt = np.ones(len(y_test), dtype=bool)
        else:
            filt = test_move_proba > move_th

        if filt.sum() < 10:
            continue

        y_filt = y_test[filt]
        pred_filt = pred[filt]
        abs_ret_filt = test_abs_ret[filt]

        acc_filt = accuracy_score(y_filt, pred_filt)

        # Profit simulé : +abs_ret quand correct, -abs_ret quand incorrect
        correct = (pred_filt == y_filt)
        pnl = np.where(correct, abs_ret_filt, -abs_ret_filt).sum()

        results_filtered.append({
            'fold': str(test_month),
            'move_th': move_th,
            'accuracy': acc_filt,
            'n_signals': int(filt.sum()),
            'pnl_pct': pnl * 100,
            'avg_move': float(abs_ret_filt.mean()) * 100,
        })

    elapsed = time.time() - t0

last_model = model

# ═══════════════════════════════════════════════════════════
# RÉSULTATS
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"RÉSULTATS — Direction 1h")
print(f"{'='*60}")

# Global
df_all = pd.DataFrame(results_all)
print(f"\nGlobal (tous les moments):")
print(f"  Accuracy OOS: {df_all['accuracy'].mean():.3f} (baseline=0.500)")
print(f"  Lift: {df_all['lift'].mean():+.3f}")
print(f"  Folds > 50%: {(df_all['accuracy'] > 0.5).sum()}/{len(df_all)}")

# Filtré par move detector
print(f"\nFiltré par Move Detector:")
df_filt = pd.DataFrame(results_filtered)
for move_th in MOVE_THRESHOLDS:
    sub = df_filt[df_filt['move_th'] == move_th]
    if len(sub) == 0:
        continue
    avg_acc = sub['accuracy'].mean()
    avg_signals = sub['n_signals'].mean()
    avg_pnl = sub['pnl_pct'].mean()
    avg_move = sub['avg_move'].mean()
    folds_pos = (sub['accuracy'] > 0.5).sum()

    label = "Tous" if move_th == 0 else f"move_proba>{move_th}"
    print(f"\n  {label}:")
    print(f"    Accuracy:     {avg_acc:.3f} (baseline=0.500)")
    print(f"    Lift:         {avg_acc - 0.5:+.3f}")
    print(f"    Signals/mois: {avg_signals:.0f}")
    print(f"    PnL/mois:     {avg_pnl:+.2f}%")
    print(f"    Avg move:     {avg_move:.3f}%")
    print(f"    Folds > 50%:  {folds_pos}/{len(sub)} ({folds_pos/len(sub)*100:.0f}%)")

# Feature importance
print(f"\nTop 15 features:")
imp = last_model.feature_importances_
top_idx = np.argsort(imp)[::-1][:15]
for idx in top_idx:
    print(f"  {feature_cols[idx]:<25} {imp[idx]:.4f}")

# Sauver
save_data = {
    'name': 'direction_1h',
    'horizon': HORIZON,
    'results_all': results_all,
    'results_filtered': results_filtered,
    'feature_cols': feature_cols,
}
os.makedirs('cache/ml', exist_ok=True)
with open('cache/ml/direction_1h.pkl', 'wb') as f:
    pickle.dump(save_data, f)
print(f"\nSauvé → cache/ml/direction_1h.pkl")
