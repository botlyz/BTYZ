"""
XGBoost ATR Barrier — TP = rr×ATR, SL = 1×ATR.

Teste RR = [1, 2, 3, 5] en walk-forward (6m train / 1m test).
Break-even automatique : 1/(rr+1)  →  50% / 33% / 25% / 17%

L'objectif : trouver le RR où precision OOS > break-even de façon consistante.

Lancement :
  .venv/bin/python src/ml/train_xgb_atr.py
  .venv/bin/python src/ml/train_xgb_atr.py --rr 3          # un seul RR
  .venv/bin/python src/ml/train_xgb_atr.py --rr 2 3 5      # plusieurs RR
"""

import os
import sys
import pickle
import logging
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from src.ml.data    import load_dataset
from src.ml.labels  import make_atr_barrier_label, label_stats
from src.ml.features import FEATURES_RAW, add_momentum_features

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level   = logging.INFO,
    format  = '%(message)s',
    handlers=[
        logging.FileHandler(f'logs/xgb_atr_{_ts}.log'),
        logging.StreamHandler(sys.stdout),
    ],
)
def print(*a, **k): logging.info(' '.join(str(x) for x in a))

# ── Config ────────────────────────────────────────────────────────────────────
RR_LIST      = [1, 2, 3, 5]
ATR_PERIOD   = 14
MAX_BARS     = 60          # 1 heure max pour tous les RR
TRAIN_MONTHS = 6
TEST_MONTHS  = 1
MIN_SAMPLES  = 300
OUTPUT_DIR   = 'cache/ml'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Walk-forward splits ───────────────────────────────────────────────────────

def make_wf_splits(index, train_months=6, test_months=1):
    splits = []
    train_start = index[0]
    end = index[-1]
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end  = train_end   + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        splits.append({'train': (train_start, train_end),
                       'test':  (train_end,   test_end)})
        train_start += pd.DateOffset(months=test_months)
    return splits


# ── Entraînement pour un RR et un Side ────────────────────────────────────────

def train_one_side(df, rr, side='long'):
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost non installé → pip install xgboost")
        return None

    be = 1.0 / (rr + 1.0)
    print(f"\n{'='*65}")
    print(f"SIDE = {side.upper()}  |  RR = {rr}  |  TP = {rr}×ATR  |  SL = 1×ATR")
    print(f"{'='*65}")

    # Label ATR dynamique
    print(f"Calcul labels ATR {side} (rr={rr}, max_bars={MAX_BARS})...")
    label = make_atr_barrier_label(
        df['close'], df['high'], df['low'],
        rr=rr, atr_period=ATR_PERIOD, max_bars=MAX_BARS,
        side=side
    )
    label_stats(label)

    df_side = df.copy()
    df_side['label'] = label
    df_side = add_momentum_features(df_side)

    feat_cols = [c for c in df_side.columns if c in FEATURES_RAW or
                 any(c.startswith(p) for p in
                     ['cvd_perp_d', 'cvd_spot_d', 'price_d', 'div_perp_', 'taker_ratio'])]

    df_clean = df_side[feat_cols + ['label']].dropna()
    print(f"Bougies après dropna : {len(df_clean):,}")

    splits = make_wf_splits(df_clean.index, TRAIN_MONTHS, TEST_MONTHS)
    print(f"Walk-forward : {len(splits)} folds ({TRAIN_MONTHS}m train / {TEST_MONTHS}m test)")

    results, importances, all_preds, all_labels = [], [], [], []

    for i, split in enumerate(splits):
        tr_s, tr_e = split['train']
        te_s, te_e = split['test']

        train_df = df_clean.loc[tr_s:tr_e]
        test_df  = df_clean.loc[te_s:te_e]

        if len(train_df) < 1000 or len(test_df) < 100:
            continue

        X_tr, y_tr = train_df[feat_cols], train_df['label']
        X_te, y_te = test_df[feat_cols],  test_df['label']

        mask_tr = y_tr != 0
        mask_te = y_te != 0

        if mask_tr.sum() < MIN_SAMPLES or mask_te.sum() < 50:
            continue

        X_tr_a = X_tr[mask_tr]
        y_tr_a = (y_tr[mask_tr] == 1).astype(int)
        X_te_a = X_te[mask_te]
        y_te_a = (y_te[mask_te] == 1).astype(int)

        scale = (y_tr_a == 0).sum() / max((y_tr_a == 1).sum(), 1)

        model = xgb.XGBClassifier(
            n_estimators     = 300,
            max_depth        = 4,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            scale_pos_weight = scale,
            eval_metric      = 'logloss',
            device           = 'cuda',
            verbosity        = 0,
            random_state     = 42,
        )
        model.fit(X_tr_a, y_tr_a)

        proba = model.predict_proba(X_te_a)[:, 1]
        preds = (proba > 0.5).astype(int)

        tp_rate   = float(y_te_a.mean())
        pred_pos  = preds == 1
        n_signals = int(pred_pos.sum())

        if n_signals > 0:
            precision = float((y_te_a.values[pred_pos] == 1).mean())
        else:
            precision = 0.0

        lift       = precision - tp_rate if n_signals > 0 else 0.0
        profitable = precision > be and n_signals > 10

        results.append({
            'fold':       i + 1,
            'test_start': te_s,
            'precision':  precision,
            'tp_rate':    tp_rate,
            'lift':       lift,
            'profitable': profitable,
            'n_signals':  n_signals,
        })

        importances.append(dict(zip(feat_cols, model.feature_importances_)))
        all_preds.append(pd.Series(proba, index=X_te_a.index))
        all_labels.append(y_te_a)

    if not results:
        return None

    results_df = pd.DataFrame(results)
    n_prof = results_df['profitable'].sum()
    n_tot  = len(results_df)

    print(f"\nRésumé {side.upper()} RR={rr} : Prec={results_df['precision'].mean():.3f} | Lift={results_df['lift'].mean():+.3f} | OK={n_prof}/{n_tot}")

    preds_series  = pd.concat(all_preds).sort_index()
    labels_series = pd.concat(all_labels).sort_index()

    output = {
        'name':       f'atr_{side}_rr{rr}',
        'side':       side,
        'rr':         rr,
        'atr_period': ATR_PERIOD,
        'max_bars':   MAX_BARS,
        'break_even': be,
        'config':     {'tp': f'{rr}×ATR', 'sl': '1×ATR', 'max_bars': MAX_BARS, 'side': side},
        'results_df': results_df,
        'importance': pd.DataFrame(importances).mean().sort_values(ascending=False),
        'preds_oos':  preds_series,
        'labels_oos': labels_series,
        'feature_cols': feat_cols,
    }

    out_path = os.path.join(OUTPUT_DIR, f'xgb_atr_{side}_rr{rr}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)

    return output


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rr', type=float, nargs='+', default=RR_LIST)
    parser.add_argument('--side', type=str, nargs='+', default=['long', 'short'])
    args = parser.parse_args()

    print("Chargement données...")
    df = load_dataset()

    for rr in args.rr:
        for side in args.side:
            train_one_side(df, rr, side=side)
