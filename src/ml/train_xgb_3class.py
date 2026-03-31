"""
XGBoost 3 classes — long / short / neutre.

Un seul modèle prédit simultanément :
  +1 → prix monte rr×ATR avant de descendre 1×ATR  (long)
  -1 → prix descend rr×ATR avant de monter 1×ATR   (short)
   0 → neutre (reste dehors)

Avantages vs deux modèles séparés :
  - Apprend les deux contextes en même temps (long ET short)
  - La classe neutre agit comme filtre naturel d'incertitude
  - Un seul modèle à déployer
  - Seuil à 0.4+ = très sélectif (3 classes, base = 0.33)

Sortie : P(long), P(short), P(neutre) par bougie
Signal  : long si P(long) > seuil  |  short si P(short) > seuil

Lancement :
  .venv/bin/python src/ml/train_xgb_3class.py
  .venv/bin/python src/ml/train_xgb_3class.py --rr 2 3
"""

import os, sys, pickle, logging, argparse, warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from src.ml.data    import load_dataset
from src.ml.labels  import make_atr_3class_label, label_stats
from src.ml.features import FEATURES_RAW, add_momentum_features

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO, format='%(message)s',
    handlers=[
        logging.FileHandler(f'logs/xgb_3class_{_ts}.log'),
        logging.StreamHandler(sys.stdout),
    ],
)
def print(*a, **k): logging.info(' '.join(str(x) for x in a))

# ── Config ────────────────────────────────────────────────────────────────────
RR_LIST      = [2, 3]
ATR_PERIOD   = 14
MAX_BARS     = 60
TRAIN_MONTHS = 6
TEST_MONTHS  = 1
MIN_SAMPLES  = 200   # par classe
OUTPUT_DIR   = 'cache/ml'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapping label numérique → classe XGBoost (doit être 0,1,2)
# -1 → 0 (short), 0 → 1 (neutre), +1 → 2 (long)
LABEL_MAP  = {-1: 0, 0: 1, 1: 2}
CLASS_NAME = {0: 'short', 1: 'neutre', 2: 'long'}


def make_wf_splits(index, train_months=6, test_months=1):
    splits, train_start = [], index[0]
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


def train_3class(df, rr):
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost non installé"); return None

    be = 1.0 / (rr + 1.0)
    print(f"\n{'='*65}")
    print(f"3-CLASS  |  RR={rr}  |  BE={be:.1%} des deux côtés")
    print(f"{'='*65}")

    print(f"Calcul label 3 classes (rr={rr}, max_bars={MAX_BARS})...")
    label_raw = make_atr_3class_label(
        df['close'], df['high'], df['low'],
        rr=rr, atr_period=ATR_PERIOD, max_bars=MAX_BARS,
    )
    label_stats(label_raw)

    df2 = df.copy()
    df2['label'] = label_raw.map(LABEL_MAP)   # -1→0, 0→1, +1→2
    df2 = add_momentum_features(df2)

    feat_cols = [c for c in df2.columns if c in FEATURES_RAW or
                 any(c.startswith(p) for p in
                     ['cvd_perp_d', 'cvd_spot_d', 'price_d', 'div_perp_', 'taker_ratio'])]

    df_clean = df2[feat_cols + ['label']].dropna()
    print(f"Bougies après dropna : {len(df_clean):,}")

    splits = make_wf_splits(df_clean.index, TRAIN_MONTHS, TEST_MONTHS)
    print(f"Walk-forward : {len(splits)} folds | BE à battre : {be:.1%}\n")

    results, importances = [], []
    preds_long, preds_short = [], []
    labels_long, labels_short = [], []

    for i, split in enumerate(splits):
        tr_s, tr_e = split['train']
        te_s, te_e = split['test']

        train_df = df_clean.loc[tr_s:tr_e]
        test_df  = df_clean.loc[te_s:te_e]
        if len(train_df) < 1000 or len(test_df) < 100:
            continue

        X_tr, y_tr = train_df[feat_cols], train_df['label']
        X_te, y_te = test_df[feat_cols],  test_df['label']

        # Skip si une classe manque
        if len(y_tr.unique()) < 3:
            continue
        counts = y_tr.value_counts()
        if counts.min() < MIN_SAMPLES:
            continue

        # Poids de classe : équilibre short/long, downweight neutre
        n_tot   = len(y_tr)
        weights = y_tr.map({
            0: n_tot / (3 * counts.get(0, 1)),   # short
            1: n_tot / (3 * counts.get(1, 1)) * 0.3,  # neutre (downweight)
            2: n_tot / (3 * counts.get(2, 1)),   # long
        })

        model = xgb.XGBClassifier(
            n_estimators     = 300,
            max_depth        = 4,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            objective        = 'multi:softmax',
            num_class        = 3,
            eval_metric      = 'mlogloss',
            device           = 'cuda',
            verbosity        = 0,
            random_state     = 42,
        )
        model.fit(X_tr, y_tr, sample_weight=weights)

        proba    = model.predict_proba(X_te)   # shape (n, 3) : [P(short), P(neutre), P(long)]
        p_short  = proba[:, 0]
        p_long   = proba[:, 2]

        # ── Métriques LONG ────────────────────────────────────────────────────
        y_true_long = (y_te == 2).astype(int)
        mask_long   = (p_long > 0.5)
        n_long      = int(mask_long.sum())
        prec_long   = float(y_true_long.values[mask_long].mean()) if n_long > 0 else 0.0
        tp_long     = float(y_true_long.mean())
        lift_long   = prec_long - tp_long if n_long > 0 else 0.0

        # ── Métriques SHORT ───────────────────────────────────────────────────
        y_true_short = (y_te == 0).astype(int)
        mask_short   = (p_short > 0.5)
        n_short      = int(mask_short.sum())
        prec_short   = float(y_true_short.values[mask_short].mean()) if n_short > 0 else 0.0
        tp_short_r   = float(y_true_short.mean())
        lift_short   = prec_short - tp_short_r if n_short > 0 else 0.0

        ok_long  = '✅' if prec_long  > be and n_long  > 5 else '❌'
        ok_short = '✅' if prec_short > be and n_short > 5 else '❌'

        print(
            f"  Fold {i+1:2d} | {te_s.date()} → {te_e.date()} | "
            f"LONG  prec={prec_long:.3f} n={n_long:4d} lift={lift_long:+.3f} {ok_long} | "
            f"SHORT prec={prec_short:.3f} n={n_short:4d} lift={lift_short:+.3f} {ok_short}"
        )

        results.append({
            'fold': i + 1, 'test_start': te_s, 'test_end': te_e,
            'tp_rate': (tp_long + tp_short_r) / 2,
            'break_even': be,
            # Long
            'prec_long':  prec_long,  'n_long':  n_long,  'lift_long':  lift_long,
            'ok_long':    prec_long > be and n_long > 5,
            # Short
            'prec_short': prec_short, 'n_short': n_short, 'lift_short': lift_short,
            'ok_short':   prec_short > be and n_short > 5,
        })
        importances.append(dict(zip(feat_cols, model.feature_importances_)))

        # Sauvegarde prédictions OOS
        idx = X_te.index
        preds_long.append(pd.Series(p_long,  index=idx))
        preds_short.append(pd.Series(p_short, index=idx))
        labels_long.append(y_true_long)
        labels_short.append(y_true_short)

    if not results:
        print("Aucun fold valide."); return None

    df_r = pd.DataFrame(results)
    print(f"\n{'─'*65}")
    print(f"RR={rr} — Résumé 3 classes OOS")
    print(f"  LONG  : prec={df_r['prec_long'].mean():.3f}  BE={be:.3f}  "
          f"folds ok={df_r['ok_long'].sum()}/{len(df_r)}")
    print(f"  SHORT : prec={df_r['prec_short'].mean():.3f}  BE={be:.3f}  "
          f"folds ok={df_r['ok_short'].sum()}/{len(df_r)}")
    ok_both = (df_r['ok_long'] | df_r['ok_short']).sum()
    print(f"  BOTH  : folds avec ≥1 signal viable = {ok_both}/{len(df_r)}")
    print(f"{'─'*65}")

    imp = pd.DataFrame(importances).mean().sort_values(ascending=False)
    print(f"\nTop 10 features :\n{imp.head(10).to_string()}")

    preds_oos = pd.DataFrame({
        'p_long':  pd.concat(preds_long).sort_index(),
        'p_short': pd.concat(preds_short).sort_index(),
    })

    output = {
        'name':         f'atr_3class_rr{rr}',
        'model_type':   '3class',
        'rr':           rr,
        'atr_period':   ATR_PERIOD,
        'max_bars':     MAX_BARS,
        'break_even':   be,
        'config':       {'tp': f'{rr}×ATR', 'sl': '1×ATR',
                         'max_bars': MAX_BARS, 'side': 'long+short'},
        'results_df':   df_r,
        'importance':   imp,
        'preds_oos':    preds_oos,
        'labels_long':  pd.concat(labels_long).sort_index(),
        'labels_short': pd.concat(labels_short).sort_index(),
        'feature_cols': feat_cols,
    }

    out_path = os.path.join(OUTPUT_DIR, f'xgb_3class_rr{rr}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSauvegardé → {out_path}")
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rr', type=float, nargs='+', default=RR_LIST)
    args = parser.parse_args()

    print("Chargement données...")
    df = load_dataset()

    for rr in args.rr:
        train_3class(df, rr)
