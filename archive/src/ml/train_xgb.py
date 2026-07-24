"""
Phase 1 — XGBoost walk-forward.

Pour chaque config Triple Barrier :
  1. Walk-forward temporel (train 6 mois → test 1 mois, avance d'1 mois)
  2. XGBoost entraîné sur train, prédit sur test
  3. Feature importance + SHAP
  4. Résultats sauvegardés dans cache/ml/

Lancement :
  .venv/bin/python src/ml/train_xgb.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from src.ml.data   import load_dataset
from src.ml.labels import make_triple_barrier_label, label_stats
from src.ml.features import (
    BARRIER_CONFIGS, FEATURES_RAW,
    get_Xy, add_momentum_features,
)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_MONTHS = 6    # fenêtre d'entraînement
TEST_MONTHS  = 1    # fenêtre de test (OOS)
MIN_SAMPLES  = 500  # minimum de trades (label != 0) par fold pour entraîner

OUTPUT_DIR   = 'cache/ml'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Walk-forward splits ───────────────────────────────────────────────────────

def make_wf_splits(index: pd.DatetimeIndex, train_months=6, test_months=1):
    """
    Génère les splits walk-forward temporels.
    Train : `train_months` mois glissants.
    Test  : `test_months` mois suivants (jamais vus).
    """
    splits = []
    start  = index[0]
    end    = index[-1]

    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end  = train_end  + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        splits.append({
            'train': (train_start, train_end),
            'test':  (train_end,   test_end),
        })
        train_start += pd.DateOffset(months=test_months)  # avance d'1 mois

    return splits


# ── Entraînement XGBoost ──────────────────────────────────────────────────────

def train_one_config(df, tp, sl, max_bars, name):
    """
    Entraîne XGBoost en walk-forward sur une config Triple Barrier.
    Retourne un dict de résultats.
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost non installé → pip install xgboost")
        return None

    print(f"\n{'='*60}")
    print(f"Config : {name}  (tp={tp*100:.1f}%, sl={sl*100:.1f}%, {max_bars}min)")
    print(f"{'='*60}")

    # Calcul du label
    print("Calcul Triple Barrier...")
    label = make_triple_barrier_label(df['close'], tp=tp, sl=sl, max_bars=max_bars)
    label_stats(label)
    df = df.copy()
    df['label'] = label

    # Features momentum
    df = add_momentum_features(df)

    # Colonnes features disponibles (brutes + momentum)
    feat_cols = [c for c in df.columns if c in FEATURES_RAW or
                 any(c.startswith(p) for p in
                     ['cvd_perp_d', 'cvd_spot_d', 'price_d', 'div_perp_', 'taker_ratio'])]

    # Drop NaN
    df_clean = df[feat_cols + ['label']].dropna()
    print(f"Bougies après dropna : {len(df_clean):,}")

    # Splits walk-forward
    splits = make_wf_splits(df_clean.index, TRAIN_MONTHS, TEST_MONTHS)
    print(f"Walk-forward : {len(splits)} folds ({TRAIN_MONTHS}m train / {TEST_MONTHS}m test)")

    results = []
    importances = []
    all_preds  = []
    all_labels = []

    for i, split in enumerate(splits):
        tr_s, tr_e = split['train']
        te_s, te_e = split['test']

        train_df = df_clean.loc[tr_s:tr_e]
        test_df  = df_clean.loc[te_s:te_e]

        if len(train_df) < 1000 or len(test_df) < 100:
            continue

        X_train = train_df[feat_cols]
        y_train = train_df['label']
        X_test  = test_df[feat_cols]
        y_test  = test_df['label']

        # Filtre : garde uniquement les labels actifs (+1 / -1)
        # pour la classification binaire (tp vs sl)
        mask_tr = y_train != 0
        mask_te = y_test  != 0

        if mask_tr.sum() < MIN_SAMPLES or mask_te.sum() < 50:
            print(f"  Fold {i+1:2d} → skip (trop peu de trades : {mask_tr.sum()})")
            continue

        X_tr_active = X_train[mask_tr]
        y_tr_active = (y_train[mask_tr] == 1).astype(int)  # 1=tp, 0=sl
        X_te_active = X_test[mask_te]
        y_te_active = (y_test[mask_te] == 1).astype(int)

        # Scale positif/négatif
        scale = (y_tr_active == 0).sum() / (y_tr_active == 1).sum()

        model = xgb.XGBClassifier(
            n_estimators      = 300,
            max_depth         = 4,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            scale_pos_weight  = scale,
            eval_metric       = 'logloss',
            device            = 'cuda',    # GPU si dispo, fallback CPU auto
            verbosity         = 0,
            random_state      = 42,
        )

        model.fit(X_tr_active, y_tr_active)

        preds = model.predict(X_te_active)
        proba = model.predict_proba(X_te_active)[:, 1]

        tp_rate   = y_te_active.mean()     # % réels TP dans ce fold

        # Métriques sur les prédictions positives (quand modèle dit "TP")
        pred_pos  = preds == 1
        n_signals = pred_pos.sum()
        if n_signals > 0 and tp_rate > 0:
            precision = (preds[pred_pos] == y_te_active.values[pred_pos]).mean()
            recall    = ((preds == 1) & (y_te_active.values == 1)).sum() / max((y_te_active == 1).sum(), 1)
        else:
            precision = 0.0
            recall    = 0.0

        # Baseline naïve : si on prédit toujours "sl" → accuracy = 1 - tp_rate
        naive_acc = 1 - tp_rate
        # Lift : est-ce qu'on fait mieux que la baseline ?
        lift = precision - tp_rate if n_signals > 0 else 0.0

        print(f"  Fold {i+1:2d} | {te_s.date()} → {te_e.date()} "
              f"| tp_rate={tp_rate:.3f} | signals={n_signals:4d} "
              f"| precision={precision:.3f} | lift={lift:+.3f}")

        results.append({
            'fold':       i + 1,
            'test_start': te_s,
            'test_end':   te_e,
            'n_trades':   mask_te.sum(),
            'tp_rate':    tp_rate,
            'n_signals':  n_signals,
            'precision':  precision,
            'recall':     recall,
            'lift':       lift,        # precision - tp_rate (> 0 = vrai alpha)
            'naive_acc':  naive_acc,
        })

        importances.append(dict(zip(feat_cols, model.feature_importances_)))
        all_preds.append(pd.Series(proba, index=X_te_active.index))
        all_labels.append(y_te_active)

    if not results:
        print("Aucun fold valide.")
        return None

    # ── Résultats agrégés ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    print(f"\nPrécision moyenne OOS  : {results_df['precision'].mean():.3f}")
    print(f"Lift moyen (prec-base) : {results_df['lift'].mean():+.3f}  ← > 0 = vrai alpha")
    print(f"Folds avec lift > 0    : {(results_df['lift'] > 0).mean():.1%}")
    print(f"tp_rate moyen          : {results_df['tp_rate'].mean():.3f}")

    # Feature importance agrégée (moyenne des folds)
    imp_df = pd.DataFrame(importances).mean().sort_values(ascending=False)
    print(f"\nTop 10 features :")
    print(imp_df.head(10).to_string())

    # Prédictions OOS concaténées
    preds_series  = pd.concat(all_preds).sort_index()
    labels_series = pd.concat(all_labels).sort_index()

    output = {
        'name':          name,
        'config':        {'tp': tp, 'sl': sl, 'max_bars': max_bars},
        'results_df':    results_df,
        'importance':    imp_df,
        'preds_oos':     preds_series,
        'labels_oos':    labels_series,
        'feature_cols':  feat_cols,
    }

    # Sauvegarde
    out_path = os.path.join(OUTPUT_DIR, f'xgb_{name}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nRésultats sauvegardés → {out_path}")

    return output


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Chargement données...")
    df = load_dataset()

    all_results = {}
    for tp, sl, max_bars, name in BARRIER_CONFIGS:
        res = train_one_config(df, tp, sl, max_bars, name)
        if res:
            all_results[name] = res

    # Comparaison finale
    print(f"\n{'='*60}")
    print("COMPARAISON DES CONFIGS")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'Précision':>10} {'Lift moy':>10} {'Lift>0':>8} {'Folds':>6}")
    print("-" * 55)
    for name, res in all_results.items():
        df_r = res['results_df']
        print(f"{name:<15} {df_r['precision'].mean():>10.3f} "
              f"{df_r['lift'].mean():>+10.3f} "
              f"{(df_r['lift']>0).mean():>8.1%} "
              f"{len(df_r):>6}")

    best = max(all_results.items(),
               key=lambda x: x[1]['results_df']['lift'].mean())
    print(f"\n→ Meilleure config : {best[0]}")
    print("→ Lance maintenant src/ml/train_lstm.py avec cette config.")
