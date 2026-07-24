"""
XGBoost — Régression R/R dynamique (2 outputs).

Le modèle prédit pour chaque bougie M1 :
  max_gain_90m  = meilleur move accessible dans les 90 prochaines minutes
  max_loss_90m  = pire move accessible dans les 90 prochaines minutes

Signal d'entrée : R/R prédit = max_gain / abs(max_loss) > seuil
TP dynamique    = max_gain prédit   (ex: +2.3%)
SL dynamique    = max_loss prédit   (ex: -0.8%)

Pas de seuil fixé à l'avance — chaque bougie a son propre TP/SL.

Lancement :
  .venv/bin/python src/ml/train_xgb_rr.py
  .venv/bin/python src/ml/train_xgb_rr.py --horizon 60
  .venv/bin/python src/ml/train_xgb_rr.py --horizon 120
"""

import os
import sys
import pickle
import logging
import datetime
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
_log_path = f"logs/xgb_rr_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_path),
    ]
)
def print(*a, **k):  # noqa
    logging.info(' '.join(str(x) for x in a))

from src.ml.data     import load_dataset
from src.ml.labels   import make_max_gain_loss_label, label_stats
from src.ml.features import FEATURES_RAW, add_momentum_features
from src.ml.train_xgb import make_wf_splits

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_MONTHS = 6
TEST_MONTHS  = 1
MIN_SAMPLES  = 1000   # min bougies par fold (régression, pas de filtre actif/neutre)
RR_THRESHOLDS = [1.2, 1.5, 2.0, 2.5, 3.0]   # seuils R/R testés en éval

OUTPUT_DIR = 'cache/ml'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Entraînement ──────────────────────────────────────────────────────────────

def train_rr(df: pd.DataFrame, horizon: int = 90, force: bool = False):
    """
    Entraîne deux XGBoost en walk-forward :
      - model_gain : prédit max_gain dans `horizon` bougies
      - model_loss : prédit max_loss dans `horizon` bougies

    Signal = R/R prédit = gain_pred / abs(loss_pred)
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost non installé → pip install xgboost")
        return None

    out_path = os.path.join(OUTPUT_DIR, f'xgb_rr_{horizon}m.pkl')
    if os.path.exists(out_path) and not force:
        print(f"\n[SKIP] xgb_rr_{horizon}m → pkl existant, relance avec --force")
        with open(out_path, 'rb') as f:
            return pickle.load(f)

    print(f"\n{'='*60}")
    print(f"XGBoost R/R — horizon {horizon} bougies ({horizon} minutes M1)")
    print(f"{'='*60}")

    # ── Labels ────────────────────────────────────────────────────────────────
    print(f"Calcul max_gain / max_loss sur {horizon} bougies...")
    labels_df = make_max_gain_loss_label(df['close'], horizon=horizon)

    print(f"  max_gain médian  : {labels_df['max_gain'].median():.3%}")
    print(f"  max_loss médian  : {labels_df['max_loss'].median():.3%}")
    print(f"  R/R médian (pot) : {labels_df['rr'].median():.2f}")
    print(f"  R/R > 1.5        : {(labels_df['rr'] > 1.5).mean():.1%} des bougies")
    print(f"  R/R > 2.0        : {(labels_df['rr'] > 2.0).mean():.1%} des bougies")

    # ── Features ──────────────────────────────────────────────────────────────
    df = add_momentum_features(df.copy())
    feat_cols = [c for c in df.columns if c in FEATURES_RAW or
                 any(c.startswith(p) for p in
                     ['cvd_perp_d', 'cvd_spot_d', 'price_d', 'div_perp_', 'taker_ratio'])]

    # Merge features + labels
    df_ml = df[feat_cols].join(labels_df[['max_gain', 'max_loss', 'rr']]).dropna()
    print(f"Bougies après dropna : {len(df_ml):,}")

    splits = make_wf_splits(df_ml.index, TRAIN_MONTHS, TEST_MONTHS)
    print(f"Walk-forward : {len(splits)} folds ({TRAIN_MONTHS}m train / {TEST_MONTHS}m test)\n")

    results    = []
    all_preds  = []   # (max_gain_pred, max_loss_pred, rr_pred, rr_actual, index)

    with tqdm(total=len(splits), desc=f'h={horizon}m', unit='fold') as pbar:
        for i, split in enumerate(splits):
            tr_s, tr_e = split['train']
            te_s, te_e = split['test']

            train_df = df_ml.loc[tr_s:tr_e]
            test_df  = df_ml.loc[te_s:te_e]

            if len(train_df) < MIN_SAMPLES or len(test_df) < 200:
                pbar.update(1)
                continue

            X_tr = train_df[feat_cols]
            X_te = test_df[feat_cols]

            # ── Model gain ────────────────────────────────────────────────────
            model_gain = xgb.XGBRegressor(
                n_estimators     = 300,
                max_depth        = 4,
                learning_rate    = 0.05,
                subsample        = 0.8,
                colsample_bytree = 0.8,
                device           = 'cuda',
                verbosity        = 0,
                random_state     = 42,
            )
            model_gain.fit(X_tr, train_df['max_gain'])
            gain_pred = model_gain.predict(X_te)

            # ── Model loss ────────────────────────────────────────────────────
            model_loss = xgb.XGBRegressor(
                n_estimators     = 300,
                max_depth        = 4,
                learning_rate    = 0.05,
                subsample        = 0.8,
                colsample_bytree = 0.8,
                device           = 'cuda',
                verbosity        = 0,
                random_state     = 42,
            )
            model_loss.fit(X_tr, train_df['max_loss'])
            loss_pred = model_loss.predict(X_te)

            # ── R/R prédit ────────────────────────────────────────────────────
            # gain_pred > 0, loss_pred < 0 (normalement)
            gain_pred_c = np.clip(gain_pred, 0, None)           # forcément positif
            loss_pred_c = np.clip(loss_pred, None, -1e-6)       # forcément négatif
            rr_pred     = gain_pred_c / np.abs(loss_pred_c)

            rr_actual   = test_df['rr'].values

            # ── Corrélations (qualité de prédiction) ─────────────────────────
            corr_gain = np.corrcoef(gain_pred, test_df['max_gain'].values)[0, 1]
            corr_loss = np.corrcoef(loss_pred, test_df['max_loss'].values)[0, 1]
            corr_rr   = np.corrcoef(rr_pred,   rr_actual)[0, 1]

            # ── Métriques par seuil R/R ───────────────────────────────────────
            baseline_rr = np.nanmedian(rr_actual)
            threshold_stats = {}
            for thr in RR_THRESHOLDS:
                mask = rr_pred > thr
                if mask.sum() > 10:
                    med_rr_signals = np.nanmedian(rr_actual[mask])
                    lift_rr        = med_rr_signals - baseline_rr
                else:
                    med_rr_signals = 0.0
                    lift_rr        = 0.0
                threshold_stats[f'n_sig_{thr}']    = int(mask.sum())
                threshold_stats[f'rr_med_{thr}']   = float(med_rr_signals)
                threshold_stats[f'lift_rr_{thr}']  = float(lift_rr)

            print(f"  Fold {i+1:2d} | {te_s.date()} → {te_e.date()} "
                  f"| corr_gain={corr_gain:+.3f} corr_loss={corr_loss:+.3f} corr_rr={corr_rr:+.3f}"
                  f"| baseline_rr={baseline_rr:.2f}"
                  f"| @RR>1.5 → n={threshold_stats['n_sig_1.5']:4d} lift={threshold_stats['lift_rr_1.5']:+.3f}")

            results.append({
                'fold':       i + 1,
                'test_start': te_s,
                'test_end':   te_e,
                'corr_gain':  corr_gain,
                'corr_loss':  corr_loss,
                'corr_rr':    corr_rr,
                'baseline_rr': baseline_rr,
                **threshold_stats,
            })

            fold_preds = pd.DataFrame({
                'gain_pred':  gain_pred,
                'loss_pred':  loss_pred,
                'rr_pred':    rr_pred,
                'gain_actual': test_df['max_gain'].values,
                'loss_actual': test_df['max_loss'].values,
                'rr_actual':  rr_actual,
            }, index=test_df.index)
            all_preds.append(fold_preds)

            pbar.update(1)
            pbar.set_postfix({'fold': i+1,
                              'corr_rr': f'{corr_rr:+.3f}',
                              'lift@1.5': f'{threshold_stats["lift_rr_1.5"]:+.3f}'})

    if not results:
        print("Aucun fold valide.")
        return None

    results_df = pd.DataFrame(results)
    preds_df   = pd.concat(all_preds).sort_index()

    print(f"\n{'─'*60}")
    print(f"RÉSULTATS AGRÉGÉS — horizon {horizon}m")
    print(f"{'─'*60}")
    print(f"Corrélation gain prédit/réel  : {results_df['corr_gain'].mean():+.3f}")
    print(f"Corrélation loss prédit/réel  : {results_df['corr_loss'].mean():+.3f}")
    print(f"Corrélation R/R prédit/réel   : {results_df['corr_rr'].mean():+.3f}  ← > 0 = modèle utile")
    print(f"R/R baseline (médian brut)    : {results_df['baseline_rr'].mean():.3f}")
    print()
    print(f"{'Seuil R/R':<12} {'Signaux/fold':>13} {'R/R médian':>12} {'Lift R/R':>10}")
    print("─" * 52)
    for thr in RR_THRESHOLDS:
        n   = results_df[f'n_sig_{thr}'].mean()
        med = results_df[f'rr_med_{thr}'].mean()
        lft = results_df[f'lift_rr_{thr}'].mean()
        print(f"  > {thr:<9} {n:>13.0f} {med:>12.3f} {lft:>+10.3f}")

    output = {
        'horizon':     horizon,
        'results_df':  results_df,
        'preds_oos':   preds_df,
        'feature_cols': feat_cols,
        'rr_thresholds': RR_THRESHOLDS,
    }

    with open(out_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nRésultats sauvegardés → {out_path}")
    print(f"Log complet            → {_log_path}")
    return output


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=90,
                        help='Horizon en bougies M1 (défaut: 90 = 1h30)')
    parser.add_argument('--force', action='store_true',
                        help='Recalcule même si pkl existant')
    args = parser.parse_args()

    print(f"Log : {_log_path}")
    print("Chargement données...")
    df = load_dataset()

    train_rr(df, horizon=args.horizon, force=args.force)
