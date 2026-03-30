"""
LSTM — Régression R/R dynamique (2 outputs).

Le modèle voit les SEQ_LEN dernières bougies et prédit :
  max_gain  = meilleur move accessible dans les HORIZON prochaines bougies
  max_loss  = pire move accessible dans les HORIZON prochaines bougies

Avantage vs XGBoost R/R :
  XGBoost voit 1 bougie → prédit des niveaux de prix (très dur sans contexte)
  LSTM voit 90 bougies  → détecte si un squeeze/accumulation se forme,
                          et prédit le potentiel du move résultant

Signal : R/R prédit = max_gain / abs(max_loss) > seuil
TP dynamique = max_gain prédit
SL dynamique = max_loss prédit

Lancement :
  .venv/bin/python src/ml/train_lstm_rr.py
  .venv/bin/python src/ml/train_lstm_rr.py --horizon 60
  .venv/bin/python src/ml/train_lstm_rr.py --seq-len 60 --horizon 90
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
_log_path = f"logs/lstm_rr_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
from src.ml.labels   import make_max_gain_loss_label
from src.ml.features import add_momentum_features
from src.ml.train_xgb import make_wf_splits

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN      = 90      # bougies de contexte (même durée que l'horizon)
HORIZON      = 90      # bougies à prédire
BATCH_SIZE   = 512
EPOCHS       = 50
HIDDEN_SIZE  = 128
N_LAYERS     = 2
DROPOUT      = 0.3
LR           = 1e-3
PATIENCE     = 5
TRAIN_MONTHS = 6
TEST_MONTHS  = 1
MIN_SAMPLES  = 1000

RR_THRESHOLDS = [1.2, 1.5, 2.0, 2.5, 3.0]

OUTPUT_DIR = 'cache/ml'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features : flow normalisé + prix (le prix est critique pour prédire des niveaux)
LSTM_FEATURES = [
    'return_1m',
    'close_z',
    'volume_z', 'taker_buy_volume_z', 'taker_sell_volume_z',
    'cvd_perp_z', 'cvd_spot_z', 'spot_taker_buy_z',
    'oi_z', 'oi_delta_z', 'ls_ratio_z', 'taker_ls_vol_z',
]


# ── Modèle LSTM 2 outputs ─────────────────────────────────────────────────────

def build_model(n_features, device):
    import torch
    import torch.nn as nn

    class LSTMRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size  = n_features,
                hidden_size = HIDDEN_SIZE,
                num_layers  = N_LAYERS,
                batch_first = True,
                dropout     = DROPOUT if N_LAYERS > 1 else 0.0,
            )
            self.dropout = nn.Dropout(DROPOUT)
            self.fc_gain = nn.Linear(HIDDEN_SIZE, 1)   # prédit max_gain
            self.fc_loss = nn.Linear(HIDDEN_SIZE, 1)   # prédit max_loss

        def forward(self, x):
            out, _   = self.lstm(x)
            out      = self.dropout(out[:, -1, :])   # dernière timestep
            gain     = self.fc_gain(out).squeeze(1)
            loss     = self.fc_loss(out).squeeze(1)
            return gain, loss

    return LSTMRegressor().to(device)


# ── Construction séquences ────────────────────────────────────────────────────

def build_sequences(X, y_gain, y_loss, seq_len):
    X_seq    = np.lib.stride_tricks.sliding_window_view(
        X, window_shape=(seq_len, X.shape[1])
    )[:, 0, :, :].astype(np.float32)

    y_gain_s = y_gain[seq_len - 1:]
    y_loss_s = y_loss[seq_len - 1:]
    return X_seq, y_gain_s, y_loss_s


# ── Entraînement ──────────────────────────────────────────────────────────────

def train_lstm_rr(df, seq_len=SEQ_LEN, horizon=HORIZON, force=False):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("PyTorch non installé → pip install torch")
        return None

    out_path = os.path.join(OUTPUT_DIR, f'lstm_rr_{horizon}m.pkl')
    if os.path.exists(out_path) and not force:
        print(f"\n[SKIP] lstm_rr_{horizon}m → pkl existant, relance avec --force")
        with open(out_path, 'rb') as f:
            return pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"LSTM R/R — seq={seq_len} bougies de contexte, horizon={horizon}m")
    print(f"Device : {device}")
    print(f"{'='*60}")

    # ── Labels ────────────────────────────────────────────────────────────────
    print(f"Calcul max_gain / max_loss sur {horizon} bougies...")
    labels_df = make_max_gain_loss_label(df['close'], horizon=horizon)
    print(f"  max_gain médian  : {labels_df['max_gain'].median():.3%}")
    print(f"  max_loss médian  : {labels_df['max_loss'].median():.3%}")
    print(f"  R/R médian (pot) : {labels_df['rr'].median():.2f}")

    # ── Features ──────────────────────────────────────────────────────────────
    feat_cols = [c for c in LSTM_FEATURES if c in df.columns]
    print(f"Features ({len(feat_cols)}) : {feat_cols}")

    df_ml = df[feat_cols].join(labels_df[['max_gain', 'max_loss', 'rr']]).dropna()
    print(f"Bougies après dropna : {len(df_ml):,}")

    splits = make_wf_splits(df_ml.index, TRAIN_MONTHS, TEST_MONTHS)
    print(f"Walk-forward : {len(splits)} folds ({TRAIN_MONTHS}m train / {TEST_MONTHS}m test)\n")

    results   = []
    all_preds = []

    with tqdm(total=len(splits), desc=f'lstm_rr h={horizon}m', unit='fold') as pbar:
        for i, split in enumerate(splits):
            tr_s, tr_e = split['train']
            te_s, te_e = split['test']

            train_df = df_ml.loc[tr_s:tr_e]
            test_df  = df_ml.loc[te_s:te_e]

            if len(train_df) < MIN_SAMPLES + seq_len or len(test_df) < 200 + seq_len:
                pbar.update(1)
                continue

            # ── Séquences ─────────────────────────────────────────────────────
            X_tr, yg_tr, yl_tr = build_sequences(
                train_df[feat_cols].values,
                train_df['max_gain'].values,
                train_df['max_loss'].values,
                seq_len,
            )
            X_te, yg_te, yl_te = build_sequences(
                test_df[feat_cols].values,
                test_df['max_gain'].values,
                test_df['max_loss'].values,
                seq_len,
            )

            # Filtre NaN dans labels
            mask_tr = ~(np.isnan(yg_tr) | np.isnan(yl_tr))
            mask_te = ~(np.isnan(yg_te) | np.isnan(yl_te))
            if mask_tr.sum() < MIN_SAMPLES or mask_te.sum() < 100:
                pbar.update(1)
                continue

            X_tr  = X_tr[mask_tr].astype(np.float32)
            yg_tr = yg_tr[mask_tr].astype(np.float32)
            yl_tr = yl_tr[mask_tr].astype(np.float32)
            X_te  = X_te[mask_te].astype(np.float32)
            yg_te = yg_te[mask_te]
            yl_te = yl_te[mask_te]

            # ── Modèle ────────────────────────────────────────────────────────
            model     = build_model(len(feat_cols), device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            criterion = nn.MSELoss()

            dataset = TensorDataset(
                torch.tensor(X_tr, device=device),
                torch.tensor(yg_tr, device=device),
                torch.tensor(yl_tr, device=device),
            )
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            best_loss = float('inf')
            no_imp    = 0

            model.train()
            for epoch in range(EPOCHS):
                epoch_loss = 0.0
                for xb, yg_b, yl_b in loader:
                    optimizer.zero_grad()
                    gain_pred, loss_pred = model(xb)
                    loss = criterion(gain_pred, yg_b) + criterion(loss_pred, yl_b)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                epoch_loss /= len(loader)

                if epoch_loss < best_loss - 1e-6:
                    best_loss = epoch_loss
                    no_imp    = 0
                else:
                    no_imp += 1
                    if no_imp >= PATIENCE:
                        break

            # ── Évaluation (par chunks pour éviter OOM) ───────────────────────
            model.eval()
            gain_parts, loss_parts = [], []
            with torch.no_grad():
                for _start in range(0, len(X_te), BATCH_SIZE):
                    xb = torch.tensor(X_te[_start:_start + BATCH_SIZE], device=device)
                    gp, lp = model(xb)
                    gain_parts.append(gp.cpu().numpy())
                    loss_parts.append(lp.cpu().numpy())
            gain_p = np.concatenate(gain_parts)
            loss_p = np.concatenate(loss_parts)

            # R/R prédit (gain forcément >0, loss forcément <0)
            gain_p_c = np.clip(gain_p, 0, None)
            loss_p_c = np.clip(loss_p, None, -1e-6)
            rr_pred  = gain_p_c / np.abs(loss_p_c)

            # R/R réel pour les mêmes bougies
            oos_index = test_df.index[seq_len - 1:][mask_te]
            rr_actual = df_ml.loc[oos_index, 'rr'].values

            # Corrélations
            corr_gain = float(np.corrcoef(gain_p, yg_te)[0, 1]) if len(gain_p) > 2 else 0.0
            corr_loss = float(np.corrcoef(loss_p, yl_te)[0, 1]) if len(loss_p) > 2 else 0.0
            corr_rr   = float(np.corrcoef(rr_pred, rr_actual)[0, 1]) if len(rr_pred) > 2 else 0.0

            baseline_rr = float(np.nanmedian(rr_actual))
            thr_stats   = {}
            for thr in RR_THRESHOLDS:
                mask = rr_pred > thr
                if mask.sum() > 10:
                    med = float(np.nanmedian(rr_actual[mask]))
                    lft = med - baseline_rr
                else:
                    med, lft = 0.0, 0.0
                thr_stats[f'n_sig_{thr}']   = int(mask.sum())
                thr_stats[f'rr_med_{thr}']  = med
                thr_stats[f'lift_rr_{thr}'] = lft

            print(f"  Fold {i+1:2d} | {te_s.date()} → {te_e.date()} "
                  f"| corr_gain={corr_gain:+.3f} corr_loss={corr_loss:+.3f} corr_rr={corr_rr:+.3f}"
                  f"| baseline={baseline_rr:.2f}"
                  f"| @RR>1.5 n={thr_stats['n_sig_1.5']:5d} lift={thr_stats['lift_rr_1.5']:+.3f}"
                  f"| ep={epoch+1}")

            results.append({
                'fold': i + 1, 'test_start': te_s, 'test_end': te_e,
                'corr_gain': corr_gain, 'corr_loss': corr_loss, 'corr_rr': corr_rr,
                'baseline_rr': baseline_rr, 'epochs': epoch + 1,
                **thr_stats,
            })

            all_preds.append(pd.DataFrame({
                'gain_pred': gain_p, 'loss_pred': loss_p, 'rr_pred': rr_pred,
                'gain_actual': yg_te, 'loss_actual': yl_te, 'rr_actual': rr_actual,
            }, index=oos_index))

            pbar.update(1)
            pbar.set_postfix({'fold': i+1, 'corr_rr': f'{corr_rr:+.3f}',
                              'lift@1.5': f'{thr_stats["lift_rr_1.5"]:+.3f}'})

    if not results:
        print("Aucun fold valide.")
        return None

    results_df = pd.DataFrame(results)
    preds_df   = pd.concat(all_preds).sort_index()

    print(f"\n{'─'*60}")
    print(f"RÉSULTATS AGRÉGÉS — LSTM R/R horizon {horizon}m, seq={seq_len}")
    print(f"{'─'*60}")
    print(f"Corrélation gain prédit/réel  : {results_df['corr_gain'].mean():+.3f}")
    print(f"Corrélation loss prédit/réel  : {results_df['corr_loss'].mean():+.3f}")
    print(f"Corrélation R/R prédit/réel   : {results_df['corr_rr'].mean():+.3f}  ← > 0 = modèle utile")
    print(f"R/R baseline (médian brut)    : {results_df['baseline_rr'].mean():.3f}")
    print(f"Epochs moyens                 : {results_df['epochs'].mean():.1f}")
    print()
    print(f"{'Seuil R/R':<12} {'Signaux/fold':>13} {'R/R médian':>12} {'Lift R/R':>10}")
    print("─" * 52)
    for thr in RR_THRESHOLDS:
        n   = results_df[f'n_sig_{thr}'].mean()
        med = results_df[f'rr_med_{thr}'].mean()
        lft = results_df[f'lift_rr_{thr}'].mean()
        print(f"  > {thr:<9} {n:>13.0f} {med:>12.3f} {lft:>+10.3f}")

    # Comparaison avec XGB R/R si dispo
    xgb_path = os.path.join(OUTPUT_DIR, f'xgb_rr_{horizon}m.pkl')
    if os.path.exists(xgb_path):
        with open(xgb_path, 'rb') as f:
            xgb_res = pickle.load(f)
        xgb_df = xgb_res['results_df']
        print(f"\n{'─'*60}")
        print(f"COMPARAISON LSTM vs XGBoost — horizon {horizon}m")
        print(f"{'─'*60}")
        print(f"{'':20} {'LSTM':>10} {'XGBoost':>10}")
        print(f"{'corr_rr':20} {results_df['corr_rr'].mean():>+10.3f} {xgb_df['corr_rr'].mean():>+10.3f}")
        for thr in [1.5, 2.0, 3.0]:
            lstm_lift = results_df[f'lift_rr_{thr}'].mean()
            xgb_lift  = xgb_df[f'lift_rr_{thr}'].mean()
            print(f"{'lift@'+str(thr):20} {lstm_lift:>+10.3f} {xgb_lift:>+10.3f}")

    output = {
        'horizon': horizon, 'seq_len': seq_len,
        'results_df': results_df, 'preds_oos': preds_df,
        'feature_cols': feat_cols, 'rr_thresholds': RR_THRESHOLDS,
    }
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nRésultats sauvegardés → {out_path}")
    print(f"Log complet            → {_log_path}")
    return output


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon',  type=int, default=90,
                        help='Horizon en bougies M1 à prédire (défaut: 90)')
    parser.add_argument('--seq-len',  type=int, default=90,
                        help='Bougies de contexte LSTM (défaut: 90)')
    parser.add_argument('--force', action='store_true',
                        help='Recalcule même si pkl existant')
    args = parser.parse_args()

    print(f"Log : {_log_path}")
    print("Chargement données...")
    df = load_dataset()

    train_lstm_rr(df, seq_len=args.seq_len, horizon=args.horizon, force=args.force)
