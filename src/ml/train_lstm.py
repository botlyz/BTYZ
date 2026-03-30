"""
Phase 2 — LSTM walk-forward.

Même logique que XGBoost mais le modèle voit les SEQ_LEN dernières bougies
→ peut détecter des patterns temporels (divergence CVD qui se forme sur 1h).

Features : colonnes _z (normalisées rolling, sans lookahead)
Label    : Triple Barrier (même que XGBoost)

Lancement :
  .venv/bin/python src/ml/train_lstm.py
"""

import os
import sys
import pickle
import warnings
import logging
import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

# ── Logging automatique ───────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
_log_path = f"logs/lstm_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_path),
    ]
)
log = logging.INFO
def print(*a, **k):  # noqa
    logging.info(' '.join(str(x) for x in a))

from src.ml.data     import load_dataset
from src.ml.labels   import make_triple_barrier_label, label_stats
from src.ml.features import BARRIER_CONFIGS
from src.ml.train_xgb import make_wf_splits

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN      = 60      # bougies de contexte (1h d'historique pour décider)
BATCH_SIZE   = 1024
EPOCHS       = 60      # plus d'epochs pour bien converger
HIDDEN_SIZE  = 128
N_LAYERS     = 2
DROPOUT      = 0.3
LR           = 1e-3
PATIENCE     = 5       # était 3, trop agressif
MIN_SAMPLES  = 500     # min trades actifs par fold

TRAIN_MONTHS = 6
TEST_MONTHS  = 1

OUTPUT_DIR = 'cache/ml'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features : flow + PRIX (return 1m + close z-score)
# Sans le prix, le modèle essaie de prédire l'évolution du prix sans le voir
LSTM_FEATURES = [
    'return_1m',                                           # ← momentum prix direct
    'close_z',                                             # ← niveau prix normalisé
    'volume_z', 'taker_buy_volume_z', 'taker_sell_volume_z',
    'cvd_perp_z', 'cvd_spot_z', 'spot_taker_buy_z',
    'oi_z', 'oi_delta_z', 'ls_ratio_z', 'taker_ls_vol_z',
]


# ── Modèle LSTM ───────────────────────────────────────────────────────────────

def build_model(n_features, device):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch non installé → pip install torch")
        return None, None

    class LSTMClassifier(nn.Module):
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
            self.fc      = nn.Linear(HIDDEN_SIZE, 1)

        def forward(self, x):
            out, _ = self.lstm(x)          # (batch, seq, hidden)
            out    = out[:, -1, :]         # dernière timestep
            out    = self.dropout(out)
            return self.fc(out).squeeze(1) # logit (pas sigmoid — BCEWithLogits)

    model = LSTMClassifier().to(device)
    return model


# ── Construction des séquences ────────────────────────────────────────────────

def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Transforme (N, features) → (N-seq_len, seq_len, features) + labels alignés.
    Chaque séquence se termine à l'indice i → label[i].
    """
    n = len(X)
    X_seq = np.lib.stride_tricks.sliding_window_view(
        X, window_shape=(seq_len, X.shape[1])
    )[:, 0, :, :]   # (N-seq_len, seq_len, features)

    y_seq = y[seq_len - 1:]  # label aligné sur la dernière bougie de la séquence
    return X_seq.astype(np.float32), y_seq


# ── Entraînement LSTM ─────────────────────────────────────────────────────────

def train_one_config(df, tp, sl, max_bars, name, pbar_global=None, force=False):
    out_path = os.path.join(OUTPUT_DIR, f'lstm_{name}.pkl')

    # ── Skip si déjà calculé ──────────────────────────────────────────────────
    if os.path.exists(out_path) and not force:
        print(f"\n[SKIP] {name} → pkl existant ({out_path}), relance avec --force pour recalculer")
        with open(out_path, 'rb') as f:
            result = pickle.load(f)
        if pbar_global:
            n_folds = len(make_wf_splits(df.index, TRAIN_MONTHS, TEST_MONTHS))
            pbar_global.update(n_folds)
        return result

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("PyTorch non installé → pip install torch")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Config : {name}  (tp={tp*100:.1f}%, sl={sl*100:.1f}%, {max_bars}min)")
    print(f"Device : {device}")
    print(f"{'='*60}")

    # Colonnes features disponibles
    feat_cols = [c for c in LSTM_FEATURES if c in df.columns]
    if len(feat_cols) < 5:
        print(f"Trop peu de features disponibles : {feat_cols}")
        return None
    print(f"Features ({len(feat_cols)}) : {feat_cols}")

    # Label Triple Barrier
    print("Calcul Triple Barrier...")
    label = make_triple_barrier_label(df['close'], tp=tp, sl=sl, max_bars=max_bars)
    label_stats(label)

    df = df.copy()
    df['label'] = label
    df_clean = df[feat_cols + ['label']].dropna()
    print(f"Bougies après dropna : {len(df_clean):,}")

    splits = make_wf_splits(df_clean.index, TRAIN_MONTHS, TEST_MONTHS)
    print(f"Walk-forward : {len(splits)} folds ({TRAIN_MONTHS}m train / {TEST_MONTHS}m test)")

    results     = []
    importances = []   # moyenne gradient par feature (proxy importance)
    all_preds   = []
    all_labels  = []

    for i, split in enumerate(splits):
        tr_s, tr_e = split['train']
        te_s, te_e = split['test']

        train_df = df_clean.loc[tr_s:tr_e]
        test_df  = df_clean.loc[te_s:te_e]

        if len(train_df) < 1000 + SEQ_LEN or len(test_df) < 100 + SEQ_LEN:
            continue

        # ── Construction séquences ────────────────────────────────────────────
        X_tr_raw = train_df[feat_cols].values
        y_tr_raw = train_df['label'].values
        X_te_raw = test_df[feat_cols].values
        y_te_raw = test_df['label'].values

        X_tr, y_tr = build_sequences(X_tr_raw, y_tr_raw, SEQ_LEN)
        X_te, y_te = build_sequences(X_te_raw, y_te_raw, SEQ_LEN)

        # Filtre : trades actifs seulement (+1 / -1)
        mask_tr = y_tr != 0
        mask_te = y_te != 0

        if mask_tr.sum() < MIN_SAMPLES or mask_te.sum() < 50:
            print(f"  Fold {i+1:2d} → skip (trop peu de trades : {mask_tr.sum()})")
            if pbar_global:
                pbar_global.update(1)
            continue

        X_tr_a = X_tr[mask_tr]
        y_tr_a = (y_tr[mask_tr] == 1).astype(np.float32)   # 1=tp, 0=sl
        X_te_a = X_te[mask_te]
        y_te_a = (y_te[mask_te] == 1).astype(np.float32)

        # Poids classe pour déséquilibre
        pos_weight = (y_tr_a == 0).sum() / max((y_tr_a == 1).sum(), 1)

        # ── Modèle + entraînement ─────────────────────────────────────────────
        model     = build_model(len(feat_cols), device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )

        dataset = TensorDataset(
            torch.tensor(X_tr_a, device=device),
            torch.tensor(y_tr_a, device=device),
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        best_loss = float('inf')
        no_imp    = 0

        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)

            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                no_imp    = 0
            else:
                no_imp += 1
                if no_imp >= PATIENCE:
                    break

        # ── Évaluation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            logits_te = model(torch.tensor(X_te_a, device=device))
            proba_te  = torch.sigmoid(logits_te).cpu().numpy()

        preds    = (proba_te > 0.5).astype(int)
        tp_rate  = y_te_a.mean()

        pred_pos  = preds == 1
        n_signals = pred_pos.sum()

        if n_signals > 0 and tp_rate > 0:
            precision = (preds[pred_pos] == y_te_a[pred_pos]).mean()
            recall    = ((preds == 1) & (y_te_a == 1)).sum() / max((y_te_a == 1).sum(), 1)
        else:
            precision = 0.0
            recall    = 0.0

        lift = precision - tp_rate if n_signals > 0 else 0.0

        print(f"  Fold {i+1:2d} | {te_s.date()} → {te_e.date()} "
              f"| tp_rate={tp_rate:.3f} | signals={n_signals:4d} "
              f"| precision={precision:.3f} | lift={lift:+.3f}"
              f"| epochs={epoch+1}")
        if pbar_global:
            pbar_global.update(1)
            pbar_global.set_postfix({'config': name, 'fold': i+1, 'lift': f'{lift:+.3f}'})

        # Index OOS aligné sur les séquences
        oos_index = test_df.index[SEQ_LEN - 1:][mask_te]

        results.append({
            'fold':       i + 1,
            'test_start': te_s,
            'test_end':   te_e,
            'n_trades':   mask_te.sum(),
            'tp_rate':    tp_rate,
            'n_signals':  n_signals,
            'precision':  precision,
            'recall':     recall,
            'lift':       lift,
            'naive_acc':  1 - tp_rate,
            'epochs':     epoch + 1,
        })

        all_preds.append(pd.Series(proba_te.flatten(), index=oos_index))
        all_labels.append(pd.Series(y_te_a.flatten(), index=oos_index))

    if not results:
        print("Aucun fold valide.")
        return None

    # ── Résultats agrégés ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    print(f"\nPrécision moyenne OOS  : {results_df['precision'].mean():.3f}")
    print(f"Lift moyen (prec-base) : {results_df['lift'].mean():+.3f}  ← > 0 = vrai alpha")
    print(f"Folds avec lift > 0    : {(results_df['lift'] > 0).mean():.1%}")
    print(f"tp_rate moyen          : {results_df['tp_rate'].mean():.3f}")
    print(f"Epochs moyens          : {results_df['epochs'].mean():.1f}")

    preds_series  = pd.concat(all_preds).sort_index()
    labels_series = pd.concat(all_labels).sort_index()

    output = {
        'name':         name,
        'config':       {'tp': tp, 'sl': sl, 'max_bars': max_bars},
        'results_df':   results_df,
        'preds_oos':    preds_series,
        'labels_oos':   labels_series,
        'feature_cols': feat_cols,
        'seq_len':      SEQ_LEN,
    }

    out_path = os.path.join(OUTPUT_DIR, f'lstm_{name}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nRésultats sauvegardés → {out_path}")
    return output


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='Recalcule même si le pkl existe déjà')
    parser.add_argument('--config', type=str, default=None,
                        help='Lance seulement cette config (ex: swing_30min)')
    args = parser.parse_args()

    print(f"Log : {_log_path}")
    if args.force:
        print("Mode --force : tous les pkl seront recalculés")
    else:
        existing = [n for _, _, _, n in BARRIER_CONFIGS
                    if os.path.exists(os.path.join(OUTPUT_DIR, f'lstm_{n}.pkl'))]
        if existing:
            print(f"PKL existants (seront skippés) : {existing}")
            print("→ relance avec --force pour tout recalculer")

    print("Chargement données...")
    df = load_dataset()

    configs = BARRIER_CONFIGS
    if args.config:
        configs = [(tp, sl, mb, n) for tp, sl, mb, n in BARRIER_CONFIGS if n == args.config]
        if not configs:
            print(f"Config '{args.config}' inconnue. Choix : {[n for _,_,_,n in BARRIER_CONFIGS]}")
            sys.exit(1)

    # Barre de progression globale — total = nb configs × nb folds estimé
    _splits_sample = make_wf_splits(df.index, TRAIN_MONTHS, TEST_MONTHS)
    _total_folds   = len(configs) * len(_splits_sample)
    print(f"\nTotal estimé : {len(configs)} configs × ~{len(_splits_sample)} folds = {_total_folds} folds\n")

    all_results = {}
    with tqdm(total=_total_folds, desc='Global', unit='fold',
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
        for tp, sl, max_bars, name in configs:
            res = train_one_config(df, tp, sl, max_bars, name,
                                   pbar_global=pbar, force=args.force)
            if res:
                all_results[name] = res

    print(f"\n{'='*60}")
    print("COMPARAISON LSTM")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'Précision':>10} {'Lift moy':>10} {'Lift>0':>8} {'Folds':>6}")
    print("-" * 55)
    for name, res in all_results.items():
        df_r = res['results_df']
        print(f"{name:<15} {df_r['precision'].mean():>10.3f} "
              f"{df_r['lift'].mean():>+10.3f} "
              f"{(df_r['lift']>0).mean():>8.1%} "
              f"{len(df_r):>6}")

    if all_results:
        best = max(all_results.items(),
                   key=lambda x: x[1]['results_df']['lift'].mean())
        print(f"\n→ Meilleure config LSTM : {best[0]}")
        print("→ Lance maintenant src/ml/evaluate.py pour le backtest VBT.")
    print(f"\nLog complet : {_log_path}")
