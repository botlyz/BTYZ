"""
LSTM 3-classes — long / short / neutre avec labels ATR.

Le modèle voit les SEQ_LEN dernières bougies M1 et prédit :
  +1 → long  (monte rr×ATR avant de descendre 1×ATR)
  -1 → short (descend rr×ATR avant de monter 1×ATR)
   0 → neutre

Avantage vs XGBoost : voit la forme de la divergence CVD/prix
sur les 30 dernières minutes, pas juste les valeurs à l'instant t.

Lancement :
  .venv/bin/python src/ml/train_lstm_3class.py
  .venv/bin/python src/ml/train_lstm_3class.py --rr 3 --seq 30
"""

import os, sys, pickle, logging, argparse, warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from src.ml.data     import load_dataset
from src.ml.labels   import make_atr_3class_label, label_stats
from src.ml.features import FEATURES_RAW, add_momentum_features

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO, format='%(message)s',
    handlers=[
        logging.FileHandler(f'logs/lstm_3class_{_ts}.log'),
        logging.StreamHandler(sys.stdout),
    ],
)
def print(*a, **k): logging.info(' '.join(str(x) for x in a))

# ── Config ────────────────────────────────────────────────────────────────────
RR_LIST      = [3]
ATR_PERIOD   = 14
MAX_BARS     = 60
SEQ_LEN      = 30        # 30 bougies M1 = 30 minutes de contexte
BATCH_SIZE   = 2048
EPOCHS       = 30
HIDDEN_SIZE  = 128
N_LAYERS     = 2
DROPOUT      = 0.3
LR           = 1e-3
PATIENCE     = 5
TRAIN_MONTHS = 6
TEST_MONTHS  = 1
MIN_SAMPLES  = 200       # par classe dans le train
OUTPUT_DIR   = 'cache/ml'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapping label → classe (même que XGBoost 3class)
LABEL_MAP  = {-1: 0, 0: 1, 1: 2}
CLASS_NAME = {0: 'short', 1: 'neutre', 2: 'long'}

# Features normalisées pour LSTM (z-score rolling = sans lookahead)
LSTM_FEATURES = [
    'return_1m', 'close_z',
    'volume_z', 'taker_buy_volume_z', 'taker_sell_volume_z',
    'cvd_perp_z', 'cvd_spot_z', 'spot_taker_buy_z',
    'oi_z', 'oi_delta_z', 'ls_ratio_z', 'taker_ls_vol_z',
]


# ── Modèle LSTM ───────────────────────────────────────────────────────────────

def build_model(n_features, device):
    import torch
    import torch.nn as nn

    class LSTM3Class(nn.Module):
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
            self.fc      = nn.Linear(HIDDEN_SIZE, 3)   # 3 classes

        def forward(self, x):
            out, _ = self.lstm(x)        # (batch, seq, hidden)
            out    = out[:, -1, :]       # dernière timestep → résumé de la séquence
            out    = self.dropout(out)
            return self.fc(out)          # logits (CrossEntropyLoss gère softmax)

    return LSTM3Class().to(device)


# ── Construction des séquences ────────────────────────────────────────────────

def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    (N, features) → (N-seq_len, seq_len, features) + labels alignés.
    Chaque séquence [i:i+seq_len] → label[i+seq_len-1].
    """
    X_seq = np.lib.stride_tricks.sliding_window_view(
        X, window_shape=(seq_len, X.shape[1])
    )[:, 0, :, :]                  # (N-seq_len+1, seq_len, features)
    y_seq = y[seq_len - 1:]        # label de la dernière bougie de la séquence
    return X_seq.astype(np.float32), y_seq


# ── Walk-forward splits ───────────────────────────────────────────────────────

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


# ── Entraînement 3-class LSTM ─────────────────────────────────────────────────

def train_3class_lstm(df, rr, seq_len=SEQ_LEN):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("PyTorch non installé → pip install torch"); return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    be = 1.0 / (rr + 1.0)

    print(f"\n{'='*65}")
    print(f"LSTM 3-CLASS  |  RR={rr}  |  BE={be:.1%}  |  seq={seq_len}  |  device={device}")
    print(f"{'='*65}")

    # ── Label ─────────────────────────────────────────────────────────────────
    print(f"Calcul label 3-class ATR (rr={rr})...")
    label_raw = make_atr_3class_label(
        df['close'], df['high'], df['low'],
        rr=rr, atr_period=ATR_PERIOD, max_bars=MAX_BARS,
    )
    label_stats(label_raw)

    df2 = df.copy()
    df2['label'] = label_raw.map(LABEL_MAP)   # -1→0, 0→1, +1→2
    df2 = add_momentum_features(df2)

    feat_cols = [c for c in LSTM_FEATURES if c in df2.columns]
    print(f"Features LSTM ({len(feat_cols)}) : {feat_cols}")

    df_clean = df2[feat_cols + ['label']].dropna()
    print(f"Bougies après dropna : {len(df_clean):,}")

    splits = make_wf_splits(df_clean.index, TRAIN_MONTHS, TEST_MONTHS)
    print(f"Walk-forward : {len(splits)} folds | seq={seq_len} | BE={be:.1%}\n")

    results     = []
    preds_long  = []
    preds_short = []
    labels_long = []
    labels_short= []

    for i, split in enumerate(splits):
        tr_s, tr_e = split['train']
        te_s, te_e = split['test']

        train_df = df_clean.loc[tr_s:tr_e]
        test_df  = df_clean.loc[te_s:te_e]

        if len(train_df) < 1000 + seq_len or len(test_df) < 100 + seq_len:
            continue

        X_tr_raw = train_df[feat_cols].values.astype(np.float32)
        y_tr_raw = train_df['label'].values.astype(np.int64)
        X_te_raw = test_df[feat_cols].values.astype(np.float32)
        y_te_raw = test_df['label'].values.astype(np.int64)

        X_tr, y_tr = build_sequences(X_tr_raw, y_tr_raw, seq_len)
        X_te, y_te = build_sequences(X_te_raw, y_te_raw, seq_len)

        # Index test aligné sur la dernière bougie de chaque séquence
        te_idx = test_df.index[seq_len - 1:]

        counts = np.bincount(y_tr, minlength=3)
        if counts.min() < MIN_SAMPLES:
            continue

        # ── Poids de classe : downweight neutre ───────────────────────────────
        n_tot = len(y_tr)
        w = np.array([
            n_tot / (3 * max(counts[0], 1)),          # short
            n_tot / (3 * max(counts[1], 1)) * 0.3,   # neutre (downweight)
            n_tot / (3 * max(counts[2], 1)),          # long
        ], dtype=np.float32)
        sample_w = torch.tensor(w[y_tr], dtype=torch.float32).to(device)

        # ── DataLoader ────────────────────────────────────────────────────────
        X_tr_t = torch.tensor(X_tr).to(device)
        y_tr_t = torch.tensor(y_tr).to(device)
        X_te_t = torch.tensor(X_te).to(device)

        ds    = TensorDataset(X_tr_t, y_tr_t, sample_w)
        loader= DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

        # ── Modèle + optimiseur ───────────────────────────────────────────────
        model     = build_model(len(feat_cols), device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(reduction='none')   # poids manuels
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5, verbose=False
        )

        best_loss, patience_cnt, best_state = float('inf'), 0, None

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            for xb, yb, wb in loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss   = (criterion(logits, yb) * wb).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)

            if avg_loss < best_loss - 1e-4:
                best_loss     = avg_loss
                patience_cnt  = 0
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    break

        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # ── Prédictions OOS ───────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            logits_te = model(X_te_t)
            proba_te  = torch.softmax(logits_te, dim=1).cpu().numpy()

        p_short = proba_te[:, 0]
        p_long  = proba_te[:, 2]

        # ── Métriques ─────────────────────────────────────────────────────────
        y_true_long  = (y_te == 2).astype(int)
        y_true_short = (y_te == 0).astype(int)

        mask_long  = (p_long  > 0.5)
        mask_short = (p_short > 0.5)
        n_long     = int(mask_long.sum())
        n_short    = int(mask_short.sum())

        prec_long  = float(y_true_long[mask_long].mean())   if n_long  > 0 else 0.0
        prec_short = float(y_true_short[mask_short].mean()) if n_short > 0 else 0.0
        tp_long    = float(y_true_long.mean())
        tp_short_r = float(y_true_short.mean())
        lift_long  = prec_long  - tp_long    if n_long  > 0 else 0.0
        lift_short = prec_short - tp_short_r if n_short > 0 else 0.0

        ok_long  = '✅' if prec_long  > be and n_long  > 5 else '❌'
        ok_short = '✅' if prec_short > be and n_short > 5 else '❌'

        print(
            f"  Fold {i+1:2d} | {te_s.date()} → {te_e.date()} | "
            f"LONG  prec={prec_long:.3f} n={n_long:5d} lift={lift_long:+.3f} {ok_long} | "
            f"SHORT prec={prec_short:.3f} n={n_short:5d} lift={lift_short:+.3f} {ok_short} | "
            f"ep={epoch+1}"
        )

        results.append({
            'fold': i+1, 'test_start': te_s, 'test_end': te_e,
            'tp_rate': (tp_long + tp_short_r) / 2,
            'break_even': be,
            'prec_long':  prec_long,  'n_long':  n_long,  'lift_long':  lift_long,
            'ok_long':    prec_long  > be and n_long  > 5,
            'prec_short': prec_short, 'n_short': n_short, 'lift_short': lift_short,
            'ok_short':   prec_short > be and n_short > 5,
        })

        preds_long.append(pd.Series(p_long,  index=te_idx[:len(p_long)]))
        preds_short.append(pd.Series(p_short, index=te_idx[:len(p_short)]))
        labels_long.append(pd.Series(y_true_long,  index=te_idx[:len(y_true_long)]))
        labels_short.append(pd.Series(y_true_short, index=te_idx[:len(y_true_short)]))

        # Libérer la VRAM explicitement entre les folds
        del X_tr_t, y_tr_t, X_te_t, sample_w, ds, loader, model, best_state
        import torch as _torch; _torch.cuda.empty_cache()

    if not results:
        print("Aucun fold valide."); return None

    df_r = pd.DataFrame(results)
    print(f"\n{'─'*65}")
    print(f"LSTM 3-CLASS RR={rr} — Résumé OOS")
    print(f"  LONG  : prec={df_r['prec_long'].mean():.3f}  BE={be:.3f}  "
          f"folds ok={df_r['ok_long'].sum()}/{len(df_r)}")
    print(f"  SHORT : prec={df_r['prec_short'].mean():.3f}  BE={be:.3f}  "
          f"folds ok={df_r['ok_short'].sum()}/{len(df_r)}")
    print(f"  BOTH  : folds avec ≥1 signal viable = "
          f"{(df_r['ok_long'] | df_r['ok_short']).sum()}/{len(df_r)}")
    print(f"{'─'*65}")

    preds_oos = pd.DataFrame({
        'p_long':  pd.concat(preds_long).sort_index(),
        'p_short': pd.concat(preds_short).sort_index(),
    })

    output = {
        'name':         f'lstm_3class_rr{rr}',
        'model_type':   '3class',
        'rr':           rr,
        'atr_period':   ATR_PERIOD,
        'max_bars':     MAX_BARS,
        'seq_len':      seq_len,
        'break_even':   be,
        'config':       {'tp': f'{rr}×ATR', 'sl': '1×ATR',
                         'max_bars': MAX_BARS, 'side': 'long+short'},
        'results_df':   df_r,
        'preds_oos':    preds_oos,
        'labels_long':  pd.concat(labels_long).sort_index(),
        'labels_short': pd.concat(labels_short).sort_index(),
        'feature_cols': feat_cols,
    }

    out_path = os.path.join(OUTPUT_DIR, f'lstm_3class_rr{rr}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSauvegardé → {out_path}")
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rr',  type=float, nargs='+', default=RR_LIST)
    parser.add_argument('--seq', type=int,   default=SEQ_LEN)
    args = parser.parse_args()

    print("Chargement données...")
    df = load_dataset()

    for rr in args.rr:
        train_3class_lstm(df, rr, seq_len=args.seq)
