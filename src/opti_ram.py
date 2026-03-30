import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pickle
import time
import multiprocessing
import optuna
from concurrent.futures import ProcessPoolExecutor, as_completed

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — modifier ici
# ─────────────────────────────────────────────────────────────────────────────
# mode 'single' = 1 seule paire | 'multi' = liste SCAN | 'full' = tout le dossier
MODE     = 'single'

pair     = 'HYPE'
tf       = '5m'
exchange = 'lighter'

SCAN = {
    'exchanges': ['lighter'],
    'timeframes': ['5m'],
    'pairs': ['HYPE', 'BTC', 'ETH', 'SOL'],
}

FULL = {
    'exchanges': ['lighter'],
    'timeframes': ['5m'],
}

# Walk-forward
N_WINDOWS   = 10
SPLIT_RATIO = 0.7

# TPE
N_TRIALS = 300

# Espace de recherche — 1 niveau, 100% allocation
# ma_window : 20 → 300  (step 10)
# env_pct   : 1% → 20%  (step 0.5%)
# sl_pct    : 1% → 10%  (step 0.5%)
SEARCH_SPACE = {
    'ma_window': (20,   300,  10),
    'env_pct':   (0.01, 0.20, 0.005),
    'sl_pct':    (0.01, 0.10, 0.005),
}

# ─────────────────────────────────────────────────────────────────────────────


def load_data(exchange, tf, pair):
    path = f'./data/raw/{exchange}/{tf}/{pair}.csv'
    if not os.path.exists(path):
        return None
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'], unit='ms')
    data = data.set_index('date').sort_index()
    data = data[~data.index.duplicated(keep='first')]
    return data


def _optimize_window(args):
    """
    Tourne dans un subprocess séparé (spawn) — chaque fenêtre est indépendante.
    Reimporte tout pour éviter les problèmes de pickling Numba.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import optuna, numpy as np
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from src.strategies.ram_dca import run_backtest

    i, train, test, n_trials, search_space = args

    MIN_TRADES  = 20
    MAX_DRAWDOWN = 0.30

    def objective(trial):
        ma_w    = trial.suggest_int(  'ma_window', *search_space['ma_window'])
        env_pct = trial.suggest_float('env_pct',   *search_space['env_pct'])
        sl      = trial.suggest_float('sl_pct',    *search_space['sl_pct'])

        try:
            pf = run_backtest(train, ma_w, [env_pct], [1.0], sl)
        except Exception:
            return -999.0

        n_trades = pf.trades.count()
        if n_trades < MIN_TRADES:
            return -999.0
        if pf.max_drawdown > MAX_DRAWDOWN:
            return -999.0

        sharpe = pf.sharpe_ratio
        ret    = pf.total_return
        dd     = pf.max_drawdown
        if np.isnan(sharpe) or np.isnan(ret):
            return -999.0

        return float(sharpe * 0.7 + ret * 0.3 * (1 - dd))

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42 + i),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    bv = study.best_value

    # Évaluation sur test avec les meilleurs params
    try:
        pf_test = run_backtest(test, bp['ma_window'], [bp['env_pct']], [1.0], bp['sl_pct'])
        test_sharpe = float(pf_test.sharpe_ratio)
        test_return = float(pf_test.total_return)
        test_max_dd = float(pf_test.max_drawdown)
        test_trades = int(pf_test.trades.count())
    except Exception:
        test_sharpe = test_return = test_max_dd = float('nan')
        test_trades = 0

    # Tous les trials (pour analyse de stabilité cross-fenêtres)
    trials_rows = []
    for t in study.trials:
        if t.value is None or not t.params:
            continue
        trials_rows.append({
            'window':    i + 1,
            'ma_window': t.params.get('ma_window'),
            'env_pct':   t.params.get('env_pct'),
            'sl_pct':    t.params.get('sl_pct'),
            'score':     t.value,
        })

    window_result = {
        'window':      i + 1,
        'train_start': str(train.index[0].date()),
        'train_end':   str(train.index[-1].date()),
        'test_start':  str(test.index[0].date()),
        'test_end':    str(test.index[-1].date()),
        'ma_window':   bp['ma_window'],
        'env_pct':     round(bp['env_pct'], 4),
        'sl_pct':      round(bp['sl_pct'],  4),
        'train_score': round(bv, 4),
        'test_sharpe': round(test_sharpe, 4),
        'test_return': round(test_return, 4),
        'test_max_dd': round(test_max_dd, 4),
        'test_trades': test_trades,
    }
    return window_result, trials_rows


def run_opti(data, pair, tf, exchange, cache_dir='./cache'):
    n_bars      = len(data)
    window_size = n_bars // N_WINDOWS
    train_size  = int(window_size * SPLIT_RATIO)
    test_size   = window_size - train_size

    windows = []
    for i in range(N_WINDOWS):
        s  = i * window_size
        te = s + train_size
        ts = te + test_size
        if ts > n_bars:
            break
        windows.append((i, data.iloc[s:te], data.iloc[te:ts]))

    n_cpus   = multiprocessing.cpu_count()
    n_workers = max(1, int(n_cpus * 0.8))

    print(f"  {pair} {tf} {exchange}")
    print(f"  {len(windows)} fenêtres × {N_TRIALS} trials = {len(windows)*N_TRIALS} backtests")
    print(f"  Workers : {n_workers}/{n_cpus}  (spawn)")

    args_list = [
        (i, tr, te, N_TRIALS, SEARCH_SPACE)
        for i, tr, te in windows
    ]

    results, all_trials = [], []
    t0 = time.time()

    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {executor.submit(_optimize_window, a): a[0] for a in args_list}
        done = 0
        for future in as_completed(futures):
            widx = futures[future]
            done += 1
            try:
                res, trials = future.result()
                results.append(res)
                all_trials.extend(trials)
                elapsed = time.time() - t0
                eta     = elapsed / done * (len(windows) - done)
                print(
                    f"  F{res['window']:02d} ✓  "
                    f"ma={res['ma_window']:3d}  "
                    f"env={res['env_pct']*100:.1f}%  "
                    f"sl={res['sl_pct']*100:.1f}%  "
                    f"test_sharpe={res['test_sharpe']:+.3f}  "
                    f"test_return={res['test_return']*100:+.1f}%  "
                    f"({done}/{len(windows)}, ETA {int(eta//60)}m{int(eta%60):02d}s)"
                )
            except Exception as e:
                print(f"  F{widx+1:02d} ✗  Erreur : {e}")

    results.sort(key=lambda x: x['window'])
    df_results = pd.DataFrame(results)
    df_trials  = pd.DataFrame(all_trials)

    elapsed = time.time() - t0
    print(f"  Terminé en {int(elapsed//60)}m{int(elapsed%60):02d}s")
    print(f"\n  Résultats test :")
    print(df_results[[
        'window', 'ma_window', 'env_pct', 'sl_pct',
        'train_score', 'test_sharpe', 'test_return', 'test_trades'
    ]].to_string(index=False))

    os.makedirs(cache_dir, exist_ok=True)
    out_path = f'{cache_dir}/ram_{pair}_{tf}_{exchange}.pickle'
    with open(out_path, 'wb') as f:
        pickle.dump({
            'results':  df_results,
            'trials':   df_trials,
            'pair':     pair,
            'tf':       tf,
            'exchange': exchange,
            'n_trials': N_TRIALS,
            'n_windows': N_WINDOWS,
            'split':    SPLIT_RATIO,
        }, f)
    print(f"  Sauvegardé → {out_path}")
    return out_path


if __name__ == '__main__':
    if MODE == 'single':
        print(f"=== RAM DCA Opti — {pair} {tf} {exchange} ===")
        data = load_data(exchange, tf, pair)
        if data is None:
            print(f"Pas de data pour {pair} {tf} {exchange}")
            sys.exit(1)
        run_opti(data, pair, tf, exchange)

    elif MODE in ('multi', 'full'):
        if MODE == 'full':
            combos = []
            for ex in FULL['exchanges']:
                for t in FULL['timeframes']:
                    d = f'./data/raw/{ex}/{t}'
                    if os.path.isdir(d):
                        for f in sorted(os.listdir(d)):
                            if f.endswith('.csv'):
                                combos.append((ex, t, f.replace('.csv', '')))
            cache_dir = f"./cache/ram_{'_'.join(FULL['exchanges'])}_{'_'.join(FULL['timeframes'])}"
        else:
            combos = [(ex, t, p) for ex in SCAN['exchanges']
                      for t in SCAN['timeframes']
                      for p in SCAN['pairs']]
            cache_dir = './cache'

        # Filtrer les déjà faits
        to_run = []
        for ex, t, p in combos:
            out = f'{cache_dir}/ram_{p}_{t}_{ex}.pickle'
            if os.path.exists(out):
                print(f"Skip {p} {t} {ex} (cache existant)")
                continue
            if not os.path.exists(f'./data/raw/{ex}/{t}/{p}.csv'):
                continue
            to_run.append((ex, t, p))

        print(f"=== RAM DCA Opti {MODE} : {len(to_run)} paires ===")
        os.makedirs(cache_dir, exist_ok=True)

        t_global = time.time()
        for i, (ex, t, p) in enumerate(to_run):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(to_run)}]")
            data = load_data(ex, t, p)
            if data is None:
                continue
            run_opti(data, p, t, ex, cache_dir=cache_dir)
            del data

        elapsed = time.time() - t_global
        print(f"\n{'='*60}")
        print(f"Tout terminé en {int(elapsed//60)}m{int(elapsed%60):02d}s")
        print(f"Résultats dans {cache_dir}/")
