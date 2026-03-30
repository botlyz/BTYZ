import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vectorbtpro import *
import numpy as np
import pandas as pd

from src.strategies.ram_dca import run_backtest

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
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

# Grille de params — 1 niveau, 100% allocation
grid_ma  = list(range(20, 320, 20))           # 15 valeurs
grid_env = list(np.arange(0.01, 0.21, 0.01))  # 20 valeurs  (1% → 20%)
grid_sl  = list(np.arange(0.01, 0.11, 0.01))  # 10 valeurs  (1% → 10%)
# Total : 15 × 20 × 10 = 3 000 combos

# ─────────────────────────────────────────────────────────────────────────────


def ram_objective(data, ma_window, env_pct, sl_pct):
    """Fonction objectif pour le grid search — retourne toutes les métriques VBT."""
    pf = run_backtest(data, ma_window, [env_pct], [1.0], sl_pct)
    return pf.stats()


def load_data(exchange, tf, pair):
    path = f'./data/raw/{exchange}/{tf}/{pair}.csv'
    if not os.path.exists(path):
        return None
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'], unit='ms')
    data = data.set_index('date').sort_index()
    data = data[~data.index.duplicated(keep='first')]
    return data


def run_opti(data, pair, tf, exchange, cache_dir='./cache'):
    import gc
    import multiprocessing
    n_cpus    = multiprocessing.cpu_count()
    n_workers = max(1, int(n_cpus * 0.8))
    print(f"Workers : {n_workers}/{n_cpus} (80%)")

    splitter = vbt.Splitter.from_n_rolling(
        data.index, n=10, length='optimize', split=0.7,
        optimize_anchor_set=1, set_labels=['train', 'test']
    )

    param_ram = vbt.parameterized(
        ram_objective,
        merge_func='concat',
        execute_kwargs=dict(
            chunk_len=n_workers,
            distribute='chunks',
            engine='pathos',
            show_progress=True,
        ),
    )
    cv_ram = vbt.split(
        param_ram,
        splitter=splitter,
        takeable_args=['data'],
        merge_func='concat',
        execute_kwargs=dict(show_progress=True),
    )

    PARAM_GRID = dict(
        ma_window=vbt.Param(grid_ma),
        env_pct=vbt.Param(grid_env),
        sl_pct=vbt.Param(grid_sl),
    )

    n_combos = len(grid_ma) * len(grid_env) * len(grid_sl)
    print(f"Combos : {n_combos}")
    print(f"Total backtests : {n_combos * splitter.n_splits * 2}")
    print("Lancement...")

    results = cv_ram(data, **PARAM_GRID)

    os.makedirs(cache_dir, exist_ok=True)
    out_path = f'{cache_dir}/ram_{pair}_{tf}_{exchange}.pickle'
    vbt.save(results, out_path)
    print(f"Sauvegardé dans {out_path}")

    del results, cv_ram, param_ram, splitter, PARAM_GRID
    try:
        from pathos.pools import _clear
        _clear()
    except Exception:
        pass
    gc.collect()
    print("Mémoire nettoyée")


if __name__ == '__main__':
    if MODE == 'single':
        print(f"=== RAM DCA — {pair} {tf} {exchange} ===")
        data = load_data(exchange, tf, pair)
        if data is None:
            print(f"Pas de data pour {pair} {tf} {exchange}")
            sys.exit(1)
        run_opti(data, pair, tf, exchange)

    elif MODE in ('multi', 'full'):
        combos = []
        if MODE == 'full':
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
                      for t in SCAN['timeframes'] for p in SCAN['pairs']]
            cache_dir = './cache'

        os.makedirs(cache_dir, exist_ok=True)

        to_run, skipped = [], 0
        for ex, t, p in combos:
            if os.path.exists(f'{cache_dir}/ram_{p}_{t}_{ex}.pickle'):
                skipped += 1
                continue
            if not os.path.exists(f'./data/raw/{ex}/{t}/{p}.csv'):
                skipped += 1
                continue
            to_run.append((ex, t, p))

        print(f"Scan {MODE} : {len(combos)} combos, {len(to_run)} à lancer, {skipped} skippées")

        import time
        t0 = time.time()
        for i, (ex, t, p) in enumerate(to_run):
            elapsed = time.time() - t0
            if i > 0:
                eta = elapsed / i * (len(to_run) - i)
                print(f"\n{'='*60}")
                print(f"  GLOBAL: {i}/{len(to_run)} | ETA {int(eta//60)}m{int(eta%60):02d}s")
            print(f"\n{'='*60}")
            print(f"  >>> {p} {t} {ex} [{i+1}/{len(to_run)}]")
            print(f"{'='*60}")
            data = load_data(ex, t, p)
            run_opti(data, p, t, ex, cache_dir=cache_dir)
            del data

        elapsed = time.time() - t0
        print(f"\nTerminé : {len(to_run)} opti en {int(elapsed//60)}m{int(elapsed%60):02d}s")
