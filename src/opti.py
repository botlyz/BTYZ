import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vectorbtpro import *
import numpy as np
import pandas as pd

from src.strategies.keltner import run_backtest

#============================================================
# CONFIG - modifier ici pour changer ce qu'on opti
#============================================================
#mode 'single' = 1 seule paire
#mode 'multi' = scan les paires listées dans SCAN
#mode 'full' = scan TOUS les csv disponibles dans data/raw/<exchange>/<tf>/
MODE = 'full'

#config single (utilisé si MODE = 'single')
pair = 'BTC'
tf = '5m'
exchange = 'lighter'

#config multi (utilisé si MODE = 'multi')
SCAN = {
    'exchanges': ['lighter'],
    'timeframes': ['5m'],
    'pairs': ['BTC', 'ETH', 'SOL', 'DOGE', 'HYPE'],
}

#config full (utilisé si MODE = 'full')
#scanne tout les csv dans data/raw/<exchange>/<tf>/
FULL = {
    'exchanges': ['lighter'],
    'timeframes': ['5m'],
}

#grille de params (commune aux deux modes)
grid_ma = list(range(20, 300, 20))
grid_atr_w = list(range(10, 80, 10))
grid_atr_m = list(np.arange(1.0, 8.0, 0.5))
grid_sl = list(np.arange(0.02, 0.14, 0.02))
#============================================================


def kc_objective(data, ma_window, atr_window, atr_mult, sl_stop):
    """fonction objectif pour le grid search, retourne le sharpe brut
    le filtrage (min trades, max dd) se fait dans l'analyse pas ici
    sinon les combos valides en train sont NaN en test (fenetres plus courtes)"""
    pf = run_backtest(data, ma_window, atr_window, atr_mult, sl_stop)
    return pf.sharpe_ratio


def load_data(exchange, tf, pair):
    """charge les données csv d'une paire"""
    path = f'./data/raw/{exchange}/{tf}/{pair}.csv'
    if not os.path.exists(path):
        return None
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'], unit='ms')
    data = data.set_index('date')
    return data


def run_opti(data, pair, tf, exchange):
    """lance le walk-forward grid search sur une paire et sauvegarde le pickle"""
    import multiprocessing
    n_cpus = multiprocessing.cpu_count()

    splitter = vbt.Splitter.from_n_rolling(
        data.index, n=10, length='optimize', split=0.7,
        optimize_anchor_set=1, set_labels=['train', 'test']
    )

    param_kc = vbt.parameterized(
        kc_objective,
        merge_func='concat',
        execute_kwargs=dict(
            chunk_len=n_cpus,
            distribute='chunks',
            engine='pathos',
            show_progress=True,
        ),
    )
    cv_kc = vbt.split(
        param_kc,
        splitter=splitter,
        takeable_args=['data'],
        merge_func='concat',
        execute_kwargs=dict(show_progress=True),
    )

    PARAM_GRID = dict(
        ma_window=vbt.Param(grid_ma),
        atr_window=vbt.Param(grid_atr_w),
        atr_mult=vbt.Param(grid_atr_m),
        sl_stop=vbt.Param(grid_sl),
    )

    n_combos = len(grid_ma) * len(grid_atr_w) * len(grid_atr_m) * len(grid_sl)
    print(f"Combos : {n_combos}")
    print(f"Total backtests : {n_combos * splitter.n_splits * 2}")
    print("Lancement...")

    results = cv_kc(data, **PARAM_GRID)

    os.makedirs('./cache', exist_ok=True)
    out_path = f'./cache/kc_wfsl_{pair}_{tf}_{exchange}.pickle'
    vbt.save(results, out_path)
    print(f"Sauvegardé dans {out_path}")
    return results


if __name__ == '__main__':
    if MODE == 'single':
        #opti sur une seule paire
        print(f"=== {pair} {tf} {exchange} ===")
        data = load_data(exchange, tf, pair)
        if data is None:
            print(f"Pas de data pour {pair} {tf} {exchange}, lance d'abord download.py")
            sys.exit(1)
        run_opti(data, pair, tf, exchange)

    elif MODE in ('multi', 'full'):
        #construire la liste de combos selon le mode
        combos = []

        if MODE == 'full':
            #scanner tous les csv dans les dossiers exchange/tf
            for ex in FULL['exchanges']:
                for t in FULL['timeframes']:
                    data_dir = f'./data/raw/{ex}/{t}'
                    if not os.path.isdir(data_dir):
                        print(f"Dossier {data_dir} introuvable, skip")
                        continue
                    for f in sorted(os.listdir(data_dir)):
                        if f.endswith('.csv'):
                            p = f.replace('.csv', '')
                            combos.append((ex, t, p))
        else:
            #multi : liste explicite
            for ex in SCAN['exchanges']:
                for t in SCAN['timeframes']:
                    for p in SCAN['pairs']:
                        combos.append((ex, t, p))

        print(f"Scan {MODE} : {len(combos)} combinaisons a tester")
        done = 0
        skipped = 0

        for ex, t, p in combos:
            print(f"\n{'='*50}")
            print(f"=== {p} {t} {ex} ({done+1}/{len(combos)}) ===")

            #skip si deja fait
            cache_path = f'./cache/kc_wfsl_{p}_{t}_{ex}.pickle'
            if os.path.exists(cache_path):
                print(f"Deja dans le cache, skip")
                skipped += 1
                done += 1
                continue

            data = load_data(ex, t, p)
            if data is None:
                print(f"Pas de data, skip (lance download.py)")
                skipped += 1
                done += 1
                continue

            run_opti(data, p, t, ex)
            done += 1

        print(f"\n{'='*50}")
        print(f"Terminé : {done - skipped} opti lancées, {skipped} skippées")
