import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vectorbtpro import *
import numpy as np
import pandas as pd

from src.strategies.keltner import run_backtest


#charger les données OHLCV
data = pd.read_csv('./data/raw/HYPEUSDT.csv')
data['date'] = pd.to_datetime(data['date'], unit='ms')
data = data.set_index('date')


def kc_objective(data, ma_window, atr_window, atr_mult, sl_stop):
    """fonction objectif pour le grid search, retourne le sharpe brut
    le filtrage (min trades, max dd) se fait dans l'analyse pas ici
    sinon les combos valides en train sont NaN en test (fenetres plus courtes)"""
    pf = run_backtest(data, ma_window, atr_window, atr_mult, sl_stop)
    return pf.sharpe_ratio


if __name__ == '__main__':
    import multiprocessing
    n_cpus = multiprocessing.cpu_count()
    print(f"CPU disponibles : {n_cpus}")

    #walk-forward : 10 fenetres glissantes, 70% train / 30% test
    #optimize_anchor_set=1 les fenetres test se chevauchent pas
    splitter = vbt.Splitter.from_n_rolling(
        data.index, n=10, length='optimize', split=0.7,
        optimize_anchor_set=1, set_labels=['train', 'test']
    )
    print(f"Fenetres : {splitter.n_splits}")

    #pipeline VBT natif : parameterized (grid search) + split (walk-forward)
    #engine='pathos' pour le multiprocessing (marche en .py pas en notebook)
    param_kc = vbt.parameterized(
            kc_objective,
            merge_func='concat',
            execute_kwargs=dict(
                chunk_len=n_cpus, #1 combo par CPU par chunk
                distribute='chunks',
                engine='pathos',
                show_progress=True,
            ),
        )
    cv_kc = vbt.split(
        param_kc,
        splitter=splitter,
        takeable_args=['data'], #'data' sera decoupé par le splitter automatiquement
        merge_func='concat',
        execute_kwargs=dict(show_progress=True),
    )

    #grille de params a tester
    PARAM_GRID = dict(
        ma_window=vbt.Param(range(20, 300, 20)),        #15 valeurs de 20 a 280
        atr_window=vbt.Param(range(10, 80, 10)),         #7 valeurs de 10 a 70
        atr_mult=vbt.Param(np.arange(1.0, 8.0, 0.5)),   #14 valeurs de 1.0 a 7.5
        sl_stop=vbt.Param(np.arange(0.02, 0.14, 0.02)), #6 valeurs de 2% a 12%
    )

    n_combos = 15 * 7 * 14 * 6
    print(f"Combos : {n_combos}")
    print(f"Total backtests : {n_combos * splitter.n_splits * 2}")
    print("Lancement...")

    results = cv_kc(data, **PARAM_GRID)

    #sauvegarder les resultats dans le cache pour l'analyse
    os.makedirs('./cache', exist_ok=True)
    vbt.save(results, './cache/kc_wfsl_results.pickle')
    print(f"\nTerminé - {len(results)} resultats sauvegardés dans ./cache/kc_wfsl_results.pickle")
