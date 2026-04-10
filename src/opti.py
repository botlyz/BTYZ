import sys
import os
import hashlib
import requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vectorbtpro import *
import numpy as np
import pandas as pd
import questionary
from questionary import Style

# ─────────────────────────────────────────────────────────────────────────────
# Notifications Telegram
# ─────────────────────────────────────────────────────────────────────────────

TG_TOKEN   = '8045706367:AAF9MV280K9NitKUiQhcjwiR7uUUWWe02g8'
TG_CHAT_ID = '1069067907'

def tg_send(msg):
    try:
        requests.post(
            f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
            json={'chat_id': TG_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'},
            timeout=10,
        )
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Grilles de paramètres
# ─────────────────────────────────────────────────────────────────────────────

RAM_ALLOC = 0.1  # allocation par trade (0.1 = 10% du capital)

GRIDS = {
    'keltner': {
        'ma_window':  list(range(20, 220, 20)),
        'atr_window': list(range(10, 220, 20)),
        'atr_mult':   list(np.arange(1.0, 8.5, 0.5)),
        'sl_stop':    list(np.arange(0.02, 0.11, 0.02)),
    },
    'ram': {
        'ma_window': list(range(20, 220, 20)),
        'env_pct':   [round(x, 4) for x in np.arange(0.005, 0.085, 0.005)] + [round(x, 4) for x in np.arange(0.09, 0.125, 0.01)],
        'sl_pct':    [round(x, 4) for x in np.arange(0.005, 0.085, 0.005)] + [round(x, 4) for x in np.arange(0.09, 0.125, 0.01)],
    },
    'ram_vol': {
        'ma_window':  list(range(20, 220, 20)),
        'env_pct':    [round(x, 4) for x in np.arange(0.005, 0.085, 0.005)] + [round(x, 4) for x in np.arange(0.09, 0.125, 0.01)],
        'sl_pct':     [round(x, 4) for x in np.arange(0.005, 0.085, 0.005)] + [round(x, 4) for x in np.arange(0.09, 0.125, 0.01)],
        'vol_window': [0, 7, 30, 90],
    },
    'ram_rsi': {
        'ma_window':   list(range(20, 220, 20)),
        'env_pct':     [round(x, 4) for x in np.arange(0.015, 0.085, 0.005)] + [round(x, 4) for x in np.arange(0.09, 0.125, 0.01)],
        'sl_pct':      [round(x, 4) for x in np.arange(0.02, 0.085, 0.005)] + [round(x, 4) for x in np.arange(0.09, 0.125, 0.01)],
        'rsi_filter':  [0, 1, 2, 3],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Fonctions objectif
# ─────────────────────────────────────────────────────────────────────────────

def keltner_objective(data, ma_window, atr_window, atr_mult, sl_stop):
    try:
        import warnings
        warnings.filterwarnings('ignore')
        from src.strategies.keltner import run_backtest
        pf = run_backtest(data, ma_window, atr_window, atr_mult, sl_stop)
        return pf.stats(silence_warnings=True)
    except Exception:
        return None


def ram_objective(data, ma_window, env_pct, sl_pct):
    try:
        import warnings
        warnings.filterwarnings('ignore')
        from src.strategies.ram_dca import run_backtest
        pf = run_backtest(data, ma_window, [env_pct], [RAM_ALLOC], sl_pct)
        return pf.stats(silence_warnings=True)
    except Exception:
        return None


def ram_vol_objective(data, ma_window, env_pct, sl_pct, vol_window):
    try:
        import warnings
        warnings.filterwarnings('ignore')
        from src.strategies.ram_dca_vol import run_backtest
        pf = run_backtest(data, ma_window, [env_pct], [RAM_ALLOC], sl_pct, vol_window=int(vol_window))
        return pf.stats(silence_warnings=True)
    except Exception:
        return None


def ram_rsi_objective(data, ma_window, env_pct, sl_pct, rsi_filter):
    try:
        import warnings
        warnings.filterwarnings('ignore')
        from src.strategies.ram_dca_rsi import run_backtest
        pf = run_backtest(data, ma_window, [env_pct], [RAM_ALLOC], sl_pct, rsi_filter=int(rsi_filter))
        return pf.stats(silence_warnings=True)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def grid_dirname(strategy, grid):
    """
    Nom de dossier lisible : strat_params_fees.
    Ex: ram_ma20-300x15_env0.005-0.08x16_sl0.005-0.08x16_fees0bps_slip2bps
    """
    from config import FEES, SLIPPAGE

    _prefix_map = {'keltner': 'kc', 'ram': 'ram', 'ram_vol': 'ram_vol', 'ram_rsi': 'ram_rsi'}
    prefix = _prefix_map.get(strategy, 'ram')

    # Résumé compact des params
    param_parts = []
    for k, v in grid.items():
        vals = [round(float(x), 8) for x in sorted(v)]
        name = k.replace('_window', '').replace('_pct', '').replace('_stop', '').replace('_mult', '')
        lo = f'{vals[0]:.3f}'.rstrip('0').rstrip('.')
        hi = f'{vals[-1]:.3f}'.rstrip('0').rstrip('.')
        param_parts.append(f'{name}{lo}-{hi}x{len(vals)}')

    fees_bps = int(round(float(FEES) * 10000))
    slip_bps = int(round(float(SLIPPAGE) * 10000))
    alloc_pct = int(round(RAM_ALLOC * 100))

    return f'{prefix}_{"_".join(param_parts)}_alloc{alloc_pct}pct_fees{fees_bps}bps_slip{slip_bps}bps'


def grid_summary(grid):
    """Résumé lisible de la grille : param min→max (n valeurs)."""
    from config import FEES, SLIPPAGE
    parts = []
    for k, v in grid.items():
        parts.append(f"{k}: {v[0]}→{v[-1]} ({len(v)} valeurs)")
    fees_bps = int(round(float(FEES) * 10000))
    slip_bps = int(round(float(SLIPPAGE) * 10000))
    parts.append(f"fees: {fees_bps}bps")
    parts.append(f"slip: {slip_bps}bps")
    return '  |  '.join(parts)


def list_pairs(exchange, tf):
    d = f'./data/raw/{exchange}/{tf}'
    if not os.path.isdir(d):
        return []
    pairs = sorted(f.replace('USDT.csv', '').replace('.csv', '')
                   for f in os.listdir(d) if f.endswith('.csv'))
    return pairs


def load_data(exchange, tf, pair):
    # Lighter : HYPE.csv | Binance um : HYPEUSDT.csv
    for suffix in ('USDT.csv', '.csv'):
        path = f'./data/raw/{exchange}/{tf}/{pair}{suffix}'
        if os.path.exists(path):
            data = pd.read_csv(path)
            data['date'] = pd.to_datetime(data['date'], unit='ms')
            data = data.set_index('date').sort_index()
            data = data[~data.index.duplicated(keep='first')]
            return data
    return None


def _chunk_grid(grid, max_combos=1500):
    """Découpe une grille en sous-grilles de max_combos combos.
    Chunke récursivement si un seul paramètre ne suffit pas."""

    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    if n_combos <= max_combos:
        return [grid]

    # Trier les params par nombre de valeurs décroissant
    sorted_keys = sorted(grid.keys(), key=lambda k: len(grid[k]), reverse=True)

    # Chunker sur le plus gros param
    chunk_key = sorted_keys[0]
    chunk_vals = grid[chunk_key]
    other_combos = n_combos // len(chunk_vals)

    vals_per_chunk = max(1, max_combos // other_combos)
    chunks = []
    for i in range(0, len(chunk_vals), vals_per_chunk):
        sub_grid = dict(grid)
        sub_grid[chunk_key] = chunk_vals[i:i + vals_per_chunk]

        # Vérifier si le sous-chunk est encore trop gros → récursion
        sub_n = 1
        for v in sub_grid.values():
            sub_n *= len(v)
        if sub_n > max_combos:
            chunks.extend(_chunk_grid(sub_grid, max_combos))
        else:
            chunks.append(sub_grid)

    return chunks


def process_chunk(pair, chunk_idx, sub_grid, strategy, exchange, tf, cache_dir):
    """Traite UN chunk d'UNE paire dans un worker Pool."""
    import warnings, gc, shutil, ctypes
    warnings.filterwarnings('ignore')

    from vectorbtpro import vbt as _vbt

    _chunk_save_dir = os.path.join(cache_dir, '.grid_chunks', pair)
    os.makedirs(_chunk_save_dir, exist_ok=True)
    _chunk_pickle = os.path.join(_chunk_save_dir, f'chunk_{chunk_idx}.pickle')

    # Fichier de progression pour ce worker (lu par le thread monitor)
    _progress_file = os.path.join(cache_dir, '.progress', f'{os.getpid()}.count')
    os.makedirs(os.path.dirname(_progress_file), exist_ok=True)

    if os.path.exists(_chunk_pickle):
        return pair, chunk_idx, True, "skip"

    _data = load_data(exchange, tf, pair)
    if _data is None:
        return pair, chunk_idx, False, "Pas de data"

    try:
        _obj_map = {
            'keltner': keltner_objective,
            'ram': ram_objective,
            'ram_vol': ram_vol_objective,
            'ram_rsi': ram_rsi_objective,
        }
        objective_fn = _obj_map[strategy]

        sub_n = 1
        for v in sub_grid.values():
            sub_n *= len(v)

        splitter = _vbt.Splitter.from_n_rolling(
            _data.index, n=10, length='optimize', split=0.7,
            optimize_anchor_set=1, set_labels=['train', 'test']
        )

        # Wrapper l'objective pour compter chaque appel (= 1 backtest)
        # functools.wraps préserve la signature pour que VBT reconnaisse 'data'
        import functools
        _bt_count = [0]
        _orig_fn = objective_fn

        @functools.wraps(_orig_fn)
        def _counting_fn(*args, **kwargs):
            result = _orig_fn(*args, **kwargs)
            _bt_count[0] += 1
            if _bt_count[0] % 10 == 0:
                try:
                    with open(_progress_file, 'w') as f:
                        f.write(str(_bt_count[0]))
                except Exception:
                    pass
            return result

        param_fn = _vbt.parameterized(
            _counting_fn,
            merge_func='concat',
            execute_kwargs=dict(show_progress=False),
        )
        _split_chunk_dir = os.path.join(cache_dir, '.split_chunks', pair, f'chunk_{chunk_idx}')

        cv_fn = _vbt.split(
            param_fn,
            splitter=splitter,
            takeable_args=['data'],
            merge_func='concat',
            execute_kwargs=dict(
                show_progress=False,
                cache_chunks=True,
                chunk_cache_dir=_split_chunk_dir,
                release_chunk_cache=True,
                chunk_collect_garbage=True,
            ),
        )

        param_grid = {k: _vbt.Param(v) for k, v in sub_grid.items()}
        chunk_results = cv_fn(_data, **param_grid)

        _vbt.save(chunk_results, _chunk_pickle)

        # Cleanup : del + flush + gc + malloc_trim (rend la RAM à l'OS)
        del chunk_results, cv_fn, param_fn, param_grid, splitter, _data
        _vbt.flush()
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        shutil.rmtree(_split_chunk_dir, ignore_errors=True)

        # Nettoyer le fichier de progression
        try:
            os.remove(_progress_file)
        except Exception:
            pass

        return pair, chunk_idx, True, "OK"
    except Exception as _e:
        try:
            del _data
        except Exception:
            pass
        _vbt.flush()
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        try:
            os.remove(_progress_file)
        except Exception:
            pass
        return pair, chunk_idx, False, str(_e)


def _process_chunk_wrapper(args):
    """Wrapper pour imap_unordered (prend un seul tuple)."""
    return process_chunk(*args)


def assemble_pair(pair, strategy, cache_dir):
    """Assemble les chunks d'une paire en un seul pickle final."""
    import gc
    _pf_map = {'keltner': 'kc_wfsl', 'ram': 'ram', 'ram_vol': 'ram_vol', 'ram_rsi': 'ram_rsi'}
    prefix = _pf_map.get(strategy, 'ram')
    _chunk_save_dir = os.path.join(cache_dir, '.grid_chunks', pair)

    if not os.path.isdir(_chunk_save_dir):
        return

    _chunk_files = sorted(f for f in os.listdir(_chunk_save_dir) if f.endswith('.pickle'))
    if not _chunk_files:
        return

    if len(_chunk_files) == 1:
        results = vbt.load(os.path.join(_chunk_save_dir, _chunk_files[0]))
    else:
        _parts = []
        for cf in _chunk_files:
            _parts.append(vbt.load(os.path.join(_chunk_save_dir, cf)))
        results = pd.concat(_parts)
        del _parts
        gc.collect()

    out_path = f'{cache_dir}/{prefix}_{pair}_5m_{os.path.basename(os.path.dirname(cache_dir)).split("_")[1] if "full_" in cache_dir else "lighter"}.pickle'
    # Déduire exchange/tf depuis le cache_dir
    _parent = os.path.basename(os.path.dirname(cache_dir))
    _parts_dir = _parent.replace('full_', '').rsplit('_', 1)
    _exchange = _parts_dir[0] if len(_parts_dir) >= 2 else 'lighter'
    _tf = _parts_dir[1] if len(_parts_dir) >= 2 else '5m'
    out_path = f'{cache_dir}/{prefix}_{pair}_{_tf}_{_exchange}.pickle'

    vbt.save(results, out_path)
    print(f"  Assemblé → {out_path}")

    del results
    gc.collect()

    import shutil
    shutil.rmtree(_chunk_save_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Menu interactif
# ─────────────────────────────────────────────────────────────────────────────

STYLE = Style([
    ('qmark',     'fg:#00b0ff bold'),
    ('question',  'bold'),
    ('answer',    'fg:#00e676 bold'),
    ('pointer',   'fg:#00b0ff bold'),
    ('selected',  'fg:#00e676'),
    ('separator', 'fg:#444444'),
    ('instruction', 'fg:#888888'),
])


def menu():
    print("\n╔══════════════════════════════╗")
    print("║      BTYZ — Optimisation     ║")
    print("╚══════════════════════════════╝\n")

    # 1. Stratégie
    strategy = questionary.select(
        "Stratégie :",
        choices=[
            questionary.Choice("Keltner Channel  (EMA + ATR)",              value='keltner'),
            questionary.Choice("RAM DCA          (SMA + enveloppe + SL)",   value='ram'),
            questionary.Choice("RAM DCA + Vol    (+ filtre ATR médiane)",   value='ram_vol'),
            questionary.Choice("RAM DCA + RSI    (+ filtre RSI neutre)",    value='ram_rsi'),
        ],
        style=STYLE,
    ).ask()
    if strategy is None:
        sys.exit(0)

    # 2. Mode
    mode = questionary.select(
        "Mode :",
        choices=[
            questionary.Choice("single — une seule paire",              value='single'),
            questionary.Choice("multi  — liste de paires",              value='multi'),
            questionary.Choice("liquid — paires par tier de liquidité", value='liquid'),
            questionary.Choice("full   — toutes les paires",            value='full'),
        ],
        style=STYLE,
    ).ask()
    if mode is None:
        sys.exit(0)

    # 3. Exchange
    exchanges_avail = sorted(
        e for e in os.listdir('./data/raw')
        if os.path.isdir(f'./data/raw/{e}')
    )
    exchange = questionary.select(
        "Exchange :",
        choices=exchanges_avail,
        style=STYLE,
    ).ask()
    if exchange is None:
        sys.exit(0)

    # 4. Timeframe
    tfs_avail = sorted(
        t for t in os.listdir(f'./data/raw/{exchange}')
        if os.path.isdir(f'./data/raw/{exchange}/{t}')
        and not t.startswith('.')
    )
    tf = questionary.select(
        "Timeframe :",
        choices=tfs_avail,
        style=STYLE,
    ).ask()
    if tf is None:
        sys.exit(0)

    # 5. Paire(s)
    pairs_avail = list_pairs(exchange, tf)
    pairs = []

    if mode == 'single':
        pair = questionary.autocomplete(
            "Paire :",
            choices=pairs_avail,
            style=STYLE,
            validate=lambda x: x in pairs_avail or f"Paire inconnue : {x}",
        ).ask()
        if pair is None:
            sys.exit(0)
        pairs = [pair]

    elif mode == 'multi':
        pairs = questionary.checkbox(
            "Paires (espace pour sélectionner) :",
            choices=pairs_avail,
            style=STYLE,
            validate=lambda x: True if len(x) > 0 else "Sélectionne au moins une paire",
        ).ask()
        if not pairs:
            sys.exit(0)

    elif mode == 'liquid':
        import json
        liq_path = './liquidity.json'
        if not os.path.exists(liq_path):
            print(f"Erreur : {liq_path} introuvable. Lance d'abord : python src/liquidity.py")
            sys.exit(1)
        with open(liq_path) as f:
            liq = json.load(f)

        tiers = questionary.checkbox(
            "Tiers de liquidité (espace pour sélectionner) :",
            choices=[
                questionary.Choice(f"★★ Très liquide  ({len(liq['tres_liquide'])} paires)", value='tres_liquide'),
                questionary.Choice(f"★  Liquide       ({len(liq['liquide'])} paires)",      value='liquide'),
                questionary.Choice(f"~  Moyen         ({len(liq['moyen'])} paires)",         value='moyen'),
            ],
            style=STYLE,
            validate=lambda x: True if len(x) > 0 else "Sélectionne au moins un tier",
        ).ask()
        if not tiers:
            sys.exit(0)

        liquid_set = set()
        for tier in tiers:
            liquid_set.update(liq[tier])
        pairs = [p for p in pairs_avail if p in liquid_set]
        if not pairs:
            print("Aucune paire disponible pour les tiers sélectionnés dans ce timeframe.")
            sys.exit(0)
        print(f"  → {len(pairs)} paires sélectionnées : {', '.join(pairs[:10])}{'...' if len(pairs) > 10 else ''}")

    else:  # full
        pairs = pairs_avail

    # 6. Récap + confirmation
    from config import FEES, SLIPPAGE, INIT_CASH
    grid = GRIDS[strategy]
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    fees_bps = int(round(float(FEES) * 10000))
    slip_bps = int(round(float(SLIPPAGE) * 10000))

    print(f"\n╔═══════════════════════════════════════════════╗")
    print(f"║             RÉCAPITULATIF                     ║")
    print(f"╠═══════════════════════════════════════════════╣")
    print(f"║  Stratégie  : {strategy:<32}║")
    print(f"║  Exchange   : {exchange:<32}║")
    print(f"║  Timeframe  : {tf:<32}║")
    print(f"║  Paires     : {len(pairs):<32}║")
    print(f"╠═══════════════════════════════════════════════╣")
    for k, v in grid.items():
        lo = f'{v[0]}'
        hi = f'{v[-1]}'
        label = f'{k}: {lo} → {hi} ({len(v)} vals)'
        print(f"║  {label:<44}║")
    print(f"║  Combos     : {n_combos:<32}║")
    print(f"║  Backtests  : {n_combos * 10 * 2:<32}║")
    print(f"╠═══════════════════════════════════════════════╣")
    print(f"║  Fees       : {fees_bps} bps ({FEES*100:.2f}% par trade){' '*(16-len(str(fees_bps)))}║")
    print(f"║  Slippage   : {slip_bps} bps ({SLIPPAGE*100:.3f}% par trade){' '*(15-len(str(slip_bps)))}║")
    print(f"║  Allocation : {int(RAM_ALLOC*100)}% du capital par trade{' '*(22-len(str(int(RAM_ALLOC*100))))}║")
    print(f"║  Capital    : {INIT_CASH:,}${' '*(31-len(f'{INIT_CASH:,}'))}║")
    print(f"╚═══════════════════════════════════════════════╝")

    ok = questionary.confirm("Lancer ?", default=True, style=STYLE).ask()
    if not ok:
        print("Annulé.")
        sys.exit(0)

    return strategy, mode, exchange, tf, pairs, grid


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time
    import subprocess
    import argparse

    # Nettoyage des workers zombies des runs précédents
    _zombie = subprocess.run(
        ['pgrep', '-f', 'python3 src/opti.py'],
        capture_output=True, text=True
    )
    _pids = [p for p in _zombie.stdout.strip().split('\n') if p and int(p) != os.getpid()]
    if _pids:
        print(f"Nettoyage de {len(_pids)} workers zombies des runs précédents...")
        subprocess.run(['kill', '-9'] + _pids, capture_output=True)
        import time as _t; _t.sleep(2)
        print("RAM libérée.")

    # Mode non-interactif : python3 src/opti.py --resume <cache_dir>
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, help='Reprendre directement sur un cache_dir existant (skip menu)')
    args, _ = parser.parse_known_args()

    if args.resume:
        # Déduire les params depuis le nom du dossier
        cache_dir = args.resume
        _base = os.path.basename(cache_dir)
        if _base.startswith('kc_'):
            prefix = 'kc_wfsl'
            strategy = 'keltner'
        elif _base.startswith('ram_rsi'):
            prefix = 'ram_rsi'
            strategy = 'ram_rsi'
        elif _base.startswith('ram_vol'):
            prefix = 'ram_vol'
            strategy = 'ram_vol'
        else:
            prefix = 'ram'
            strategy = 'ram'
        grid = GRIDS[strategy]

        # Déduire exchange/tf depuis le parent
        _parent = os.path.basename(os.path.dirname(cache_dir))  # full_lighter_5m
        _parts = _parent.replace('full_', '').rsplit('_', 1)
        exchange = _parts[0] if len(_parts) >= 2 else 'lighter'
        tf = _parts[1] if len(_parts) >= 2 else '5m'

        # Charger les paires depuis liquidity.json ou data
        pairs = list_pairs(exchange, tf)
        mode = 'full'

        print(f"\n[RESUME] {cache_dir}")
        print(f"Stratégie: {strategy} | Exchange: {exchange} | TF: {tf} | Paires: {len(pairs)}")
    else:
        strategy, mode, exchange, tf, pairs, grid = menu()

    dirname = grid_dirname(strategy, grid)
    _pf_map2 = {'keltner': 'kc_wfsl', 'ram': 'ram', 'ram_vol': 'ram_vol', 'ram_rsi': 'ram_rsi'}
    prefix = _pf_map2.get(strategy, 'ram')

    if mode in ('full', 'liquid'):
        cache_dir = f'./cache/full_{exchange}_{tf}/{dirname}'
    else:
        cache_dir = f'./cache/{dirname}'

    print(f"\nGrille : {grid_summary(grid)}")
    print(f"Cache → {cache_dir}/\n")
    os.makedirs(cache_dir, exist_ok=True)

    # Filtrer les paires déjà dans le cache (pickle final existe)
    to_run  = []
    skipped = 0
    for p in pairs:
        out = f'{cache_dir}/{prefix}_{p}_{tf}_{exchange}.pickle'
        if os.path.exists(out):
            print(f"Skip {p} (cache existant)")
            skipped += 1
            continue
        to_run.append(p)

    if not to_run:
        print("Rien à faire — tout est déjà dans le cache.")
        sys.exit(0)

    # Chunker la grille
    grid_chunks = _chunk_grid(grid, max_combos=200)
    n_chunks = len(grid_chunks)
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    # Construire la liste de tâches : (pair, chunk_idx, sub_grid)
    tasks = []
    for p in to_run:
        for chunk_idx, sub_grid in enumerate(grid_chunks):
            tasks.append((p, chunk_idx, sub_grid))

    print(f"\n{len(to_run)} paires × {n_chunks} chunks = {len(tasks)} tâches")
    print(f"Combos/chunk : ~{n_combos // n_chunks} | Total combos/paire : {n_combos}")

    import multiprocessing
    import multiprocessing as _mp
    import psutil
    import signal

    n_cpus = multiprocessing.cpu_count()
    n_parallel = min(n_cpus, len(tasks))
    print(f"Parallélisme : {n_parallel} workers (Pool + malloc_trim)\n")

    t0 = time.time()
    done_chunks = 0
    failed = []

    _task_args = [
        (p, chunk_idx, sub_grid, strategy, exchange, tf, cache_dir)
        for p, chunk_idx, sub_grid in tasks
    ]

    # Log file
    _log_path = os.path.join(cache_dir, 'opti.log')
    def _main_log(msg):
        _ram = psutil.virtual_memory()
        _ts = time.strftime('%H:%M:%S')
        _line = f"[{_ts}|RAM {_ram.used/1024**3:.1f}/{_ram.total/1024**3:.0f}G ({_ram.percent:.0f}%)] {msg}"
        print(_line)
        with open(_log_path, 'a') as _f:
            _f.write(_line + '\n')

    from tqdm import tqdm

    _bt_per_chunk = (n_combos // n_chunks) * 20
    _total_bt = n_combos * 20 * len(to_run)

    import threading, glob as _glob

    _main_log(f"Démarrage : {len(tasks)} tasks, {n_parallel} workers, {_total_bt:,} backtests total")

    # Dossier de progression
    _progress_dir = os.path.join(cache_dir, '.progress')
    os.makedirs(_progress_dir, exist_ok=True)
    # Nettoyer les anciens fichiers de progression
    for f in _glob.glob(os.path.join(_progress_dir, '*.count')):
        os.remove(f)

    pool = _mp.Pool(processes=n_parallel)

    def _sigint_handler(sig, frame):
        print("\n\nCtrl+C reçu — arrêt...")
        pool.terminate()
        pool.join()
        sys.exit(1)
    signal.signal(signal.SIGINT, _sigint_handler)

    pbar = tqdm(total=_total_bt, desc="Backtests", unit='bt', unit_scale=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

    # Thread monitor : lit les fichiers de progression des workers toutes les 0.5s
    _stop_monitor = threading.Event()
    _monitor_state = {'last_total': 0, 'completed_bt': 0}

    def _monitor_progress():
        while not _stop_monitor.is_set():
            try:
                worker_files = _glob.glob(os.path.join(_progress_dir, '*.count'))
                _live_count = 0
                for wf in worker_files:
                    try:
                        with open(wf, 'r') as f:
                            _live_count += int(f.read().strip() or '0')
                    except Exception:
                        pass
                _new_total = _monitor_state['completed_bt'] + _live_count
                delta = _new_total - _monitor_state['last_total']
                if delta > 0:
                    pbar.update(delta)
                    _monitor_state['last_total'] = _new_total
                ram_pct = psutil.virtual_memory().percent
                pbar.set_postfix_str(f"RAM {ram_pct:.0f}% | {done_chunks}/{len(tasks)} chunks")
            except Exception:
                pass
            _stop_monitor.wait(0.5)

    _monitor_thread = threading.Thread(target=_monitor_progress, daemon=True)
    _monitor_thread.start()

    try:
        for result in pool.imap_unordered(_process_chunk_wrapper, _task_args):
            pair, cidx, success, msg = result
            done_chunks += 1

            # Ajouter les backtests de ce chunk au total complété
            sub_n = 1
            for _ta in _task_args:
                if _ta[0] == pair and _ta[1] == cidx:
                    for v in _ta[2].values():
                        sub_n *= len(v)
                    break
            _monitor_state['completed_bt'] += sub_n * 20

            ram_pct = psutil.virtual_memory().percent
            status = "✓" if success else f"✗ {msg}"
            _main_log(f"{status} {pair} chunk {cidx+1}/{n_chunks} [{done_chunks}/{len(tasks)}] RAM {ram_pct:.0f}%")

            if not success:
                failed.append(f"{pair}/chunk{cidx}")

            if ram_pct > 90:
                _main_log(f"🚨 RAM CRITIQUE {ram_pct:.0f}% — ARRÊT")
                tg_send(f"🚨 Opti arrêtée — RAM {ram_pct:.0f}%")
                pool.terminate()
                break

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        print("\nArrêt en cours...")
        pool.terminate()
        pool.join()
        sys.exit(1)
    finally:
        _stop_monitor.set()
        _monitor_thread.join(timeout=2)
        pbar.close()

    # Phase 2 : Assembler les chunks en pickles finaux (séquentiel, rapide)
    print(f"\nAssemblage des chunks...")
    assembled = 0
    for p in to_run:
        out = f'{cache_dir}/{prefix}_{p}_{tf}_{exchange}.pickle'
        if os.path.exists(out):
            continue
        _chunk_dir = os.path.join(cache_dir, '.grid_chunks', p)
        _expected = [f'chunk_{i}.pickle' for i in range(n_chunks)]
        _existing = [f for f in _expected if os.path.exists(os.path.join(_chunk_dir, f))]
        if len(_existing) == n_chunks:
            assemble_pair(p, strategy, cache_dir)
            assembled += 1
        else:
            print(f"  ⚠ {p} : {len(_existing)}/{n_chunks} chunks — incomplet, skip")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Terminé : {assembled} paires assemblées, {done_chunks} chunks en {int(elapsed//60)}m{int(elapsed%60):02d}s")
    if failed:
        print(f"Échecs : {', '.join(failed)}")
    print(f"Résultats dans {cache_dir}/")
    print(f"{'='*60}")
    tg_send(f"✅ <b>Opti terminée</b>\n{assembled}/{len(to_run)} paires en {int(elapsed//60)}m{int(elapsed%60):02d}s\nÉchecs: {len(failed)}\n→ {cache_dir}/")
