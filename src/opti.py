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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def grid_dirname(strategy, grid):
    """
    Nom de dossier lisible : strat_params_fees.
    Ex: ram_ma20-300x15_env0.005-0.08x16_sl0.005-0.08x16_fees0bps_slip2bps
    """
    from config import FEES, SLIPPAGE

    prefix = 'kc' if strategy == 'keltner' else 'ram'

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


def run_opti(data, strategy, pair, tf, exchange, grid, cache_dir='./cache'):
    import gc

    splitter = vbt.Splitter.from_n_rolling(
        data.index, n=10, length='optimize', split=0.7,
        optimize_anchor_set=1, set_labels=['train', 'test']
    )

    objective_fn = keltner_objective if strategy == 'keltner' else ram_objective

    param_fn = vbt.parameterized(
        objective_fn,
        merge_func='concat',
        execute_kwargs=dict(show_progress=True),
    )
    _split_chunk_dir = os.path.join(cache_dir, '.split_chunks', pair)

    cv_fn = vbt.split(
        param_fn,
        splitter=splitter,
        takeable_args=['data'],
        merge_func='concat',
        execute_kwargs=dict(
            show_progress=True,
            cache_chunks=True,
            chunk_cache_dir=_split_chunk_dir,
            release_chunk_cache=True,
            chunk_collect_garbage=True,
        ),
    )

    param_grid = {k: vbt.Param(v) for k, v in grid.items()}

    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    print(f"Combos : {n_combos}")
    print(f"Total backtests : {n_combos * splitter.n_splits * 2}")
    print("Lancement...\n")

    results = cv_fn(data, **param_grid)

    os.makedirs(cache_dir, exist_ok=True)
    prefix   = 'kc_wfsl' if strategy == 'keltner' else 'ram'
    out_path = f'{cache_dir}/{prefix}_{pair}_{tf}_{exchange}.pickle'
    vbt.save(results, out_path)
    print(f"\nSauvegardé → {out_path}")

    del results, cv_fn, param_fn, splitter, param_grid
    vbt.clear_cache()
    gc.collect()

    # Nettoyer les chunks temporaires
    import shutil
    shutil.rmtree(_split_chunk_dir, ignore_errors=True)


def process_pair(p, strategy, exchange, tf, grid, cache_dir):
    """Traite une paire dans un process séparé (appelé par ProcessPoolExecutor)."""
    import warnings
    warnings.filterwarnings('ignore')
    _data = load_data(exchange, tf, p)
    if _data is None:
        return p, False, "Pas de data"
    try:
        run_opti(_data, strategy, p, tf, exchange, grid, cache_dir=cache_dir)
        return p, True, "OK"
    except Exception as _e:
        return p, False, str(_e)


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
            questionary.Choice("Keltner Channel  (EMA + ATR)",  value='keltner'),
            questionary.Choice("RAM DCA          (SMA + enveloppe + SL)", value='ram'),
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
        prefix = 'kc_wfsl' if _base.startswith('kc_') else 'ram'
        strategy = 'keltner' if prefix == 'kc_wfsl' else 'ram'
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
    prefix = 'kc_wfsl' if strategy == 'keltner' else 'ram'

    if mode in ('full', 'liquid'):
        cache_dir = f'./cache/full_{exchange}_{tf}/{dirname}'
    else:
        cache_dir = f'./cache/{dirname}'

    print(f"\nGrille : {grid_summary(grid)}")
    print(f"Cache → {cache_dir}/\n")
    os.makedirs(cache_dir, exist_ok=True)

    # Filtrer les paires déjà dans le cache
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

    import multiprocessing
    n_cpus = multiprocessing.cpu_count()
    # Paralléliser au niveau des paires (pas des combos)
    # Chaque paire tourne en serial → pas de deadlock interne
    n_parallel = min(n_cpus, len(to_run))  # 1 paire par cœur
    print(f"\n{len(to_run)} paires à traiter, {skipped} skippées")
    print(f"Parallélisme : {n_parallel} paires simultanées (serial par paire)\n")

    t0 = time.time()
    done = 0
    failed = []

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=n_parallel, max_tasks_per_child=1) as pool:
        futures = {pool.submit(process_pair, p, strategy, exchange, tf, grid, cache_dir): p for p in to_run}
        for future in as_completed(futures):
            p = futures[future]
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (len(to_run) - done) if done > 0 else 0
            try:
                pair, success, msg = future.result(timeout=30 * 60)
                if success:
                    print(f"  ✓ {pair} [{done}/{len(to_run)}] ETA {int(eta//60)}m{int(eta%60):02d}s")
                else:
                    print(f"  ✗ {pair} : {msg} [{done}/{len(to_run)}]")
                    failed.append(pair)
                    tg_send(f"⚠️ <b>Opti erreur</b> sur {pair}\n{msg}")
            except Exception as e:
                print(f"  ✗ {p} : {e} [{done}/{len(to_run)}]")
                failed.append(p)
                tg_send(f"⚠️ <b>Opti crash</b> sur {p}\n{e}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Terminé : {done} paires en {int(elapsed//60)}m{int(elapsed%60):02d}s")
    if failed:
        print(f"Échecs : {', '.join(failed)}")
    print(f"Résultats dans {cache_dir}/")
    print(f"{'='*60}")
    tg_send(f"✅ <b>Opti terminée</b>\n{done}/{len(to_run)} paires en {int(elapsed//60)}m{int(eta%60):02d}s\nÉchecs: {len(failed)}\n→ {cache_dir}/")
