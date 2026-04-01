import sys
import os
import hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vectorbtpro import *
import numpy as np
import pandas as pd
import questionary
from questionary import Style

# ─────────────────────────────────────────────────────────────────────────────
# Grilles de paramètres
# ─────────────────────────────────────────────────────────────────────────────

GRIDS = {
    'keltner': {
        'ma_window':  list(range(20, 220, 20)),
        'atr_window': list(range(10, 220, 20)),
        'atr_mult':   list(np.arange(1.0, 8.5, 0.5)),
        'sl_stop':    list(np.arange(0.02, 0.11, 0.02)),
    },
    'ram': {
        'ma_window': list(range(20, 320, 20)),
        'env_pct':   [round(x, 4) for x in np.arange(0.005, 0.085, 0.005)],  # 0.5% → 8% step 0.5%
        'sl_pct':    [round(x, 4) for x in np.arange(0.005, 0.085, 0.005)],  # 0.5% → 8% step 0.5%
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Fonctions objectif
# ─────────────────────────────────────────────────────────────────────────────

def keltner_objective(data, ma_window, atr_window, atr_mult, sl_stop):
    try:
        from src.strategies.keltner import run_backtest
        pf = run_backtest(data, ma_window, atr_window, atr_mult, sl_stop)
        return pf.stats()
    except Exception:
        return None


def ram_objective(data, ma_window, env_pct, sl_pct):
    try:
        from src.strategies.ram_dca import run_backtest
        pf = run_backtest(data, ma_window, [env_pct], [1.0], sl_pct)
        return pf.stats()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def grid_hash(grid):
    """
    Retourne un identifiant court (8 hex) unique pour une grille de paramètres.
    Inclut fees + slippage : changer la config crée un nouveau dossier cache.
    Utilise json.dumps pour un sérialisation stable (pas de np.float64 dans str).
    """
    import json
    from config import FEES, SLIPPAGE
    canonical = {k: [round(float(x), 8) for x in sorted(v)] for k, v in grid.items()}
    sig = json.dumps(canonical, sort_keys=True) + f'|fees={float(FEES)}|slip={float(SLIPPAGE)}'
    h = hashlib.md5(sig.encode()).hexdigest()[:8]
    return h


def grid_summary(grid):
    """Résumé lisible de la grille : param min→max (n valeurs)."""
    parts = []
    for k, v in grid.items():
        parts.append(f"{k}: {v[0]}→{v[-1]} ({len(v)} valeurs)")
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
    import multiprocessing
    n_cpus    = multiprocessing.cpu_count()
    n_workers = max(1, int(n_cpus * 0.8))
    print(f"\nWorkers : {n_workers}/{n_cpus} (80%)")

    splitter = vbt.Splitter.from_n_rolling(
        data.index, n=10, length='optimize', split=0.7,
        optimize_anchor_set=1, set_labels=['train', 'test']
    )

    objective_fn = keltner_objective if strategy == 'keltner' else ram_objective

    param_fn = vbt.parameterized(
        objective_fn,
        merge_func='concat',
        execute_kwargs=dict(
            chunk_len=n_workers * 4,
            distribute='chunks',
            engine='pathos',
            init_kwargs=dict(nodes=n_workers),
            show_progress=True,
        ),
    )
    cv_fn = vbt.split(
        param_fn,
        splitter=splitter,
        takeable_args=['data'],
        merge_func='concat',
        execute_kwargs=dict(show_progress=True),
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
    try:
        from pathos.pools import _clear
        _clear()
    except Exception:
        pass
    gc.collect()


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
    grid = GRIDS[strategy]
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    print(f"\n┌─────────────────────────────────────────┐")
    print(f"│  Stratégie : {strategy:<28}│")
    print(f"│  Exchange  : {exchange:<28}│")
    print(f"│  Timeframe : {tf:<28}│")
    print(f"│  Paires    : {len(pairs):<28}│")
    print(f"│  Combos    : {n_combos:<28}│")
    print(f"│  Backtests : {n_combos * 10 * 2:<28}│")
    print(f"└─────────────────────────────────────────┘")

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

    strategy, mode, exchange, tf, pairs, grid = menu()

    gh = grid_hash(grid)
    prefix = 'kc_wfsl' if strategy == 'keltner' else 'ram'

    if mode in ('full', 'liquid'):
        cache_dir = f'./cache/full_{exchange}_{tf}/{prefix}_grid_{gh}'
    else:
        cache_dir = f'./cache/{prefix}_grid_{gh}'

    print(f"\nGrille [{gh}] : {grid_summary(grid)}")
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

    print(f"\n{len(to_run)} paires à traiter, {skipped} skippées\n")

    t0 = time.time()
    for i, p in enumerate(to_run):
        elapsed = time.time() - t0
        if i > 0:
            eta = elapsed / i * (len(to_run) - i)
            print(f"\n{'='*60}")
            print(f"  GLOBAL: {i}/{len(to_run)} | ETA {int(eta//60)}m{int(eta%60):02d}s")
        print(f"{'='*60}")
        print(f"  >>> {p} {tf} {exchange}  [{i+1}/{len(to_run)}]")
        print(f"{'='*60}")

        data = load_data(exchange, tf, p)
        if data is None:
            print(f"  Pas de data pour {p}, skip")
            continue
        run_opti(data, strategy, p, tf, exchange, grid, cache_dir=cache_dir)
        del data

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Terminé : {len(to_run)} paires en {int(elapsed//60)}m{int(elapsed%60):02d}s")
    print(f"Résultats dans {cache_dir}/")
    print(f"{'='*60}")
