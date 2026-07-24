"""
Classifie les paires Lighter en 4 niveaux de liquidité :
  tres_liquide / liquide / moyen / poubelle

Méthode :
  1. Récupère strategy_index via l'API Lighter
  2. Lit les CSV 1m (si dispo) → calcule flat_trading_% et gap_p99_%
  3. Applique des seuils par classe d'actif (crypto, commodité, forex, action)
  4. Regroupe par niveau et sauvegarde en JSON

Usage :
  python src/liquidity.py
  python src/liquidity.py --data data/raw/lighter/1m --out liquidity.json
"""

import os, sys, json, argparse, glob
import numpy as np
import pandas as pd
import urllib.request

LIGHTER_API = 'https://mainnet.zklighter.elliot.ai/api/v1'
HEADERS = {
    'Origin':     'https://app.lighter.xyz',
    'Referer':    'https://app.lighter.xyz/',
    'User-Agent': 'Mozilla/5.0',
    'Accept':     'application/json',
}

# ── Seuils par classe d'actif ─────────────────────────────────────────────────
# (flat_trading_%, gap_p99_%) → niveau
# Format : {strat_group: [(flat_max, gap_max, niveau), ...]}  du + strict au + lâche
THRESHOLDS = {
    # crypto : deux règles pour "liquide"
    #   1) flat < 4%  ET gap < 0.10%  (cas standard)
    #   2) flat < 8%  ET gap < 0.001% (gap quasi nul = MM parfaitement continu, ex XPL)
    'crypto':    [(1.0,  0.005, 'tres_liquide'),
                  (4.0,  0.10,  'liquide'),
                  (8.0,  0.001, 'liquide'),
                  (8.0,  0.20,  'moyen')],
    'commodity': [(20.0, 0.03,  'tres_liquide'),
                  (55.0, 0.30,  'liquide'),
                  (70.0, 0.50,  'moyen')],
    # forex sur Lighter : flat > 28% pour toutes les paires → jamais liquide
    # seuil moyen abaissé à 25% pour que GBPUSD+ → poubelle
    'forex':     [(25.0, 0.05,  'moyen')],
    'stock_us':  [(30.0, 0.05,  'tres_liquide'),
                  (40.0, 0.15,  'liquide'),
                  (55.0, 0.30,  'moyen')],
    'stock_kr':  [(40.0, 0.20,  'tres_liquide'),
                  (50.0, 0.40,  'liquide'),
                  (65.0, 0.60,  'moyen')],
}

STRAT_TO_GROUP = {
    0: None,         # inactif
    2: 'crypto',
    3: 'commodity',
    4: 'forex',
    5: 'stock_us',
    6: 'stock_kr',
    7: 'crypto',     # crypto sans MM dédié — mêmes seuils
}


def fetch_markets():
    """Retourne {symbol: {market_id, strategy_index, status}} depuis l'API."""
    url = f'{LIGHTER_API}/orderBookDetails'
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.load(r)
    out = {}
    for b in data['order_book_details']:
        sym = b['symbol']
        out[sym] = {
            'market_id':      b['market_id'],
            'strategy_index': b.get('strategy_index', -1),
            'status':         b.get('status', 'unknown'),
        }
    return out


def compute_metrics(csv_path):
    """
    Calcule flat_trading_% et gap_p99_% depuis un CSV 1m.
    flat_trading_% : % de bougies plates (high==low) pendant les heures actives.
    gap_p99_%      : 99e centile des écarts |open[t+1] - close[t]| / close[t].
    """
    df = pd.read_csv(csv_path)
    if len(df) < 200:
        return None, None

    df = df.sort_values('date')
    df['ts']   = pd.to_datetime(df['date'], unit='ms', utc=True)
    df         = df.set_index('ts')
    df['flat'] = (df['high'] == df['low']).astype(int)

    # Heures actives : heures de la semaine avec >20 % de bougies non-plates
    df['dow_hour'] = df.index.dayofweek * 24 + df.index.hour
    activity       = df.groupby('dow_hour')['flat'].apply(lambda s: 1 - s.mean())
    trading_hours  = activity[activity > 0.20].index
    in_trading     = df['dow_hour'].isin(trading_hours)

    if in_trading.sum() < 100:
        return None, None

    flat_pct = float(df.loc[in_trading, 'flat'].mean() * 100)

    price_gap = ((df['open'].shift(-1) - df['close']) / df['close']).abs() * 100
    gap_p99   = float(price_gap[in_trading].quantile(0.99))

    return round(flat_pct, 3), round(gap_p99, 4)


def classify(group, flat_pct, gap_p99):
    """Retourne 'tres_liquide', 'liquide', 'moyen' ou 'poubelle'."""
    if group is None:
        return 'poubelle'
    for flat_max, gap_max, level in THRESHOLDS[group]:
        if flat_pct <= flat_max and gap_p99 <= gap_max:
            return level
    return 'poubelle'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw/lighter/1m', help='Dossier CSV 1m')
    parser.add_argument('--out',  default='liquidity.json',       help='Fichier JSON de sortie')
    args = parser.parse_args()

    print('Récupération des marchés Lighter...')
    markets = fetch_markets()
    print(f'  {len(markets)} marchés trouvés')

    csv_files = {
        os.path.basename(f).replace('.csv', ''): f
        for f in glob.glob(os.path.join(args.data, '*.csv'))
    }
    print(f'  {len(csv_files)} CSV 1m disponibles\n')

    result   = {'tres_liquide': [], 'liquide': [], 'moyen': [], 'poubelle': []}
    details  = {}

    for sym, info in sorted(markets.items()):
        strat  = info['strategy_index']
        status = info['status']
        group  = STRAT_TO_GROUP.get(strat)

        # Inactif → poubelle directement
        if status == 'inactive' or group is None:
            result['poubelle'].append(sym)
            details[sym] = {'niveau': 'poubelle', 'raison': 'inactif'}
            continue

        # Calcul métriques si CSV dispo
        if sym in csv_files:
            flat_pct, gap_p99 = compute_metrics(csv_files[sym])
        else:
            flat_pct, gap_p99 = None, None

        if flat_pct is None:
            # Pas de données : on ne peut pas classer → moyen par défaut
            niveau = 'moyen'
            raison = 'pas de données 1m'
        else:
            niveau = classify(group, flat_pct, gap_p99)
            raison = f'flat={flat_pct}% gap_p99={gap_p99}%'

        result[niveau].append(sym)
        details[sym] = {
            'niveau':   niveau,
            'group':    group,
            'flat_%':   flat_pct,
            'gap_p99%': gap_p99,
            'raison':   raison,
        }

        icon = {'tres_liquide': '★★', 'liquide': '★ ', 'moyen': '~ ', 'poubelle': '✗ '}[niveau]
        print(f'  {icon} {sym:15} {raison}')

    # Tri alphabétique dans chaque catégorie
    for k in result:
        result[k].sort()

    # JSON de sortie
    output = {
        'tres_liquide': result['tres_liquide'],
        'liquide':      result['liquide'],
        'moyen':        result['moyen'],
        'poubelle':     result['poubelle'],
        'details':      details,
    }

    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'\n{"─"*50}')
    print(f'  ★★ Très liquide : {len(result["tres_liquide"])} paires')
    print(f'  ★  Liquide      : {len(result["liquide"])} paires')
    print(f'  ~  Moyen        : {len(result["moyen"])} paires')
    print(f'  ✗  Poubelle     : {len(result["poubelle"])} paires')
    print(f'\nJSON sauvegardé → {args.out}')


if __name__ == '__main__':
    main()
