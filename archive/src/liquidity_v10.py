"""
Classifie les paires Lighter en 4 niveaux de liquidité :
  tres_liquide / liquide / moyen / poubelle

Méthode :
  1. Récupère strategy_index + volume/trades/OI via l'API Lighter
  2. Lit les CSV 1m (si dispo) → calcule flat_%, gap_p99 (stocks/forex/comm),
     en excluant les weekends pour les non-crypto
  3. Crypto : classification par activité réelle (vol $, trades/j, flat%)
     Stocks/forex/commodity : classification par flat% + gap_p99
  4. Regroupe par niveau et sauvegarde en JSON

Usage :
  python src/liquidity.py
  python src/liquidity.py --data data/raw/lighter/1m --out liquidity.json
"""

import os, json, argparse, glob
import pandas as pd
import urllib.request

LIGHTER_API = 'https://mainnet.zklighter.elliot.ai/api/v1'
HEADERS = {
    'Origin':     'https://app.lighter.xyz',
    'Referer':    'https://app.lighter.xyz/',
    'User-Agent': 'Mozilla/5.0',
    'Accept':     'application/json',
}

# ── Seuils flat% + gap_p99 pour non-crypto (stocks, forex, commodity) ─────────
# Crypto est traité séparément (voir classify_crypto) — la métrique gap
# est invalide sur Lighter (MM cote en continu = p99 pollué par les flat bars).
THRESHOLDS = {
    'commodity': [(20.0, 0.03,  'tres_liquide'),
                  (55.0, 0.30,  'liquide'),
                  (70.0, 0.50,  'moyen')],
    'forex':     [(25.0, 0.05,  'moyen')],
    'stock_us':  [(40.0, 0.05,  'tres_liquide'),
                  (55.0, 0.15,  'liquide'),
                  (65.0, 0.30,  'moyen')],
    'stock_kr':  [(40.0, 0.20,  'tres_liquide'),
                  (55.0, 0.40,  'liquide'),
                  (65.0, 0.60,  'moyen')],
}

STRAT_TO_GROUP = {
    0: None,         # inactif
    2: 'crypto',
    3: 'commodity',
    4: 'forex',
    5: 'stock_us',
    6: 'stock_kr',
    7: 'crypto',     # crypto sans MM dédié — mêmes règles
}


def fetch_markets():
    """Retourne {symbol: {market_id, strategy_index, status, vol, trades, OI}}."""
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
            'daily_vol_usd':  float(b.get('daily_quote_token_volume') or 0),
            'daily_trades':   float(b.get('daily_trades_count') or 0),
            'open_interest':  float(b.get('open_interest') or 0),
        }
    return out


RECENT_WINDOW_BARS = 130_000  # ~90 jours à 1m

def compute_metrics(csv_path, group):
    """
    Retourne (flat_%, gap_p99_%) sur fenêtre récente.
    Non-crypto : weekends exclus (marchés fermés) sinon flat% explose.
    """
    df = pd.read_csv(csv_path)
    if len(df) < 200:
        return None, None

    df = df.sort_values('date').tail(RECENT_WINDOW_BARS)
    df['ts'] = pd.to_datetime(df['date'], unit='ms', utc=True)
    df = df.set_index('ts')

    # Weekends exclus pour les marchés fermés le week-end
    if group in ('stock_us', 'stock_kr', 'forex', 'commodity'):
        df = df[df.index.dayofweek < 5]

    if len(df) < 100:
        return None, None

    df['flat'] = (df['high'] == df['low']).astype(int)

    # Heures actives : heures de la semaine avec >20 % de bougies non-plates
    df['dow_hour'] = df.index.dayofweek * 24 + df.index.hour
    activity      = df.groupby('dow_hour')['flat'].apply(lambda s: 1 - s.mean())
    trading_hours = activity[activity > 0.20].index
    in_trading    = df['dow_hour'].isin(trading_hours)

    if in_trading.sum() < 100:
        return None, None

    flat_pct = float(df.loc[in_trading, 'flat'].mean() * 100)

    price_gap = ((df['open'].shift(-1) - df['close']) / df['close']).abs() * 100
    gap_p99   = float(price_gap[in_trading].quantile(0.99))

    return round(flat_pct, 3), round(gap_p99, 4)


# ── Règles crypto ─────────────────────────────────────────────────────────────
# Une crypto est vivante si :
#   - vol/j ≥ 70k$  (sinon pas assez de flux pour trader)
#   - trades/j ≥ 400 (sinon MM seul, pas de vrai trading)
#   - pas (flat > 65% ET vol < 500k$) — MM présent mais volume trop faible
CRYPTO_MIN_VOL_USD     = 70_000
CRYPTO_MIN_TRADES      = 400
CRYPTO_DEAD_FLAT       = 65.0
CRYPTO_DEAD_VOL_FLOOR  = 500_000

def crypto_alive(vol, trades, flat_pct):
    if vol < CRYPTO_MIN_VOL_USD:    return False
    if trades < CRYPTO_MIN_TRADES:  return False
    if flat_pct > CRYPTO_DEAD_FLAT and vol < CRYPTO_DEAD_VOL_FLOOR: return False
    return True

def classify_crypto(flat_pct, vol, trades):
    if not crypto_alive(vol, trades, flat_pct):
        return 'poubelle'
    if flat_pct <= 1.0:   return 'tres_liquide'
    if flat_pct <= 10.0:  return 'liquide'
    return 'moyen'

def classify(group, flat_pct, gap_p99, vol=0, trades=0):
    if group is None:
        return 'poubelle'
    if group == 'crypto':
        return classify_crypto(flat_pct, vol, trades)
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

    result  = {'tres_liquide': [], 'liquide': [], 'moyen': [], 'poubelle': []}
    details = {}

    for sym, info in sorted(markets.items()):
        strat  = info['strategy_index']
        status = info['status']
        group  = STRAT_TO_GROUP.get(strat)

        if status == 'inactive' or group is None:
            result['poubelle'].append(sym)
            details[sym] = {'niveau': 'poubelle', 'raison': 'inactif'}
            continue

        if sym in csv_files:
            flat_pct, gap_p99 = compute_metrics(csv_files[sym], group)
        else:
            flat_pct, gap_p99 = None, None

        vol    = info.get('daily_vol_usd', 0)
        trades = info.get('daily_trades', 0)

        if flat_pct is None:
            niveau = 'poubelle'
            raison = 'pas de données 1m'
        else:
            niveau = classify(group, flat_pct, gap_p99, vol, trades)
            if group == 'crypto':
                raison = f'flat={flat_pct}% vol=${int(vol):,} trades/j={int(trades)}'
            else:
                raison = f'flat={flat_pct}% gap_p99={gap_p99}%'

        result[niveau].append(sym)
        details[sym] = {
            'niveau':      niveau,
            'group':       group,
            'flat_%':      flat_pct,
            'gap_p99%':    gap_p99,
            'vol_usd':     int(vol),
            'trades/j':    int(trades),
            'raison':      raison,
        }

        icon = {'tres_liquide': '★★', 'liquide': '★ ', 'moyen': '~ ', 'poubelle': '✗ '}[niveau]
        print(f'  {icon} {sym:15} {raison}')

    for k in result:
        result[k].sort()

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
