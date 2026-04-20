"""
Classifie les paires Lighter en 4 niveaux de liquidité :
  tres_liquide / liquide / moyen / poubelle

Méthode (basée sur snapshot orderbook live, pas CSV) :
  1. Récupère la liste des marchés via /orderBookDetails (vol, trades, status)
  2. Pour chaque marché actif, snapshot du carnet via /orderBookOrders
  3. Calcule deux métriques :
     - slip_bps   : coût d'exécution d'un trade $TARGET_NOTIONAL (worst side)
     - max_gap_bps: plus gros écart entre deux ordres consécutifs du top of book
  4. Gate d'activité pour les crypto (trades/j) — filtre MM statique sans flow
  5. Classification + sauvegarde JSON

Usage :
  python src/liquidity.py
  python src/liquidity.py --out liquidity.json --target 10000
"""

import json, argparse, time
import urllib.request

LIGHTER_API = 'https://mainnet.zklighter.elliot.ai/api/v1'
HEADERS = {
    'Origin':     'https://app.lighter.xyz',
    'Referer':    'https://app.lighter.xyz/',
    'User-Agent': 'Mozilla/5.0',
    'Accept':     'application/json',
}

STRAT_CRYPTO = {2, 7}  # indexes crypto (MM dédié ou générique)

# ── Paramètres de classification ──────────────────────────────────────────────
TARGET_NOTIONAL = 10_000        # $ — taille de trade cible pour slip
ORDERBOOK_LIMIT = 250           # profondeur snapshot par côté

# Gate d'activité crypto : MM statique sans trades réels = poubelle
CRYPTO_MIN_TRADES = 400
CRYPTO_SLIP_WHEN_IDLE = 5       # si trades < min ET slip > X → poubelle

# Seuils durs (au-delà : poubelle quoi qu'il arrive)
MAX_BOOK_GAP_BPS = 40
MAX_SLIP_BPS     = 80

# Étages (book + slip conjoints)
TL_SLIP, TL_GAP  = 3, 3
LIQ_SLIP, LIQ_GAP = 20, 20


def get_json(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.load(r)
        except Exception:
            if i == retries - 1: raise
            time.sleep(1.5)


def fetch_markets():
    """Retourne {symbol: {market_id, strategy_index, status, vol, trades}}."""
    details = get_json(f'{LIGHTER_API}/orderBookDetails')
    books   = get_json(f'{LIGHTER_API}/orderBooks')
    sym_to_id = {b['symbol']: b['market_id'] for b in books['order_books']}
    out = {}
    for b in details['order_book_details']:
        sym = b['symbol']
        out[sym] = {
            'market_id':      sym_to_id.get(sym, b.get('market_id')),
            'strategy_index': b.get('strategy_index', -1),
            'status':         b.get('status', 'unknown'),
            'daily_vol_usd':  float(b.get('daily_quote_token_volume') or 0),
            'daily_trades':   float(b.get('daily_trades_count') or 0),
        }
    return out


def fetch_book(market_id):
    """Snapshot orderbook. Retourne (bids, asks, mid) ou None."""
    d = get_json(f'{LIGHTER_API}/orderBookOrders?market_id={market_id}&limit={ORDERBOOK_LIMIT}')
    bids = sorted([(float(o['price']), float(o['remaining_base_amount']))
                   for o in d.get('bids', [])], reverse=True)
    asks = sorted([(float(o['price']), float(o['remaining_base_amount']))
                   for o in d.get('asks', [])])
    if not bids or not asks:
        return None
    mid = (bids[0][0] + asks[0][0]) / 2
    return bids, asks, mid


def analyse_book(bids, asks, mid, target=TARGET_NOTIONAL):
    """
    Retourne dict :
      - slip_bps    : max(ask_slip, bid_slip) pour exécuter target $ (None si depth insuffisante)
      - max_gap_bps : plus gros écart (en bps du mid) entre ordres consécutifs près du top
    """
    def side(orders):
        cum, max_gap, prev = 0.0, 0.0, mid
        hit = None
        for px, sz in orders:
            g = abs(px - prev) / mid * 10000
            if g > max_gap: max_gap = g
            cum += px * sz
            prev = px
            if cum >= target and hit is None:
                hit = abs(px - mid) / mid * 10000
                break
        return max_gap, hit

    a_gap, a_slip = side(asks)
    b_gap, b_slip = side(bids)
    slip = max(a_slip, b_slip) if (a_slip is not None and b_slip is not None) else None
    return {'slip_bps': slip, 'max_gap_bps': max(a_gap, b_gap)}


def classify(slip_bps, max_gap_bps, is_crypto, trades):
    if slip_bps is None:
        return 'poubelle'
    # Gate activité crypto : MM statique = poubelle
    if is_crypto and trades < CRYPTO_MIN_TRADES and slip_bps > CRYPTO_SLIP_WHEN_IDLE:
        return 'poubelle'
    # Book trop dégradé
    if max_gap_bps > MAX_BOOK_GAP_BPS or slip_bps > MAX_SLIP_BPS:
        return 'poubelle'
    # Étages
    if slip_bps <= TL_SLIP and max_gap_bps <= TL_GAP:
        return 'tres_liquide'
    if slip_bps <= LIQ_SLIP and max_gap_bps <= LIQ_GAP:
        return 'liquide'
    return 'moyen'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',    default='liquidity.json')
    parser.add_argument('--target', type=float, default=TARGET_NOTIONAL,
                        help='Taille de trade cible en $ (défaut 10000)')
    parser.add_argument('--sleep',  type=float, default=0.25,
                        help='Délai entre requêtes API (sec)')
    args = parser.parse_args()

    print('Récupération des marchés Lighter...')
    markets = fetch_markets()
    active = [(s, i) for s, i in markets.items() if i['status'] != 'inactive']
    print(f'  {len(markets)} marchés totaux, {len(active)} actifs')

    result  = {'tres_liquide': [], 'liquide': [], 'moyen': [], 'poubelle': []}
    details = {}

    for sym, info in sorted(markets.items()):
        status   = info['status']
        strat    = info['strategy_index']
        mid_id   = info['market_id']
        vol      = info['daily_vol_usd']
        trades   = info['daily_trades']
        is_crypto = strat in STRAT_CRYPTO

        if status == 'inactive' or mid_id is None:
            result['poubelle'].append(sym)
            details[sym] = {'niveau': 'poubelle', 'raison': 'inactif'}
            continue

        try:
            book = fetch_book(mid_id)
        except Exception as e:
            book = None
            err = str(e)[:60]
        else:
            err = None

        if book is None:
            niveau = 'poubelle'
            raison = f'book vide ({err})' if err else 'book vide'
            result[niveau].append(sym)
            details[sym] = {'niveau': niveau, 'raison': raison}
            print(f'  ✗  {sym:15} {raison}')
            time.sleep(args.sleep)
            continue

        bids, asks, mid = book
        m = analyse_book(bids, asks, mid, target=args.target)
        slip, gap = m['slip_bps'], m['max_gap_bps']
        niveau = classify(slip, gap, is_crypto, trades)

        slip_s = f'{slip:.1f}' if slip is not None else 'MISS'
        raison = f'slip={slip_s}bps gap={gap:.1f}bps trades/j={int(trades)}'

        result[niveau].append(sym)
        details[sym] = {
            'niveau':      niveau,
            'strat':       strat,
            'is_crypto':   is_crypto,
            'slip_bps':    round(slip, 2) if slip is not None else None,
            'max_gap_bps': round(gap, 2),
            'vol_usd':     int(vol),
            'trades/j':    int(trades),
            'raison':      raison,
        }

        icon = {'tres_liquide': '★★', 'liquide': '★ ',
                'moyen': '~ ', 'poubelle': '✗ '}[niveau]
        print(f'  {icon} {sym:15} {raison}')
        time.sleep(args.sleep)

    for k in result:
        result[k].sort()

    output = {
        'tres_liquide': result['tres_liquide'],
        'liquide':      result['liquide'],
        'moyen':        result['moyen'],
        'poubelle':     result['poubelle'],
        'details':      details,
        'params': {
            'target_notional': args.target,
            'max_book_gap_bps': MAX_BOOK_GAP_BPS,
            'max_slip_bps': MAX_SLIP_BPS,
            'tl_slip_bps': TL_SLIP, 'tl_gap_bps': TL_GAP,
            'liq_slip_bps': LIQ_SLIP, 'liq_gap_bps': LIQ_GAP,
            'crypto_min_trades': CRYPTO_MIN_TRADES,
        },
    }

    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'\n{"─"*50}')
    print(f'  ★★ Très liquide : {len(result["tres_liquide"])} paires')
    print(f'  ★  Liquide      : {len(result["liquide"])} paires')
    print(f'  ~  Moyen        : {len(result["moyen"])} paires')
    print(f'  ✗  Poubelle     : {len(result["poubelle"])} paires')
    tradables = len(result['tres_liquide']) + len(result['liquide']) + len(result['moyen'])
    print(f'  Total tradables: {tradables}')
    print(f'\nJSON sauvegardé → {args.out}')


if __name__ == '__main__':
    main()
