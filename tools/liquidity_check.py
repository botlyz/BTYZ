"""analyse de liquidité sur Lighter
mesure le spread, la depth à differents niveaux, le slippage, le volume etc
score composite pour évaluer si une paire est tradable"""
import requests
import math

BASE = "https://mainnet.zklighter.elliot.ai/api/v1"
DEPTH_LEVELS = [0.5, 1, 2, 5, 10, 20]  #niveaux de profondeur en %
SLIPPAGE_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000]  #tailles de test en USD


def get_all_markets():
    """recup tous les marchés actifs"""
    details = requests.get(f"{BASE}/orderBookDetails?filter=all").json()
    markets = details.get("order_book_details", []) + details.get("spot_order_book_details", [])
    return [m for m in markets if m["status"] == "active"]


def find_market(symbol, markets):
    """trouve un marché par symbol (exact puis partiel)"""
    for m in markets:
        if m["symbol"].upper() == symbol.upper():
            return m
    for m in markets:
        if symbol.upper() in m["symbol"].upper():
            return m
    return None


def compute_depth(orders, mid, pct, side='ask'):
    """calcule la profondeur en USD à ±pct% du mid price"""
    total = 0
    for o in orders:
        price = float(o["price"])
        size = float(o["remaining_base_amount"])
        if side == 'ask' and price <= mid * (1 + pct / 100):
            total += price * size
        elif side == 'bid' and price >= mid * (1 - pct / 100):
            total += price * size
    return total


def compute_slippage(orders, mid, size_usd):
    """simule un ordre market de size_usd et retourne le slippage en %"""
    remaining = size_usd
    filled_base = 0
    spent = 0
    for o in orders:
        price = float(o["price"])
        sz = float(o["remaining_base_amount"])
        avail = price * sz
        if avail >= remaining:
            filled_base += remaining / price
            spent += remaining
            remaining = 0
            break
        filled_base += sz
        spent += avail
        remaining -= avail

    if filled_base == 0:
        return 100.0, 0  #pas de liquidité du tout
    avg_price = spent / filled_base
    slip = abs(avg_price - mid) / mid * 100
    fill_pct = (size_usd - remaining) / size_usd * 100
    return slip, fill_pct


def get_liquidity_score(symbol, markets=None):
    """analyse complète de la liquidité d'une paire"""
    if markets is None:
        markets = get_all_markets()

    market = find_market(symbol, markets)
    if not market:
        print(f"  '{symbol}' non trouvé")
        return None

    market_id = market["market_id"]
    volume_24h = float(market.get("daily_quote_token_volume", 0))
    trades_24h = int(market.get("daily_trades_count", 0))
    oi = float(market.get("open_interest", 0))

    #recup le carnet (max depth)
    ob = requests.get(f"{BASE}/orderBookOrders", params={"market_id": market_id, "limit": 250}).json()
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])

    if not bids or not asks:
        print(f"  {market['symbol']}: carnet vide")
        return None

    best_bid = float(bids[0]["price"])
    best_ask = float(asks[0]["price"])
    mid = (best_bid + best_ask) / 2
    spread_pct = (best_ask - best_bid) / mid * 100

    #depth à chaque niveau
    depths = {}
    for pct in DEPTH_LEVELS:
        ask_d = compute_depth(asks, mid, pct, 'ask')
        bid_d = compute_depth(bids, mid, pct, 'bid')
        depths[pct] = {'ask': ask_d, 'bid': bid_d, 'total': ask_d + bid_d}

    #slippage pour differentes tailles
    slippages = {}
    for size in SLIPPAGE_SIZES:
        slip_buy, fill_buy = compute_slippage(asks, mid, size)
        slip_sell, fill_sell = compute_slippage(bids, mid, size)
        slippages[size] = {
            'buy': slip_buy, 'sell': slip_sell,
            'avg': (slip_buy + slip_sell) / 2,
            'fill_buy': fill_buy, 'fill_sell': fill_sell,
        }

    #=== SCORE ===
    #spread (0-25pts) : 0% = 25, 0.1% = 15, 0.5% = 5, 1%+ = 0
    spread_score = max(0, 25 * (1 - spread_pct / 0.5))

    #depth ±1% (0-25pts) : basé sur log scale
    #$100k+ = 25, $10k = 15, $1k = 5, <$100 = 0
    d1 = depths.get(1, {}).get('total', 0)
    depth_score = min(25, 5 * math.log10(max(1, d1)))

    #depth ±5% (0-15pts) : la profondeur réelle pour des ordres plus gros
    d5 = depths.get(5, {}).get('total', 0)
    depth5_score = min(15, 3 * math.log10(max(1, d5)))

    #slippage $10k (0-15pts) : combien ça coute de rentrer/sortir
    slip_10k = slippages.get(10_000, {}).get('avg', 100)
    slip_score = max(0, 15 * (1 - slip_10k / 1.0))

    #volume 24h (0-10pts) : activité
    vol_score = min(10, 1.43 * math.log10(max(1, volume_24h)))

    #OI (0-10pts) : interet ouvert = combien de capital est engagé
    oi_score = min(10, 1.43 * math.log10(max(1, oi)))

    total_score = spread_score + depth_score + depth5_score + slip_score + vol_score + oi_score

    #=== AFFICHAGE ===
    print(f"\n{'='*60}")
    print(f"  {market['symbol']}  —  Score: {total_score:.1f}/100")
    print(f"{'='*60}")
    print(f"  Prix mid:     ${mid:,.4f}")
    print(f"  Spread:       {spread_pct:.4f}%  ({spread_score:.1f}/25)")
    print()

    print(f"  Depth:")
    for pct in DEPTH_LEVELS:
        d = depths[pct]
        marker = ' <--' if pct == 1 else ''
        print(f"    ±{pct:>4}% :  asks ${d['ask']:>12,.0f}  |  bids ${d['bid']:>12,.0f}  |  total ${d['total']:>12,.0f}{marker}")
    print(f"    Score depth ±1%: {depth_score:.1f}/25, ±5%: {depth5_score:.1f}/15")
    print()

    print(f"  Slippage (achat market):")
    for size in SLIPPAGE_SIZES:
        s = slippages[size]
        fill_warn = '' if s['fill_buy'] >= 99.9 else f'  (fill: {s["fill_buy"]:.0f}%)'
        print(f"    ${size:>7,} :  buy {s['buy']:.4f}%  |  sell {s['sell']:.4f}%  |  avg {s['avg']:.4f}%{fill_warn}")
    print(f"    Score slip $10k: {slip_score:.1f}/15")
    print()

    print(f"  Volume 24h:   ${volume_24h:>14,.0f}  ({vol_score:.1f}/10)")
    print(f"  Trades 24h:   {trades_24h:>14,}")
    print(f"  OI:           ${oi:>14,.0f}  ({oi_score:.1f}/10)")
    print(f"{'='*60}\n")

    return {
        'symbol': market['symbol'],
        'score': total_score,
        'spread': spread_pct,
        'depth_1pct': d1,
        'depth_5pct': d5,
        'slip_10k': slip_10k,
        'volume_24h': volume_24h,
        'oi': oi,
    }


def rank_all(min_score=0):
    """rank toutes les paires par score de liquidité"""
    markets = get_all_markets()
    perps = [m for m in markets if m.get('market_type') == 'perp']
    print(f'{len(perps)} paires perp a analyser...\n')

    results = []
    for m in perps:
        try:
            r = get_liquidity_score(m['symbol'], markets)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  {m['symbol']}: erreur {e}")

    results.sort(key=lambda x: x['score'], reverse=True)
    print(f"\n{'='*80}")
    print(f"  RANKING LIQUIDITE LIGHTER ({len(results)} paires)")
    print(f"{'='*80}")
    print(f"  {'Rank':<5} {'Symbol':<12} {'Score':>6} {'Spread':>8} {'Depth±1%':>12} {'Depth±5%':>12} {'Slip$10k':>10} {'Vol24h':>14} {'OI':>12}")
    print(f"  {'-'*75}")
    for i, r in enumerate(results):
        if r['score'] < min_score:
            continue
        print(f"  {i+1:<5} {r['symbol']:<12} {r['score']:>5.1f} {r['spread']:>7.4f}% ${r['depth_1pct']:>10,.0f} ${r['depth_5pct']:>10,.0f} {r['slip_10k']:>9.4f}% ${r['volume_24h']:>12,.0f} ${r['oi']:>10,.0f}")
    print(f"{'='*80}")
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--rank':
        min_score = float(sys.argv[2]) if len(sys.argv) > 2 else 0
        rank_all(min_score)
    elif len(sys.argv) > 1:
        markets = get_all_markets()
        for sym in sys.argv[1:]:
            get_liquidity_score(sym, markets)
    else:
        #par defaut, quelques paires de test
        markets = get_all_markets()
        for sym in ['BTC', 'ETH', 'HYPE', 'YZY']:
            get_liquidity_score(sym, markets)
