"""telechargement de données OHLCV
- exchanges classiques (binance etc) : via VBT/CCXT (import lourd)
- lighter : API directe + 100 proxys async, worker pool (import léger)
"""
import os
import sys
import time
import datetime


#=== CONFIG ===
PROXY_URL = os.environ.get("PROXY_URL", "")
LIGHTER_API = "https://mainnet.zklighter.elliot.ai/api/v1"
LIGHTER_START = 1737100800  #17 jan 2025 en timestamp secondes
MAX_CANDLES = 500  #max par requete lighter
RATE_LIMIT = 1.0  #1 req/s par proxy


#================================================================
# LIGHTER - worker pool async (1 worker = 1 proxy = 1 req/s)
#================================================================

def load_proxies():
    """telecharge la liste de proxys depuis webshare"""
    import urllib.request
    resp = urllib.request.urlopen(PROXY_URL)
    lines = resp.read().decode().strip().split('\n')
    proxies = []
    for line in lines:
        ip, port, user, pw = line.strip().split(':')
        proxies.append(f'http://{user}:{pw}@{ip}:{port}')
    print(f'{len(proxies)} proxys chargés')
    return proxies


def tf_to_seconds(tf):
    """convertit un timeframe en secondes (1m=60, 5m=300, 1h=3600 etc)"""
    units = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
    return int(tf[:-1]) * units[tf[-1]]


def _print_progress(progress, msg=''):
    """affiche la barre de progression globale"""
    done = progress['done']
    total = progress['total']
    pct = done / total * 100 if total > 0 else 0
    elapsed = time.time() - progress['start']

    if done > 0 and done < total:
        eta = elapsed / done * (total - done)
        eta_str = f'ETA {int(eta)}s'
    elif done >= total:
        eta_str = f'total {int(elapsed)}s'
    else:
        eta_str = '...'

    bar_len = 30
    filled = int(bar_len * done / total) if total > 0 else 0
    bar = '█' * filled + '░' * (bar_len - filled)

    print(f'\r  [{bar}] {done}/{total} ({pct:.0f}%) {eta_str} | {msg:<50}', end='', flush=True)
    if done >= total:
        print()


async def download_lighter_all(tf='1h', symbols=None):
    """telecharge TOUTES les paires futures de lighter en async
    architecture : worker pool, chaque worker a son proxy et fait 1 req/s
    une queue globale de jobs (market_id, chunk_start, chunk_end) est consommée par les workers"""
    import asyncio
    import aiohttp
    import pandas as pd

    #recup la liste des marchés perp
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{LIGHTER_API}/orderBooks') as resp:
            data = await resp.json()

    markets = [(ob['market_id'], ob['symbol'])
               for ob in data['order_books']
               if ob.get('market_type') == 'perp'
               and (symbols is None or ob['symbol'] in symbols)]
    print(f'{len(markets)} marchés perp trouvés sur Lighter')

    output_dir = f'data/raw/lighter/{tf}'
    os.makedirs(output_dir, exist_ok=True)
    proxies = load_proxies()

    tf_secs = tf_to_seconds(tf)
    end_ts = int(time.time())
    chunk_size = MAX_CANDLES * tf_secs

    #preparer tous les jobs : (market_id, symbol, chunk_idx, start, end)
    #et les structures pour stocker les resultats
    market_chunks = {}  #symbol -> {total_chunks, existing_df, candles: [None]*n}
    jobs = []

    for market_id, symbol in markets:
        csv_path = f'{output_dir}/{symbol}.csv'
        start_ts = LIGHTER_START
        existing_df = None

        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            if len(existing_df) > 0:
                last_ts = int(existing_df['date'].iloc[-1]) // 1000
                start_ts = last_ts + tf_secs

        if start_ts >= end_ts:
            market_chunks[symbol] = {'skip': True}
            continue

        chunks = []
        t = start_ts
        while t < end_ts:
            chunk_end = min(t + chunk_size, end_ts)
            chunks.append((t, chunk_end))
            t = chunk_end

        market_chunks[symbol] = {
            'skip': False,
            'existing_df': existing_df,
            'candles': [None] * len(chunks),
            'total': len(chunks),
        }

        for idx, (cs, ce) in enumerate(chunks):
            jobs.append((market_id, symbol, idx, cs, ce))

    n_skipped = sum(1 for v in market_chunks.values() if v.get('skip'))
    n_to_dl = len(markets) - n_skipped
    print(f'{n_to_dl} paires a télécharger, {n_skipped} deja a jour, {len(jobs)} requetes a faire')

    if not jobs:
        print('Rien a faire.')
        return

    #progress : on track les paires terminées (pas les requetes)
    progress = {
        'done': 0, 'total': n_to_dl,
        'start': time.time(),
    }
    #compteur de chunks done par symbol pour savoir quand sauvegarder
    chunks_done = {s: 0 for s in market_chunks if not market_chunks[s].get('skip')}
    total_candles = 0
    lock = asyncio.Lock()

    def _save_symbol(symbol):
        """assemble et sauvegarde un symbol dès que tous ses chunks sont finis"""
        nonlocal total_candles
        info = market_chunks[symbol]
        rows = []
        for chunk_candles in info['candles']:
            if chunk_candles:
                for c in chunk_candles:
                    rows.append({
                        'date': c['t'],
                        'open': c['o'],
                        'high': c['h'],
                        'low': c['l'],
                        'close': c['c'],
                        'volume': c['v'],
                    })

        if not rows:
            return

        new_df = pd.DataFrame(rows)
        new_df = new_df.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)

        existing_df = info.get('existing_df')
        if existing_df is not None and len(existing_df) > 0:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
        else:
            combined = new_df

        csv_path = f'{output_dir}/{symbol}.csv'
        combined.to_csv(csv_path, index=False)
        total_candles += len(new_df)

    #queue de jobs
    queue = asyncio.Queue()
    for job in jobs:
        await queue.put(job)

    async def worker(proxy, session):
        """1 worker = 1 proxy, consomme la queue, 1 req/s"""
        while True:
            try:
                market_id, symbol, chunk_idx, start, end = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            url = f'{LIGHTER_API}/candles'
            params = {
                'market_id': market_id,
                'resolution': tf,
                'start_timestamp': start,
                'end_timestamp': end,
                'count_back': MAX_CANDLES,
            }

            candles = []
            for attempt in range(5):
                try:
                    async with session.get(url, params=params, proxy=proxy,
                                           timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 429:
                            await asyncio.sleep(2 + attempt * 2)
                            continue
                        data = await resp.json()
                        if data.get('code') == 200:
                            candles = data.get('c', [])
                        break
                except Exception:
                    await asyncio.sleep(1 + attempt)

            market_chunks[symbol]['candles'][chunk_idx] = candles

            #checker si tous les chunks de ce symbol sont finis
            async with lock:
                chunks_done[symbol] += 1
                if chunks_done[symbol] >= market_chunks[symbol]['total']:
                    #tous les chunks sont la, sauvegarder immediatement
                    _save_symbol(symbol)
                    progress['done'] += 1
                    _print_progress(progress, f'{symbol} sauvegardé')

            await asyncio.sleep(RATE_LIMIT)

    #lancer les workers (1 par proxy)
    connector = aiohttp.TCPConnector(limit=len(proxies), ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [worker(proxy, session) for proxy in proxies]
        await asyncio.gather(*workers)

    elapsed = time.time() - progress['start']
    print(f'\nTerminé : {n_to_dl} paires, {total_candles} candles, {n_skipped} deja a jour ({int(elapsed)}s)')


#================================================================
# CCXT/VBT - pour les exchanges classiques (binance, bybit etc)
#================================================================

def _save_symbol_vbt(data, symbol, output_dir):
    """sauvegarde un symbol vbt en pickle + csv"""
    from vectorbtpro import vbt

    csv_path = f'{output_dir}/{symbol}.csv'
    pickle_path = f'{output_dir}/{symbol}.pickle'

    sym_data = data.select(symbol) if len(data.symbols) > 1 else data
    vbt.save(sym_data, pickle_path)

    df = sym_data.get()
    df.columns = [c.lower() for c in df.columns]
    df.index.name = 'date'
    df.index = df.index.astype('int64') // 10**6
    df.to_csv(csv_path)
    print(f'  {symbol} -> {csv_path} ({len(df)} lignes)')


def download_symbols(symbols, start='2017-01-01', end=None,
                     timeframe='1h', exchange='binance'):
    """telecharge via VBT/CCXT (binance, bybit etc)"""
    from vectorbtpro import vbt

    if end is None:
        end = datetime.datetime.now().strftime('%Y-%m-%d')

    output_dir = f'data/raw/{exchange}/{timeframe}'
    os.makedirs(output_dir, exist_ok=True)

    to_update = []
    to_download = []
    for sym in symbols:
        pickle_path = f'{output_dir}/{sym}.pickle'
        if os.path.exists(pickle_path):
            to_update.append(sym)
        else:
            to_download.append(sym)

    for sym in to_update:
        pickle_path = f'{output_dir}/{sym}.pickle'
        print(f'Update {sym}...')
        data = vbt.load(pickle_path)
        data = data.update()
        _save_symbol_vbt(data, sym, output_dir)

    if to_download:
        print(f'Download {to_download} sur {exchange} ({start} -> {end}, {timeframe})...')
        data = vbt.CCXTData.pull(
            to_download,
            start=start, end=end,
            timeframe=timeframe, exchange=exchange,
            execute_kwargs=dict(engine='threadpool'),
        )
        for sym in to_download:
            _save_symbol_vbt(data, sym, output_dir)

    print('Done.')


#================================================================
# CLI
#================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage :")
        print("  python src/download.py --exchange lighter --tf 1h")
        print("  python src/download.py --exchange binance --tf 1h BTCUSDT ETHUSDT")
        print("  python src/download.py --tf 5m BTCUSDT  (binance par defaut)")
        sys.exit(1)

    exchange = 'binance'
    timeframe = '1h'
    symbols = []

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--exchange':
            exchange = args[i + 1]
            i += 2
        elif args[i] == '--tf':
            timeframe = args[i + 1]
            i += 2
        else:
            symbols.append(args[i])
            i += 1

    print(f"Exchange : {exchange}")
    print(f"Timeframe : {timeframe}")

    if exchange == 'lighter':
        print("Mode Lighter : download async avec proxys")
        import asyncio
        asyncio.run(download_lighter_all(tf=timeframe, symbols=symbols if symbols else None))
    else:
        if not symbols:
            print("Erreur : il faut au moins 1 symbol pour les exchanges classiques")
            sys.exit(1)
        print(f"Paires : {symbols}")
        download_symbols(symbols, exchange=exchange, timeframe=timeframe)
