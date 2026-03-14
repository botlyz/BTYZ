"""telechargement de données OHLCV via CCXT (integré dans VBT PRO)"""
import os
import sys
from vectorbtpro import vbt
import datetime


def _save_symbol(data, symbol, output_dir):
    """sauvegarde un symbol en pickle + csv"""
    csv_path = f'{output_dir}/{symbol}.csv'
    pickle_path = f'{output_dir}/{symbol}.pickle'

    sym_data = data.select(symbol) if len(data.symbols) > 1 else data
    vbt.save(sym_data, pickle_path)

    #csv pour lisibilité et import notebooks
    df = sym_data.get()
    df.columns = [c.lower() for c in df.columns]
    df.index.name = 'date'
    df.index = df.index.astype('int64') // 10**6
    df.to_csv(csv_path)
    print(f'  {symbol} -> {csv_path} ({len(df)} lignes)')


def download_symbols(symbols, start='2017-01-01', end=None,
                     timeframe='1h', exchange='binance'):
    """telecharge les données OHLCV de plusieurs symboles en parallele
    utilise le threadpool natif de VBT pour fetch les paires en meme temps
    si le pickle existe deja, on update juste les nouvelles bougies"""

    if end is None:
        end = datetime.datetime.now().strftime('%Y-%m-%d')

    output_dir = f'data/raw/{exchange}/{timeframe}'
    os.makedirs(output_dir, exist_ok=True)

    #separer les symbols a update vs ceux a telecharger from scratch
    to_update = []
    to_download = []
    for sym in symbols:
        pickle_path = f'{output_dir}/{sym}.pickle'
        if os.path.exists(pickle_path):
            to_update.append(sym)
        else:
            to_download.append(sym)

    #update les pickles existants (1 par 1, chaque pickle a son propre state)
    for sym in to_update:
        pickle_path = f'{output_dir}/{sym}.pickle'
        print(f'Update {sym}...')
        data = vbt.load(pickle_path)
        data = data.update()
        _save_symbol(data, sym, output_dir)

    #premier download en batch avec threadpool VBT natif
    if to_download:
        print(f'Download {to_download} sur {exchange} ({start} -> {end}, {timeframe})...')
        data = vbt.CCXTData.pull(
            to_download,
            start=start,
            end=end,
            timeframe=timeframe,
            exchange=exchange,
            execute_kwargs=dict(engine='threadpool'),
        )
        #sauvegarder chaque symbol separement (pickle + csv)
        for sym in to_download:
            _save_symbol(data, sym, output_dir)

    print('Done.')


def download_symbol(symbol, start='2017-01-01', end=None,
                    timeframe='1h', exchange='binance'):
    """telecharge un seul symbol (wrapper pour compat)"""
    download_symbols([symbol], start=start, end=end,
                     timeframe=timeframe, exchange=exchange)


if __name__ == '__main__':
    #usage : python src/download.py BTCUSDT ETHUSDT
    #        python src/download.py --exchange bybit --tf 5m BTCUSDT SOLUSDT

    if len(sys.argv) < 2:
        print("Usage : python src/download.py [--exchange binance] [--tf 1h] <symbol1> <symbol2> ...")
        print("Ex : python src/download.py BTCUSDT ETHUSDT")
        print("Ex : python src/download.py --tf 5m BTCUSDT HYPEUSDT")
        print("Ex : python src/download.py --exchange bybit --tf 15m BTCUSDT SOLUSDT")
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
    print(f"Paires : {symbols}")

    download_symbols(symbols, exchange=exchange, timeframe=timeframe)
