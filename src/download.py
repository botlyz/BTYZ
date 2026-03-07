"""telechargement de données OHLCV via CCXT (integré dans VBT PRO)"""
import os
from vectorbtpro import vbt


def download_symbol(symbol, start='2020-01-01', end='2026-03-01',
                    timeframe='1h', exchange='binance', output_dir='data/raw'):
    """telecharge les données OHLCV d'un symbole et sauvegarde en CSV
    utilise CCXT sous le capot via vbt.CCXTData"""

    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/{symbol}.csv'

    print(f'Telechargement {symbol} ({start} - {end}, {timeframe})...')
    data = vbt.CCXTData.pull(
        symbol,
        start=start,
        end=end,
        timeframe=timeframe,
        exchange=exchange,
    )
    df = data.get() #recuperer le dataframe
    df.to_csv(output_path)
    print(f'Sauvegardé dans {output_path} ({len(df)} lignes)')
    return df


if __name__ == '__main__':
    #telecharger les 3 symboles qu'on utilise
    for symbol in ['BTCUSDT', 'ETHUSDT', 'HYPEUSDT']:
        download_symbol(symbol)
