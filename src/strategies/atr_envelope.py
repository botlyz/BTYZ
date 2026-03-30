import numpy as np
import pandas as pd
from vectorbtpro import vbt

from config import FEES, INIT_CASH


def run_backtest(data, ma_window, atr_window, atr_mult, size=0.3):
    """Backtest Enveloppe ATR -- mean reversion sur SMA +/- ATR"""

    #Enveloppe ATR = SMA +/- multiplicateur x ATR
    #pareil que Keltner mais avec une SMA (plus lisse, moins reactif)
    sma = vbt.MA.run(data['close'], window=ma_window).ma
    atr = vbt.ATR.run(data['high'], data['low'], data['close'], window=atr_window).atr

    #shift pour pas tricher (look-ahead bias)
    sma_shifted = sma.shift(1)
    atr_shifted = atr.shift(1)
    bande_inf = sma_shifted - atr_mult * atr_shifted
    bande_sup = sma_shifted + atr_mult * atr_shifted

    #--- Signaux Long ---
    long_entries = data['low'] <= bande_inf #prix touche la bande basse, achat
    long_exits = data['high'] >= sma_shifted #retour a la moyenne, on sort

    #--- Signaux Short ---
    short_entries = data['high'] >= bande_sup #prix touche la bande haute, vente
    short_exits = data['low'] <= sma_shifted #retour a la moyenne, on sort

    #prix d'execution custom
    price = pd.Series(np.nan, index=data.index)
    price[long_exits] = sma_shifted[long_exits]
    price[short_exits] = sma_shifted[short_exits]
    price[long_entries] = bande_inf[long_entries]
    price[short_entries] = bande_sup[short_entries]

    pf = vbt.Portfolio.from_signals(
        data['close'],
        long_entries=long_entries, long_exits=long_exits,
        short_entries=short_entries, short_exits=short_exits,
        price=price,
        open=data['open'], high=data['high'], low=data['low'],
        size=size, size_type='valuepercent',
        fees=FEES, init_cash=INIT_CASH, accumulate=False,
    )
    return pf
