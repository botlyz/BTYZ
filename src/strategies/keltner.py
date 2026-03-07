import numpy as np
import pandas as pd
from vectorbtpro import vbt

from config import FEES, INIT_CASH


def run_backtest(data, ma_window, atr_window, atr_mult, sl_stop=None, size=0.3):
    """Backtest Keltner Channel -- mean reversion sur EMA +/- ATR"""

    #Keltner = EMA +/- multiplicateur x ATR
    #EMA reagit plus vite qu'une SMA donc mieux pour le mean reversion
    ema = vbt.MA.run(data['close'], window=ma_window, wtype='Exp').ma
    atr = vbt.ATR.run(data['high'], data['low'], data['close'], window=atr_window).atr

    #shift de 1 bougie pour eviter le look-ahead bias
    #sinon on regarde la bougie actuelle pour decider d'entrer dessus = triche
    ema_shifted = ema.shift(1)
    atr_shifted = atr.shift(1)
    bande_inf = ema_shifted - atr_mult * atr_shifted #bande basse = zone de survente
    bande_sup = ema_shifted + atr_mult * atr_shifted #bande haute = zone de surachat

    #long quand le prix touche la bande inf (survente, on achete le retour)
    long_entries = data['low'] <= bande_inf
    long_exits = data['high'] >= ema_shifted #on sort quand ca revient a la moyenne

    #short quand le prix touche la bande sup (surachat, on vend le retour)
    short_entries = data['high'] >= bande_sup
    short_exits = data['low'] <= ema_shifted

    #prix d'execution custom pour etre realiste
    #on entre au prix de la bande, on sort au prix de l'EMA
    price = pd.Series(np.nan, index=data.index)
    price[long_exits] = ema_shifted[long_exits]
    price[short_exits] = ema_shifted[short_exits]
    price[long_entries] = bande_inf[long_entries]
    price[short_entries] = bande_sup[short_entries]

    pf_kwargs = dict(
        close=data['close'],
        long_entries=long_entries, long_exits=long_exits,
        short_entries=short_entries, short_exits=short_exits,
        price=price,
        open=data['open'], high=data['high'], low=data['low'],
        size=size, size_type='valuepercent',
        fees=FEES, init_cash=INIT_CASH, accumulate=False,
    )

    #sl_stop = stop loss en pourcentage, optionnel
    #ex: sl_stop=0.05 si le trade perd 5% on coupe
    if sl_stop is not None:
        pf_kwargs['sl_stop'] = sl_stop

    return vbt.Portfolio.from_signals(**pf_kwargs)
