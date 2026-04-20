"""
Enveloppe ATR — mean reversion avec SL dynamique ATR.
Enveloppe = SMA ± atr_mult × ATR
SL = avg_entry ± sl_mult × ATR (dynamique, basé sur l'ATR au moment de l'entrée)
Cooldown post-SL : attend que le prix retouche la MA avant de re-trader.

Paramètres :
- ma_window   : période de la SMA
- atr_window  : période de l'ATR
- atr_mult    : multiplicateur ATR pour l'enveloppe
- sl_mult     : multiplicateur ATR pour le SL
"""
import numpy as np
import pandas as pd
from numba import njit
from vectorbtpro import vbt

from config import FEES, INIT_CASH, SLIPPAGE


@njit(cache=True)
def atr_envelope_nb(high, low, close, ma, upper_envs, lower_envs,
                    allocations, sl_mult, atr_vals):
    """
    Mean reversion avec enveloppe ATR + SL dynamique ATR + cooldown.

    Paramètres
    ----------
    high, low, close : arrays 1D OHLC
    ma               : moving average
    upper_envs       : (n_levels, n_bars) bandes hautes (MA + atr_mult × ATR)
    lower_envs       : (n_levels, n_bars) bandes basses (MA - atr_mult × ATR)
    allocations      : (n_levels,) fraction du capital par niveau
    sl_mult          : multiplicateur ATR pour le stop loss
    atr_vals         : ATR à chaque bougie

    Retourne
    --------
    target_size  : (n_bars,) taille cible en % du capital
    exec_price   : (n_bars,) prix d'exécution
    """
    n = len(close)
    n_levels = len(allocations)

    target_size = np.full(n, np.nan)
    exec_price  = np.full(n, np.nan)

    pos_dir       = 0
    level_idx     = 0
    avg_entry     = 0.0
    qty           = 0.0
    sl_cooldown   = False
    pending_sl    = False
    pending_sl_px = 0.0
    entry_atr     = 0.0

    for i in range(1, n):
        # SL différé
        if pending_sl:
            target_size[i] = 0.0
            exec_price[i]  = pending_sl_px
            pos_dir     = 0
            level_idx   = 0
            avg_entry   = 0.0
            qty         = 0.0
            sl_cooldown = True
            pending_sl  = False
            continue

        ma_prev = ma[i - 1]
        if np.isnan(ma_prev):
            continue

        # Cooldown post-SL
        if sl_cooldown:
            touched = low[i] <= ma_prev <= high[i]
            crossed = min(close[i - 1], close[i]) <= ma_prev <= max(close[i - 1], close[i])
            if touched or crossed:
                sl_cooldown = False
            else:
                continue

        # TP : retour à la MA
        if pos_dir != 0:
            exit_hit = (pos_dir == 1 and high[i] >= ma_prev) or \
                       (pos_dir == -1 and low[i] <= ma_prev)
            if exit_hit:
                target_size[i] = 0.0
                exec_price[i]  = ma_prev
                pos_dir = 0
                level_idx = 0
                avg_entry = 0.0
                qty = 0.0
                continue

        # Entrées DCA multi-niveaux
        tmp_dir   = pos_dir
        tmp_idx   = level_idx
        tmp_avg   = avg_entry
        tmp_qty   = qty
        traded    = False
        wp_sum    = 0.0
        alloc_sum = 0.0

        # Long : bandes inférieures
        if tmp_dir >= 0:
            while tmp_idx < n_levels:
                lim = lower_envs[tmp_idx, i - 1]
                alloc = allocations[tmp_idx]
                if low[i] <= lim:
                    prev_val = tmp_avg * tmp_qty
                    tmp_qty  += alloc
                    tmp_avg  = (prev_val + lim * alloc) / tmp_qty
                    wp_sum   += lim * alloc
                    alloc_sum += alloc
                    tmp_idx  += 1
                    tmp_dir   = 1
                    traded    = True
                else:
                    break

        # Short : bandes supérieures
        if tmp_dir <= 0:
            while tmp_idx < n_levels:
                lim = upper_envs[tmp_idx, i - 1]
                alloc = allocations[tmp_idx]
                if high[i] >= lim:
                    prev_val = tmp_avg * tmp_qty
                    tmp_qty  += alloc
                    tmp_avg  = (prev_val + lim * alloc) / tmp_qty
                    wp_sum   += lim * alloc
                    alloc_sum += alloc
                    tmp_idx  += 1
                    tmp_dir   = -1
                    traded    = True
                else:
                    break

        # SL dynamique basé sur ATR
        sl_hit = False
        sl_px  = 0.0
        if tmp_dir != 0:
            _atr = atr_vals[i - 1] if not np.isnan(atr_vals[i - 1]) else 0.0
            if traded and pos_dir == 0:
                entry_atr = _atr

            _sl_dist = sl_mult * (entry_atr if entry_atr > 0 else _atr)

            if tmp_dir == 1:
                sl_level = tmp_avg - _sl_dist
                if low[i] <= sl_level:
                    sl_hit = True
                    sl_px  = sl_level
            else:
                sl_level = tmp_avg + _sl_dist
                if high[i] >= sl_level:
                    sl_hit = True
                    sl_px  = sl_level

        # Mise à jour finale
        if sl_hit and traded and pos_dir == 0:
            pos_dir   = tmp_dir
            level_idx = tmp_idx
            avg_entry = tmp_avg
            qty       = tmp_qty
            target_size[i] = qty if pos_dir == 1 else -qty
            exec_price[i]  = wp_sum / alloc_sum if alloc_sum > 0 else np.nan
            pending_sl    = True
            pending_sl_px = sl_px
        elif sl_hit:
            target_size[i] = 0.0
            exec_price[i]  = sl_px
            pos_dir     = 0
            level_idx   = 0
            avg_entry   = 0.0
            qty         = 0.0
            sl_cooldown = True
        elif traded:
            pos_dir   = tmp_dir
            level_idx = tmp_idx
            avg_entry = tmp_avg
            qty       = tmp_qty
            target_size[i] = qty if pos_dir == 1 else -qty
            exec_price[i]  = wp_sum / alloc_sum if alloc_sum > 0 else np.nan
            if pos_dir == 0:
                entry_atr = atr_vals[i - 1] if not np.isnan(atr_vals[i - 1]) else 0.0

    return target_size, exec_price


def run_backtest(data, ma_window, atr_window, atr_mult, sl_mult,
                 ohlc4=False, leverage=1.0, freq=None):
    """
    Backtest enveloppe ATR avec SL dynamique.

    Paramètres
    ----------
    data        : DataFrame OHLCV (index datetime)
    ma_window   : période de la SMA
    atr_window  : période de l'ATR
    atr_mult    : multiplicateur ATR pour l'enveloppe
    sl_mult     : multiplicateur ATR pour le SL
    """
    if ohlc4:
        src = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    else:
        src = data['close']

    ma = vbt.MA.run(src, window=ma_window).ma
    atr = vbt.ATR.run(data['high'], data['low'], data['close'], window=atr_window).atr

    n_bars = len(data)

    upper_envs = np.zeros((1, n_bars))
    lower_envs = np.zeros((1, n_bars))
    upper_envs[0] = (ma + atr_mult * atr).values
    lower_envs[0] = (ma - atr_mult * atr).values

    target_size, exec_price = atr_envelope_nb(
        data['high'].values,
        data['low'].values,
        data['close'].values,
        ma.values,
        upper_envs,
        lower_envs,
        np.array([1.0], dtype=np.float64),
        float(sl_mult),
        atr.values,
    )

    size_s  = pd.Series(target_size, index=data.index)
    price_s = pd.Series(exec_price,  index=data.index)

    if freq is None:
        _delta = data.index.to_series().diff().dropna()
        _median_sec = _delta.dt.total_seconds().median()
        _freq_map = {60: '1min', 180: '3min', 300: '5min', 900: '15min',
                     1800: '30min', 3600: '1h', 7200: '2h', 14400: '4h', 86400: '1D'}
        freq = _freq_map.get(int(_median_sec), f'{int(_median_sec)}s')

    pf = vbt.Portfolio.from_orders(
        close=data['close'],
        size=size_s,
        price=price_s,
        size_type='TargetPercent',
        init_cash=INIT_CASH,
        leverage=leverage,
        fees=FEES,
        slippage=SLIPPAGE,
        freq=freq,
    )
    return pf
