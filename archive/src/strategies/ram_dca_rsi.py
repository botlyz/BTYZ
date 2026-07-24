"""
RAM DCA + filtre RSI neutre.
Identique à ram_dca.py mais n'entre que si RSI est dans une zone neutre (pas de momentum).
Nouveau paramètre : rsi_filter (0=off, 1=RSI60 40-60, 2=RSI80 42-58, 3=RSI120 42-58).
"""
import numpy as np
import pandas as pd
from numba import njit
from vectorbtpro import vbt

from config import FEES, INIT_CASH, SLIPPAGE

# Presets RSI validés par le scan alpha
RSI_PRESETS = {
    0: None,                    # off
    1: (60, 40.0, 60.0),        # RSI(60) 40-60
    2: (80, 42.0, 58.0),        # RSI(80) 42-58
    3: (120, 42.0, 58.0),       # RSI(120) 42-58
}


@njit(cache=True)
def _rsi_nb(close, period):
    """RSI calculé en Numba."""
    n = len(close)
    out = np.full(n, 50.0)
    if period >= n:
        return out
    # Seed : moyenne des gains/losses sur la première fenêtre
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        d = close[i] - close[i - 1]
        if d > 0:
            avg_gain += d
        else:
            avg_loss -= d
    avg_gain /= period
    avg_loss /= period
    if avg_loss == 0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    # Wilder smoothing
    for i in range(period + 1, n):
        d = close[i] - close[i - 1]
        gain = d if d > 0 else 0.0
        loss = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            out[i] = 100.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


@njit(cache=True)
def ram_dca_rsi_nb(high, low, close, ma, upper_envs, lower_envs,
                   allocations, sl_pct, can_trade):
    """
    RAM DCA avec filtre can_trade.
    Identique au noyau original — les exits/SL/cooldown fonctionnent même si can_trade=False.
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

    for i in range(1, n):
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

        if sl_cooldown:
            touched = low[i] <= ma_prev <= high[i]
            crossed = min(close[i - 1], close[i]) <= ma_prev <= max(close[i - 1], close[i])
            if touched or crossed:
                sl_cooldown = False
            else:
                continue

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

        if not can_trade[i]:
            continue

        tmp_dir   = pos_dir
        tmp_idx   = level_idx
        tmp_avg   = avg_entry
        tmp_qty   = qty
        traded    = False
        wp_sum    = 0.0
        alloc_sum = 0.0

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

        sl_hit = False
        sl_px  = 0.0
        if tmp_dir != 0:
            if tmp_dir == 1:
                sl_level = tmp_avg * (1.0 - sl_pct)
                if low[i] <= sl_level:
                    sl_hit = True
                    sl_px  = sl_level
            else:
                sl_level = tmp_avg * (1.0 + sl_pct)
                if high[i] >= sl_level:
                    sl_hit = True
                    sl_px  = sl_level

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

    return target_size, exec_price


def compute_rsi_filter(close, rsi_filter):
    """
    Retourne un array booléen can_trade basé sur le preset RSI.

    rsi_filter : int (0=off, 1=RSI60 40-60, 2=RSI80 42-58, 3=RSI120 42-58)
    """
    n = len(close)
    preset = RSI_PRESETS.get(int(rsi_filter))
    if preset is None:
        return np.ones(n, dtype=np.bool_)

    period, lo, hi = preset
    rsi_vals = _rsi_nb(close, period)
    return (rsi_vals > lo) & (rsi_vals < hi)


def run_backtest(data, ma_window, envelope_levels, allocations, sl_pct,
                 rsi_filter=0, ohlc4=False, leverage=1.0, freq=None):
    """
    Backtest RAM DCA + filtre RSI.

    Paramètres
    ----------
    data              : DataFrame OHLCV (index datetime)
    ma_window         : période de la MA
    envelope_levels   : liste de fractions pour les bandes
    allocations       : liste d'allocations par niveau
    sl_pct            : stop loss en fraction
    rsi_filter        : preset RSI (0=off, 1=RSI60 40-60, 2=RSI80 42-58, 3=RSI120 42-58)
    ohlc4             : si True, MA sur (O+H+L+C)/4
    leverage          : levier
    """
    assert len(envelope_levels) == len(allocations)

    if ohlc4:
        src = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    else:
        src = data['close']

    ma = vbt.MA.run(src, window=ma_window).ma

    n_levels = len(envelope_levels)
    n_bars   = len(data)

    upper_envs = np.zeros((n_levels, n_bars))
    lower_envs = np.zeros((n_levels, n_bars))
    for i, pct in enumerate(envelope_levels):
        upper_envs[i] = (ma * (1.0 + pct)).values
        lower_envs[i] = (ma * (1.0 - pct)).values

    can_trade = compute_rsi_filter(data['close'].values, rsi_filter)

    target_size, exec_price = ram_dca_rsi_nb(
        data['high'].values,
        data['low'].values,
        data['close'].values,
        ma.values,
        upper_envs,
        lower_envs,
        np.array(allocations, dtype=np.float64),
        float(sl_pct),
        can_trade,
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
