"""
RAM DCA + filtre volatilité ATR.
Identique à ram_dca.py mais n'entre que si ATR(24h) > médiane rolling.
Nouveau paramètre : vol_window (fenêtre de la médiane rolling en jours, 0 = off).
"""
import numpy as np
import pandas as pd
from numba import njit
from vectorbtpro import vbt

from config import FEES, INIT_CASH, SLIPPAGE


@njit(cache=True)
def ram_dca_vol_nb(high, low, close, ma, upper_envs, lower_envs,
                   allocations, sl_pct, can_trade):
    """
    RAM DCA identique au noyau original mais avec un filtre can_trade.
    can_trade[i] = True → les entrées sont autorisées sur la bougie i.
    Les exits (TP, SL) et le cooldown fonctionnent normalement même si can_trade = False.
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

        # ── FILTRE VOL : skip les entrées si volatilité trop basse ──
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


def compute_vol_filter(high, low, close, vol_window_days, tf_minutes=5):
    """
    Calcule le masque can_trade basé sur ATR(24h) > médiane rolling.

    Paramètres
    ----------
    high, low, close : arrays 1D
    vol_window_days   : fenêtre de la médiane rolling en jours (0 = off = tout True)
    tf_minutes        : timeframe en minutes (5 pour 5min)

    Retourne
    --------
    can_trade : array booléen (True = vol suffisante pour trader)
    """
    n = len(close)

    if vol_window_days == 0:
        return np.ones(n, dtype=np.bool_)

    # ATR 24h (en bougies)
    atr_period = int(1440 / tf_minutes)  # 288 pour 5min

    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    # ATR rolling
    atr = np.full(n, np.nan)
    cs = np.cumsum(tr)
    for i in range(atr_period, n):
        atr[i] = (cs[i] - cs[i - atr_period]) / atr_period

    # Médiane rolling (fenêtre en bougies)
    med_window = vol_window_days * atr_period
    rolling_med = np.full(n, np.nan)
    for i in range(med_window, n):
        rolling_med[i] = np.median(atr[i - med_window:i])

    # can_trade = ATR actuel > médiane rolling
    can_trade = np.zeros(n, dtype=np.bool_)
    for i in range(med_window, n):
        if not np.isnan(atr[i]) and not np.isnan(rolling_med[i]):
            can_trade[i] = atr[i] > rolling_med[i]

    return can_trade


def run_backtest(data, ma_window, envelope_levels, allocations, sl_pct,
                 vol_window=0, ohlc4=False, leverage=1.0, freq=None):
    """
    Backtest RAM DCA + filtre volatilité.

    Paramètres
    ----------
    data              : DataFrame OHLCV (index datetime)
    ma_window         : période de la MA
    envelope_levels   : liste de fractions pour les bandes
    allocations       : liste d'allocations par niveau
    sl_pct            : stop loss en fraction
    vol_window        : fenêtre médiane ATR en jours (0 = off, 7/30/90)
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

    # Détecter le timeframe en minutes
    _delta = data.index.to_series().diff().dropna()
    _median_sec = _delta.dt.total_seconds().median()
    tf_minutes = int(_median_sec / 60)

    # Filtre vol
    can_trade = compute_vol_filter(
        data['high'].values, data['low'].values, data['close'].values,
        vol_window, tf_minutes,
    )

    target_size, exec_price = ram_dca_vol_nb(
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
