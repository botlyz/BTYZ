import numpy as np
import pandas as pd
from numba import njit
from vectorbtpro import vbt

from config import FEES, INIT_CASH


@njit
def ram_dca_nb(high, low, close, ma, upper_envs, lower_envs, allocations, sl_pct):
    """
    DCA mean reversion multi-niveaux avec stop loss et cooldown post-SL.

    Logique :
    - Long  : prix touche une bande inférieure → entre avec l'allocation du niveau
    - Short : prix touche une bande supérieure → entre avec l'allocation du niveau
    - Sortie : prix revient à la MA (take profit)
    - SL    : si le prix dépasse avg_entry * (1 ± sl_pct) → clôture + cooldown
    - Cooldown : après un SL, bloque les entrées jusqu'au prochain contact avec la MA

    Paramètres
    ----------
    high, low, close : arrays 1D de prix OHLC
    ma               : moving average (même index que close)
    upper_envs       : (n_levels, n_bars) bandes hautes
    lower_envs       : (n_levels, n_bars) bandes basses
    allocations      : (n_levels,) fraction du capital par niveau (ex: [0.2, 0.3, 0.5])
    sl_pct           : stop loss en fraction (ex: 0.05 = 5%)

    Retourne
    --------
    target_size  : (n_bars,) taille cible en % du capital (nan = rien à faire)
    exec_price   : (n_bars,) prix d'exécution (nan = rien à faire)
    """
    n = len(close)
    n_levels = len(allocations)

    target_size = np.full(n, np.nan)
    exec_price  = np.full(n, np.nan)

    pos_dir       = 0     # 0: flat, 1: long, -1: short
    level_idx     = 0     # prochain niveau disponible
    avg_entry     = 0.0
    qty           = 0.0
    sl_cooldown   = False

    for i in range(1, n):
        ma_prev = ma[i - 1]
        if np.isnan(ma_prev):
            continue

        # ── Cooldown post-SL : attendre que le prix touche/traverse la MA ─────
        if sl_cooldown:
            touched = low[i] <= ma_prev <= high[i]
            crossed = min(close[i - 1], close[i]) <= ma_prev <= max(close[i - 1], close[i])
            if touched or crossed:
                sl_cooldown = False
            else:
                continue

        # ── Sortie (take profit au retour à la MA) ───────────────────────────
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

        # ── Entrées DCA multi-niveaux ────────────────────────────────────────
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

        # Short : bandes supérieures (seulement si pas passé long)
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

        # ── Stop loss (évalué sur le nouvel état) ────────────────────────────
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

        # ── Mise à jour finale ───────────────────────────────────────────────
        if sl_hit:
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


def run_backtest(data, ma_window, envelope_levels, allocations, sl_pct,
                 ohlc4=False, leverage=1.0, freq=None):
    """
    Backtest RAM DCA multi-niveaux.

    Paramètres
    ----------
    data            : DataFrame OHLCV (index datetime)
    ma_window       : période de la MA (SMA ou OHLC4 SMA)
    envelope_levels : liste de fractions pour les bandes (ex: [0.01, 0.02, 0.03])
    allocations     : liste d'allocations par niveau (ex: [0.2, 0.3, 0.5], somme <= 1)
    sl_pct          : stop loss en fraction (ex: 0.05)
    ohlc4           : si True, MA calculée sur (O+H+L+C)/4 plutôt que close
    leverage        : levier (1 = pas de levier)
    """
    assert len(envelope_levels) == len(allocations), "envelope_levels et allocations doivent avoir la même taille"

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

    target_size, exec_price = ram_dca_nb(
        data['high'].values,
        data['low'].values,
        data['close'].values,
        ma.values,
        upper_envs,
        lower_envs,
        np.array(allocations, dtype=np.float64),
        float(sl_pct),
    )

    size_s  = pd.Series(target_size, index=data.index)
    price_s = pd.Series(exec_price,  index=data.index)

    # Auto-détecter la fréquence depuis l'index si non fournie
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
        freq=freq,
    )
    return pf
