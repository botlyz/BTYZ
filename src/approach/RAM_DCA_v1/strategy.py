"""RAM_DCA — DCA mean-reversion multi-niveaux (3 niveaux).

Logique (kernel Numba ram_dca_nb):
- 3 bandes upper/lower SMA × (1 ± env_lvl_i)
- Long si low touche une bande inférieure, short si high touche une bande supérieure
- Multi-niveau DCA : ajoute à la position quand bande suivante touchée
- TP : retour à la MA
- SL : avg_entry × (1 ± sl_pct)
- Cooldown post-SL: attend retour à la MA avant nouvelle entrée

Params optimisés:
- ma_window
- env_lvl_1 / env_lvl_2 / env_lvl_3   (croissants : lvl_1 < lvl_2 < lvl_3)
- alloc_1 / alloc_2 / alloc_3         (allocations par niveau, somme normalisée à 1)
- sl_pct
- ohlc4
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from numba import njit
from vectorbtpro import vbt

from engine.strategy_interface import BaseStrategy

# Runtime hints — set by engine before each backtest
_target_fees: float = 0.0001
_target_slippage: float = 0.0002   # 2 bps default
_target_freq: str = "5min"
_init_cash: float = 10_000.0


@njit(cache=True)
def _ram_dca_nb(high, low, close, ma, upper_envs, lower_envs, allocations, sl_pct):
    n = len(close)
    n_levels = len(allocations)

    target_size = np.full(n, np.nan)
    exec_price = np.full(n, np.nan)

    pos_dir = 0
    level_idx = 0
    avg_entry = 0.0
    qty = 0.0
    sl_cooldown = False
    pending_sl = False
    pending_sl_px = 0.0

    for i in range(1, n):
        if pending_sl:
            target_size[i] = 0.0
            exec_price[i] = pending_sl_px
            pos_dir = 0
            level_idx = 0
            avg_entry = 0.0
            qty = 0.0
            sl_cooldown = True
            pending_sl = False
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
                exec_price[i] = ma_prev
                pos_dir = 0
                level_idx = 0
                avg_entry = 0.0
                qty = 0.0
                continue

        tmp_dir = pos_dir
        tmp_idx = level_idx
        tmp_avg = avg_entry
        tmp_qty = qty
        traded = False
        wp_sum = 0.0
        alloc_sum = 0.0

        if tmp_dir >= 0:
            while tmp_idx < n_levels:
                lim = lower_envs[tmp_idx, i - 1]
                alloc = allocations[tmp_idx]
                if low[i] <= lim:
                    prev_val = tmp_avg * tmp_qty
                    tmp_qty += alloc
                    tmp_avg = (prev_val + lim * alloc) / tmp_qty
                    wp_sum += lim * alloc
                    alloc_sum += alloc
                    tmp_idx += 1
                    tmp_dir = 1
                    traded = True
                else:
                    break

        if tmp_dir <= 0:
            while tmp_idx < n_levels:
                lim = upper_envs[tmp_idx, i - 1]
                alloc = allocations[tmp_idx]
                if high[i] >= lim:
                    prev_val = tmp_avg * tmp_qty
                    tmp_qty += alloc
                    tmp_avg = (prev_val + lim * alloc) / tmp_qty
                    wp_sum += lim * alloc
                    alloc_sum += alloc
                    tmp_idx += 1
                    tmp_dir = -1
                    traded = True
                else:
                    break

        sl_hit = False
        sl_px = 0.0
        if tmp_dir != 0:
            if tmp_dir == 1:
                sl_level = tmp_avg * (1.0 - sl_pct)
                if low[i] <= sl_level:
                    sl_hit = True
                    sl_px = sl_level
            else:
                sl_level = tmp_avg * (1.0 + sl_pct)
                if high[i] >= sl_level:
                    sl_hit = True
                    sl_px = sl_level

        if sl_hit and traded and pos_dir == 0:
            pos_dir = tmp_dir
            level_idx = tmp_idx
            avg_entry = tmp_avg
            qty = tmp_qty
            target_size[i] = qty if pos_dir == 1 else -qty
            exec_price[i] = wp_sum / alloc_sum if alloc_sum > 0 else np.nan
            pending_sl = True
            pending_sl_px = sl_px
        elif sl_hit:
            target_size[i] = 0.0
            exec_price[i] = sl_px
            pos_dir = 0
            level_idx = 0
            avg_entry = 0.0
            qty = 0.0
            sl_cooldown = True
        elif traded:
            pos_dir = tmp_dir
            level_idx = tmp_idx
            avg_entry = tmp_avg
            qty = tmp_qty
            target_size[i] = qty if pos_dir == 1 else -qty
            exec_price[i] = wp_sum / alloc_sum if alloc_sum > 0 else np.nan

    return target_size, exec_price


# Discrete envelope levels (% écart à la MA): 0.5%, 1%, 2%, …, 15%
ENVELOPE_LEVELS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                   0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
# Gap minimum entre 2 bandes consécutives (en fraction)
ENV_MIN_GAP = 0.01  # 1%
# Si pas trouvé en bornes hautes, Optuna replie au max → on évite l'index out of range


class Strategy(BaseStrategy):
    """RAM DCA — DCA mean-reversion 1-3 bandes avec hard SL + cooldown."""

    def param_space(self, trial) -> Dict[str, Any]:
        ma_window = trial.suggest_int("ma_window", 20, 200)
        n_bands_target = trial.suggest_int("n_bands", 1, 3)

        # Pick bandes croissantes — chaque suivante doit être au moins ENV_MIN_GAP au-dessus
        bands = []
        # Bande 1
        i1 = trial.suggest_int("band_1_idx", 0, len(ENVELOPE_LEVELS) - 1)
        bands.append(ENVELOPE_LEVELS[i1])

        # Bande 2 (si n_bands ≥ 2)
        if n_bands_target >= 2:
            min_j = next((j for j, lv in enumerate(ENVELOPE_LEVELS)
                          if lv >= bands[-1] + ENV_MIN_GAP - 1e-9), None)
            if min_j is not None and min_j < len(ENVELOPE_LEVELS):
                i2 = trial.suggest_int("band_2_idx", min_j, len(ENVELOPE_LEVELS) - 1)
                bands.append(ENVELOPE_LEVELS[i2])

        # Bande 3 (si n_bands ≥ 3 ET on a pu placer la 2)
        if n_bands_target >= 3 and len(bands) >= 2:
            min_k = next((k for k, lv in enumerate(ENVELOPE_LEVELS)
                          if lv >= bands[-1] + ENV_MIN_GAP - 1e-9), None)
            if min_k is not None and min_k < len(ENVELOPE_LEVELS):
                i3 = trial.suggest_int("band_3_idx", min_k, len(ENVELOPE_LEVELS) - 1)
                bands.append(ENVELOPE_LEVELS[i3])

        n_bands = len(bands)

        # Allocations (pas 10%) — chaque alloc ∈ {10%, 20%, …, 100%}
        # Contrainte : somme ≤ 100% (= 10 × 10%). Pas de normalisation.
        # On encode en entiers (1=10%, …, 10=100%) puis on cap dynamiquement
        # la suggestion pour garantir au moins 10% restant par bande restante.
        allocs_int: list[int] = []
        for i in range(n_bands):
            remaining_bands = n_bands - i - 1
            used = sum(allocs_int)
            # max pour cette bande = 10 - (somme déjà allouée) - (min réservé aux restantes)
            max_here = 10 - used - remaining_bands
            max_here = max(1, max_here)
            ai = trial.suggest_int(f"alloc_{i+1}_int", 1, max_here)
            allocs_int.append(ai)
        allocs = [round(a * 0.1, 2) for a in allocs_int]  # ex: 0.3, 0.2, 0.4

        sl_pct = round(trial.suggest_float("sl_pct", 0.01, 0.10, step=0.005), 4)

        return {
            "ma_window": ma_window,
            "n_bands": n_bands,
            "env_levels": bands,          # list[float], len = n_bands
            "allocations": allocs,        # list[float], somme=1, len = n_bands
            "sl_pct": sl_pct,
            "ohlc4": False,               # fixé — pas optimisé
        }

    def run_backtest(self, data, params):
        if data is None or len(data) < params["ma_window"] + 10:
            return None

        envelope_levels = list(params["env_levels"])
        allocations = np.array(params["allocations"], dtype=np.float64)
        n_levels = len(envelope_levels)
        if n_levels == 0 or len(allocations) != n_levels:
            return None

        if params.get("ohlc4"):
            src = (data["open"] + data["high"] + data["low"] + data["close"]) / 4
        else:
            src = data["close"]

        ma = vbt.MA.run(src, window=params["ma_window"]).ma

        n_levels = len(envelope_levels)
        n_bars = len(data)
        upper = np.zeros((n_levels, n_bars))
        lower = np.zeros((n_levels, n_bars))
        for i, pct in enumerate(envelope_levels):
            upper[i] = (ma * (1.0 + pct)).values
            lower[i] = (ma * (1.0 - pct)).values

        ts, px = _ram_dca_nb(
            data["high"].values, data["low"].values, data["close"].values,
            ma.values, upper, lower, allocations,
            float(params["sl_pct"]),
        )
        size_s = pd.Series(ts, index=data.index)
        price_s = pd.Series(px, index=data.index)

        pf = vbt.Portfolio.from_orders(
            close=data["close"],
            size=size_s,
            price=price_s,
            size_type="TargetPercent",
            init_cash=_init_cash,
            leverage=1.0,
            fees=_target_fees,
            slippage=_target_slippage,
            freq=_target_freq,
        )
        return pf
