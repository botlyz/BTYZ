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


class Strategy(BaseStrategy):
    """RAM DCA — 3-level mean-reversion DCA with hard SL + cooldown."""

    def param_space(self, trial) -> Dict[str, Any]:
        ma_window = trial.suggest_int("ma_window", 20, 200)
        # Bandes croissantes — on tire un base level puis deux multiplicateurs
        env_lvl_1 = round(trial.suggest_float("env_lvl_1", 0.003, 0.025, step=0.001), 4)
        env_step_2 = round(trial.suggest_float("env_step_2", 0.003, 0.025, step=0.001), 4)
        env_step_3 = round(trial.suggest_float("env_step_3", 0.003, 0.030, step=0.001), 4)
        env_lvl_2 = round(env_lvl_1 + env_step_2, 4)
        env_lvl_3 = round(env_lvl_2 + env_step_3, 4)
        # Allocations relatives (somme normalisée)
        a1 = trial.suggest_float("alloc_1_raw", 0.1, 1.0, step=0.05)
        a2 = trial.suggest_float("alloc_2_raw", 0.1, 1.0, step=0.05)
        a3 = trial.suggest_float("alloc_3_raw", 0.1, 1.0, step=0.05)
        total = a1 + a2 + a3
        alloc = [round(a1 / total, 4), round(a2 / total, 4), round(a3 / total, 4)]
        sl_pct = round(trial.suggest_float("sl_pct", 0.01, 0.10, step=0.005), 4)
        ohlc4 = trial.suggest_categorical("ohlc4", [False, True])
        return {
            "ma_window": ma_window,
            "env_lvl_1": env_lvl_1,
            "env_lvl_2": env_lvl_2,
            "env_lvl_3": env_lvl_3,
            "alloc_1": alloc[0],
            "alloc_2": alloc[1],
            "alloc_3": alloc[2],
            "sl_pct": sl_pct,
            "ohlc4": ohlc4,
        }

    def run_backtest(self, data, params):
        if data is None or len(data) < params["ma_window"] + 10:
            return None

        envelope_levels = [params["env_lvl_1"], params["env_lvl_2"], params["env_lvl_3"]]
        allocations = np.array(
            [params["alloc_1"], params["alloc_2"], params["alloc_3"]],
            dtype=np.float64,
        )

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
            slippage=0.0,
            freq=_target_freq,
        )
        return pf
