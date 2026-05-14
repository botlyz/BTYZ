"""RAM_DCA_RSI — Single-envelope mean-reversion + RSI regime filter.

Variante simplifiée de RAM_DCA_v1 :
- **1 seule enveloppe** (pas de DCA multi-niveaux)
- **Allocation fixe 10%** du capital par ordre (TargetPercent = 0.10)
- **Filtre RSI** sélectionné par Optuna :
    * `rsi_mode = inside`  → trade seulement si RSI dans la zone neutre [100-level, level]
    * `rsi_mode = outside` → trade seulement si RSI hors de cette zone (zones extrêmes)
- **SL max 10%** (range 1-10%)
- **Sortie TP** : retour à la MA (comme RAM_DCA_v1)
- **Cooldown post-SL** : attend retour à la MA avant nouvelle entrée

Params optimisés:
- ma_window         : int [20, 200]
- env_pct           : choice [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] %
- sl_pct            : float [0.01, 0.10] step 0.005
- rsi_length        : choice [14, 30, 60, 90, 120]
- rsi_level         : choice [50, 55, 60, 65, 70]  (bande symétrique [100-level, level])
- rsi_mode          : choice ["inside", "outside"]
- ohlc4             : bool
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

# Grilles discrètes pour Optuna
ENV_PCT_CHOICES = [0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
                   0.06, 0.07, 0.08, 0.09, 0.10]
RSI_LENGTHS = [14, 30, 60, 90, 120]                # 5 longueurs
RSI_LEVELS  = [50, 55, 60, 65, 70]                 # 5 niveaux symétriques
RSI_MODES   = ["inside", "outside"]

ALLOC_FIXED = 0.10  # 10% du capital par ordre — fixe


@njit
def _rsi_nb(close, period):
    """RSI Wilder en Numba — retourne array float64 (0–100, 50 par défaut)."""
    n = len(close)
    out = np.full(n, 50.0)
    if period >= n:
        return out
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


@njit
def _ram_single_env_nb(high, low, close, ma, upper_env, lower_env,
                       sl_pct, alloc, can_trade):
    """Single-envelope mean reversion kernel.

    upper_env, lower_env : 1D arrays (1 seule bande)
    alloc                : float — taille fixe par ordre (TargetPercent)
    can_trade            : 1D bool array (RSI filter)
    """
    n = len(close)
    target_size = np.full(n, np.nan)
    exec_price  = np.full(n, np.nan)

    pos_dir       = 0      # 0=flat, +1=long, -1=short
    avg_entry     = 0.0
    sl_cooldown   = False
    pending_sl    = False
    pending_sl_px = 0.0

    for i in range(1, n):
        # Realise le SL différé du bar précédent
        if pending_sl:
            target_size[i] = 0.0
            exec_price[i]  = pending_sl_px
            pos_dir     = 0
            avg_entry   = 0.0
            sl_cooldown = True
            pending_sl  = False
            continue

        ma_prev = ma[i - 1]
        if np.isnan(ma_prev):
            continue

        # Cooldown : attend retour à la MA avant nouveau trade
        if sl_cooldown:
            touched = low[i] <= ma_prev <= high[i]
            crossed = min(close[i - 1], close[i]) <= ma_prev <= max(close[i - 1], close[i])
            if touched or crossed:
                sl_cooldown = False
            else:
                continue

        # Exit : retour à la MA (TP)
        if pos_dir != 0:
            exit_hit = (pos_dir == 1 and high[i] >= ma_prev) or \
                       (pos_dir == -1 and low[i] <= ma_prev)
            if exit_hit:
                target_size[i] = 0.0
                exec_price[i]  = ma_prev
                pos_dir   = 0
                avg_entry = 0.0
                continue

        # Filtre RSI : pas de nouvelle entrée si can_trade=False
        if not can_trade[i]:
            continue

        # Pas de nouvelle entrée si déjà en position (single env = pas de DCA)
        if pos_dir != 0:
            continue

        # Détection entrée — long si touche bande basse, short si touche bande haute
        lim_lo = lower_env[i - 1]
        lim_hi = upper_env[i - 1]
        traded = False
        new_dir = 0
        entry_px = 0.0

        if low[i] <= lim_lo:
            new_dir = 1
            entry_px = lim_lo
            traded = True
        elif high[i] >= lim_hi:
            new_dir = -1
            entry_px = lim_hi
            traded = True

        if not traded:
            continue

        # Check SL immédiat (rare mais possible si gap)
        if new_dir == 1:
            sl_level = entry_px * (1.0 - sl_pct)
            sl_hit = low[i] <= sl_level
        else:
            sl_level = entry_px * (1.0 + sl_pct)
            sl_hit = high[i] >= sl_level

        if sl_hit:
            # Entrée + SL même barre → on entre, et on planifie le close au bar suivant
            target_size[i] = alloc if new_dir == 1 else -alloc
            exec_price[i]  = entry_px
            pos_dir   = new_dir
            avg_entry = entry_px
            pending_sl    = True
            pending_sl_px = sl_level
        else:
            target_size[i] = alloc if new_dir == 1 else -alloc
            exec_price[i]  = entry_px
            pos_dir   = new_dir
            avg_entry = entry_px

    return target_size, exec_price


def _build_can_trade(close, rsi_length, rsi_level, rsi_mode):
    """Retourne array bool : True = permis de trader.

    inside  → trade si  (100-level) < RSI < level     (zone neutre)
    outside → trade si  RSI <= (100-level)  OR  RSI >= level   (zones extrêmes)
    """
    rsi = _rsi_nb(close, rsi_length)
    lo = 100.0 - rsi_level
    hi = rsi_level
    if rsi_mode == "inside":
        return (rsi > lo) & (rsi < hi)
    else:  # outside
        return (rsi <= lo) | (rsi >= hi)


class Strategy(BaseStrategy):
    """RAM_DCA_RSI — single-env mean-reversion + RSI inside/outside filter."""

    def score(self, metrics):
        from engine.scoring import score_robust
        return score_robust(metrics)

    def param_space(self, trial) -> Dict[str, Any]:
        # MA window discrétisé en pas de 10 → 20, 30, …, 200 (19 valeurs au lieu de 181)
        ma_window  = trial.suggest_int("ma_window", 20, 200, step=10)
        env_idx    = trial.suggest_int("env_idx", 0, len(ENV_PCT_CHOICES) - 1)
        env_pct    = ENV_PCT_CHOICES[env_idx]
        sl_pct     = round(trial.suggest_float("sl_pct", 0.01, 0.10, step=0.005), 4)

        rsi_len_idx = trial.suggest_int("rsi_len_idx", 0, len(RSI_LENGTHS) - 1)
        rsi_length  = RSI_LENGTHS[rsi_len_idx]

        rsi_lvl_idx = trial.suggest_int("rsi_level_idx", 0, len(RSI_LEVELS) - 1)
        rsi_level   = RSI_LEVELS[rsi_lvl_idx]

        rsi_mode = trial.suggest_categorical("rsi_mode", RSI_MODES)
        # ohlc4 fixé False — sortie de l'espace de recherche pour limiter le nombre de params

        return {
            "ma_window":  ma_window,
            "env_pct":    env_pct,
            "sl_pct":     sl_pct,
            "rsi_length": rsi_length,
            "rsi_level":  rsi_level,
            "rsi_mode":   rsi_mode,
            "ohlc4":      False,
        }

    def compute_target_arrays(self, data, params):
        if data is None or len(data) < params["ma_window"] + params["rsi_length"] + 10:
            return None, None

        if params.get("ohlc4"):
            src = (data["open"] + data["high"] + data["low"] + data["close"]) / 4
        else:
            src = data["close"]
        ma = vbt.MA.run(src, window=params["ma_window"]).ma
        env_pct = float(params["env_pct"])
        upper = (ma * (1.0 + env_pct)).values
        lower = (ma * (1.0 - env_pct)).values

        can_trade = _build_can_trade(
            data["close"].values,
            int(params["rsi_length"]),
            float(params["rsi_level"]),
            str(params["rsi_mode"]),
        )

        ts, px = _ram_single_env_nb(
            data["high"].values, data["low"].values, data["close"].values,
            ma.values, upper, lower,
            float(params["sl_pct"]), float(ALLOC_FIXED),
            can_trade,
        )
        return pd.Series(ts, index=data.index), pd.Series(px, index=data.index)

    def run_backtest(self, data, params):
        size_s, price_s = self.compute_target_arrays(data, params)
        if size_s is None:
            return None
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
