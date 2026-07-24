"""GAPFILL_v3 — GAPFILL_v1 + coût d'exécution OFF-HOURS par paire (conditions réelles).

Différence vs v1 : les fees ne sont plus un forfait global (--bps CLI) mais le
demi-spread off-hours mesuré sur les ticks Lighter de CHAQUE paire
(data/exec_costs_offhours.json, généré par scripts/build_exec_costs_offhours.py,
validé sur fills réels : AAPL est. 2.1 vs réel 1.87 bps). Le _target_slippage 2 bps
reste en plus (buffer latence/staleness). Hook moteur : `_load_pair_fees(pair)`
(wfa_runner + mccv_runner). Le --bps CLI ne sert que de fallback si paire absente.

Thèse (reformulée après audit) :
  Le perp Lighter trade 24/7 alors que le marché réel ferme (nuit + weekend). On a
  VÉRIFIÉ que la dérive close→open est du vrai price discovery (corr 0.98 avec l'open
  réel, overnight ET weekend) — donc on NE fade PAS la dérive directionnelle.
  Ce qu'on récolte = les MICRO-OSCILLATIONS de Lighter autour de sa trajectoire pendant
  la fermeture : le prix s'écarte de >entry_thresh du dernier close réel puis revient
  vers lui (<exit_thresh) AVANT la réouverture. Edge brut mince (~0.4%/trade) → viable
  uniquement à coût réel bas (taker Lighter ~2-3 bps), pas à 10 bps (marge intégrateur).

Données (loader engine.data_loader.load_gapfill, source="gapfill") :
  DataFrame indexé 5min, colonnes : open, high, low, close (= prix Lighter, ce qu'on
  trade), fair_value (dernier close réel CONNU, yfinance ffill), real_age_min (âge en
  min du dernier bar réel COMPLÉTÉ). Anti-look-ahead données géré dans le loader
  (yfinance décalé en fin de barre).

Anti-look-ahead exécution : décision calculée sur le bar i, exécutée au close du bar
  i+1 (lag 1 bar via `_lag`). Convention identique au kernel FUNDING_ARB_v1.

Params optimisés :
  - entry_thresh : écart |Lighter - fair| pour entrer en fade   [0.2% .. 2.0%]
  - exit_thresh  : écart sous lequel on considère le gap rempli  [0.05% .. 0.5%]
  - recent_min   : âge (min) du dernier bar réel au-delà duquel marché = fermé
                   (doit être > cadence yfinance 60min)          {90,120,150,180}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from vectorbtpro import vbt

from engine.strategy_interface import BaseStrategy
from engine.scoring import score_robust

# Runtime hints (override par moteur via _target_fees / _target_freq).
# _target_fees défaut = 2 bps = coût taker Lighter réel (PAS la marge intégrateur 10bps).
_target_fees: float = 0.0002
_target_slippage: float = 0.0002     # 2 bps (non overridé par le moteur)
_target_freq: str = "5min"
_init_cash: float = 10_000.0

# --- v2 : coût off-hours par paire (hook moteur) ---
_EXEC_COSTS_FILE = Path(__file__).resolve().parents[3] / "data" / "exec_costs_offhours.json"
_exec_costs_cache: dict | None = None


def _load_pair_fees(pair: str) -> float | None:
    """Demi-spread off-hours de la paire (fraction, par fill) — None si inconnue
    (le moteur retombe alors sur le --bps CLI)."""
    global _exec_costs_cache
    if _exec_costs_cache is None:
        try:
            with open(_EXEC_COSTS_FILE) as f:
                _exec_costs_cache = json.load(f)
        except Exception:
            _exec_costs_cache = {}
    info = _exec_costs_cache.get(pair)
    if not info:
        return None
    bps = float(info.get("half_spread_offhours_bps") or 0)
    return bps * 1e-4 if bps > 0 else None


def _lag(a: np.ndarray) -> np.ndarray:
    """Décale un signal booléen de 1 bar (décide [i-1], exécute close[i])."""
    return np.r_[False, a[:-1]]


class Strategy(BaseStrategy):
    # Indique au notebook d'analyse de charger via load_gapfill (fair_value + real_age_min)
    DATA_SOURCE = "gapfill"

    def param_space(self, trial) -> Dict[str, Any]:
        entry = trial.suggest_float("entry_thresh", 0.002, 0.02)
        # exit borné sous entry pour garder un edge brut positif
        exit_hi = max(0.0006, entry * 0.6)
        exit_thresh = trial.suggest_float("exit_thresh", 0.0005, exit_hi)
        recent_min = trial.suggest_categorical("recent_min", [90, 120, 150, 180])
        # sl_pct haut (0.10) ≈ pas de stop pour ces gaps → l'opti choisit si le SL aide
        sl_pct = trial.suggest_float("sl_pct", 0.01, 0.10)
        return {"entry_thresh": entry, "exit_thresh": exit_thresh,
                "recent_min": recent_min, "sl_pct": sl_pct}

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]):
        # Source de vérité UNIQUE = compute_target_arrays -> from_orders.
        # Garantit un BT 1:1 avec le replay global du notebook (qui appelle aussi
        # compute_target_arrays). Convention from_orders du framework (RAM/FUNDING).
        target, price = self.compute_target_arrays(data, params)
        if target is None or price is None:
            return None
        pf = vbt.Portfolio.from_orders(
            close=data["close"],
            size=target,
            price=price,
            size_type="TargetPercent",
            init_cash=_init_cash,
            leverage=1.0,
            fees=_target_fees,
            slippage=_target_slippage,
            freq=_target_freq,
        )
        return pf

    def _signals(self, data: pd.DataFrame, params: Dict[str, Any]):
        """Logique de signaux partagée (run_backtest + compute_target_arrays).

        Returns (long_entries, short_entries, exits, px, idx) — tous lagués 1 bar
        (décision [i-1], exécution close[i]) — ou None si données insuffisantes.
        """
        if data is None or len(data) < 200:
            return None
        if not {"close", "fair_value", "real_age_min"}.issubset(data.columns):
            return None

        entry_thresh = float(params["entry_thresh"])
        exit_thresh = float(params["exit_thresh"])
        recent_min = float(params["recent_min"])

        px = data["close"].to_numpy(dtype=float)
        fair = data["fair_value"].to_numpy(dtype=float)
        closed = data["real_age_min"].to_numpy(dtype=float) > recent_min
        dev = (px - fair) / fair  # >0 = Lighter au-dessus du fair value

        long_entries = _lag(closed & (dev < -entry_thresh))
        short_entries = _lag(closed & (dev > entry_thresh))
        gap_filled = np.abs(dev) < exit_thresh
        exits = _lag(gap_filled | (~closed))
        return long_entries, short_entries, exits, px, data.index

    def compute_target_arrays(self, data: pd.DataFrame, params: Dict[str, Any]):
        """Pour le replay VBT global du notebook (§4).

        Émet une cible TargetPercent en {-1, 0, +1} (1.0 = 100% long, -1.0 = short)
        UNIQUEMENT aux changements de position (entrée/sortie), NaN = hold sinon —
        convention from_orders du notebook (sinon rééquilibrage à chaque barre =
        sur-trading).

        Machine à états (= logique live exacte, toutes les 5 min) :
          1. STOP-LOSS sur la position courante : décision sur px[i-1] (lag 1 bar),
             coupe si le prix s'écarte de sl_pct du prix d'entrée. Déclenche un
             cooldown (pas de ré-entrée tant que le gap n'est pas rempli / réouvert)
             pour éviter le whipsaw immédiat.
          2. SORTIE standard : gap rempli (|dev|<exit) OU marché rouvert (déjà dans `exits`).
          3. ENTRÉE / RETOURNEMENT : sur signal opposé, on inverse directement
             (long↔short), sinon on ouvre depuis flat. Bloqué pendant le cooldown.
        Returns (target_size, exec_price) = (Series cible sparse, Series close Lighter).
        """
        sig = self._signals(data, params)
        if sig is None:
            return None, None
        long_e, short_e, exits, px, idx = sig
        sl_pct = float(params.get("sl_pct", 0.10))

        n = len(px)
        target = np.full(n, np.nan)
        s = 0
        entry_px = 0.0
        cooldown = False
        for i in range(n):
            new_s = s

            # 1) Stop-loss (décision sur px[i-1], exécution à i)
            if s != 0 and sl_pct > 0 and i >= 1:
                if (s == 1 and px[i - 1] <= entry_px * (1.0 - sl_pct)) or \
                   (s == -1 and px[i - 1] >= entry_px * (1.0 + sl_pct)):
                    new_s = 0
                    cooldown = True

            # 2) Sortie standard (gap rempli / réouverture)
            if new_s != 0 and exits[i]:
                new_s = 0

            # Release cooldown quand gap rempli ou marché rouvert
            if cooldown and exits[i]:
                cooldown = False

            # 3) Entrée / retournement (bloqué pendant cooldown)
            if not cooldown:
                if long_e[i]:
                    new_s = 1
                elif short_e[i]:
                    new_s = -1

            if new_s != s:
                target[i] = float(new_s)
                if new_s != 0:
                    entry_px = px[i]
                s = new_s
        return pd.Series(target, index=idx), pd.Series(px, index=idx)

    def score(self, metrics: Dict[str, Any]) -> float:
        return score_robust(metrics)
