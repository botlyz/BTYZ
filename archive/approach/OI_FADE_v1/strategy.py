"""OI_FADE_v1 — fade du pump surpeuplé (OI Binance croissant + move intense).

Thèse (validée en screen, t=2.3-3.0 dédup) : un move violent ACCOMPAGNÉ d'une
expansion d'open interest = les retardataires s'empilent au sommet -> reversion
sur 24-48h quand ils se font secouer. Le payeur = le FOMO tardif.

Signal (1h, source "oi" = OHLCV Lighter + OI Binance) :
  doi   = variation OI sur doi_w barres ; r = ret prix sur doi_w barres
  z-scores ROLLING (past-only, fenêtre 30j) -> inten = max(z_doi,0) * |z_r|
  entrée SHORT quand inten > quantile rolling(30j, pctl) ET r > 0 (pump)
  entrée LONG symétrique optionnelle (dump surpeuplé) — désactivée : 96% des
  événements sont des pumps, le long n'a rien montré au screen.
  Sortie : horizon fixe hold barres (td_stop) + SL dur sl_pct.

Params optimisés : doi_w, pctl, hold, sl_pct. Fenêtres z/quantile FIXES (30j).
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np, pandas as pd
from vectorbtpro import vbt
from engine.strategy_interface import BaseStrategy

_target_fees: float = 0.0003
_target_slippage: float = 0.0002
_target_freq: str = "1h"
_init_cash: float = 10_000.0

ROLL = 240          # 10j en barres 1h : z-scores + quantile (past-only).
# NB : le moteur score les folds sur la tranche test SEULE (30j=720 barres 1h),
# une fenêtre rolling 30j y serait NaN partout -> 10j max structurel.


class Strategy(BaseStrategy):
    DATA_SOURCE = "oi"          # marimo : charger via engine.data_loader.load_oi

    def score(self, metrics):
        from engine.scoring import score_robust
        return score_robust(metrics)

    def param_space(self, trial) -> Dict[str, Any]:
        return {
            "doi_w": trial.suggest_int("doi_w", 2, 8, step=2),
            "pctl": round(trial.suggest_float("pctl", 0.85, 0.99, step=0.02), 2),
            "hold": trial.suggest_int("hold", 12, 60, step=12),
            "sl_pct": round(trial.suggest_float("sl_pct", 0.04, 0.16, step=0.04), 2),
        }

    def _signals(self, data, params):
        c, oi = data["close"], data["oi"]
        w = int(params["doi_w"])
        doi = oi.pct_change(w)
        r = c.pct_change(w)
        zd = (doi - doi.rolling(ROLL).mean()) / doi.rolling(ROLL).std()
        zr = (r - r.rolling(ROLL).mean()) / r.rolling(ROLL).std()
        inten = zd.clip(lower=0) * zr.abs()
        thr = inten.rolling(ROLL).quantile(float(params["pctl"]))
        se = ((inten > thr) & (r > 0) & (doi > 0)).fillna(False)
        return se

    def run_backtest(self, data, params):
        if data is None or "oi" not in data.columns or len(data) < ROLL + 100:
            return None
        se = self._signals(data, params)
        if not se.any():
            return None
        return vbt.Portfolio.from_signals(
            open=data["open"], high=data["high"], low=data["low"], close=data["close"],
            short_entries=se,
            td_stop=pd.Timedelta(hours=int(params["hold"])),
            sl_stop=float(params["sl_pct"]),
            fees=_target_fees, slippage=_target_slippage,
            init_cash=_init_cash, freq=_target_freq)

    def compute_target_arrays(self, data, params):
        """Replay marimo : size sparse (-1 à l'entrée, 0 à la sortie hold/SL)."""
        if data is None or "oi" not in data.columns or len(data) < ROLL + 100:
            return None, None
        se = self._signals(data, params).to_numpy()
        c = data["close"].to_numpy(); hi = data["high"].to_numpy()
        hold = int(params["hold"]); sl = float(params["sl_pct"])
        n = len(c); size = np.full(n, np.nan)
        i = 0
        while i < n:
            if se[i]:
                size[i] = -1.0
                entry = c[i]; j = i + 1
                while j < n and j - i < hold and hi[j] < entry * (1 + sl):
                    j += 1
                if j < n:
                    size[j] = 0.0
                i = j + 1
            else:
                i += 1
        return pd.Series(size, index=data.index), pd.Series(c, index=data.index)
