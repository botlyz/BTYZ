"""Cœur backtest partagé — vbt.Portfolio.from_signals uniquement.

La stratégie ne produit que des Signals ; le moteur backteste. Métriques via
engine.metrics.extract_from_portfolio (jamais de math manuelle).
"""
from __future__ import annotations

import pandas as pd

from quantlab import config


def _td(td_stop, tf: str) -> pd.Timedelta:
    """td_stop en barres -> pd.Timedelta selon la tf."""
    return int(td_stop) * pd.Timedelta(config.FREQ_MAP[tf])


def _stop_kwargs(sig, tf: str) -> dict:
    kw = {}
    if sig.sl_stop is not None:
        kw["sl_stop"] = float(sig.sl_stop)
    if sig.tp_stop is not None:
        kw["tp_stop"] = float(sig.tp_stop)
    if sig.td_stop is not None:
        kw["td_stop"] = _td(sig.td_stop, tf)
    return kw


def run_signals_backtest(strategy, data: pd.DataFrame, params: dict, *,
                         fees: float, slippage: float = config.DEFAULT_SLIPPAGE,
                         init_cash: float = config.DEFAULT_INIT_CASH,
                         tf: str):
    """Backtest mono-paire. `fees`/`slippage` en fraction (0.0003 = 3 bps)."""
    import vectorbtpro as vbt

    sig = strategy.signals(data, strategy.full_params(params)).align(data.index)
    return vbt.Portfolio.from_signals(
        close=data["close"],
        entries=sig.long_entries,
        exits=sig.long_exits,
        short_entries=sig.short_entries,
        short_exits=sig.short_exits,
        fees=fees, slippage=slippage, init_cash=init_cash,
        freq=config.FREQ_MAP[tf],
        **_stop_kwargs(sig, tf),
    )


def pooled_backtest(strategy, datas: dict[str, pd.DataFrame], params, *,
                    fees: float, tf: str,
                    weights: dict[str, float] | None = None):
    """Backtest multi-paires mêmes params, cash partagé -> un portefeuille poolé.

    Colonnes = paires (close en DataFrame large sur l'union des index, ffill),
    signaux calculés par paire sur SES données puis réalignés (False hors index).
    `weights` : fraction de la valeur du portefeuille par ordre et par paire
    (défaut équipondéré 1/n).
    """
    import vectorbtpro as vbt

    pairs = list(datas)
    if not pairs:
        raise ValueError("pooled_backtest: aucune paire")
    fp = strategy.full_params(params)

    close = pd.concat({p: datas[p]["close"] for p in pairs}, axis=1).sort_index()
    close.columns = pairs
    union = close.index
    close = close.ffill()

    frames = {name: {} for name in
              ("long_entries", "long_exits", "short_entries", "short_exits")}
    stop_kw: dict = {}
    for p in pairs:
        sig = strategy.signals(datas[p], fp).align(datas[p].index)
        stop_kw = _stop_kwargs(sig, tf)  # mêmes params -> mêmes stops partout
        for name in frames:
            s = getattr(sig, name)
            if s is None:
                s = pd.Series(False, index=datas[p].index)
            frames[name][p] = s.reindex(union, fill_value=False)
    wide = {name: pd.DataFrame(cols)[pairs] for name, cols in frames.items()}

    if weights is None:
        weights = {p: 1.0 / len(pairs) for p in pairs}
    # ligne (1, n_pairs) broadcastée par colonne (une Series indexée par paire
    # ne broadcaste pas contre un index datetime)
    import numpy as np
    size = np.array([[float(weights.get(p, 0.0)) for p in pairs]])

    return vbt.Portfolio.from_signals(
        close=close,
        entries=wide["long_entries"],
        exits=wide["long_exits"],
        short_entries=wide["short_entries"],
        short_exits=wide["short_exits"],
        size=size, size_type="valuepercent",
        fees=fees, slippage=config.DEFAULT_SLIPPAGE,
        init_cash=config.DEFAULT_INIT_CASH,
        group_by=True, cash_sharing=True,
        freq=config.FREQ_MAP[tf],
        **stop_kw,
    )


def portfolio_returns(pf) -> pd.Series:
    """Returns par barre en Series (portefeuille groupé -> déjà agrégé)."""
    r = pf.returns
    if callable(r):
        r = r()
    if isinstance(r, pd.DataFrame):
        if r.shape[1] == 1:
            return r.iloc[:, 0]
        raise ValueError("portfolio non groupé: passe group_by=True/cash_sharing "
                         "ou extrais la colonne voulue")
    return r
