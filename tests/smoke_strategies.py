"""Smoke test des stratégies pilotes (OI_FADE_v1, SMA_CROSS_v1).

Exécution : cd BTYZ/src && python ../tests/smoke_strategies.py
Vérifie : contrat, signaux sur données réelles, causalité (préfixe = moitié),
backtest vbt (ordre de grandeur du nombre de trades).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from quantlab.contract import BaseStrategy, Signals, validate_strategy  # noqa: E402
from engine.data_loader import load_ohlcv  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    tag = "OK  " if cond else "FAIL"
    print(f"  [{tag}] {msg}")
    if not cond:
        FAILURES.append(msg)


def load_strategy(sid: str):
    path = ROOT / "strategies" / sid / "strategy.py"
    spec = importlib.util.spec_from_file_location(f"strategies.{sid}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Strategy


def series_equal_on_prefix(full: pd.Series | None, half: pd.Series | None,
                           n: int) -> bool:
    if full is None and half is None:
        return True
    if full is None or half is None:
        return False
    return bool(full.iloc[:n].equals(half.iloc[:n]))


def causality_check(strat, data: pd.DataFrame, params: dict) -> bool:
    n = len(data) // 2
    sig_full = strat.signals(data, params)
    sig_half = strat.signals(data.iloc[:n].copy(), params)
    ok = True
    for name in ("long_entries", "long_exits", "short_entries", "short_exits"):
        f, h = getattr(sig_full, name), getattr(sig_half, name)
        if not series_equal_on_prefix(f, h, n):
            diff = int((f.iloc[:n] != h.iloc[:n]).sum()) if f is not None and h is not None else -1
            print(f"    causalité KO sur {name} ({diff} barres divergentes)")
            ok = False
    return ok


def bool_series_clean(sig: Signals) -> bool:
    for name in ("long_entries", "long_exits", "short_entries", "short_exits"):
        s = getattr(sig, name)
        if s is None:
            continue
        if s.isna().any() or s.dtype != bool:
            print(f"    {name}: dtype={s.dtype}, NaN={int(s.isna().sum())}")
            return False
    return True


def run_backtest(strat, data: pd.DataFrame, sig: Signals):
    from vectorbtpro import vbt
    kw = {}
    if sig.td_stop is not None:
        kw["td_stop"] = pd.Timedelta(hours=int(sig.td_stop))
    if sig.sl_stop is not None:
        kw["sl_stop"] = float(sig.sl_stop)
    return vbt.Portfolio.from_signals(
        open=data["open"], high=data["high"], low=data["low"], close=data["close"],
        entries=sig.long_entries, exits=sig.long_exits,
        short_entries=sig.short_entries, short_exits=sig.short_exits,
        fees=0.0003, freq="1h", **kw)


def test_strategy(sid: str, pairs: list[str], source: str) -> None:
    print(f"\n=== {sid} ===")
    cls = load_strategy(sid)
    check(cls.__name__ == "Strategy" and issubclass(cls, BaseStrategy),
          "classe Strategy(BaseStrategy)")
    errs = validate_strategy(cls)
    check(errs == [], f"validate_strategy -> {errs}")
    strat = cls()
    check(strat.n_free_params() <= 4, f"param_space <= 4 dims ({strat.n_free_params()})")

    data = None
    pair = None
    for p in pairs:
        d = load_ohlcv(p, tf="1h", source=source)
        if d is not None and len(d) > 2000:
            data, pair = d, p
            break
    check(data is not None, f"données 1h source={source} (paire {pair})")
    if data is None:
        return
    print(f"    {pair}: {len(data)} barres, {data.index[0]} -> {data.index[-1]}")

    params = strat.full_params({})
    sig = strat.signals(data, params).align(data.index)
    check(bool_series_clean(sig), "séries booléennes sans NaN")
    n_entries = int(sum(int(s.sum()) for s in (sig.long_entries, sig.short_entries)
                        if s is not None))
    check(n_entries > 0, f"entrées > 0 ({n_entries} sur l'historique)")

    check(causality_check(strat, data, params),
          "causalité: signals(moitié) == signals(full) sur le préfixe")

    pf = run_backtest(strat, data, sig)
    n_trades = int(pf.trades.count())
    months = (data.index[-1] - data.index[0]).days / 30.4
    print(f"    backtest: {n_trades} trades sur {months:.1f} mois "
          f"({n_trades / months:.2f}/mois), total_return={pf.total_return:.2%}")
    check(n_trades > 0, "backtest produit des trades")


def main() -> None:
    test_strategy("OI_FADE_v1", ["BTC", "ETH", "SOL"], "oi")
    test_strategy("SMA_CROSS_v1", ["BTC", "ETH", "SOL"], "lighter")
    print()
    if FAILURES:
        print(f"SMOKE TEST FAILED ({len(FAILURES)} erreurs)")
        sys.exit(1)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
