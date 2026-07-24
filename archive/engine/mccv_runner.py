"""Monte-Carlo Cross-Validation: random target dates, rolling re-opti.

For each random `target` date:
  1. train window  = [target - train_days, target)
  2. oos window    = [target, target + test_days)
  3. Optuna TPE on train → best params
  4. Backtest best params on oos
  5. Save (target, train_sh, oos_sh, oos_ret, oos_dd, oos_pf, oos_wr, oos_n, *best_params)

Output: results/<APPROACH_ID>/mccv/<PAIR>_<TF>_<BPS>bps.json
"""
from __future__ import annotations

import json
import math
import random
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import (
    BAR_MINUTES, DEFAULT_MIN_TRADES_PER_FOLD, DEFAULT_TEST_DAYS, DEFAULT_TRAIN_DAYS,
    DEFAULT_TRIALS_PER_FOLD, FREQ_MAP, RESULTS_ROOT,
)
from .data_loader import load_ohlcv
from .metrics import extract_from_portfolio


def _gen_targets(index: pd.DatetimeIndex, train_days: int, test_days: int,
                 n_targets: int, seed: int) -> list[pd.Timestamp]:
    """Pick n_targets random dates such that both train + oos windows fit."""
    start = index[0] + pd.Timedelta(days=train_days + 1)
    end = index[-1] - pd.Timedelta(days=test_days + 1)
    if start >= end:
        return []
    rng = random.Random(seed)
    span_s = (end - start).total_seconds()
    seen = set()
    out = []
    tries = 0
    while len(out) < n_targets and tries < n_targets * 10:
        tries += 1
        offset = rng.uniform(0, span_s)
        ts = start + pd.Timedelta(seconds=offset)
        # snap to nearest bar in index
        ts = index[index.searchsorted(ts, side="left")]
        date_only = ts.normalize()
        if date_only in seen:
            continue
        seen.add(date_only)
        out.append(ts)
    return sorted(out)


def _run_one_target(args):
    """Subprocess worker: TPE train on [target-train_days, target), evaluate on [target, target+test_days)."""
    (approach_id, target_iso, train_vals, train_cols, train_index,
     oos_vals, oos_cols, oos_index,
     vbt_freq, fees, seed, trials, min_trades) = args

    warnings.filterwarnings("ignore")
    src_root = Path(__file__).resolve().parent.parent
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from engine.approach_loader import instantiate_strategy
    from engine.tpe_search import run_tpe_fold

    train_df = pd.DataFrame(train_vals, columns=train_cols, index=train_index)
    oos_df   = pd.DataFrame(oos_vals,   columns=oos_cols,   index=oos_index)

    strat = instantiate_strategy(approach_id)
    mod = sys.modules[f"approach.{approach_id}.strategy"]
    if hasattr(mod, "_target_fees"):
        mod._target_fees = fees
    if hasattr(mod, "_target_freq"):
        mod._target_freq = vbt_freq

    res = run_tpe_fold(
        train_data=train_df, test_data=oos_df,
        param_space_fn=strat.param_space,
        run_backtest_fn=strat.run_backtest,
        score_fn=strat.score,
        trials=trials,
        min_trades_per_fold=min_trades,
        seed=seed,
        fold_idx=0,
        n_jobs=1,
    )
    if res is None:
        return {"target": target_iso, "error": "tpe_failed"}
    train_m = res["train_metrics"]
    oos_m = res["test_metrics"]
    row = {
        "target": target_iso,
        "train_sh": round(float(train_m.get("sharpe_ratio") or 0), 3),
        "oos_sh":   round(float(oos_m.get("sharpe_ratio") or 0), 3),
        "oos_ret":  round(float(oos_m.get("total_return_pct") or 0), 3),
        "oos_dd":   round(float(oos_m.get("max_drawdown_pct") or 0), 3),
        "oos_pf":   round(float(oos_m.get("profit_factor") or oos_m.get("trades_profit_factor") or 0), 3),
        "oos_wr":   round(float(oos_m.get("win_rate_pct") or 0), 2),
        "oos_n":    int(oos_m.get("total_trades", 0) or oos_m.get("trades_count", 0) or 0),
    }
    for k, v in res["params"].items():
        row[k] = (bool(v) if isinstance(v, (bool, np.bool_)) else
                  (int(v) if isinstance(v, (np.integer,)) else
                   float(v) if isinstance(v, (np.floating,)) else v))
    return row


def run_mccv(approach_id: str, pair: str, tf: str, bps: int,
             n_targets: int = 12, seed: int = 42,
             trials: int = DEFAULT_TRIALS_PER_FOLD,
             train_days: int = DEFAULT_TRAIN_DAYS,
             test_days: int = DEFAULT_TEST_DAYS,
             min_trades: int = DEFAULT_MIN_TRADES_PER_FOLD,
             workers: int = 4,
             source: str = "auto") -> dict | None:
    vbt_freq = FREQ_MAP.get(tf, tf)
    ohlcv = load_ohlcv(pair, vbt_freq, source=source)
    if ohlcv is None:
        print(f"[{pair}/{tf}/{bps}bps] OHLCV missing")
        return None

    bar_min = BAR_MINUTES.get(vbt_freq, 15)
    train_bars = int(train_days * 1440 / bar_min)
    test_bars  = int(test_days * 1440 / bar_min)

    targets = _gen_targets(ohlcv.index, train_days, test_days, n_targets, seed)
    if not targets:
        print(f"[{pair}/{tf}/{bps}bps] data too short for {n_targets} targets")
        return None

    fees = bps * 1e-4
    # Per-pair fees override (même hook que wfa_runner) : si le module stratégie
    # expose `_load_pair_fees(pair)`, on l'utilise au lieu du bps CLI (fallback).
    try:
        from engine.approach_loader import load_strategy_module
        _mod = load_strategy_module(approach_id)
        if hasattr(_mod, "_load_pair_fees"):
            override = _mod._load_pair_fees(pair)
            if override and override > 0:
                print(f"[{pair}/{tf}] fees override per-pair = {override*1e4:.2f} bps/fill (vs global {fees*1e4:.2f})")
                fees = float(override)
    except Exception as _e:
        print(f"[{pair}/{tf}] per-pair fees lookup failed: {_e}")
    args_list = []
    for tgt in targets:
        idx = ohlcv.index.searchsorted(tgt, side="left")
        train = ohlcv.iloc[max(0, idx - train_bars):idx]
        oos   = ohlcv.iloc[idx:idx + test_bars]
        if len(train) < 500 or len(oos) < 50:
            continue
        args_list.append((
            approach_id, tgt.isoformat(),
            train.values, list(train.columns), train.index,
            oos.values,   list(oos.columns),   oos.index,
            vbt_freq, fees, seed, trials, min_trades,
        ))

    rows = []
    with ProcessPoolExecutor(max_workers=min(workers, len(args_list))) as ex:
        futures = {ex.submit(_run_one_target, a): a[1] for a in args_list}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                if r and "error" not in r:
                    rows.append(r)
            except Exception as e:
                print(f"  target {futures[fut]} error: {e}")

    out = {
        "approach_id": approach_id,
        "pair": pair,
        "tf": tf,
        "bps": bps,
        "n_targets": len(rows),
        "targets": [r["target"] for r in rows],
        "rows": rows,
    }
    # Filename inclut train/test pour éviter écrasement entre configs grid
    out_path = RESULTS_ROOT / approach_id / "mccv" / f"{pair}_{tf}_{bps}bps_{train_days}t_{test_days}o.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out["train_days"] = train_days
    out["test_days"] = test_days
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{pair}/{tf}/{bps}bps/{train_days}t/{test_days}o] MCCV saved: {len(rows)} targets → {out_path}")
    return out
