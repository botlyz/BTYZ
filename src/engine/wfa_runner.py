"""Walk-Forward orchestrator: grid (tf × bps × pair) × folds.

Each fold runs Optuna TPE in a subprocess (ProcessPoolExecutor). Results land in
results/<APPROACH_ID>/full/<tf>_<bps>bps/<PAIR>/{summary.json, all_folds.csv, trades/fold_*.parquet}.
"""
from __future__ import annotations

import gc
import json
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    BAR_MINUTES, DEFAULT_FOLD_WORKERS, DEFAULT_MIN_TRADES_PER_FOLD, DEFAULT_PAIR_WORKERS,
    DEFAULT_STEP_DAYS, DEFAULT_TEST_DAYS, DEFAULT_TRAIN_DAYS, DEFAULT_TRIALS_PER_FOLD,
    DEFAULT_WARMUP_BARS, FREQ_MAP, RESULTS_ROOT, TF_ALIAS,
)
from .data_loader import load_ohlcv
from .trades import extract_trades_df
from .walk_forward import compute_folds


def _clean(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 10)
    except Exception:
        return str(v) if not isinstance(v, int) else v


# ── Subprocess fold worker ─────────────────────────────────────────────────

def _run_fold_worker(args):
    """Pickled args. Re-imports strategy in subprocess, runs Optuna TPE on this fold."""
    (approach_id, fold_idx,
     train_vals, train_cols, train_index,
     test_vals,  test_cols,  test_index,
     vbt_freq, fees, seed, trials, min_trades_per_fold) = args

    import warnings as _w
    _w.filterwarnings("ignore")
    # Add BTYZ/src to path so engine + approach packages are importable in subprocess
    src_root = Path(__file__).resolve().parent.parent
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from engine.approach_loader import instantiate_strategy
    from engine.tpe_search import run_tpe_fold

    train_df = pd.DataFrame(train_vals, columns=train_cols, index=train_index)
    test_df  = pd.DataFrame(test_vals,  columns=test_cols,  index=test_index)

    strat = instantiate_strategy(approach_id)
    # Strategy may consume runtime hints from module-level (fees, freq)
    mod = sys.modules[f"approach.{approach_id}.strategy"]
    if hasattr(mod, "_target_fees"):
        mod._target_fees = fees
    if hasattr(mod, "_target_freq"):
        mod._target_freq = vbt_freq
    # _target_slippage stays at strategy module default (2 bps unless engine overrides)
    # Approche optionnelle : si le module expose `_per_pair_fees_resolver(pair) → float`,
    # on l'utilise pour override fees par paire (utile pour cross-exchange).
    # Le pair name doit avoir été passé via les args (extension future).

    result = run_tpe_fold(
        train_data=train_df,
        test_data=test_df,
        param_space_fn=strat.param_space,
        run_backtest_fn=strat.run_backtest,
        score_fn=strat.score,
        trials=trials,
        min_trades_per_fold=min_trades_per_fold,
        seed=seed + fold_idx,
        fold_idx=fold_idx,
        n_jobs=1,
    )
    if result is None:
        return None
    result["fold"] = fold_idx
    return result


# ── Per-pair orchestration ─────────────────────────────────────────────────

def run_pair(approach_id: str, pair: str, tf: str, fees: float, out_dir: Path,
             seed: int = 42, trials: int = DEFAULT_TRIALS_PER_FOLD,
             workers: int = DEFAULT_FOLD_WORKERS,
             train_days: int = DEFAULT_TRAIN_DAYS,
             test_days: int = DEFAULT_TEST_DAYS,
             step_days: int = DEFAULT_STEP_DAYS,
             min_trades_per_fold: int = DEFAULT_MIN_TRADES_PER_FOLD,
             warmup_bars: int = DEFAULT_WARMUP_BARS,
             source: str = "auto") -> dict | None:
    vbt_freq = FREQ_MAP.get(tf, tf)
    ohlcv = load_ohlcv(pair, vbt_freq, source=source)
    if ohlcv is None:
        tqdm.write(f"  [{pair}] OHLCV introuvable — skip")
        return None

    # Per-pair fees override : si le module strategie expose `_load_pair_fees(pair)`,
    # on l'utilise au lieu du `fees` global (utile pour cross-exchange où taker + half-spread
    # diffèrent par paire). `bps` CLI sert alors juste de baseline si la fonction renvoie None.
    try:
        from engine.approach_loader import load_strategy_module
        _mod = load_strategy_module(approach_id)
        if hasattr(_mod, "_load_pair_fees"):
            override = _mod._load_pair_fees(pair)
            if override and override > 0:
                tqdm.write(f"  [{pair}] fees override per-pair = {override*1e4:.2f} bps/fill (vs global {fees*1e4:.2f})")
                fees = float(override)
    except Exception as _e:
        tqdm.write(f"  [{pair}] per-pair fees lookup failed: {_e}")

    bar_min = BAR_MINUTES.get(vbt_freq, 15)
    folds = compute_folds(len(ohlcv), train_days, test_days, step_days, vbt_freq)
    if not folds:
        tqdm.write(f"  [{pair}] Pas assez de données — skip")
        return None
    tqdm.write(f"  [{pair}] {len(folds)} folds | {len(ohlcv)} barres {vbt_freq}")

    fold_args = []
    for fold_idx, (s, te, end) in enumerate(folds):
        train = ohlcv.iloc[s:te]
        test  = ohlcv.iloc[te:end]
        fold_args.append((
            approach_id, fold_idx,
            train.values, list(train.columns), train.index,
            test.values,  list(test.columns),  test.index,
            vbt_freq, fees, seed, trials, min_trades_per_fold,
        ))

    fold_results_raw: list[dict | None] = [None] * len(folds)
    with ProcessPoolExecutor(max_workers=min(workers, len(folds))) as ex:
        futures = {ex.submit(_run_fold_worker, args): args[1] for args in fold_args}
        for fut in as_completed(futures):
            fidx = futures[fut]
            try:
                fold_results_raw[fidx] = fut.result()
            except Exception as e:
                tqdm.write(f"    [{pair}] fold {fidx} erreur: {e}")

    # In-parent re-run for trade extraction (single process, single strategy instance)
    from .approach_loader import instantiate_strategy
    strat = instantiate_strategy(approach_id)
    mod = sys.modules[f"approach.{approach_id}.strategy"]
    if hasattr(mod, "_target_fees"):
        mod._target_fees = fees
    if hasattr(mod, "_target_freq"):
        mod._target_freq = vbt_freq
    # _target_slippage stays at strategy module default (2 bps unless engine overrides)

    pair_dir = out_dir / pair
    pair_dir.mkdir(parents=True, exist_ok=True)
    trades_dir = pair_dir / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)

    fold_results_final = []
    for fold_idx, (s, te, end) in enumerate(folds):
        res = fold_results_raw[fold_idx]
        if res is None:
            continue
        params  = res["params"]
        train_m = res.get("train_metrics", {})
        test_m  = res.get("test_metrics", {})

        wi = max(0, te - warmup_bars)
        slice_w = ohlcv.iloc[wi:end]
        test_start = ohlcv.index[te]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pf = strat.run_backtest(slice_w, params)
                trades_df = extract_trades_df(pf, test_start)
                if not trades_df.empty:
                    trades_df.to_parquet(trades_dir / f"fold_{fold_idx}.parquet", index=False)
            except Exception as e:
                tqdm.write(f"    [{pair}] fold {fold_idx} trade extraction erreur: {e}")

        fold_results_final.append({
            "fold": fold_idx,
            "params": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                           (int(v) if isinstance(v, (np.integer,)) else
                            float(v) if isinstance(v, (np.floating,)) else v))
                       for k, v in params.items()},
            "train_metrics": {k: _clean(v) for k, v in train_m.items()},
            "test_metrics":  {k: _clean(v) for k, v in test_m.items()},
        })

    if not fold_results_final:
        tqdm.write(f"  [{pair}] Aucun fold valide")
        return None

    summary = {
        "approach_id": approach_id,
        "pair": pair,
        "tf": vbt_freq,
        "fees": fees,
        "n_folds": len(fold_results_final),
        "folds": fold_results_final,
    }
    (pair_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Flat CSV view
    rows = []
    for fr in fold_results_final:
        row = {"fold": fr["fold"]}
        for k, v in fr["params"].items():
            row[f"p_{k}"] = v
        for k, v in fr["train_metrics"].items():
            row[f"train_{k}"] = v
        for k, v in fr["test_metrics"].items():
            row[f"test_{k}"] = v
        rows.append(row)
    pd.DataFrame(rows).to_csv(pair_dir / "all_folds.csv", index=False)

    test_returns = [fr["test_metrics"].get("total_return_pct") or 0 for fr in fold_results_final]
    avg_ret = float(np.mean(test_returns)) if test_returns else 0.0
    n_prof  = sum(1 for r in test_returns if r > 0)
    tqdm.write(f"  [{pair}] {len(fold_results_final)} folds OK | avg_return={avg_ret:.1f}% | {n_prof}/{len(fold_results_final)} prof.")
    gc.collect()
    return summary


# ── Grid orchestrator ─────────────────────────────────────────────────────

def run_grid(approach_id: str, tfs: Sequence[str], bps_list: Sequence[int],
             pairs: Sequence[str], **kwargs) -> None:
    from .approach_loader import load_strategy_module
    load_strategy_module(approach_id)  # pre-warm once before any thread/process is spawned

    out_root = RESULTS_ROOT / approach_id / "full"
    out_root.mkdir(parents=True, exist_ok=True)

    grid = [(tf, bps) for tf in tfs for bps in bps_list]
    total_runs = len(grid) * len(pairs)
    pair_workers = kwargs.pop("pair_workers", DEFAULT_PAIR_WORKERS)

    print("=" * 70)
    print(f"BTYZ engine — approach={approach_id}")
    print(f"  TF        : {list(tfs)}")
    print(f"  BPS       : {list(bps_list)}")
    print(f"  Pairs     : {len(pairs)} — {list(pairs)}")
    print(f"  Output    : {out_root}")
    print(f"  Total runs: {len(grid)} configs × {len(pairs)} pairs = {total_runs}")
    print("=" * 70)

    t0 = time.time()
    progress = tqdm(total=total_runs, unit="run", ncols=90,
                    bar_format="{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]")

    train_days = kwargs.get("train_days", DEFAULT_TRAIN_DAYS)
    test_days  = kwargs.get("test_days",  DEFAULT_TEST_DAYS)
    step_days  = kwargs.get("step_days",  DEFAULT_STEP_DAYS)

    for tf, bps in grid:
        run_tag = f"{tf}_{bps}bps_{train_days}d{test_days}d{step_days}d"
        out_dir = out_root / run_tag
        progress.set_description(run_tag)

        todo = [p for p in pairs if not (out_dir / p / "summary.json").exists()]
        skipped = len(pairs) - len(todo)
        if skipped:
            tqdm.write(f"  [SKIP] {run_tag}: {skipped} pairs already done")
            progress.update(skipped)
        if not todo:
            continue

        def _do(pair):
            try:
                run_pair(approach_id, pair, tf, bps * 1e-4, out_dir, **kwargs)
            except Exception as e:
                tqdm.write(f"  ERROR {run_tag}/{pair}: {e}")
            progress.update(1)

        with ThreadPoolExecutor(max_workers=pair_workers) as pair_pool:
            futs = [pair_pool.submit(_do, p) for p in todo]
            for _ in as_completed(futs):
                pass

    progress.close()
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"DONE en {elapsed/3600:.1f}h ({elapsed:.0f}s) — results in {out_root}")
