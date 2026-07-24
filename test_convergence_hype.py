"""Convergence test for ATR_ENV_v1 on HYPE 5m — mirrors production WFA exactly.

Runs all WFA folds (90/30/30) in parallel via ProcessPoolExecutor (1 process / fold,
n_jobs=1 inside each fold → same as engine's wfa_runner). Each fold tracks the
best_value at each trial. Aggregates plateau-trial across folds (median, max, p90)
so we can pick a `--trials` value that covers the slowest-converging fold.

Usage:  cd /home/devbox/BTYZ && .venv/bin/python3 test_convergence_hype.py
"""
from __future__ import annotations
import sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

PAIR = "HYPE"
TF = "5m"
BAR_MIN = 5
TRAIN_DAYS = 90
TEST_DAYS = 30
STEP_DAYS = 30
TRIALS = 2000
SEED = 42
MIN_TRADES = 30


def load_5m(pair: str) -> pd.DataFrame:
    fp = PROJ_ROOT / "data" / "raw" / "lighter" / "1m" / f"{pair}.csv"
    df = pd.read_csv(fp)
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.resample("5min", origin="epoch", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["close"])


def compute_folds(n_bars: int, train_days: int, test_days: int, step_days: int, bar_min: int):
    """Same logic as engine.walk_forward.compute_folds."""
    train_bars = int(train_days * 1440 / bar_min)
    test_bars  = int(test_days  * 1440 / bar_min)
    step_bars  = int(step_days  * 1440 / bar_min)
    folds = []
    i = 0
    while True:
        s = i * step_bars
        train_end = s + train_bars
        test_end  = train_end + test_bars
        if test_end > n_bars:
            break
        if (train_end - s) >= 500 and (test_end - train_end) >= 100:
            folds.append((s, train_end, test_end))
        i += 1
    return folds


def run_one_fold(args):
    """Subprocess worker — runs Optuna TPE on a single fold, tracks best_value per trial.
    Mirrors production wfa_runner exactly: n_jobs=1, seed = SEED + fold_idx.
    """
    fold_idx, train_vals, train_cols, train_index = args

    import sys as _sys
    src_root = PROJ_ROOT / "src"
    if str(src_root) not in _sys.path:
        _sys.path.insert(0, str(src_root))
    import warnings; warnings.filterwarnings("ignore")

    import optuna
    from optuna.samplers import TPESampler
    from approach.ATR_ENV_v1.strategy import Strategy
    from engine.metrics import extract_from_portfolio
    from engine.scoring import score_robust

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_df = pd.DataFrame(train_vals, columns=train_cols, index=train_index)
    strat = Strategy()

    progression = []  # list of best_value per trial

    def objective(trial):
        params = strat.param_space(trial)
        try:
            pf = strat.run_backtest(train_df, params)
            if pf is None:
                return -10.0
            m = extract_from_portfolio(pf)
            tr = m.get("total_trades", 0) or 0
            abs_min = max(3, MIN_TRADES // 4)
            if tr < abs_min:
                return -10.0 + tr / abs_min
            sc = score_robust(m)
            if tr < MIN_TRADES:
                sc += -3.0 * (1.0 - tr / MIN_TRADES)
            else:
                sc += min(0.75, 0.25 * ((tr - MIN_TRADES) / MIN_TRADES))
            return sc
        except Exception:
            return -10.0

    def cb(study, trial):
        v = study.best_value if study.best_trial is not None else -10.0
        progression.append((trial.number + 1, v))

    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED + fold_idx, multivariate=True, warn_independent_sampling=False),
    )
    study.optimize(
        objective, n_trials=TRIALS, callbacks=[cb],
        gc_after_trial=True, catch=(Exception,), n_jobs=1,
    )
    elapsed = time.time() - t0

    return {
        "fold": fold_idx,
        "elapsed_s": elapsed,
        "best_value": float(study.best_value) if study.best_trial else None,
        "best_params": dict(study.best_trial.user_attrs.get("full_params", study.best_trial.params)) if study.best_trial else None,
        "progression": progression,
    }


def detect_plateau(progression, eps=0.005, window=200):
    arr = np.array(progression)
    if len(arr) <= window:
        return None
    scores = arr[:, 1].astype(float)
    trials = arr[:, 0].astype(int)
    for i in range(len(scores) - window):
        if scores[i + window] - scores[i] < eps:
            return int(trials[i])
    return None


def main():
    df = load_5m(PAIR)
    folds = compute_folds(len(df), TRAIN_DAYS, TEST_DAYS, STEP_DAYS, BAR_MIN)
    print(f"[CONV] HYPE 5m: {len(df)} bars total, {len(folds)} folds (90/30/30)")
    for i, (s, te, tx) in enumerate(folds):
        print(f"   fold {i}:  train {df.index[s]} → {df.index[te-1]}  ({te-s} bars)")
    print(f"\n[CONV] Running TPE {TRIALS} trials/fold, {len(folds)} folds in parallel (process pool)\n")

    # Build args for each fold
    jobs = []
    for fi, (s, te, _) in enumerate(folds):
        train_slice = df.iloc[s:te]
        jobs.append((fi, train_slice.values, list(train_slice.columns), train_slice.index))

    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=min(12, len(folds))) as ex:
        futures = {ex.submit(run_one_fold, j): j[0] for j in jobs}
        done = 0
        for f in as_completed(futures):
            r = f.result()
            results[r["fold"]] = r
            done += 1
            print(f"[CONV]   fold {r['fold']:>2} done in {r['elapsed_s']:.0f}s  best={r['best_value']:+.4f}  ({done}/{len(folds)} folds)")

    total_elapsed = time.time() - t0
    print(f"\n[CONV] All folds done in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)\n")

    # Per-fold plateau detection
    print(f"{'Fold':<6} {'BestVal':>9} {'PlateauTrial':>14} {'BestParams':<60}")
    print("-" * 110)
    plateau_trials = []
    for fi in sorted(results):
        r = results[fi]
        plat = detect_plateau(r["progression"])
        if plat is not None:
            plateau_trials.append(plat)
        plat_str = str(plat) if plat is not None else "—"
        params_str = ", ".join(f"{k}={v}" for k, v in r["best_params"].items()) if r["best_params"] else "(none)"
        print(f"{fi:<6} {r['best_value']:>+9.4f} {plat_str:>14} {params_str:<60}")

    print()
    if plateau_trials:
        arr = np.array(plateau_trials)
        print(f"[CONV] Plateau trial across {len(plateau_trials)} folds:")
        print(f"   median  : {int(np.median(arr))}")
        print(f"   max     : {int(np.max(arr))}")
        print(f"   p90     : {int(np.percentile(arr, 90))}")
        print(f"   → RECOMMENDED --trials = {int(np.percentile(arr, 90))} (covers 90% of folds)")
    else:
        print(f"[CONV] No fold plateaued within {TRIALS} trials — increase TRIALS")

    # Save progression CSVs (1 per fold) + summary
    out_dir = PROJ_ROOT / "cache" / "convergence_HYPE"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fi, r in results.items():
        pd.DataFrame(r["progression"], columns=["trial", "best_value"]).to_csv(
            out_dir / f"fold_{fi:02d}.csv", index=False
        )
    summary = []
    for fi in sorted(results):
        r = results[fi]
        summary.append({
            "fold": fi, "best_value": r["best_value"],
            "plateau_trial": detect_plateau(r["progression"]),
            "elapsed_s": r["elapsed_s"],
            **(r["best_params"] or {}),
        })
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    print(f"\n[CONV] Saved per-fold progressions → {out_dir}/fold_NN.csv")
    print(f"[CONV] Summary               → {out_dir}/summary.csv")


if __name__ == "__main__":
    main()
