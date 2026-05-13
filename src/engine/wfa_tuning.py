"""WFA hyper-tuning: comparer plusieurs (train_days, test_days, step_days)
sur un set de paires, pour décider du meilleur découpage walk-forward.

Métriques agrégées par config (sur tous folds × pairs):
- mean / median test Sharpe ratio
- % folds rentables (test_return_pct > 0)
- WFE corr (Pearson train_sharpe vs test_sharpe — overfitting indicator)
- mean test return %, mean max DD %
- N total folds (= coût de re-opti)

Output:
- results/<APPROACH>/wfa_tuning/<train>d_<test>d_<step>d/<pair>/summary.json
- results/<APPROACH>/wfa_tuning/_summary.json  (ranking final)

Usage:
  python -m engine.wfa_tuning \
    --approach RAM_DCA_v1 \
    --tf 5m --bps 1 \
    --pairs BTC ETH HYPE \
    --trials 80 \
    --workers 2 --pair-workers 4
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from engine.config import RESULTS_ROOT
from engine.wfa_runner import run_pair


# Default configurations to compare
DEFAULT_CONFIGS = [
    # (train_days, test_days, step_days, tag)
    (30,  7,   7,   "30d_7d_7d"),
    (30,  14,  14,  "30d_14d_14d"),
    (60,  14,  14,  "60d_14d_14d"),
    (60,  21,  21,  "60d_21d_21d"),
    (90,  21,  21,  "90d_21d_21d"),   # current default
    (90,  21,  7,   "90d_21d_7d"),    # overlapping (3x folds)
    (90,  30,  30,  "90d_30d_30d"),
    (180, 30,  30,  "180d_30d_30d"),
]


def aggregate_config(approach: str, cfg_tag: str, pairs: Sequence[str]) -> dict:
    """Read summary.json per pair, aggregate metrics across folds × pairs."""
    base = RESULTS_ROOT / approach / "wfa_tuning" / cfg_tag
    train_sh, test_sh, test_ret, test_dd = [], [], [], []
    n_folds_total, n_pairs_ok = 0, 0
    durations = []
    for pair in pairs:
        sj = base / pair / "summary.json"
        if not sj.exists():
            continue
        d = json.loads(sj.read_text())
        folds = d.get("folds", [])
        if not folds:
            continue
        n_pairs_ok += 1
        for fr in folds:
            tm = fr.get("train_metrics", {}) or {}
            te = fr.get("test_metrics", {}) or {}
            try:
                train_sh.append(float(tm.get("sharpe_ratio") or 0))
                test_sh.append(float(te.get("sharpe_ratio") or 0))
                test_ret.append(float(te.get("total_return_pct") or 0))
                test_dd.append(float(te.get("max_drawdown_pct") or 0))
            except Exception:
                pass
            n_folds_total += 1

    if n_folds_total == 0:
        return {"cfg_tag": cfg_tag, "n_folds": 0, "error": "no_data"}

    train_arr = np.array(train_sh)
    test_arr  = np.array(test_sh)
    ret_arr   = np.array(test_ret)
    dd_arr    = np.array(test_dd)

    corr = float("nan")
    if len(train_arr) > 1 and train_arr.std() > 0 and test_arr.std() > 0:
        corr = float(np.corrcoef(train_arr, test_arr)[0, 1])

    pct_pos = float((ret_arr > 0).mean() * 100)
    mean_test_sh = float(test_arr.mean())
    median_test_sh = float(np.median(test_arr))
    mean_train_sh = float(train_arr.mean())

    # Composite score:
    # +mean OOS sharpe (cap 5) + 1.5 × WFE_corr_clipped + pct_positive_bonus - DD penalty
    sh_score = min(max(mean_test_sh, -3), 5)
    corr_score = 1.5 * (corr if not np.isnan(corr) else 0)
    pos_score = (pct_pos / 100) * 1.5      # max 1.5
    dd_pen = max(0, dd_arr.mean() - 20) * 0.02
    score = sh_score + corr_score + pos_score - dd_pen

    return {
        "cfg_tag": cfg_tag,
        "n_pairs_ok": n_pairs_ok,
        "n_folds_total": n_folds_total,
        "avg_folds_per_pair": round(n_folds_total / n_pairs_ok, 1) if n_pairs_ok else 0,
        "mean_train_sh": round(mean_train_sh, 3),
        "mean_test_sh": round(mean_test_sh, 3),
        "median_test_sh": round(median_test_sh, 3),
        "sh_degradation": round(mean_train_sh - mean_test_sh, 3),
        "pct_positive_folds": round(pct_pos, 1),
        "mean_test_ret_pct": round(float(ret_arr.mean()), 3),
        "mean_test_dd_pct": round(float(dd_arr.mean()), 3),
        "wfe_corr_sharpe": round(corr, 3) if not np.isnan(corr) else None,
        "composite_score": round(score, 3),
    }


def main():
    ap = argparse.ArgumentParser(description="WFA hyper-tuning over (train, test, step) days")
    ap.add_argument("--approach", required=True)
    ap.add_argument("--tf", default="5m")
    ap.add_argument("--bps", type=int, default=1)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--workers", type=int, default=2, help="Fold workers (subprocess)")
    ap.add_argument("--pair-workers", type=int, default=4, help="Parallel pairs (threads)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="Filter by cfg_tag (e.g. 90d_21d_21d). If empty, run all DEFAULT_CONFIGS.")
    ap.add_argument("--source", default="auto", choices=["auto", "lighter", "binance"])
    args = ap.parse_args()

    configs = DEFAULT_CONFIGS
    if args.configs:
        configs = [c for c in configs if c[3] in args.configs]
        if not configs:
            print(f"No matching configs in {DEFAULT_CONFIGS}")
            return

    base = RESULTS_ROOT / args.approach / "wfa_tuning"
    base.mkdir(parents=True, exist_ok=True)

    fees = args.bps * 1e-4

    print("=" * 70)
    print(f"WFA tuning — approach={args.approach}  tf={args.tf}  bps={args.bps}")
    print(f"  Pairs   : {args.pairs}")
    print(f"  Trials  : {args.trials} / fold")
    print(f"  Configs : {[c[3] for c in configs]}")
    print(f"  Output  : {base}")
    print("=" * 70)

    t_start = time.time()
    for (train_d, test_d, step_d, tag) in configs:
        out_dir = base / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{tag}] train={train_d}d test={test_d}d step={step_d}d — pairs {args.pairs}")

        todo = [p for p in args.pairs if not (out_dir / p / "summary.json").exists()]
        if not todo:
            print(f"  [SKIP] all pairs already done for {tag}")
            continue

        t0 = time.time()

        def _do(pair):
            try:
                run_pair(
                    approach_id=args.approach, pair=pair, tf=args.tf, fees=fees, out_dir=out_dir,
                    seed=args.seed, trials=args.trials, workers=args.workers,
                    train_days=train_d, test_days=test_d, step_days=step_d,
                    source=args.source,
                )
            except Exception as e:
                print(f"  ERR {tag}/{pair}: {e}")

        with ThreadPoolExecutor(max_workers=args.pair_workers) as pool:
            futs = [pool.submit(_do, p) for p in todo]
            for _ in as_completed(futs):
                pass

        print(f"  done in {(time.time() - t0)/60:.1f} min")
        gc.collect()

    # ── Aggregate & rank ──
    print("\n" + "=" * 70)
    print("FINAL RANKING")
    print("=" * 70)

    summaries = [aggregate_config(args.approach, tag, args.pairs) for (_, _, _, tag) in configs]
    summaries.sort(key=lambda s: s.get("composite_score", -999), reverse=True)

    out_summary = {
        "approach": args.approach,
        "tf": args.tf,
        "bps": args.bps,
        "pairs": list(args.pairs),
        "trials_per_fold": args.trials,
        "elapsed_sec": round(time.time() - t_start, 1),
        "ranking": summaries,
    }
    out_path = base / "_summary.json"
    out_path.write_text(json.dumps(out_summary, indent=2, ensure_ascii=False))

    cols = ("cfg_tag", "n_folds_total", "mean_test_sh", "pct_positive_folds",
            "wfe_corr_sharpe", "mean_test_ret_pct", "mean_test_dd_pct",
            "sh_degradation", "composite_score")
    widths = [16, 13, 13, 19, 16, 17, 17, 16, 17]
    header = " ".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for s in summaries:
        row = " ".join(f"{str(s.get(c, '-')):>{w}}" for c, w in zip(cols, widths))
        print(row)
    print(f"\n→ Best: {summaries[0]['cfg_tag']} (score={summaries[0]['composite_score']})")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
