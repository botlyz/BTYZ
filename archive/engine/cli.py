"""CLI entry-point for the BTYZ engine.

Usage examples:
  # Walk-Forward grid (3m × {1,2}bps × pairs)
  python -m engine.cli wfa --approach ATR_ENV_v1 --tf 3m 5m --bps 1 2 --pairs BTC ETH

  # MCCV (random-date OOS) for one config
  python -m engine.cli mccv --approach ATR_ENV_v1 --pair BTC --tf 3m --bps 1 --n-targets 12

  # List discoverable approaches
  python -m engine.cli list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make BTYZ/src importable when called as `python -m engine.cli`
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from engine.approach_loader import list_approaches
from engine.config import (
    DEFAULT_FOLD_WORKERS, DEFAULT_MIN_TRADES_PER_FOLD, DEFAULT_PAIR_WORKERS,
    DEFAULT_STEP_DAYS, DEFAULT_TEST_DAYS, DEFAULT_TRAIN_DAYS, DEFAULT_TRIALS_PER_FOLD,
    DEFAULT_WARMUP_BARS, PROJECT_ROOT,
)


def _cmd_list(_args):
    apps = list_approaches()
    if not apps:
        print("No approaches found in src/approach/")
        return
    print("Available approaches:")
    for a in apps:
        print(f"  - {a}")


def _cmd_wfa(args):
    from engine.wfa_runner import run_grid
    pairs = args.pairs or _load_default_pairs(args.tf)
    run_grid(
        approach_id=args.approach,
        tfs=args.tf,
        bps_list=args.bps,
        pairs=pairs,
        seed=args.seed,
        trials=args.trials,
        workers=args.workers,
        pair_workers=args.pair_workers,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        min_trades_per_fold=args.min_trades,
        warmup_bars=args.warmup,
        source=args.source,
    )


def _cmd_mccv(args):
    from engine.mccv_runner import run_mccv
    for pair in args.pairs:
        for tf in args.tf:
            for bps in args.bps:
                run_mccv(
                    approach_id=args.approach,
                    pair=pair,
                    tf=tf,
                    bps=bps,
                    n_targets=args.n_targets,
                    seed=args.seed,
                    trials=args.trials,
                    train_days=args.train_days,
                    test_days=args.test_days,
                    min_trades=args.min_trades,
                    workers=args.workers,
                    source=args.source,
                )


def _load_default_pairs(tfs):
    """Fall back to liquidity.json if --pairs not specified."""
    liq = PROJECT_ROOT / "liquidity.json"
    if not liq.exists():
        return ["BTC", "ETH"]
    data = json.loads(liq.read_text())
    candidates = []
    for lvl in ("tres_liquide", "liquide", "moyen"):
        candidates.extend(data.get(lvl, []))
    seen = set()
    return [p for p in candidates if not (p in seen or seen.add(p))]


def build_parser():
    p = argparse.ArgumentParser(prog="engine", description="BTYZ generic optimization engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    # list
    sub.add_parser("list", help="List discoverable approaches under src/approach/")

    # wfa
    wfa = sub.add_parser("wfa", help="Run walk-forward grid optimization")
    wfa.add_argument("--approach", required=True, help="Approach ID under src/approach/")
    wfa.add_argument("--tf", nargs="+", default=["3m", "5m", "15m"], help="Timeframes")
    wfa.add_argument("--bps", nargs="+", type=int, default=[0, 1, 2], help="Fees in bps")
    wfa.add_argument("--pairs", nargs="+", default=None, help="Pairs (defaults to liquidity.json)")
    wfa.add_argument("--trials", type=int, default=DEFAULT_TRIALS_PER_FOLD)
    wfa.add_argument("--workers", type=int, default=DEFAULT_FOLD_WORKERS, help="Fold workers (subprocess pool)")
    wfa.add_argument("--pair-workers", type=int, default=DEFAULT_PAIR_WORKERS, help="Parallel pairs (threads)")
    wfa.add_argument("--seed", type=int, default=42)
    wfa.add_argument("--train-days", type=int, default=DEFAULT_TRAIN_DAYS)
    wfa.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    wfa.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS)
    wfa.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES_PER_FOLD)
    wfa.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_BARS)
    wfa.add_argument("--source", default="auto", choices=["auto", "lighter", "binance", "cross_exchange", "gapfill"])

    # mccv
    mccv = sub.add_parser("mccv", help="Run MCCV (random-date OOS) for a config")
    mccv.add_argument("--approach", required=True)
    mccv.add_argument("--pairs", nargs="+", required=True)
    mccv.add_argument("--tf", nargs="+", required=True)
    mccv.add_argument("--bps", nargs="+", type=int, required=True)
    mccv.add_argument("--n-targets", type=int, default=12)
    mccv.add_argument("--trials", type=int, default=DEFAULT_TRIALS_PER_FOLD)
    mccv.add_argument("--workers", type=int, default=4)
    mccv.add_argument("--seed", type=int, default=42)
    mccv.add_argument("--train-days", type=int, default=DEFAULT_TRAIN_DAYS)
    mccv.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    mccv.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES_PER_FOLD)
    mccv.add_argument("--source", default="auto", choices=["auto", "lighter", "binance", "cross_exchange", "gapfill"])

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    {
        "list": _cmd_list,
        "wfa": _cmd_wfa,
        "mccv": _cmd_mccv,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
