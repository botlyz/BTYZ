#!/usr/bin/env python3
"""WFA tick-level pour LIQ_FADE_v1 (fade des cascades de liquidation).

Réplique la logique du moteur BTYZ (folds train/test glissants, Optuna par fold,
parallèle, results CSV) mais sur TICKS au lieu de bougies — car la stratégie est
event-driven sub-seconde, incompatible avec le découpage barre-par-barre du moteur.

Pour chaque paire :
  - charge les ticks (filtre px>0),
  - découpe en folds par TEMPS (train_days / test_days / step_days),
  - Optuna optimise (gap_s, delay_s, hold_s, min_notional) sur le train,
  - évalue les meilleurs params sur le test (jamais vu),
  - sauve results/LIQ_FADE_v1/full/<cfg>/<PAIR>/all_folds.csv

Coût AR par paire = 2×half_spread (exec_costs.json) + 2×integrator_bps.

Usage :
  python scripts/wfa_liq_fade.py --pairs ASTER LIT ZEC --trials 80
  python scripts/wfa_liq_fade.py --all --bps 3 --trials 80 --workers 6
"""
import argparse
import glob
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICKS = os.path.join(ROOT, "data", "lighter_ticks")
EC = json.load(open(os.path.join(ROOT, "data", "exec_costs.json")))
MAX_TICKS = 150_000_000   # skip les géants (mémoire) : BTC/ETH traités à part


def _score(ret, min_trades):
    """Sharpe par-trade annualisé approx, pénalisé si trop peu de trades."""
    n = len(ret)
    if n < min_trades:
        return -10.0 + n / max(min_trades, 1)
    sd = ret.std()
    if sd <= 0:
        return -10.0
    return float(ret.mean() / sd * np.sqrt(n))   # ~Sharpe (annualisation neutre vs trials)


def run_pair(pair, train_days, test_days, step_days, trials, integ_bps,
             min_trades, seed):
    from approach.LIQ_FADE_v1.kernel import run_liq_fade
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    f = os.path.join(TICKS, f"{pair}.parquet")
    if not os.path.exists(f):
        return pair, None
    n_meta = pq.ParquetFile(f).metadata.num_rows
    if n_meta > MAX_TICKS:
        return pair, "skip_giant"
    df = pq.read_table(f, columns=["timestamp", "px", "sz", "is_maker_ask",
                                    "trade_type"]).to_pandas()
    df = df[df["px"] > 0].reset_index(drop=True)
    if len(df) < 5000 or (df["trade_type"] != "trade").sum() < 100:
        return pair, None

    half = EC.get(pair, {}).get("half_spread_bps", 3.0)
    cost_bps = 2 * half + 2 * integ_bps       # aller-retour

    t0, t1 = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    DAY = 86400_000
    folds = []
    s = t0
    while s + (train_days + test_days) * DAY <= t1:
        tr_end = s + train_days * DAY
        te_end = tr_end + test_days * DAY
        folds.append((s, tr_end, te_end))
        s += step_days * DAY
    if not folds:
        return pair, "no_fold"

    # index temps pour slicing rapide
    ts = df["timestamp"].to_numpy()
    rows = []
    for fi, (a, b, c) in enumerate(folds):
        tr = df.iloc[np.searchsorted(ts, a):np.searchsorted(ts, b)]
        te = df.iloc[np.searchsorted(ts, b):np.searchsorted(ts, c)]
        if len(tr) < 2000 or len(te) < 500:
            continue

        def objective(trial):
            gap = trial.suggest_int("gap_s", 1, 10)
            delay = trial.suggest_int("delay_s", 1, 10)
            hold = trial.suggest_int("hold_s", 60, 1800, step=60)
            mn = trial.suggest_categorical("min_notional",
                                           [5000, 10000, 25000, 50000, 100000])
            r = run_liq_fade(tr, gap_s=gap, delay_s=delay, hold_s=hold, stop_frac=0.0,
                             buffer_mult=0.1, base_cost_bps=cost_bps, min_notional=mn)
            return _score(r["ret"], min_trades)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed + fi))
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        p = study.best_params

        # évaluation OOS
        rt = run_liq_fade(te, gap_s=p["gap_s"], delay_s=p["delay_s"], hold_s=p["hold_s"],
                          stop_frac=0.0, buffer_mult=0.1, base_cost_bps=cost_bps,
                          min_notional=p["min_notional"])
        ret = rt["ret"]
        n = len(ret)
        if n > 0:
            eq = np.cumprod(1 + ret)
            sh = _score(ret, 1)
            rows.append(dict(fold=fi, **{f"p_{k}": v for k, v in p.items()},
                             test_trades=n, test_sharpe=round(sh, 3),
                             test_ret_mean_bps=round(ret.mean() * 1e4, 2),
                             test_winrate=round((ret > 0).mean() * 100, 1),
                             test_total_pct=round((eq[-1] - 1) * 100, 2),
                             cost_ar_bps=round(cost_bps, 1)))
        else:
            rows.append(dict(fold=fi, **{f"p_{k}": v for k, v in p.items()},
                             test_trades=0, test_sharpe=0.0, test_ret_mean_bps=0.0,
                             test_winrate=0.0, test_total_pct=0.0,
                             cost_ar_bps=round(cost_bps, 1)))
    return pair, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--bps", type=int, default=3, help="integrator fee bps (par fill)")
    ap.add_argument("--train-days", type=int, default=90)
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument("--step-days", type=int, default=30)
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-trades", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.all:
        pairs = sorted(os.path.basename(p)[:-8] for p in glob.glob(f"{TICKS}/*.parquet"))
    else:
        pairs = args.pairs or ["ASTER", "LIT", "ZEC"]

    cfg = f"{args.train_days}t_{args.test_days}o_{args.bps}bps"
    out = os.path.join(ROOT, "results", "LIQ_FADE_v1", "full", cfg)
    os.makedirs(out, exist_ok=True)
    print(f"WFA LIQ_FADE | {len(pairs)} paires | {cfg} | trials={args.trials} | workers={args.workers}")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_pair, p, args.train_days, args.test_days, args.step_days,
                          args.trials, args.bps, args.min_trades, args.seed): p
                for p in pairs}
        for fut in as_completed(futs):
            pair, res = fut.result()
            done += 1
            if isinstance(res, list) and res:
                d = os.path.join(out, pair)
                os.makedirs(d, exist_ok=True)
                dfres = pd.DataFrame(res)
                dfres.to_csv(os.path.join(d, "all_folds.csv"), index=False)
                med = dfres["test_sharpe"].median()
                print(f"  [{done}/{len(pairs)}] {pair:10} {len(res)} folds | Sharpe OOS méd {med:+.2f}")
            else:
                print(f"  [{done}/{len(pairs)}] {pair:10} — {res}")
    print(f"\nFini -> {out}")


if __name__ == "__main__":
    main()
