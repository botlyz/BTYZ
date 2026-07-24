"""Run gap-fill backtest on all TradFi candidates in parallel (24 workers).

Filters to symbols with both Lighter 5m and yfinance data + enough history.
Reports a summary table sorted by Sharpe. First-pass edge discovery.

Run: .venv/bin/python3 src/approach/gapfill/run_all_gapfill.py
"""
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_gapfill import load_aligned, run_backtest, LGT_DIR, YF_DIR

MIN_BARS = 8000  # ~28 days of 5min data minimum (need several weekends)


def candidates():
    syms = []
    for yf_fp in sorted(YF_DIR.glob("*.csv")):
        sym = yf_fp.stem
        lgt_fp = LGT_DIR / f"{sym}.csv"
        if not lgt_fp.exists():
            continue
        syms.append(sym)
    return syms


def worker(sym):
    try:
        df = load_aligned(sym)
        if df is None or len(df) < MIN_BARS:
            return {"sym": sym, "status": "skip (insufficient data)", "bars": len(df) if df is not None else 0}
        pf = run_backtest(sym)
        if pf is None:
            return {"sym": sym, "status": "skip"}
        st = pf.stats()
        def g(k, d=0.0):
            try:
                v = st[k]
                return float(v) if not pd.isna(v) else d
            except Exception:
                return d
        return {
            "sym": sym,
            "status": "ok",
            "bars": len(df),
            "closed_pct": round((~df["market_open"]).mean() * 100, 1),
            "return_pct": round(g("Total Return [%]"), 2),
            "sharpe": round(g("Sharpe Ratio"), 2),
            "max_dd": round(g("Max Drawdown [%]"), 2),
            "trades": int(g("Total Trades")),
            "win_rate": round(g("Win Rate [%]"), 1),
            "profit_factor": round(g("Profit Factor"), 2),
        }
    except Exception as e:
        return {"sym": sym, "status": f"error: {type(e).__name__}: {e}"}


def main():
    syms = candidates()
    print(f"Running gap-fill backtest on {len(syms)} candidates (24 workers)...\n")
    results = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(worker, s): s for s in syms}
        for fut in as_completed(futs):
            results.append(fut.result())

    ok = [r for r in results if r.get("status") == "ok"]
    ok.sort(key=lambda r: -r["sharpe"])

    print(f"{'SYM':10}{'RET%':>9}{'SHARPE':>8}{'MAXDD':>8}{'TRADES':>8}{'WIN%':>7}{'PF':>6}{'CLOSED%':>9}{'BARS':>8}")
    for r in ok:
        print(f"{r['sym']:10}{r['return_pct']:>9.1f}{r['sharpe']:>8.2f}{r['max_dd']:>8.1f}"
              f"{r['trades']:>8}{r['win_rate']:>7.1f}{r['profit_factor']:>6.2f}{r['closed_pct']:>9.1f}{r['bars']:>8}")

    skipped = [r for r in results if r.get("status") != "ok"]
    if skipped:
        print(f"\nSkipped/errors ({len(skipped)}):")
        for r in sorted(skipped, key=lambda x: x["sym"]):
            print(f"  {r['sym']:10} {r['status']}")

    # Save CSV
    if ok:
        out = Path("data/lighter/gapfill_results.csv")
        pd.DataFrame(ok).to_csv(out, index=False)
        print(f"\nSaved -> {out}")
        # Aggregate stats
        df = pd.DataFrame(ok)
        profitable = df[df["return_pct"] > 0]
        print(f"\n=== Aggregate ===")
        print(f"  Profitable: {len(profitable)}/{len(df)} ({len(profitable)/len(df)*100:.0f}%)")
        print(f"  Median Sharpe: {df['sharpe'].median():.2f}")
        print(f"  Median Return: {df['return_pct'].median():.1f}%")
        print(f"  Top 5 by Sharpe: {', '.join(df.nlargest(5,'sharpe')['sym'])}")


if __name__ == "__main__":
    main()
