"""Marimo notebook to analyze any BTYZ engine approach.

Drop a folder under src/approach/<ID>/strategy.py, launch a wfa run
(`python -m engine.cli wfa --approach <ID> ...`), then open this notebook
and pick the approach in the selector.

Sections:
  §0 selectors (approach / run config / pair)
  §1 fold table + params
  §2 cross-run macro view (fee sensitivity)
  §3 param stability per fold
  §4 WFE check (train vs test sharpe scatter)
  §5 walk-forward concatenated equity (rebuilt from trades parquet)
  §6 MCCV table (if available)
"""
import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _imports():
    import sys
    sys.path.insert(0, "/home/devbox/BTYZ/src")
    import json
    from pathlib import Path
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px

    RESULTS = Path("/home/devbox/BTYZ/results")
    return RESULTS, Path, go, json, mo, np, pd, px


@app.cell
def _list_approaches(RESULTS, mo):
    approaches = sorted([p.name for p in RESULTS.iterdir() if p.is_dir() and (p / "full").exists()])
    approach = mo.ui.dropdown(options=approaches, value=approaches[0] if approaches else None, label="Approach")
    approach
    return approach,


@app.cell
def _list_runs(RESULTS, approach, mo):
    if approach.value is None:
        runs = []
    else:
        full = RESULTS / approach.value / "full"
        runs = sorted([p.name for p in full.iterdir() if p.is_dir()]) if full.exists() else []
    run = mo.ui.dropdown(options=runs, value=runs[0] if runs else None, label="Run (tf_bps)")
    run
    return run, runs


@app.cell
def _list_pairs(RESULTS, approach, mo, run):
    if approach.value is None or run.value is None:
        pairs = []
    else:
        run_dir = RESULTS / approach.value / "full" / run.value
        pairs = sorted([p.name for p in run_dir.iterdir() if p.is_dir() and (p / "summary.json").exists()])
    pair = mo.ui.dropdown(options=pairs, value=pairs[0] if pairs else None, label="Pair")
    pair
    return pair, pairs


@app.cell
def _load_summary(RESULTS, approach, json, mo, pair, run):
    if not (approach.value and run.value and pair.value):
        mo.stop(True, mo.md("⚠️ Sélectionne approach / run / pair."))
    sj_path = RESULTS / approach.value / "full" / run.value / pair.value / "summary.json"
    with open(sj_path) as f:
        summary = json.load(f)
    mo.md(f"**{summary['approach_id']} · {summary['pair']} · {summary['tf']} · fees={summary['fees']}** — {summary['n_folds']} folds")
    return summary,


@app.cell
def _folds_df(pd, summary):
    rows = []
    for fr in summary["folds"]:
        row = {"fold": fr["fold"]}
        for k, v in fr["params"].items():
            row[f"p_{k}"] = v
        for k, v in fr.get("train_metrics", {}).items():
            row[f"train_{k}"] = v
        for k, v in fr.get("test_metrics", {}).items():
            row[f"test_{k}"] = v
        rows.append(row)
    folds_df = pd.DataFrame(rows)
    return folds_df,


@app.cell
def _section_1_fold_table(folds_df, mo):
    keep = ["fold"] + [c for c in folds_df.columns if c.startswith("p_")] \
         + ["train_sharpe_ratio", "train_total_return_pct",
            "test_sharpe_ratio", "test_total_return_pct",
            "test_max_drawdown_pct", "test_total_trades"]
    keep = [c for c in keep if c in folds_df.columns]
    mo.md("## §1 Fold-by-fold view"), mo.ui.table(folds_df[keep])
    return


@app.cell
def _section_2_cross_run(RESULTS, approach, go, json, mo, np, pair):
    """Fee sensitivity per run config (same pair across all runs)."""
    if not (approach.value and pair.value):
        return
    full = RESULTS / approach.value / "full"
    rows = []
    for run_dir in sorted(full.iterdir()):
        sj = run_dir / pair.value / "summary.json"
        if not sj.exists():
            continue
        d = json.loads(sj.read_text())
        test_rets = [f["test_metrics"].get("total_return_pct") or 0 for f in d["folds"]]
        test_sharpes = [f["test_metrics"].get("sharpe_ratio") or 0 for f in d["folds"]]
        n_prof = sum(1 for r in test_rets if r > 0)
        rows.append({
            "run": run_dir.name,
            "n_folds": len(d["folds"]),
            "avg_ret_pct": float(np.mean(test_rets)) if test_rets else 0,
            "med_sharpe": float(np.median(test_sharpes)) if test_sharpes else 0,
            "n_prof": n_prof,
        })
    if not rows:
        return
    fig = go.Figure()
    fig.add_bar(x=[r["run"] for r in rows], y=[r["avg_ret_pct"] for r in rows], name="avg test return %")
    fig.update_layout(height=320, title=f"{approach.value} / {pair.value} — fee sensitivity across runs")
    mo.md("## §2 Cross-run fee sensitivity"), mo.ui.plotly(fig)
    return


@app.cell
def _section_3_param_stability(folds_df, go, mo, summary):
    param_cols = [c for c in folds_df.columns if c.startswith("p_")]
    fig = go.Figure()
    for c in param_cols:
        vals = folds_df[c].astype(float, errors="ignore")
        rng = vals.max() - vals.min()
        norm = (vals - vals.min()) / rng if rng else vals * 0
        fig.add_scatter(x=folds_df["fold"], y=norm, mode="lines+markers", name=c.replace("p_", ""))
    fig.update_layout(height=350, title="Param stability across folds (normalized 0–1)")
    mo.md("## §3 Param stability per fold"), mo.ui.plotly(fig)
    return


@app.cell
def _section_4_wfe_check(folds_df, go, mo, np):
    if "train_sharpe_ratio" not in folds_df.columns or "test_sharpe_ratio" not in folds_df.columns:
        return
    fig = go.Figure()
    fig.add_scatter(x=folds_df["train_sharpe_ratio"], y=folds_df["test_sharpe_ratio"],
                    mode="markers+text", text=folds_df["fold"], textposition="top center")
    m = float(max(folds_df["train_sharpe_ratio"].abs().max(), folds_df["test_sharpe_ratio"].abs().max()) or 1)
    fig.add_shape(type="line", x0=-m, y0=-m, x1=m, y1=m, line=dict(color="grey", dash="dash"))
    fig.update_layout(height=400, title="WFE: train sharpe vs test sharpe (fold #)",
                      xaxis_title="train_sharpe", yaxis_title="test_sharpe")
    try:
        corr = np.corrcoef(folds_df["train_sharpe_ratio"].fillna(0), folds_df["test_sharpe_ratio"].fillna(0))[0, 1]
    except Exception:
        corr = float("nan")
    mo.md(f"## §4 WFE check (corr={corr:.3f})"), mo.ui.plotly(fig)
    return


@app.cell
def _section_5_wf_equity(RESULTS, approach, go, mo, pair, pd, run, summary):
    """Concatenate per-fold equity from trades parquet to get the walk-forward equity curve."""
    if not summary:
        return
    pair_dir = RESULTS / approach.value / "full" / run.value / pair.value
    trades_dir = pair_dir / "trades"
    pieces = []
    cumul = 1.0
    for fr in summary["folds"]:
        i = fr["fold"]
        fp = trades_dir / f"fold_{i}.parquet"
        if not fp.exists():
            continue
        t = pd.read_parquet(fp)
        if t.empty:
            continue
        t = t.sort_values("exit_time")
        # Build cumulative pnl normalized at fold start
        pnl = t["pnl"].cumsum() / 10000.0 + 1.0  # init_cash = 10_000
        ts = pd.to_datetime(t["exit_time"], utc=True)
        # rebase to running cumulative product
        scaled = (pnl / pnl.iloc[0] - 1.0) * cumul + cumul
        pieces.append(pd.DataFrame({"ts": ts, "eq": scaled}))
        cumul = float(scaled.iloc[-1])

    if not pieces:
        return
    eq = pd.concat(pieces, ignore_index=True).sort_values("ts")
    fig = go.Figure()
    fig.add_scatter(x=eq["ts"], y=eq["eq"], mode="lines", name="WF equity (×init)")
    fig.update_layout(height=380, title=f"§5 Walk-forward concatenated equity — {pair.value} ({run.value})")
    mo.ui.plotly(fig)
    return


@app.cell
def _section_6_mccv(RESULTS, approach, go, json, mo, pair, pd):
    """If MCCV results exist, show oos sharpe distribution."""
    mccv_dir = RESULTS / approach.value / "mccv"
    if not mccv_dir.exists() or not pair.value:
        return
    matches = list(mccv_dir.glob(f"{pair.value}_*.json"))
    if not matches:
        return
    fig = go.Figure()
    summary_rows = []
    for fp in matches:
        d = json.loads(fp.read_text())
        rows = d.get("rows", [])
        oos = [r["oos_sh"] for r in rows]
        if not oos:
            continue
        fig.add_box(y=oos, name=f"{d['tf']}_{d['bps']}bps")
        summary_rows.append({
            "config": f"{d['tf']}_{d['bps']}bps",
            "n_targets": len(rows),
            "median_oos_sh": float(pd.Series(oos).median()),
            "pct_positive": round(sum(1 for s in oos if s > 0) / len(oos) * 100, 1),
        })
    fig.update_layout(height=400, title=f"§6 MCCV OOS Sharpe distribution — {pair.value}")
    mo.md("## §6 MCCV"), mo.ui.plotly(fig), mo.ui.table(pd.DataFrame(summary_rows))
    return


if __name__ == "__main__":
    app.run()
