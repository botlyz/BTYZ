"""BTYZ engine — analyse marimo générique.

Inspiré de AlphaSearchV2/analyse_042x.py.

Pour chaque approche déposée sous src/approach/<ID>/ et tournée via
`python -m engine.cli wfa ...`, ce notebook charge les résultats et offre:

§0 Selectors : approach / run_cfg (tf_bps) / pair
§1 Fold-by-fold table (params + train/test metrics)
§2 Cross-run fee sensitivity (par paire, sur tous les bps disponibles)
§3 Stabilité des paramètres par fold (line chart normalisé)
§4 WFE check (train vs test sharpe scatter + return par fold + KPI stats)
§5 Equity walk-forward (concaténée depuis trades parquet)
§6 Table régimes (avec selection='single' → §7 trades du fold cliqué)
§7 Trades du régime sélectionné

Run : `marimo edit /home/devbox/BTYZ/notebooks/analyse/analyse_engine.py`
"""
import marimo

__generated_with = "0.13.2"
app = marimo.App(width="full")


# ─────────────────────────────────────────────────────────────────────
# Imports + constantes
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _imports():
    import sys as _sys
    _sys.path.insert(0, "/home/devbox/BTYZ/src")

    import json
    import pathlib
    import warnings as _w

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.renderers.default = "png"
    _w.filterwarnings("ignore", category=RuntimeWarning)
    return go, json, mo, np, pathlib, pd, pio


@app.cell
def _constants(pathlib):
    RESULTS_ROOT = pathlib.Path("/home/devbox/BTYZ/results")
    DATA_1M = pathlib.Path("/home/devbox/BTYZ/data/raw/lighter/1m")
    return DATA_1M, RESULTS_ROOT


@app.cell
def _discover(RESULTS_ROOT):
    """Liste les approaches qui ont au moins 1 run avec un summary.json."""
    approaches = []
    if RESULTS_ROOT.exists():
        for d in sorted(RESULTS_ROOT.iterdir()):
            if not d.is_dir():
                continue
            full = d / "full"
            if not full.exists():
                continue
            for run_dir in full.iterdir():
                if any((p / "summary.json").exists() for p in run_dir.iterdir() if p.is_dir()):
                    approaches.append(d.name)
                    break
    return (sorted(set(approaches)),)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _helpers(DATA_1M, RESULTS_ROOT, json, pd):
    def safe_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    def safe_int(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0

    def list_runs(approach_id):
        d = RESULTS_ROOT / approach_id / "full"
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir())

    def list_pairs(approach_id, run_cfg):
        d = RESULTS_ROOT / approach_id / "full" / run_cfg
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir()
                      if p.is_dir() and (p / "summary.json").exists())

    def load_summary(approach_id, run_cfg, pair):
        sj = RESULTS_ROOT / approach_id / "full" / run_cfg / pair / "summary.json"
        if not sj.exists():
            return None
        with open(sj) as f:
            return json.load(f)

    def fmt_param(v):
        """Lists → '[a, b, c]', floats → 4 decimals, else raw."""
        if isinstance(v, (list, tuple)):
            if all(isinstance(x, float) for x in v):
                return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"
            return str(list(v))
        if isinstance(v, float):
            return f"{v:.6g}"
        return v

    def flatten_summary(summary):
        """summary → list[dict] (1 row per fold)."""
        rows = []
        for fd in summary.get("folds", []):
            params = fd.get("params", {})
            tm = fd.get("train_metrics", {}) or {}
            te = fd.get("test_metrics", {}) or {}
            row = {"fold": fd.get("fold", 0)}
            for k, v in params.items():
                row[f"p_{k}"] = v
            for prefix, m in [("train", tm), ("test", te)]:
                row[f"{prefix}_start"]       = m.get("start_index")
                row[f"{prefix}_end"]         = m.get("end_index")
                row[f"{prefix}_sharpe"]      = safe_float(m.get("sharpe_ratio"))
                row[f"{prefix}_return_pct"]  = safe_float(m.get("total_return_pct"))
                row[f"{prefix}_trades"]      = safe_int(m.get("total_trades") or m.get("trades_count"))
                row[f"{prefix}_dd_pct"]      = safe_float(m.get("max_drawdown_pct"))
                row[f"{prefix}_dd_dur_days"] = safe_float(m.get("max_drawdown_duration") or m.get("dd_dur_days"))
                row[f"{prefix}_sortino"]    = safe_float(m.get("sortino_ratio"))
                row[f"{prefix}_calmar"]     = safe_float(m.get("calmar_ratio"))
                row[f"{prefix}_pf"]         = safe_float(m.get("profit_factor"))
                row[f"{prefix}_wr"]         = safe_float(m.get("win_rate_pct"))
            rows.append(row)
        df = pd.DataFrame(rows)
        for c in ["train_start", "train_end", "test_start", "test_end"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        return df

    def load_ohlcv(pair, tf):
        """1m CSV → resample to tf (e.g. '5min')."""
        fp = DATA_1M / f"{pair}.csv"
        if not fp.exists():
            return None
        df = pd.read_csv(fp, low_memory=False, usecols=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        df = df.set_index("date").sort_index()
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        return df.resample(tf, label="left", closed="left").agg(agg).dropna()

    def fees_from_run(run_cfg):
        try:
            bps = int(run_cfg.split("_")[-1].replace("bps", ""))
            return bps * 1e-4
        except Exception:
            return 0.0

    return (flatten_summary, fees_from_run, fmt_param, list_pairs, list_runs,
            load_ohlcv, load_summary, safe_float, safe_int)


# ─────────────────────────────────────────────────────────────────────
# §0 Selectors
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _selectors(_discover, mo):
    approaches = _discover[0] if isinstance(_discover, tuple) else _discover
    return (approaches,)


@app.cell
def _approach_selector(approaches, mo):
    mo.stop(not approaches, mo.callout(
        mo.md("Aucune approche trouvée dans `results/`. Lance d'abord `python -m engine.cli wfa --approach <ID> ...`"),
        kind="warn"
    ))
    approach = mo.ui.dropdown(
        options={a: a for a in approaches},
        value=approaches[0],
        label="Approach",
    )
    mo.output.replace(mo.hstack([approach], justify="start", gap=2))
    return (approach,)


@app.cell
def _run_selector(approach, list_runs, mo):
    runs = list_runs(approach.value) if approach.value else []
    mo.stop(not runs, mo.callout(mo.md(f"Aucun run trouvé pour `{approach.value}`"), kind="warn"))
    run_cfg = mo.ui.dropdown(
        options={r: r for r in runs},
        value=runs[0],
        label="Run (tf_bps)",
    )
    mo.output.replace(mo.hstack([run_cfg], justify="start", gap=2))
    return (run_cfg,)


@app.cell
def _pair_selector(approach, list_pairs, mo, run_cfg):
    pairs = list_pairs(approach.value, run_cfg.value) if approach.value and run_cfg.value else []
    mo.stop(not pairs, mo.callout(mo.md(f"Aucune paire dans `{run_cfg.value}`"), kind="warn"))
    pair = mo.ui.dropdown(
        options={p: p for p in pairs},
        value=pairs[0],
        label="Paire",
    )
    mo.output.replace(mo.hstack([pair], justify="start", gap=2))
    return (pair,)


# ─────────────────────────────────────────────────────────────────────
# Load summary + flatten
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _load(approach, flatten_summary, load_summary, mo, pair, run_cfg):
    summary = load_summary(approach.value, run_cfg.value, pair.value)
    mo.stop(summary is None, mo.callout(mo.md("summary.json introuvable."), kind="warn"))
    folds_df = flatten_summary(summary)
    mo.stop(folds_df.empty, mo.callout(mo.md("Aucun fold valide dans ce summary."), kind="warn"))
    return folds_df, summary


# ─────────────────────────────────────────────────────────────────────
# §1 Fold-by-fold table
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section1(approach, fmt_param, folds_df, mo, pair, pd, run_cfg, summary):
    _param_cols = [c for c in folds_df.columns if c.startswith("p_")]
    _display = ["fold"] + _param_cols + [
        "train_sharpe", "test_sharpe",
        "train_return_pct", "test_return_pct",
        "test_dd_pct", "test_dd_dur_days", "test_trades",
    ]
    _display = [c for c in _display if c in folds_df.columns]

    _df = folds_df[_display].copy()
    # Format lists & floats for display
    for _c in _param_cols:
        _df[_c] = _df[_c].apply(fmt_param)
    for _c in _df.select_dtypes(include="float").columns:
        _df[_c] = _df[_c].round(4)

    mo.output.replace(mo.vstack([
        mo.md(f"## §1 Folds — **{approach.value}** · {pair.value} · {run_cfg.value} "
              f"(fees={summary.get('fees')}, n_folds={summary.get('n_folds')})"),
        mo.ui.table(_df.reset_index(drop=True), selection=None, page_size=30),
    ]))


# ─────────────────────────────────────────────────────────────────────
# §2 Cross-run fee sensitivity
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section2(RESULTS_ROOT, approach, go, json, list_runs, mo, np, pair, pd, safe_float):
    runs = list_runs(approach.value)
    rows = []
    for rc in runs:
        sj = RESULTS_ROOT / approach.value / "full" / rc / pair.value / "summary.json"
        if not sj.exists():
            continue
        d = json.loads(sj.read_text())
        test_rets = [safe_float(f.get("test_metrics", {}).get("total_return_pct")) for f in d.get("folds", [])]
        test_shs  = [safe_float(f.get("test_metrics", {}).get("sharpe_ratio"))   for f in d.get("folds", [])]
        clean_rets = [r for r in test_rets if not np.isnan(r)]
        clean_shs  = [s for s in test_shs  if not np.isnan(s)]
        n_prof = sum(1 for r in clean_rets if r > 0)
        rows.append({
            "run": rc,
            "n_folds": len(d.get("folds", [])),
            "avg_test_ret_pct": float(np.mean(clean_rets)) if clean_rets else 0.0,
            "median_test_sharpe": float(np.median(clean_shs)) if clean_shs else 0.0,
            "n_prof_folds": n_prof,
            "pct_prof": (n_prof / len(clean_rets) * 100) if clean_rets else 0.0,
        })

    if not rows:
        mo.output.replace(mo.md("## §2 Cross-run fee sensitivity\n_Pas de données._"))
    else:
        rs = pd.DataFrame(rows)
        _colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in rs["avg_test_ret_pct"]]
        _fig = go.Figure([go.Bar(
            x=rs["run"], y=rs["avg_test_ret_pct"], marker_color=_colors,
            text=[f"{r:.1f}%" for r in rs["avg_test_ret_pct"]], textposition="outside",
        )])
        _fig.update_layout(
            title=f"{pair.value} — return WF moyen par run (fee sensitivity)",
            yaxis_title="avg test return %", height=320,
            shapes=[dict(type="line", x0=-0.5, x1=len(rs)-0.5, y0=0, y1=0,
                         line=dict(color="white", dash="dot"))],
        )
        _fig2 = go.Figure([go.Bar(
            x=rs["run"], y=rs["pct_prof"], marker_color=["#3498db"] * len(rs),
            text=[f"{int(n)}/{int(t)}" for n, t in zip(rs["n_prof_folds"], rs["n_folds"])],
            textposition="outside",
        )])
        _fig2.update_layout(
            title=f"{pair.value} — % folds rentables par run",
            yaxis_title="% folds rentables", height=280,
        )
        for _c in rs.select_dtypes(include="float").columns:
            rs[_c] = rs[_c].round(3)
        mo.output.replace(mo.vstack([
            mo.md(f"## §2 Cross-run fee sensitivity — {pair.value}"),
            mo.ui.table(rs, selection=None),
            mo.ui.plotly(_fig),
            mo.ui.plotly(_fig2),
        ]))


# ─────────────────────────────────────────────────────────────────────
# §3 Param stability per fold
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section3(folds_df, go, mo, pair, pd, run_cfg):
    _param_cols = [c for c in folds_df.columns if c.startswith("p_")]
    _fig = go.Figure()
    for _pc in _param_cols:
        _vals = folds_df[_pc]
        # Skip list-valued params (env_levels, allocations) — plot via separate trace
        if _vals.apply(lambda x: isinstance(x, list)).any():
            continue
        _num = pd.to_numeric(_vals, errors="coerce")
        if _num.isna().all():
            continue
        _mn, _mx = _num.min(), _num.max()
        _norm = (_num - _mn) / (_mx - _mn + 1e-12) if _mx != _mn else _num * 0 + 0.5
        _fig.add_trace(go.Scatter(
            x=folds_df["fold"], y=_norm, mode="lines+markers", name=_pc[2:],
            hovertemplate=f"<b>{_pc[2:]}</b>: %{{customdata:.4g}}<extra></extra>",
            customdata=_num,
        ))
    _fig.update_layout(
        title=f"{pair.value} · {run_cfg.value} — params par fold (normalisés 0→1)",
        xaxis_title="Fold", yaxis_title="Valeur normalisée", height=350,
    )
    mo.output.replace(mo.vstack([
        mo.md(f"## §3 Stabilité params — {pair.value}"),
        mo.ui.plotly(_fig),
    ]))


# ─────────────────────────────────────────────────────────────────────
# §4 WFE check
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section4(folds_df, go, mo, pair):
    _df = folds_df.dropna(subset=["train_sharpe", "test_sharpe"])
    if _df.empty:
        mo.output.replace(mo.md("## §4 WFE check\n_Pas assez de données._"))
    else:
        _corr = _df["train_sharpe"].corr(_df["test_sharpe"])
        _pct_pos = (_df["test_return_pct"] > 0).mean() * 100

        _fig_scatter = go.Figure()
        _fig_scatter.add_trace(go.Scatter(
            x=_df["train_sharpe"], y=_df["test_sharpe"],
            mode="markers+text", text=_df["fold"].astype(str), textposition="top center",
            marker=dict(size=10, color=_df["fold"], colorscale="Viridis",
                        showscale=True, colorbar=dict(title="Fold")),
        ))
        _lim = max(_df["train_sharpe"].abs().max(), _df["test_sharpe"].abs().max()) * 1.1 or 1.0
        _fig_scatter.add_shape(type="line", x0=-_lim, y0=-_lim, x1=_lim, y1=_lim,
                               line=dict(dash="dot", color="gray"))
        _fig_scatter.update_layout(
            title=f"{pair.value} — Sharpe train vs test (corr={_corr:.2f})",
            xaxis_title="Train Sharpe", yaxis_title="Test Sharpe", height=380,
        )

        _colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in _df["test_return_pct"]]
        _fig_bar = go.Figure([go.Bar(
            x=_df["fold"], y=_df["test_return_pct"], marker_color=_colors,
            name="Test return %",
        )])
        _fig_bar.add_trace(go.Scatter(
            x=_df["fold"], y=_df["train_return_pct"], mode="lines+markers",
            name="Train return %", line=dict(dash="dot", color="#3498db"),
        ))
        _fig_bar.update_layout(
            title=f"{pair.value} — Train vs Test return % par fold",
            xaxis_title="Fold", yaxis_title="Return %", height=300,
        )

        mo.output.replace(mo.vstack([
            mo.md(f"## §4 WFE check — {pair.value}"),
            mo.hstack([
                mo.stat(label="Corr train/test Sharpe", value=f"{_corr:.2f}"),
                mo.stat(label="Mean test Sharpe", value=f'{_df["test_sharpe"].mean():.2f}'),
                mo.stat(label="% folds rentables", value=f"{_pct_pos:.0f}%"),
                mo.stat(label="N folds", value=str(len(_df))),
            ], justify="start", gap=4),
            mo.ui.plotly(_fig_scatter),
            mo.ui.plotly(_fig_bar),
        ]))


# ─────────────────────────────────────────────────────────────────────
# §5 Walk-forward equity (rebuilt from trades parquet)
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section5(RESULTS_ROOT, approach, folds_df, go, mo, pair, pd, run_cfg):
    """Concaténation des PnL par fold depuis trades parquet — pas de re-run kernel."""
    trades_dir = RESULTS_ROOT / approach.value / "full" / run_cfg.value / pair.value / "trades"
    if not trades_dir.exists():
        mo.output.replace(mo.md("## §5 Equity walk-forward\n_Pas de trades parquet._"))
    else:
        # Use init_cash=10_000 (BTYZ default) — equity in $ relative to start
        INIT = 10_000.0
        pieces = []
        cumul = INIT
        fold_starts = []
        for fr in sorted(folds_df.to_dict(orient="records"), key=lambda r: r["fold"]):
            i = int(fr["fold"])
            fp = trades_dir / f"fold_{i}.parquet"
            if not fp.exists():
                continue
            t = pd.read_parquet(fp)
            if t.empty:
                continue
            t = t.sort_values("exit_time")
            # Cumulative dollar PnL within this fold, then rebase to running cumul
            pnl_dollar = t["pnl"].cumsum() + cumul
            ts = pd.to_datetime(t["exit_time"], utc=True)
            pieces.append(pd.DataFrame({"ts": ts, "eq": pnl_dollar}))
            cumul = float(pnl_dollar.iloc[-1])
            fold_starts.append(ts.iloc[0])

        if not pieces:
            mo.output.replace(mo.md("## §5 Equity walk-forward\n_Aucun fold avec trades._"))
        else:
            eq = pd.concat(pieces, ignore_index=True).sort_values("ts").reset_index(drop=True)
            running_max = eq["eq"].cummax()
            dd_pct = (eq["eq"] / running_max - 1) * 100
            total_ret = (eq["eq"].iloc[-1] / INIT - 1) * 100
            max_dd = float(dd_pct.min())
            n_trades_total = sum(len(p) for p in pieces)

            _fig = go.Figure()
            _fig.add_trace(go.Scatter(
                x=eq["ts"], y=eq["eq"], mode="lines",
                name="Equity ($)", line=dict(color="#3498db", width=1.6),
            ))
            for _ts in fold_starts:
                _fig.add_vline(x=_ts, line=dict(color="white", dash="dash", width=0.5), opacity=0.3)
            _fig.update_layout(
                title=f"{pair.value} · {run_cfg.value} — equity WF concaténée ({len(pieces)} folds · {n_trades_total} trades)",
                yaxis_title="Equity ($)", height=380,
            )

            _fig_dd = go.Figure()
            _fig_dd.add_trace(go.Scatter(
                x=eq["ts"], y=dd_pct, mode="lines", fill="tozeroy",
                line=dict(color="#e74c3c", width=1), name="Drawdown %",
            ))
            _fig_dd.update_layout(
                title="Drawdown %", yaxis_title="%", height=220,
            )

            mo.output.replace(mo.vstack([
                mo.md(f"## §5 Walk-forward equity — {pair.value}"),
                mo.hstack([
                    mo.stat(label="Return total", value=f"{total_ret:.1f}%"),
                    mo.stat(label="Max DD", value=f"{max_dd:.1f}%"),
                    mo.stat(label="N trades", value=str(n_trades_total)),
                    mo.stat(label="N folds OK", value=f"{len(pieces)}/{len(folds_df)}"),
                ], justify="start", gap=4),
                mo.ui.plotly(_fig),
                mo.ui.plotly(_fig_dd),
            ]))


# ─────────────────────────────────────────────────────────────────────
# §6 Régimes (clic → §7)
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section6(fmt_param, folds_df, mo, pair, run_cfg):
    _df = folds_df.copy()
    _param_cols = [c for c in _df.columns if c.startswith("p_")]

    for _c in ["train_start", "train_end", "test_start", "test_end"]:
        if _c in _df.columns:
            _df[_c] = _df[_c].dt.strftime("%Y-%m-%d %H:%M").fillna("")

    for _c in _param_cols:
        _df[_c] = _df[_c].apply(fmt_param)

    _display = (
        ["fold", "train_start", "train_end", "test_start", "test_end"]
        + _param_cols
        + ["test_sharpe", "test_return_pct", "test_trades",
           "test_dd_pct", "test_dd_dur_days", "test_pf", "test_wr"]
    )
    _display = [c for c in _display if c in _df.columns]
    _df_show = _df[_display].reset_index(drop=True)
    for _c in _df_show.select_dtypes(include="float").columns:
        _df_show[_c] = _df_show[_c].round(4)

    regimes_table = mo.ui.table(_df_show, selection="single", page_size=30)

    mo.output.replace(mo.vstack([
        mo.md(f"## §6 Régimes WF — {pair.value} · {run_cfg.value}"),
        mo.md("_Un régime = fenêtre train → params optimaux → perf test. Clique pour voir les trades §7._"),
        regimes_table,
    ]))
    return (regimes_table,)


# ─────────────────────────────────────────────────────────────────────
# §7 Trades du régime sélectionné
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section7(RESULTS_ROOT, approach, folds_df, mo, pair, pd, regimes_table, run_cfg):
    _sel = regimes_table.value
    if _sel is None or len(_sel) == 0:
        mo.output.replace(mo.md("## §7 Trades du régime\n_Clique sur une ligne du tableau §6 pour afficher les trades._"))
    else:
        _fold_idx = int(_sel.iloc[0]["fold"])
        _trades_fp = RESULTS_ROOT / approach.value / "full" / run_cfg.value / pair.value / "trades" / f"fold_{_fold_idx}.parquet"
        if not _trades_fp.exists():
            mo.output.replace(mo.callout(mo.md(f"Pas de trades parquet pour fold {_fold_idx}."), kind="warn"))
        else:
            _t = pd.read_parquet(_trades_fp).copy()
            _fold_row = folds_df[folds_df["fold"] == _fold_idx]
            if not _fold_row.empty:
                _r = _fold_row.iloc[0]
                _params = {k[2:]: v for k, v in _r.items() if k.startswith("p_")}
            else:
                _params = {}

            _n = len(_t)
            _wr = (_t["return_pct"] > 0).mean() * 100 if _n > 0 else 0
            _pnl_total = _t["pnl"].sum() if _n > 0 else 0
            _wins = _t[_t["pnl"] > 0]["pnl"].sum()
            _losses = abs(_t[_t["pnl"] < 0]["pnl"].sum())
            _pf = _wins / _losses if _losses > 0 else float("inf")

            for _c in _t.select_dtypes(include="float").columns:
                _t[_c] = _t[_c].round(6)

            mo.output.replace(mo.vstack([
                mo.md(f"## §7 Trades — Fold {_fold_idx} · {pair.value}"),
                mo.md(f"_Params: `{_params}`_"),
                mo.hstack([
                    mo.stat(label="N trades", value=str(_n)),
                    mo.stat(label="Win rate", value=f"{_wr:.0f}%"),
                    mo.stat(label="PnL total", value=f"{_pnl_total:.2f}$"),
                    mo.stat(label="Profit factor", value=f"{_pf:.2f}"),
                ], justify="start", gap=4),
                mo.ui.table(_t.reset_index(drop=True), selection=None, page_size=50),
            ]))


if __name__ == "__main__":
    app.run()
