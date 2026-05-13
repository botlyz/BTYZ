"""BTYZ engine — analyse marimo générique (version simple).

§0  Cross-run table (sélecteur principal) — trie par sharpe moyen desc
§1  Fold-by-fold table (params + train/test metrics)
§2  Stabilité des paramètres par fold (line chart normalisé)
§3  WFE check (train vs test sharpe scatter + return par fold + KPI)
§4  Equity walk-forward (concaténée depuis trades parquet)
§5  Régimes table (selection='single' → §6 VBT replay du fold)
§6  VBT replay : price chart + entry/exit markers + equity + trades

Règles marimo:
- noms exportés sans préfixe `_` ; locales avec `_`
- pas de `return` inside if/else (utiliser mo.stop pour halt)
- un seul `return` par cellule au tout en fin (ou pas de return)
"""
import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


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
    return go, json, mo, np, pathlib, pd


@app.cell
def _constants(pathlib):
    RESULTS_ROOT = pathlib.Path("/home/devbox/BTYZ/results")
    DATA_1M = pathlib.Path("/home/devbox/BTYZ/data/raw/lighter/1m")
    return DATA_1M, RESULTS_ROOT


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

    def load_summary(approach_id, run_cfg, pair):
        sj = RESULTS_ROOT / approach_id / "full" / run_cfg / pair / "summary.json"
        if not sj.exists():
            return None
        with open(sj) as f:
            return json.load(f)

    def fmt_param(v):
        if isinstance(v, (list, tuple)):
            if all(isinstance(x, float) for x in v):
                return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"
            return str(list(v))
        if isinstance(v, float):
            return f"{v:.6g}"
        return v

    def flatten_summary(summary):
        rows = []
        for fd in summary.get("folds", []):
            params = fd.get("params", {})
            tm = fd.get("train_metrics", {}) or {}
            te = fd.get("test_metrics", {}) or {}
            row = {"fold": fd.get("fold", 0)}
            for k, v in params.items():
                row[f"p_{k}"] = v
            for prefix, m in [("train", tm), ("test", te)]:
                row[f"{prefix}_start"] = m.get("start_index")
                row[f"{prefix}_end"] = m.get("end_index")
                row[f"{prefix}_sharpe"] = safe_float(m.get("sharpe_ratio"))
                row[f"{prefix}_return_pct"] = safe_float(m.get("total_return_pct"))
                row[f"{prefix}_trades"] = safe_int(m.get("total_trades") or m.get("trades_count"))
                row[f"{prefix}_dd_pct"] = safe_float(m.get("max_drawdown_pct"))
                row[f"{prefix}_dd_dur_days"] = safe_float(m.get("max_drawdown_duration") or m.get("dd_dur_days"))
                row[f"{prefix}_pf"] = safe_float(m.get("profit_factor"))
                row[f"{prefix}_wr"] = safe_float(m.get("win_rate_pct"))
            rows.append(row)
        df = pd.DataFrame(rows)
        for c in ["train_start", "train_end", "test_start", "test_end"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        return df

    def load_ohlcv(pair, tf):
        fp = DATA_1M / f"{pair}.csv"
        if not fp.exists():
            return None
        df = pd.read_csv(fp, low_memory=False, usecols=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        df = df.set_index("date").sort_index()
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        return df.resample(tf, label="left", closed="left").agg(agg).dropna()

    def parse_tf_bps(run_name):
        parts = run_name.split("_")
        tf_raw = parts[0] if parts else ""
        tf = {"1m": "1min", "3m": "3min", "5m": "5min",
              "15m": "15min", "30m": "30min", "1h": "1h",
              "2h": "2h", "4h": "4h", "1d": "1D"}.get(tf_raw, tf_raw)
        bps = 0
        if len(parts) > 1:
            try:
                bps = int(parts[-1].replace("bps", ""))
            except ValueError:
                bps = 0
        return tf, bps

    return flatten_summary, fmt_param, load_ohlcv, load_summary, parse_tf_bps


# ─────────────────────────────────────────────────────────────────────
# §0 Cross-run aggregated table — PRIMARY SELECTOR
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _build_cross_run(RESULTS_ROOT, json, np, pd):
    rows = []
    if RESULTS_ROOT.exists():
        for _ap in sorted(RESULTS_ROOT.iterdir()):
            if not _ap.is_dir():
                continue
            _full = _ap / "full"
            if not _full.exists():
                continue
            for _run in sorted(_full.iterdir()):
                if not _run.is_dir():
                    continue
                _parts = _run.name.split("_")
                _tf = _parts[0] if _parts else "?"
                _bps = _parts[-1] if len(_parts) > 1 else "?"
                for _pair in sorted(_run.iterdir()):
                    _sj = _pair / "summary.json"
                    if not _sj.exists():
                        continue
                    try:
                        _d = json.loads(_sj.read_text())
                    except Exception:
                        continue
                    _ts, _rets, _dds = [], [], []
                    for _fr in _d.get("folds", []):
                        _tm = _fr.get("test_metrics", {}) or {}
                        for _arr, _key in ((_ts, "sharpe_ratio"),
                                           (_rets, "total_return_pct"),
                                           (_dds, "max_drawdown_pct")):
                            try:
                                _v = float(_tm.get(_key) or 0)
                                _arr.append(_v)
                            except (TypeError, ValueError):
                                pass
                    if not _ts:
                        continue
                    rows.append({
                        "approach": _ap.name,
                        "run": _run.name,
                        "tf": _tf,
                        "bps": _bps,
                        "pair": _pair.name,
                        "n_folds": len(_d.get("folds", [])),
                        "mean_sharpe": round(float(np.mean(_ts)), 2),
                        "median_sharpe": round(float(np.median(_ts)), 2),
                        "pct_positive": round((np.array(_rets) > 0).mean() * 100, 1) if _rets else 0.0,
                        "mean_ret_pct": round(float(np.mean(_rets)), 2) if _rets else 0.0,
                        "mean_dd_pct": round(float(np.mean(_dds)), 2) if _dds else 0.0,
                    })
    cross_run_df = pd.DataFrame(rows)
    if not cross_run_df.empty:
        cross_run_df = cross_run_df.sort_values("mean_sharpe", ascending=False).reset_index(drop=True)
    return (cross_run_df,)


@app.cell
def _cross_table_ui(cross_run_df, mo, pd):
    if cross_run_df.empty:
        cross_table = mo.ui.table(
            pd.DataFrame({"info": ["Aucun résultat dans results/. Lance python -m engine.cli wfa ..."]}),
            selection=None,
        )
        _content = mo.vstack([mo.md("## §0 Cross-run table"), cross_table])
    else:
        cross_table = mo.ui.table(cross_run_df, selection="single", page_size=25)
        _content = mo.vstack([
            mo.md("## §0 Cross-run table — trie cliquable, défaut = sharpe moyen décroissant"),
            mo.md("_Clique une ligne pour configurer les sections du dessous._"),
            cross_table,
        ])
    mo.output.replace(_content)
    return (cross_table,)


@app.cell
def _selection(cross_run_df, cross_table, mo):
    mo.stop(cross_run_df.empty, mo.callout(
        mo.md("Aucune donnée. Lance une opti WFA d'abord."), kind="warn"
    ))
    _sel = cross_table.value if cross_table is not None else None
    if _sel is not None and len(_sel) > 0:
        _row = _sel.iloc[0]
    else:
        _row = cross_run_df.iloc[0]
    approach_id = _row["approach"]
    run_cfg = _row["run"]
    pair = _row["pair"]
    mo.output.replace(mo.callout(
        mo.md(f"**Configuré sur :** `{approach_id}` / `{run_cfg}` / `{pair}` — "
              f"sharpe moyen `{_row.get('mean_sharpe', 0):.2f}` · "
              f"{_row.get('n_folds', 0)} folds · "
              f"{_row.get('pct_positive', 0):.0f}% folds positifs"),
        kind="success",
    ))
    return approach_id, pair, run_cfg


@app.cell
def _load_folds(approach_id, flatten_summary, load_summary, mo, pair, run_cfg):
    summary = load_summary(approach_id, run_cfg, pair)
    mo.stop(summary is None, mo.callout(mo.md("summary.json introuvable."), kind="warn"))
    folds_df = flatten_summary(summary)
    mo.stop(folds_df.empty, mo.callout(mo.md("Aucun fold valide."), kind="warn"))
    return folds_df, summary


# ─────────────────────────────────────────────────────────────────────
# §1 Fold-by-fold table
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section1(approach_id, fmt_param, folds_df, mo, pair, run_cfg, summary):
    _param_cols = [c for c in folds_df.columns if c.startswith("p_")]
    _display = ["fold"] + _param_cols + [
        "train_sharpe", "test_sharpe",
        "train_return_pct", "test_return_pct",
        "test_dd_pct", "test_dd_dur_days", "test_trades",
    ]
    _display = [c for c in _display if c in folds_df.columns]
    _df = folds_df[_display].copy()
    for _c in _param_cols:
        _df[_c] = _df[_c].apply(fmt_param)
    for _c in _df.select_dtypes(include="float").columns:
        _df[_c] = _df[_c].round(4)
    mo.output.replace(mo.vstack([
        mo.md(f"## §1 Folds — **{approach_id}** · {pair} · {run_cfg} "
              f"(fees={summary.get('fees')}, n_folds={summary.get('n_folds')})"),
        mo.ui.table(_df.reset_index(drop=True), selection=None, page_size=30),
    ]))


# ─────────────────────────────────────────────────────────────────────
# §2 Stabilité params per fold (fix bool subtraction)
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section2_stability(folds_df, go, mo, pair, pd, run_cfg):
    _param_cols = [c for c in folds_df.columns if c.startswith("p_")]
    _fig = go.Figure()
    for _pc in _param_cols:
        _vals = folds_df[_pc]
        if _vals.apply(lambda x: isinstance(x, (list, tuple))).any():
            continue
        _num = pd.to_numeric(_vals.apply(lambda x: int(x) if isinstance(x, bool) else x),
                             errors="coerce").astype(float)
        if _num.isna().all():
            continue
        _mn = float(_num.min())
        _mx = float(_num.max())
        if _mx == _mn:
            _norm = _num * 0 + 0.5
        else:
            _norm = (_num - _mn) / (_mx - _mn)
        _fig.add_trace(go.Scatter(
            x=folds_df["fold"], y=_norm, mode="lines+markers", name=_pc[2:],
            hovertemplate=f"<b>{_pc[2:]}</b>: %{{customdata:.4g}}<extra></extra>",
            customdata=_num,
        ))
    _fig.update_layout(
        title=f"{pair} · {run_cfg} — params par fold (normalisés 0→1)",
        xaxis_title="Fold", yaxis_title="Valeur normalisée", height=350,
    )
    mo.output.replace(mo.vstack([
        mo.md(f"## §2 Stabilité params — {pair}"),
        mo.ui.plotly(_fig),
    ]))


# ─────────────────────────────────────────────────────────────────────
# §3 WFE check
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section3_wfe(folds_df, go, mo, pair):
    _df_full = folds_df.dropna(subset=["train_sharpe", "test_sharpe"])
    if _df_full.empty:
        mo.output.replace(mo.md("## §3 WFE check — _pas assez de données._"))
    else:
        _corr = _df_full["train_sharpe"].corr(_df_full["test_sharpe"])
        _pct_pos = (_df_full["test_return_pct"] > 0).mean() * 100
        _fig_scatter = go.Figure()
        _fig_scatter.add_trace(go.Scatter(
            x=_df_full["train_sharpe"], y=_df_full["test_sharpe"],
            mode="markers+text", text=_df_full["fold"].astype(str), textposition="top center",
            marker=dict(size=10, color=_df_full["fold"], colorscale="Viridis",
                        showscale=True, colorbar=dict(title="Fold")),
        ))
        _lim = max(_df_full["train_sharpe"].abs().max(), _df_full["test_sharpe"].abs().max()) * 1.1 or 1.0
        _fig_scatter.add_shape(type="line", x0=-_lim, y0=-_lim, x1=_lim, y1=_lim,
                               line=dict(dash="dot", color="gray"))
        _fig_scatter.update_layout(
            title=f"{pair} — Sharpe train vs test (corr={_corr:.2f})",
            xaxis_title="Train Sharpe", yaxis_title="Test Sharpe", height=380,
        )
        _colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in _df_full["test_return_pct"]]
        _fig_bar = go.Figure([go.Bar(
            x=_df_full["fold"], y=_df_full["test_return_pct"], marker_color=_colors,
            name="Test return %",
        )])
        _fig_bar.add_trace(go.Scatter(
            x=_df_full["fold"], y=_df_full["train_return_pct"], mode="lines+markers",
            name="Train return %", line=dict(dash="dot", color="#3498db"),
        ))
        _fig_bar.update_layout(
            title=f"{pair} — Train vs Test return % par fold",
            xaxis_title="Fold", yaxis_title="Return %", height=300,
        )
        mo.output.replace(mo.vstack([
            mo.md(f"## §3 WFE check — {pair}"),
            mo.hstack([
                mo.stat(label="Corr train/test Sharpe", value=f"{_corr:.2f}"),
                mo.stat(label="Mean test Sharpe", value=f'{_df_full["test_sharpe"].mean():.2f}'),
                mo.stat(label="% folds rentables", value=f"{_pct_pos:.0f}%"),
                mo.stat(label="N folds", value=str(len(_df_full))),
            ], justify="start", gap=4),
            mo.ui.plotly(_fig_scatter),
            mo.ui.plotly(_fig_bar),
        ]))


# ─────────────────────────────────────────────────────────────────────
# §4 Walk-forward equity (from trades parquet)
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section4_equity(RESULTS_ROOT, approach_id, folds_df, go, mo, pair, pd, run_cfg):
    _trades_dir = RESULTS_ROOT / approach_id / "full" / run_cfg / pair / "trades"
    if not _trades_dir.exists():
        mo.output.replace(mo.md("## §4 Equity walk-forward — _pas de trades parquet._"))
    else:
        _INIT = 10_000.0
        _pieces = []
        _cumul = _INIT
        _fold_starts = []
        for _fr in sorted(folds_df.to_dict(orient="records"), key=lambda r: r["fold"]):
            _i = int(_fr["fold"])
            _fp = _trades_dir / f"fold_{_i}.parquet"
            if not _fp.exists():
                continue
            _t = pd.read_parquet(_fp)
            if _t.empty:
                continue
            _t = _t.sort_values("exit_time")
            _pnl_dollar = _t["pnl"].cumsum() + _cumul
            _ts = pd.to_datetime(_t["exit_time"], utc=True)
            _pieces.append(pd.DataFrame({"ts": _ts, "eq": _pnl_dollar}))
            _cumul = float(_pnl_dollar.iloc[-1])
            _fold_starts.append(_ts.iloc[0])

        if not _pieces:
            mo.output.replace(mo.md("## §4 Equity walk-forward — _aucun fold avec trades._"))
        else:
            _eq = pd.concat(_pieces, ignore_index=True).sort_values("ts").reset_index(drop=True)
            _running_max = _eq["eq"].cummax()
            _dd_pct = (_eq["eq"] / _running_max - 1) * 100
            _total_ret = (_eq["eq"].iloc[-1] / _INIT - 1) * 100
            _max_dd = float(_dd_pct.min())
            _n_trades_total = sum(len(p) for p in _pieces)

            _fig = go.Figure()
            _fig.add_trace(go.Scatter(
                x=_eq["ts"], y=_eq["eq"], mode="lines",
                name="Equity ($)", line=dict(color="#3498db", width=1.6),
            ))
            for _x in _fold_starts:
                _fig.add_vline(x=_x, line=dict(color="white", dash="dash", width=0.5), opacity=0.3)
            _fig.update_layout(
                title=f"{pair} · {run_cfg} — equity WF "
                      f"({len(_pieces)} folds · {_n_trades_total} trades)",
                yaxis_title="Equity ($)", height=380,
            )
            _fig_dd = go.Figure()
            _fig_dd.add_trace(go.Scatter(
                x=_eq["ts"], y=_dd_pct, mode="lines", fill="tozeroy",
                line=dict(color="#e74c3c", width=1), name="Drawdown %",
            ))
            _fig_dd.update_layout(title="Drawdown %", yaxis_title="%", height=220)
            mo.output.replace(mo.vstack([
                mo.md(f"## §4 Walk-forward equity — {pair}"),
                mo.hstack([
                    mo.stat(label="Return total", value=f"{_total_ret:.1f}%"),
                    mo.stat(label="Max DD", value=f"{_max_dd:.1f}%"),
                    mo.stat(label="N trades", value=str(_n_trades_total)),
                    mo.stat(label="N folds OK", value=f"{len(_pieces)}/{len(folds_df)}"),
                ], justify="start", gap=4),
                mo.ui.plotly(_fig),
                mo.ui.plotly(_fig_dd),
            ]))


# ─────────────────────────────────────────────────────────────────────
# §5 Régimes (clic → §6)
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section5_regimes(fmt_param, folds_df, mo, pair, run_cfg):
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
        mo.md(f"## §5 Régimes WF — {pair} · {run_cfg}"),
        mo.md("_Un régime = fenêtre train → params optimaux → perf test. Clique pour §6 (VBT replay)._"),
        regimes_table,
    ]))
    return (regimes_table,)


# ─────────────────────────────────────────────────────────────────────
# §6 VBT replay du fold cliqué (addtraces entries/exits)
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section6_replay(approach_id, folds_df, go, load_ohlcv, mo,
                     pair, parse_tf_bps, regimes_table, run_cfg):
    import sys as _sys
    import warnings as _w
    _sys.path.insert(0, "/home/devbox/BTYZ/src")
    _w.filterwarnings("ignore")

    _sel = regimes_table.value if regimes_table is not None else None
    if _sel is None or len(_sel) == 0:
        mo.output.replace(mo.md("## §6 VBT replay — _clique une ligne dans §5._"))
    else:
        _fold_idx = int(_sel.iloc[0]["fold"])
        _fold_row = folds_df[folds_df["fold"] == _fold_idx]
        if _fold_row.empty:
            mo.output.replace(mo.callout(mo.md(f"Fold {_fold_idx} introuvable."), kind="warn"))
        else:
            _r = _fold_row.iloc[0]
            _params = {k[2:]: v for k, v in _r.items() if k.startswith("p_")}
            _test_start = _r.get("test_start")
            _test_end = _r.get("test_end")
            _tf_str, _bps = parse_tf_bps(run_cfg)
            _fees = _bps * 1e-4

            _ohlcv = load_ohlcv(pair, _tf_str)
            if _ohlcv is None:
                mo.output.replace(mo.callout(mo.md(f"OHLCV {pair} introuvable."), kind="warn"))
            else:
                _si = int(_ohlcv.index.searchsorted(_test_start))
                _te = int(_ohlcv.index.searchsorted(_test_end, side="right"))
                _wi = max(0, _si - 300)
                _slc = _ohlcv.iloc[_wi:_te]

                _error = None
                _pf = None
                try:
                    from engine.approach_loader import instantiate_strategy
                    _strat = instantiate_strategy(approach_id)
                    _mod = _sys.modules.get(f"approach.{approach_id}.strategy")
                    if _mod is not None:
                        if hasattr(_mod, "_target_fees"):
                            _mod._target_fees = _fees
                        if hasattr(_mod, "_target_freq"):
                            _mod._target_freq = _tf_str
                    with _w.catch_warnings():
                        _w.simplefilter("ignore")
                        _pf = _strat.run_backtest(_slc, _params)
                except Exception as _e:
                    _error = str(_e)

                if _error is not None:
                    mo.output.replace(mo.callout(mo.md(f"Erreur VBT replay : `{_error}`"), kind="danger"))
                elif _pf is None:
                    mo.output.replace(mo.callout(mo.md("VBT replay returned None."), kind="warn"))
                else:
                    _equity = _pf.value.loc[_test_start:]
                    _trades = _pf.trades.records_readable.copy()
                    _idx = _pf.wrapper.index
                    if "Entry Index" in _trades.columns:
                        _trades["entry_time"] = _idx[_trades["Entry Index"].clip(upper=len(_idx)-1).astype(int)]
                    if "Exit Index" in _trades.columns:
                        _trades["exit_time"] = _idx[_trades["Exit Index"].clip(upper=len(_idx)-1).astype(int)]
                    if "entry_time" in _trades.columns:
                        _trades = _trades[_trades["entry_time"] >= _test_start]
                    _trades = _trades.rename(columns={
                        "Avg Entry Price": "entry_price", "Avg Exit Price": "exit_price",
                        "Size": "size", "Return": "return_pct", "PnL": "pnl", "Direction": "side",
                        "Entry Fees": "entry_fees", "Exit Fees": "exit_fees", "Status": "status",
                    })

                    _price_test = _slc["close"].loc[_test_start:]
                    _fig_px = go.Figure()
                    _fig_px.add_trace(go.Scatter(
                        x=_price_test.index, y=_price_test.values,
                        mode="lines", name="Close",
                        line=dict(color="#888", width=1),
                    ))
                    if "side" in _trades.columns and not _trades.empty:
                        _longs = _trades[_trades["side"].str.lower() == "long"]
                        _shorts = _trades[_trades["side"].str.lower() == "short"]
                        if not _longs.empty:
                            _fig_px.add_trace(go.Scatter(
                                x=_longs["entry_time"], y=_longs["entry_price"],
                                mode="markers", name="Long entry",
                                marker=dict(symbol="triangle-up", color="#2ecc71", size=11,
                                            line=dict(width=1, color="white")),
                            ))
                            _fig_px.add_trace(go.Scatter(
                                x=_longs["exit_time"], y=_longs["exit_price"],
                                mode="markers", name="Long exit",
                                marker=dict(symbol="x", color="#2ecc71", size=9,
                                            line=dict(width=1, color="white")),
                            ))
                        if not _shorts.empty:
                            _fig_px.add_trace(go.Scatter(
                                x=_shorts["entry_time"], y=_shorts["entry_price"],
                                mode="markers", name="Short entry",
                                marker=dict(symbol="triangle-down", color="#e74c3c", size=11,
                                            line=dict(width=1, color="white")),
                            ))
                            _fig_px.add_trace(go.Scatter(
                                x=_shorts["exit_time"], y=_shorts["exit_price"],
                                mode="markers", name="Short exit",
                                marker=dict(symbol="x", color="#e74c3c", size=9,
                                            line=dict(width=1, color="white")),
                            ))
                    _fig_px.update_layout(
                        title=f"{pair} fold {_fold_idx} — Price + entries/exits",
                        height=420, xaxis_title="Time", yaxis_title="Price",
                    )

                    _fig_eq = go.Figure()
                    _fig_eq.add_trace(go.Scatter(
                        x=_equity.index, y=_equity.values,
                        mode="lines", name="Equity ($)",
                        line=dict(color="#3498db", width=1.5),
                    ))
                    _fig_eq.update_layout(
                        title=f"{pair} fold {_fold_idx} — Equity ($)",
                        height=300, yaxis_title="Equity ($)",
                    )

                    _keep = ["entry_time", "exit_time", "side", "entry_price", "exit_price",
                             "size", "return_pct", "pnl"]
                    _keep = [c for c in _keep if c in _trades.columns]
                    _trades_show = _trades[_keep].copy()
                    if "return_pct" in _trades_show.columns:
                        _trades_show["return_pct"] = (_trades_show["return_pct"] * 100).round(4)
                    for _c in _trades_show.select_dtypes(include="float").columns:
                        _trades_show[_c] = _trades_show[_c].round(6)

                    _n = len(_trades_show)
                    _wr = (_trades_show["return_pct"] > 0).mean() * 100 if _n > 0 else 0
                    _pnl_total = _trades_show["pnl"].sum() if _n > 0 else 0
                    _ret_total = (_equity.iloc[-1] / _equity.iloc[0] - 1) * 100 if len(_equity) > 1 else 0

                    mo.output.replace(mo.vstack([
                        mo.md(f"## §6 VBT replay — Fold {_fold_idx} · {pair} ({_tf_str}, {_bps}bps)"),
                        mo.md(f"_Params: `{_params}`_"),
                        mo.hstack([
                            mo.stat(label="N trades", value=str(_n)),
                            mo.stat(label="Win rate", value=f"{_wr:.0f}%"),
                            mo.stat(label="Return", value=f"{_ret_total:.2f}%"),
                            mo.stat(label="PnL", value=f"${_pnl_total:.2f}"),
                        ], justify="start", gap=4),
                        mo.ui.plotly(_fig_px),
                        mo.ui.plotly(_fig_eq),
                        mo.ui.table(_trades_show.reset_index(drop=True), selection=None, page_size=50),
                    ]))


if __name__ == "__main__":
    app.run()
