"""BTYZ engine — analyse marimo générique.

§0  Cross-run table — filtres + tri sharpe moyen desc (sélecteur principal)
§1  Fold-by-fold table (params + train/test metrics) pour la combo sélectionnée
§2  Stabilité des paramètres par fold (line chart normalisé 0→1)
§3  WFE check (train vs test sharpe scatter + return par fold + KPI)
§4  VBT replay walk-forward — re-run kernel par fold, concat OOS, resample 1D

Règles marimo:
- Noms exportés sans préfixe `_` ; locales préfixées `_`
- Pas de `return` inside if/else (mo.stop pour halt)
- Single return par cellule en fin

Données:
- BASE_OHLCV = data/raw/lighter/1m/<PAIR>.csv (les vraies données brutes
  Lighter en 1-minute). On RESAMPLE à la timeframe du run (3min, 5min, 15min…).
  Le suffixe '1m' du dossier indique juste la granularité source : il n'y
  a pas de répertoire 5min séparé, on construit le 5m à partir du 1m.
"""
import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


# ──────────────────────────────────────────────────────────────────────
# Imports / constants / helpers
# ──────────────────────────────────────────────────────────────────────

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
    BASE_OHLCV = pathlib.Path("/home/devbox/BTYZ/data/raw/lighter/1m")
    LIQUIDITY_JSON = pathlib.Path("/home/devbox/BTYZ/liquidity.json")
    return BASE_OHLCV, LIQUIDITY_JSON, RESULTS_ROOT


@app.cell
def _helpers(BASE_OHLCV, LIQUIDITY_JSON, RESULTS_ROOT, json, pd):
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
        """Charge le CSV 1m de Lighter puis resample à la timeframe demandée."""
        fp = BASE_OHLCV / f"{pair}.csv"
        if not fp.exists():
            return None
        df = pd.read_csv(fp, low_memory=False,
                         usecols=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        df = df.set_index("date").sort_index()
        agg = {"open": "first", "high": "max", "low": "min",
               "close": "last", "volume": "sum"}
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

    def load_liquidity_map():
        if not LIQUIDITY_JSON.exists():
            return {}
        with open(LIQUIDITY_JSON) as f:
            d = json.load(f)
        out = {}
        for level in ("tres_liquide", "liquide", "moyen", "poubelle"):
            for p in d.get(level, []):
                out[p] = level
        return out

    return (flatten_summary, fmt_param, load_liquidity_map,
            load_ohlcv, load_summary, parse_tf_bps)


# ──────────────────────────────────────────────────────────────────────
# §0 Build cross-run aggregate
# ──────────────────────────────────────────────────────────────────────

@app.cell
def _build_cross_run(RESULTS_ROOT, json, load_liquidity_map, np, pd):
    """Parcourt tous results/<approach>/full/<run>/<pair>/summary.json
    et calcule des stats simples (sharpe moyen/médian, %positive, DD, DD_dur, PF, WR).
    Trié par mean_sharpe décroissant.
    """
    liq_map = load_liquidity_map()
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
                _bps_str = _parts[-1] if len(_parts) > 1 else "?bps"
                for _pair in sorted(_run.iterdir()):
                    _sj = _pair / "summary.json"
                    if not _sj.exists():
                        continue
                    try:
                        _d = json.loads(_sj.read_text())
                    except Exception:
                        continue
                    _ts, _rets, _dds, _ddurs, _pfs, _wrs = [], [], [], [], [], []
                    for _fr in _d.get("folds", []):
                        _tm = _fr.get("test_metrics", {}) or {}
                        for _arr, _key in (
                            (_ts, "sharpe_ratio"),
                            (_rets, "total_return_pct"),
                            (_dds, "max_drawdown_pct"),
                            (_ddurs, "max_drawdown_duration"),
                            (_pfs, "profit_factor"),
                            (_wrs, "win_rate_pct"),
                        ):
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
                        "bps": _bps_str,
                        "pair": _pair.name,
                        "liquidity": liq_map.get(_pair.name, "unknown"),
                        "n_folds": len(_d.get("folds", [])),
                        "mean_sharpe": round(float(np.mean(_ts)), 2),
                        "median_sharpe": round(float(np.median(_ts)), 2),
                        "pct_positive": round((np.array(_rets) > 0).mean() * 100, 1) if _rets else 0.0,
                        "mean_ret_pct": round(float(np.mean(_rets)), 2) if _rets else 0.0,
                        "mean_dd_pct": round(float(np.mean(_dds)), 2) if _dds else 0.0,
                        "mean_dd_dur_days": round(float(np.mean(_ddurs)), 2) if _ddurs else 0.0,
                        "mean_pf": round(float(np.mean(_pfs)), 2) if _pfs else 0.0,
                        "mean_wr": round(float(np.mean(_wrs)), 2) if _wrs else 0.0,
                    })
    cross_run_df = pd.DataFrame(rows)
    if not cross_run_df.empty:
        cross_run_df = cross_run_df.sort_values("mean_sharpe", ascending=False).reset_index(drop=True)
    return (cross_run_df,)


# ──────────────────────────────────────────────────────────────────────
# §0 Filtres UI
# ──────────────────────────────────────────────────────────────────────

@app.cell
def _filters_ui(cross_run_df, mo):
    mo.stop(cross_run_df.empty, mo.callout(
        mo.md("Aucun résultat dans `results/`. Lance d'abord `python -m engine.cli wfa ...`"),
        kind="warn"
    ))
    _approaches = sorted(cross_run_df["approach"].unique().tolist())
    _bpss = sorted(cross_run_df["bps"].unique().tolist())
    _liq_levels = ["tres_liquide", "liquide", "moyen", "poubelle", "unknown"]
    _liq_available = [l for l in _liq_levels if l in cross_run_df["liquidity"].unique().tolist()]

    f_approach = mo.ui.multiselect(
        options=_approaches, value=_approaches, label="Approaches",
    )
    f_bps = mo.ui.multiselect(
        options=_bpss, value=_bpss, label="BPS",
    )
    f_liq = mo.ui.multiselect(
        options=_liq_available, value=_liq_available, label="Liquidity",
    )
    f_min_sharpe = mo.ui.number(
        start=-10.0, stop=10.0, step=0.1, value=-10.0, label="Min mean Sharpe",
    )
    f_min_pos = mo.ui.slider(
        start=0, stop=100, step=5, value=0, label="Min % positive folds",
    )
    mo.output.replace(mo.vstack([
        mo.md("### §0 Filtres"),
        mo.hstack([f_approach, f_bps, f_liq], justify="start", gap=2),
        mo.hstack([f_min_sharpe, f_min_pos], justify="start", gap=2),
    ]))
    return f_approach, f_bps, f_liq, f_min_pos, f_min_sharpe


@app.cell
def _filtered_table(cross_run_df, f_approach, f_bps, f_liq,
                    f_min_pos, f_min_sharpe, mo, pd):
    _df = cross_run_df.copy()
    if f_approach.value:
        _df = _df[_df["approach"].isin(f_approach.value)]
    if f_bps.value:
        _df = _df[_df["bps"].isin(f_bps.value)]
    if f_liq.value:
        _df = _df[_df["liquidity"].isin(f_liq.value)]
    _df = _df[_df["mean_sharpe"] >= f_min_sharpe.value]
    _df = _df[_df["pct_positive"] >= f_min_pos.value]
    _df = _df.reset_index(drop=True)

    if _df.empty:
        cross_table = mo.ui.table(
            pd.DataFrame({"info": ["Aucun résultat ne match les filtres."]}),
            selection=None,
        )
        _msg = mo.md(f"## §0 Cross-run table — _0 résultat avec ces filtres_")
    else:
        cross_table = mo.ui.table(_df, selection="single", page_size=30)
        _msg = mo.md(f"## §0 Cross-run table — **{len(_df)} combos** (sharpe moyen desc)")
    mo.output.replace(mo.vstack([
        _msg,
        mo.md("_Clique une ligne pour configurer les sections du dessous._"),
        cross_table,
    ]))
    filtered_df = _df
    return cross_table, filtered_df


@app.cell
def _selection(cross_table, filtered_df, mo):
    mo.stop(filtered_df.empty, mo.callout(
        mo.md("Aucune donnée — les filtres ne matchent rien."), kind="warn"
    ))
    _sel = cross_table.value if cross_table is not None else None
    if _sel is not None and len(_sel) > 0:
        _row = _sel.iloc[0]
    else:
        _row = filtered_df.iloc[0]
    approach_id = _row["approach"]
    run_cfg = _row["run"]
    pair = _row["pair"]
    mo.output.replace(mo.callout(
        mo.md(f"**Sélectionné** : `{approach_id}` / `{run_cfg}` / `{pair}` ({_row.get('liquidity', '?')}) — "
              f"sharpe moyen `{_row.get('mean_sharpe', 0):.2f}`, "
              f"médian `{_row.get('median_sharpe', 0):.2f}`, "
              f"{_row.get('pct_positive', 0):.0f}% folds positifs, "
              f"PF `{_row.get('mean_pf', 0):.2f}`, "
              f"DD `{_row.get('mean_dd_pct', 0):.2f}%` "
              f"({_row.get('mean_dd_dur_days', 0):.1f}j)"),
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


# ──────────────────────────────────────────────────────────────────────
# §1 Fold-by-fold table
# ──────────────────────────────────────────────────────────────────────

@app.cell
def _section1(approach_id, fmt_param, folds_df, mo, pair, run_cfg, summary):
    _param_cols = [c for c in folds_df.columns if c.startswith("p_")]
    _display = ["fold"] + _param_cols + [
        "train_sharpe", "test_sharpe",
        "train_return_pct", "test_return_pct",
        "test_dd_pct", "test_dd_dur_days",
        "test_pf", "test_wr", "test_trades",
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


# ──────────────────────────────────────────────────────────────────────
# §2 Stabilité params par fold
# ──────────────────────────────────────────────────────────────────────

@app.cell
def _section2_stability(folds_df, go, mo, pair, pd, run_cfg):
    _param_cols = [c for c in folds_df.columns if c.startswith("p_")]
    _fig = go.Figure()
    for _pc in _param_cols:
        _vals = folds_df[_pc]
        if _vals.apply(lambda x: isinstance(x, (list, tuple))).any():
            continue
        _num = pd.to_numeric(
            _vals.apply(lambda x: int(x) if isinstance(x, bool) else x),
            errors="coerce",
        ).astype(float)
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
        mo.md(f"## §2 Stabilité des paramètres — {pair}"),
        mo.ui.plotly(_fig),
    ]))


# ──────────────────────────────────────────────────────────────────────
# §3 WFE check
# ──────────────────────────────────────────────────────────────────────

@app.cell
def _section3_wfe(folds_df, go, mo, pair):
    _df_full = folds_df.dropna(subset=["train_sharpe", "test_sharpe"])
    if _df_full.empty:
        mo.output.replace(mo.md("## §3 WFE check — _pas assez de données._"))
    else:
        _corr = _df_full["train_sharpe"].corr(_df_full["test_sharpe"])
        _pct_pos = (_df_full["test_return_pct"] > 0).mean() * 100
        _fig_sc = go.Figure()
        _fig_sc.add_trace(go.Scatter(
            x=_df_full["train_sharpe"], y=_df_full["test_sharpe"],
            mode="markers+text", text=_df_full["fold"].astype(str),
            textposition="top center",
            marker=dict(size=10, color=_df_full["fold"],
                        colorscale="Viridis", showscale=True,
                        colorbar=dict(title="Fold")),
        ))
        _lim = max(_df_full["train_sharpe"].abs().max(),
                   _df_full["test_sharpe"].abs().max()) * 1.1 or 1.0
        _fig_sc.add_shape(type="line", x0=-_lim, y0=-_lim, x1=_lim, y1=_lim,
                          line=dict(dash="dot", color="gray"))
        _fig_sc.update_layout(
            title=f"{pair} — Sharpe train vs test (corr={_corr:.2f})",
            xaxis_title="Train Sharpe", yaxis_title="Test Sharpe", height=380,
        )
        _colors = ["#2ecc71" if v > 0 else "#e74c3c"
                   for v in _df_full["test_return_pct"]]
        _fig_bar = go.Figure([go.Bar(
            x=_df_full["fold"], y=_df_full["test_return_pct"],
            marker_color=_colors, name="Test return %",
        )])
        _fig_bar.add_trace(go.Scatter(
            x=_df_full["fold"], y=_df_full["train_return_pct"],
            mode="lines+markers", name="Train return %",
            line=dict(dash="dot", color="#3498db"),
        ))
        _fig_bar.update_layout(
            title=f"{pair} — Train vs Test return % par fold",
            xaxis_title="Fold", yaxis_title="Return %", height=300,
        )
        mo.output.replace(mo.vstack([
            mo.md(f"## §3 WFE check — {pair}"),
            mo.hstack([
                mo.stat(label="Corr train/test Sharpe", value=f"{_corr:.2f}"),
                mo.stat(label="Mean test Sharpe",
                        value=f'{_df_full["test_sharpe"].mean():.2f}'),
                mo.stat(label="% folds rentables", value=f"{_pct_pos:.0f}%"),
                mo.stat(label="N folds", value=str(len(_df_full))),
            ], justify="start", gap=4),
            mo.ui.plotly(_fig_sc),
            mo.ui.plotly(_fig_bar),
        ]))


# ──────────────────────────────────────────────────────────────────────
# §4 VBT replay walk-forward — re-run kernel par fold, concat, resample 1D
# ──────────────────────────────────────────────────────────────────────

@app.cell
def _section4_vbt_wf(approach_id, folds_df, go, load_ohlcv, mo, pair,
                     parse_tf_bps, pd, run_cfg, summary):
    import sys as _sys
    _sys.path.insert(0, "/home/devbox/BTYZ/src")
    import warnings as _w
    _w.filterwarnings("ignore")

    _tf_str, _bps = parse_tf_bps(run_cfg)
    _fees = _bps * 1e-4

    _ohlcv_full = load_ohlcv(pair, _tf_str)
    if _ohlcv_full is None:
        mo.output.replace(mo.callout(
            mo.md(f"OHLCV {pair} introuvable dans `data/raw/lighter/1m/`."), kind="warn"
        ))
    else:
        _equity_pieces = []
        _fold_starts = []
        _fold_returns = []
        _n_trades_total = 0
        _capital = 10_000.0
        _error = None
        try:
            from engine.approach_loader import instantiate_strategy
            _strat = instantiate_strategy(approach_id)
            _mod = _sys.modules.get(f"approach.{approach_id}.strategy")
            if _mod is not None:
                if hasattr(_mod, "_target_fees"):
                    _mod._target_fees = _fees
                if hasattr(_mod, "_target_freq"):
                    _mod._target_freq = _tf_str

            for _fr in sorted(folds_df.to_dict(orient="records"),
                              key=lambda r: r["fold"]):
                _params = {k[2:]: v for k, v in _fr.items()
                           if k.startswith("p_")}
                _test_start = _fr.get("test_start")
                _test_end = _fr.get("test_end")
                if pd.isna(_test_start) or pd.isna(_test_end):
                    continue

                _si = int(_ohlcv_full.index.searchsorted(_test_start))
                _te = int(_ohlcv_full.index.searchsorted(_test_end, side="right"))
                _wi = max(0, _si - 300)
                _slc = _ohlcv_full.iloc[_wi:_te]
                if len(_slc) < 50:
                    continue

                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    _pf = _strat.run_backtest(_slc, _params)
                if _pf is None:
                    continue

                _value_test = _pf.value.loc[_test_start:_test_end]
                if len(_value_test) < 2:
                    continue
                # Rebase: each fold restart from current cumulative capital
                _fold_growth = _value_test / _value_test.iloc[0]
                _fold_equity = _fold_growth * _capital
                _equity_pieces.append(_fold_equity)
                _fold_starts.append(_value_test.index[0])
                _capital = float(_fold_equity.iloc[-1])
                _fold_returns.append(float(_fold_growth.iloc[-1] - 1) * 100)
                _n_trades_total += int(_pf.trades.count() or 0)
        except Exception as _e:
            _error = str(_e)

        if _error is not None:
            mo.output.replace(mo.callout(
                mo.md(f"VBT replay erreur : `{_error}`"), kind="danger"
            ))
        elif not _equity_pieces:
            mo.output.replace(mo.md("## §4 VBT replay WF — _aucun fold valide._"))
        else:
            _equity = pd.concat(_equity_pieces).sort_index()
            _equity = _equity[~_equity.index.duplicated(keep="first")]
            # Resample 1D pour léger
            _eq_1d = _equity.resample("1D").last().ffill()
            _running_max = _eq_1d.cummax()
            _dd_pct = (_eq_1d / _running_max - 1) * 100
            _total_ret = (_eq_1d.iloc[-1] / 10_000.0 - 1) * 100
            _max_dd = float(_dd_pct.min())
            _max_dd_idx = _dd_pct.idxmin()
            _dd_dur_days = 0
            try:
                _under = _dd_pct < -0.01
                _runs = (_under != _under.shift()).cumsum()
                _drawdown_runs = _runs[_under].value_counts().max()
                _dd_dur_days = int(_drawdown_runs) if pd.notna(_drawdown_runs) else 0
            except Exception:
                pass

            _fig_eq = go.Figure()
            _fig_eq.add_trace(go.Scatter(
                x=_eq_1d.index, y=_eq_1d.values, mode="lines",
                name="Equity ($)",
                line=dict(color="#3498db", width=1.6),
            ))
            for _i, _ts in enumerate(_fold_starts):
                _fig_eq.add_vline(
                    x=_ts, line=dict(color="rgba(255,255,255,0.35)",
                                     dash="dash", width=0.8),
                    annotation_text=f"f{_i}", annotation_position="top",
                )
            _fig_eq.update_layout(
                title=f"{pair} · {run_cfg} — VBT replay walk-forward OOS "
                      f"(equity concaténée, resample 1D · {len(_equity_pieces)} folds · "
                      f"{_n_trades_total} trades)",
                yaxis_title="Equity ($, base 10_000)", height=430,
                xaxis_title="Date",
            )

            _fig_dd = go.Figure()
            _fig_dd.add_trace(go.Scatter(
                x=_dd_pct.index, y=_dd_pct.values, mode="lines",
                fill="tozeroy",
                line=dict(color="#e74c3c", width=1),
                name="Drawdown %",
            ))
            _fig_dd.update_layout(
                title="Drawdown %", yaxis_title="%", height=220,
                xaxis_title="Date",
            )

            _fig_folds = go.Figure([go.Bar(
                x=list(range(len(_fold_returns))), y=_fold_returns,
                marker_color=["#2ecc71" if r > 0 else "#e74c3c"
                              for r in _fold_returns],
                text=[f"{r:+.2f}%" for r in _fold_returns],
                textposition="outside",
            )])
            _fig_folds.update_layout(
                title="Return par fold (VBT replay)",
                xaxis_title="Fold", yaxis_title="Return %", height=280,
            )

            mo.output.replace(mo.vstack([
                mo.md(f"## §4 VBT replay walk-forward — {pair}"),
                mo.md(
                    f"_Re-run du kernel `{approach_id}` sur chaque fold OOS avec "
                    f"les params optimisés, concaténation des equities, "
                    f"resample 1D pour affichage léger._"
                ),
                mo.hstack([
                    mo.stat(label="Return total", value=f"{_total_ret:.1f}%"),
                    mo.stat(label="Max DD", value=f"{_max_dd:.1f}%"),
                    mo.stat(label="DD dur (j)", value=f"{_dd_dur_days}"),
                    mo.stat(label="N trades", value=str(_n_trades_total)),
                    mo.stat(label="N folds", value=str(len(_equity_pieces))),
                ], justify="start", gap=4),
                mo.ui.plotly(_fig_eq),
                mo.ui.plotly(_fig_dd),
                mo.ui.plotly(_fig_folds),
            ]))


if __name__ == "__main__":
    app.run()
