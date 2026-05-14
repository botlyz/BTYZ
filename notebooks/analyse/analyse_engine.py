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

    return (
        flatten_summary,
        fmt_param,
        load_liquidity_map,
        load_ohlcv,
        load_summary,
        parse_tf_bps,
    )


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
                # bps = segment qui contient "bps" (peut ne pas être en dernier
                # depuis qu'on suffixe le tag avec le découpage walk-forward)
                _bps_str = next((p for p in _parts if "bps" in p), "?bps")
                # wfa tag = segment matchant le pattern <int>d<int>d<int>d
                import re as _re
                _wfa = next((p for p in _parts if _re.fullmatch(r"\d+d\d+d\d+d", p)), "")
                for _pair in sorted(_run.iterdir()):
                    _sj = _pair / "summary.json"
                    if not _sj.exists():
                        continue
                    try:
                        _d = json.loads(_sj.read_text())
                    except Exception:
                        continue
                    _ts, _rets, _dds, _ddurs, _pfs, _wrs, _trs = [], [], [], [], [], [], []
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
                        # trades : prend total_trades sinon trades_count
                        try:
                            _vtr = _tm.get("total_trades")
                            if _vtr is None:
                                _vtr = _tm.get("trades_count")
                            _trs.append(float(_vtr or 0))
                        except (TypeError, ValueError):
                            pass
                    if not _ts:
                        continue
                    rows.append({
                        "approach": _ap.name,
                        "run": _run.name,
                        "tf": _tf,
                        "bps": _bps_str,
                        "wfa": _wfa,
                        "pair": _pair.name,
                        "liquidity": liq_map.get(_pair.name, "unknown"),
                        "n_folds": len(_d.get("folds", [])),
                        "mean_sharpe": round(float(np.mean(_ts)), 2),
                        "median_sharpe": round(float(np.median(_ts)), 2),
                        "std_sharpe": round(float(np.std(_ts)), 2),
                        "pct_positive": round((np.array(_rets) > 0).mean() * 100, 1) if _rets else 0.0,
                        "mean_ret_pct": round(float(np.mean(_rets)), 2) if _rets else 0.0,
                        "mean_dd_pct": round(float(np.mean(_dds)), 2) if _dds else 0.0,
                        "mean_dd_dur_days": round(float(np.mean(_ddurs)), 2) if _ddurs else 0.0,
                        "mean_trades": round(float(np.mean(_trs)), 1) if _trs else 0.0,
                        "median_trades": round(float(np.median(_trs)), 1) if _trs else 0.0,
                        "std_trades": round(float(np.std(_trs)), 1) if _trs else 0.0,
                        "mean_pf": round(float(np.mean(_pfs)), 2) if _pfs else 0.0,
                        "mean_wr": round(float(np.mean(_wrs)), 2) if _wrs else 0.0,
                    })
    cross_run_df = pd.DataFrame(rows)
    if not cross_run_df.empty:
        cross_run_df = cross_run_df.sort_values("mean_sharpe", ascending=False).reset_index(drop=True)
    return (cross_run_df,)


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
    _max_folds = int(cross_run_df["n_folds"].max()) if "n_folds" in cross_run_df.columns else 20
    f_min_folds = mo.ui.slider(
        start=1, stop=max(_max_folds, 1), step=1, value=1,
        label="Min folds", show_value=True,
    )
    mo.output.replace(mo.vstack([
        mo.md("### §0 Filtres"),
        mo.hstack([f_approach, f_bps, f_liq], justify="start", gap=2),
        mo.hstack([f_min_sharpe, f_min_pos, f_min_folds], justify="start", gap=2),
    ]))
    return f_approach, f_bps, f_liq, f_min_folds, f_min_pos, f_min_sharpe


@app.cell
def _():
    return


@app.cell
def _filtered_table(
    cross_run_df,
    f_approach,
    f_bps,
    f_liq,
    f_min_folds,
    f_min_pos,
    f_min_sharpe,
    mo,
    pd,
):
    _df = cross_run_df.copy()
    if f_approach.value:
        _df = _df[_df["approach"].isin(f_approach.value)]
    if f_bps.value:
        _df = _df[_df["bps"].isin(f_bps.value)]
    if f_liq.value:
        _df = _df[_df["liquidity"].isin(f_liq.value)]
    _df = _df[_df["mean_sharpe"] >= f_min_sharpe.value]
    _df = _df[_df["pct_positive"] >= f_min_pos.value]
    _df = _df[_df["n_folds"] >= f_min_folds.value]
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
    return


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
    return


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
    return


@app.cell
def _section4_controls(mo):
    leverage_slider = mo.ui.slider(
        start=1.0, stop=10.0, step=0.5, value=1.0,
        label="Levier (multiplie la taille des positions)",
        show_value=True,
    )
    # Position size en % du capital — utilisé pour les stratégies à taille fixe
    # (RAM_DCA_RSI_v1) en remplaçant l'alloc hardcodée (10% par défaut).
    # Ignoré silencieusement pour les stratégies à allocations dynamiques (RAM_DCA_v1).
    size_pct_slider = mo.ui.slider(
        start=5, stop=100, step=5, value=10,
        label="Position size % (override alloc fixe — ignoré si la strat utilise allocations dynamiques)",
        show_value=True,
    )
    mo.output.replace(mo.vstack([
        mo.md("### §4 — Contrôles"),
        leverage_slider,
        size_pct_slider,
    ]))
    return leverage_slider, size_pct_slider


@app.cell
def _section4_vbt_wf(
    approach_id,
    folds_df,
    go,
    leverage_slider,
    load_ohlcv,
    mo,
    np,
    pair,
    parse_tf_bps,
    pd,
    run_cfg,
    size_pct_slider,
):
    import sys as _sys
    _sys.path.insert(0, "/home/devbox/BTYZ/src")
    import warnings as _w
    _w.filterwarnings("ignore")

    import vectorbtpro as vbt

    _tf_str, _bps = parse_tf_bps(run_cfg)
    _fees = _bps * 1e-4
    _slippage = 0.0002

    _ohlcv_full = load_ohlcv(pair, _tf_str)
    if _ohlcv_full is None:
        mo.output.replace(mo.callout(
            mo.md(f"OHLCV {pair} introuvable dans `data/raw/lighter/1m/`."),
            kind="warn",
        ))
    else:
        _error = None
        _pf = None
        _fold_starts = []
        _n_folds_used = 0
        try:
            from engine.approach_loader import instantiate_strategy
            _strat = instantiate_strategy(approach_id)
            _mod = _sys.modules.get(f"approach.{approach_id}.strategy")
            if _mod is not None:
                if hasattr(_mod, "_target_fees"):
                    _mod._target_fees = _fees
                if hasattr(_mod, "_target_freq"):
                    _mod._target_freq = _tf_str

            # Build stitched arrays globally — one slot per bar of the full OHLCV
            _N = len(_ohlcv_full)
            _g_size = np.full(_N, np.nan)
            _g_price = np.full(_N, np.nan)

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
                if _te <= _si:
                    continue
                _wi = max(0, _si - 300)
                _slc = _ohlcv_full.iloc[_wi:_te]
                if len(_slc) < 50:
                    continue

                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    _ts, _ep = _strat.compute_target_arrays(_slc, _params)
                if _ts is None or _ep is None:
                    continue

                _wl = _si - _wi
                _test_len = _te - _si
                _g_size[_si:_te] = _ts.values[_wl:_wl + _test_len]
                _g_price[_si:_te] = _ep.values[_wl:_wl + _test_len]
                _fold_starts.append(_ohlcv_full.index[_si])
                _n_folds_used += 1

            if _n_folds_used == 0:
                raise RuntimeError(
                    "Aucun fold valide — la stratégie expose-t-elle "
                    "`compute_target_arrays()` ?"
                )

            _size_s = pd.Series(_g_size, index=_ohlcv_full.index)
            _price_s = pd.Series(_g_price, index=_ohlcv_full.index)

            with _w.catch_warnings():
                _w.simplefilter("ignore")
                _lev = float(leverage_slider.value)
                # Si la stratégie expose ALLOC_FIXED (ex: RAM_DCA_RSI_v1), on
                # surcharge la taille avec le slider. Pour les strats à
                # allocations dynamiques (RAM_DCA_v1), on laisse les sizes
                # natives et le slider est ignoré (scale=1.0).
                _alloc_fixed = getattr(_mod, "ALLOC_FIXED", None) if _mod else None
                if _alloc_fixed and _alloc_fixed > 0:
                    _size_scale = (float(size_pct_slider.value) / 100.0) / float(_alloc_fixed)
                else:
                    _size_scale = 1.0
                _size_lev = _size_s * _lev * _size_scale
                _pf = vbt.Portfolio.from_orders(
                    close=_ohlcv_full["close"],
                    size=_size_lev,
                    price=_price_s,
                    size_type="TargetPercent",
                    init_cash=10_000.0,
                    leverage=_lev,
                    fees=_fees,
                    slippage=_slippage,
                    freq=_tf_str,
                )
        except Exception as _e:
            _error = str(_e)

        if _error is not None:
            mo.output.replace(mo.callout(
                mo.md(f"VBT replay erreur : `{_error}`"), kind="danger",
            ))
        elif _pf is None:
            mo.output.replace(mo.md("## §4 VBT replay WF — _portfolio vide._"))
        else:
            # Restrict to WF window
            _wf_start = folds_df["test_start"].dropna().iloc[0]
            _wf_end   = folds_df["test_end"].dropna().iloc[-1]

            with _w.catch_warnings():
                _w.simplefilter("ignore")
                _pf_wf = _pf.loc[_wf_start:_wf_end]
                _stats = _pf_wf.stats()
            _stats_df = pd.DataFrame({
                "Métrique": _stats.index.astype(str),
                "Valeur":   [str(v) for v in _stats.values],
            })

            # ── Compute per-fold indicators (test slice only) ──
            # For each fold, compute MA + bands using THAT fold's params.
            _fold_indicators = []
            _prev_params = None
            for _fr in sorted(folds_df.to_dict(orient="records"),
                              key=lambda r: r["fold"]):
                _p = {k[2:]: v for k, v in _fr.items() if k.startswith("p_")}
                _t0 = _fr.get("test_start")
                _t1 = _fr.get("test_end")
                if pd.isna(_t0) or pd.isna(_t1):
                    continue
                _si = int(_ohlcv_full.index.searchsorted(_t0))
                _te = int(_ohlcv_full.index.searchsorted(_t1, side="right"))
                if _te <= _si:
                    continue
                _maw = int(_p.get("ma_window", 0) or 0)
                _wi = max(0, _si - max(_maw, 1))
                _close = _ohlcv_full["close"].iloc[_wi:_te]
                _ma_full = (_close.rolling(_maw, min_periods=_maw).mean().values
                            if _maw > 0 else np.full(len(_close), np.nan))
                _wl = _si - _wi
                _ma_test = _ma_full[_wl:_wl + (_te - _si)]

                _envs = _p.get("env_levels")
                _up_test = np.full_like(_ma_test, np.nan, dtype=float)
                _lo_test = np.full_like(_ma_test, np.nan, dtype=float)
                if isinstance(_envs, (list, tuple)) and len(_envs) > 0:
                    _wid = float(max(_envs))
                    _up_test = _ma_test * (1.0 + _wid)
                    _lo_test = _ma_test * (1.0 - _wid)
                elif "atr_mult" in _p and "atr_window" in _p:
                    _atrw = int(_p["atr_window"])
                    _wi2 = max(0, _si - _atrw)
                    _hh = _ohlcv_full["high"].iloc[_wi2:_te].values
                    _ll = _ohlcv_full["low"].iloc[_wi2:_te].values
                    _cc = _ohlcv_full["close"].iloc[_wi2:_te].values
                    _pc = np.roll(_cc, 1); _pc[0] = _cc[0]
                    _tr = np.maximum.reduce([
                        _hh - _ll, np.abs(_hh - _pc), np.abs(_ll - _pc),
                    ])
                    _atr = pd.Series(_tr).rolling(_atrw, min_periods=_atrw).mean().values
                    _wl2 = _si - _wi2
                    _atr_test = _atr[_wl2:_wl2 + (_te - _si)]
                    _mult = float(_p["atr_mult"])
                    _up_test = _ma_test + _mult * _atr_test
                    _lo_test = _ma_test - _mult * _atr_test

                _fold_indicators.append({
                    "fold":       int(_fr.get("fold", 0)),
                    "index":      _ohlcv_full.index[_si:_te],
                    "ma":         _ma_test,
                    "range_hi":   _up_test,
                    "range_lo":   _lo_test,
                    "params":     _p,
                    "test_start": _t0,
                    "params_changed": _prev_params is not None
                                       and _prev_params != _p,
                })
                _prev_params = _p

            # ── VBT plot: pf.loc[wf].resample('1D').plot() ──
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                _pf_wf_ds = _pf_wf.resample("1D")
                _fig_vbt = _pf_wf_ds.plot()

            # Remove resampled 'Close' trace — replaced by candlesticks
            _fig_vbt.data = tuple(
                t for t in _fig_vbt.data
                if getattr(t, "name", "") != "Close"
            )

            # Candlestick downsamplé (max ~800 bougies)
            _ohlcv_wf = _ohlcv_full.loc[_wf_start:_wf_end]
            _step_c = max(1, len(_ohlcv_wf) // 800)
            _cslc = _ohlcv_wf.iloc[::_step_c]
            _fig_vbt.add_trace(go.Candlestick(
                x=_cslc.index,
                open=_cslc["open"], high=_cslc["high"],
                low=_cslc["low"],   close=_cslc["close"],
                name="OHLCV",
                increasing_line_color="#2ecc71", decreasing_line_color="#e74c3c",
                increasing_fillcolor="#2ecc71",  decreasing_fillcolor="#e74c3c",
            ))

            # Per-fold indicators (downsamplés, ~80-150 pts/fold)
            _pts_per_fold = max(80, 2000 // max(1, len(_fold_indicators)))
            _lbl_seen = set()
            def _show(label):
                v = label not in _lbl_seen
                _lbl_seen.add(label)
                return v

            for _fi in _fold_indicators:
                _s = max(1, len(_fi["index"]) // _pts_per_fold)
                _ip = _fi["index"][::_s]
                _fig_vbt.add_trace(go.Scatter(
                    x=_ip, y=_fi["range_hi"][::_s], mode="lines",
                    line=dict(color="#3498db", width=1, dash="dash"),
                    name="Upper band", showlegend=_show("hi"),
                    legendgroup="hi", opacity=0.8,
                ))
                _fig_vbt.add_trace(go.Scatter(
                    x=_ip, y=_fi["range_lo"][::_s], mode="lines",
                    line=dict(color="#e74c3c", width=1, dash="dash"),
                    name="Lower band", showlegend=_show("lo"),
                    legendgroup="lo", opacity=0.8,
                ))
                _fig_vbt.add_trace(go.Scatter(
                    x=_ip, y=_fi["ma"][::_s], mode="lines",
                    line=dict(color="#f39c12", width=1, dash="dot"),
                    name="MA", showlegend=_show("ma"),
                    legendgroup="ma", opacity=0.75,
                ))
                # Séparateur de fold — orange si params changent, gris sinon
                _fig_vbt.add_vline(
                    x=str(_fi["test_start"]),
                    line=dict(
                        color="rgba(230,126,34,0.6)" if _fi["params_changed"]
                              else "rgba(180,180,180,0.35)",
                        width=1.5 if _fi["params_changed"] else 0.8,
                        dash="solid" if _fi["params_changed"] else "dash",
                    ),
                )

            _fig_vbt.update_layout(
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                paper_bgcolor="#0f0f1a",
                plot_bgcolor="#161625",
                height=700,
                title=dict(
                    text=(f"{pair} · {run_cfg} · {approach_id} — "
                          f"{_n_folds_used} folds · lev x{_lev:.1f}"),
                    font=dict(color="#ddd", size=12),
                ),
                legend=dict(bgcolor="rgba(0,0,0,0.3)",
                            font=dict(color="#ccc", size=10)),
            )

            _ret_tot = float((_pf_wf.value.iloc[-1] / _pf_wf.value.iloc[0] - 1) * 100)
            _max_dd  = float((_pf_wf.value / _pf_wf.value.cummax() - 1).min() * 100)
            _n_tr    = int(_pf_wf.trades.count() or 0)

            mo.output.replace(mo.vstack([
                mo.md(f"## §4 Walk-Forward — {pair} · {run_cfg}"),
                mo.hstack([
                    mo.stat(label="Return total", value=f"{_ret_tot:.1f}%"),
                    mo.stat(label="Max DD",       value=f"{_max_dd:.1f}%"),
                    mo.stat(label="Trades",       value=str(_n_tr)),
                    mo.stat(label="Folds",        value=f"{_n_folds_used}"),
                    mo.stat(label="Levier",       value=f"x{_lev:.1f}"),
                ], gap=4, justify="start"),
                mo.ui.plotly(_fig_vbt),
                mo.md("---"),
                mo.md("### pf.stats() — période walk-forward complète"),
                mo.ui.table(_stats_df.reset_index(drop=True),
                            selection=None, page_size=60),
            ]))
    return


if __name__ == "__main__":
    app.run()
