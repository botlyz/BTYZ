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

    def load_ohlcv(pair, tf, with_cross_exchange=False):
        """Charge le CSV 1m de Lighter puis resample à la timeframe demandée.
        Merge aussi signed_rate / apr depuis data/raw/lighter/funding/ si dispo
        (utilise par FUNDING_HARVEST_v1).

        Si with_cross_exchange=True (ex: FUNDING_ARB_v1), attache aussi le funding
        Hyperliquid → colonnes signed_rate_lighter, signed_rate_hl, spread. Drop les
        lignes où l'un des deux funding manque.
        """
        fp = BASE_OHLCV / f"{pair}.csv"
        if not fp.exists():
            return None
        df = pd.read_csv(fp, low_memory=False,
                         usecols=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        df = df.set_index("date").sort_index()
        agg = {"open": "first", "high": "max", "low": "min",
               "close": "last", "volume": "sum"}
        ohlcv = df.resample(tf, label="left", closed="left").agg(agg).dropna()
        # Attache funding columns si le fichier existe (no-op pour les strats non-funding)
        try:
            from engine.data_loader import _maybe_attach_funding
            ohlcv = _maybe_attach_funding(ohlcv, pair, tf)
        except Exception:
            pass
        # Cross-exchange : attache aussi le funding HL
        if with_cross_exchange:
            try:
                from engine.data_loader import _attach_hl_funding
                _xe = _attach_hl_funding(ohlcv, pair)
                if _xe is not None and len(_xe) > 0:
                    ohlcv = _xe
            except Exception:
                pass
        return ohlcv

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
    # Position size en % du capital. Deux modes selon la strat :
    # - Strats à alloc fixe (FUNDING_HARVEST, ATR_ENV, RAM_DCA_RSI) :
    #     slider = % du capital alloué par position (remplace ALLOC_FIXED).
    # - Strats à alloc dynamique (RAM_DCA_v1) :
    #     slider = MULTIPLICATEUR sur le sizing WFA-optimisé. 100 = original, 200 = ×2.
    size_pct_slider = mo.ui.slider(
        start=5, stop=200, step=5, value=100,
        label="Position size % (alloc fixe) OU multiplicateur % (alloc dynamique)",
        show_value=True,
    )
    # Override du bps fees (sinon parsé depuis le nom du run). Match les fees prod.
    bps_slider = mo.ui.slider(
        start=1, stop=10, step=1, value=1,
        label="BPS fees (override le bps du run_cfg)",
        show_value=True,
    )
    mo.output.replace(mo.vstack([
        mo.md("### §4 — Contrôles"),
        leverage_slider,
        size_pct_slider,
        bps_slider,
    ]))
    return bps_slider, leverage_slider, size_pct_slider


@app.cell
def _section4_vbt_wf(
    approach_id,
    bps_slider,
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

    _tf_str, _ = parse_tf_bps(run_cfg)
    _bps = float(bps_slider.value)
    _fees = _bps * 1e-4
    _slippage = 0.0002

    # Detect cross-exchange strategies (e.g., FUNDING_ARB_v1) via DATA_SOURCE attribute
    _need_xe = False
    try:
        from engine.approach_loader import load_strategy_module
        _mod_check = load_strategy_module(approach_id)
        _cls = getattr(_mod_check, "Strategy", None)
        if _cls is not None and getattr(_cls, "DATA_SOURCE", None) == "cross_exchange":
            _need_xe = True
        # Override fees per-pair pour les strats cross-exchange (match le WFA runner)
        if _need_xe and hasattr(_mod_check, "_load_pair_fees"):
            _per_pair = _mod_check._load_pair_fees(pair)
            if _per_pair and _per_pair > 0:
                _fees = float(_per_pair)
                _slippage = 0.0  # half-spread déjà inclus dans _per_pair
    except Exception:
        pass
    _ohlcv_full = load_ohlcv(pair, _tf_str, with_cross_exchange=_need_xe)
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
            import importlib as _importlib
            # Force reload du module strat pour propager les modifs hot (Marimo
            # garde en cache sys.modules entre les ré-exécutions de cellule).
            _mod_name = f"approach.{approach_id}.strategy"
            if _mod_name in _sys.modules:
                try:
                    _importlib.reload(_sys.modules[_mod_name])
                except Exception:
                    pass
            _strat = instantiate_strategy(approach_id)
            _mod = _sys.modules.get(_mod_name)
            if _mod is not None:
                if hasattr(_mod, "_target_fees"):
                    _mod._target_fees = _fees
                if hasattr(_mod, "_target_freq"):
                    _mod._target_freq = _tf_str
                if hasattr(_mod, "_target_slippage"):
                    _mod._target_slippage = _slippage

            # ── REPLAY WALK-FORWARD vbt-native ──
            # Per-fold: vbt.PF.from_orders(..., last_state=, save_state=True)
            # chaine ensuite avec vbt.PF.row_stack(*pfs, chained=True).
            # Avantages:
            #   - capital roll-forward natif (compound entre folds)
            #   - position carry-over realiste (live-like)
            #   - le pf chaine est un vrai vbt.Portfolio => .value, .stats(),
            #     .resample().plot(), .trades.count() etc. natifs.
            _lev = float(leverage_slider.value)
            _alloc_fixed = getattr(_mod, "ALLOC_FIXED", None) if _mod else None
            if _alloc_fixed and _alloc_fixed > 0:
                # Strat à alloc fixe (FUNDING_HARVEST, ATR_ENV, RAM_DCA_RSI...) :
                # slider % du capital désiré → scale tel que out_size = slider/100
                _size_scale = (float(size_pct_slider.value) / 100.0) / float(_alloc_fixed)
            else:
                # Strat à alloc dynamique (RAM_DCA_v1 multi-bandes) : pas de ALLOC_FIXED.
                # Le slider devient un MULTIPLICATEUR sur le sizing WFA-optimisé.
                # 100 = sizing original, 200 = double tout, 50 = moitié.
                _size_scale = float(size_pct_slider.value) / 100.0

            _pfs = []
            _last_state = None
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
                _slc_full = _ohlcv_full.iloc[_wi:_te]
                if len(_slc_full) < 50:
                    continue

                # Kernel signals on full slice (warmup helps indicator state)
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    _ts, _ep = _strat.compute_target_arrays(_slc_full, _params)
                if _ts is None or _ep is None:
                    continue

                # Restrict to test window
                _wl = _si - _wi
                _test_size_raw = _ts.iloc[_wl:]
                _test_price = _ep.iloc[_wl:]
                _test_data = _ohlcv_full.iloc[_si:_te]
                # Apply ALLOC * slider * leverage
                _alloc_eff = _alloc_fixed if (_alloc_fixed and _alloc_fixed > 0) else 1.0
                _test_size = _test_size_raw * _alloc_eff * _size_scale * _lev

                # Optional cash_dividends (per-share, vbt scale auto par position)
                # pour les strats avec revenu externe scale (ex: FUNDING_HARVEST).
                # cash_dividends est CORRECT avec compound + slider, cash_earnings non.
                _cd = None
                if hasattr(_strat, "compute_cash_dividends"):
                    try:
                        _cd = _strat.compute_cash_dividends(_test_data, _params)
                    except Exception:
                        _cd = None

                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    pf_fold = vbt.Portfolio.from_orders(
                        close=_test_data["close"],
                        size=_test_size,
                        price=_test_price,
                        size_type="TargetPercent",
                        init_cash=10_000.0,
                        last_state=_last_state,
                        save_state=True,
                        cash_dividends=_cd,
                        leverage=_lev,
                        fees=_fees,
                        slippage=_slippage,
                        freq=_tf_str,
                    )
                _pfs.append(pf_fold)
                _last_state = pf_fold.last_state
                _fold_starts.append(_ohlcv_full.index[_si])
                _n_folds_used += 1

            if _n_folds_used == 0:
                raise RuntimeError(
                    "Aucun fold valide — la stratégie expose-t-elle "
                    "`compute_target_arrays()` ?"
                )

            # Stack en un seul vbt.Portfolio chaine
            # vbt bug: row_stack(*objs) avec len==1 tente d'itérer sur le Portfolio.
            # Workaround: si 1 seul fold, on saute le stack.
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                if len(_pfs) == 1:
                    _pf = _pfs[0]
                else:
                    _pf = vbt.Portfolio.row_stack(*_pfs, chained=True)
        except Exception as _e:
            _error = str(_e)

        if _error is not None:
            mo.output.replace(mo.callout(
                mo.md(f"VBT replay erreur : `{_error}`"), kind="danger",
            ))
        elif _pf is None:
            mo.output.replace(mo.md("## §4 VBT replay WF — _portfolio vide._"))
        else:
            # Restrict to WF window — vbt natif (_pf est un vbt.Portfolio chaine)
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

            # ── VBT native plot: pf.loc[wf].resample('1D').plot() ──
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


@app.cell
def _section5_picker(filtered_df, mo, pd):
    """§5.0 — Tableau cochable des candidats pour portfolio multi-paires."""
    if filtered_df.empty:
        portfolio_table = mo.ui.table(
            pd.DataFrame({"info": ["Aucune ligne — élargis les filtres §0."]}),
            selection=None,
        )
        mo.output.replace(mo.md("## §5 Portfolio multi-paires — _filtres §0 vides._"))
    else:
        portfolio_table = mo.ui.table(
            filtered_df, selection="multi", page_size=30,
            label="Sélection portfolio (cochage multi)",
        )
        mo.output.replace(mo.vstack([
            mo.md("## §5 Portfolio multi-paires — sélection"),
            mo.md("_Coche **plusieurs lignes** dans le tableau ci-dessous puis clique "
                  "**Lancer le backtest** pour simuler le portefeuille mutualisé._"),
            portfolio_table,
        ]))
    return (portfolio_table,)


@app.cell
def _section5_controls(mo):
    """§5.1 — Contrôles portfolio AUTONOMES (init_cash, alloc, bps, levier).
    Indépendants de §4 pour ne pas perturber le backtest unitaire."""
    init_cash_p = mo.ui.number(
        start=1000, stop=1_000_000, step=1000, value=10_000,
        label="Init cash $",
    )
    alloc_pair_p = mo.ui.slider(
        start=10, stop=200, step=5, value=100,
        label="% capital TOTAL à diviser entre paires (100% = sat)", show_value=True,
    )
    bps_p = mo.ui.slider(
        start=1, stop=20, step=1, value=10,
        label="BPS fees", show_value=True,
    )
    lev_p = mo.ui.slider(
        start=1.0, stop=10.0, step=0.5, value=1.0,
        label="Levier", show_value=True,
    )
    use_extra_p = mo.ui.checkbox(
        value=False,
        label="Re-opti extension : 150 trials Optuna sur les 90 derniers jours avant la fin WFA, puis applique aux bars post-WFA (cache JSON par paire)",
    )
    run_btn = mo.ui.run_button(label="▶ Lancer le backtest multi-paires", kind="success")
    mo.output.replace(mo.vstack([
        mo.md("### §5 — Contrôles portfolio"),
        mo.hstack([init_cash_p, alloc_pair_p, bps_p, lev_p],
                  gap=3, justify="start"),
        use_extra_p,
        run_btn,
    ]))
    return alloc_pair_p, bps_p, init_cash_p, lev_p, run_btn, use_extra_p


@app.cell
def _section5_portfolio(
    alloc_pair_p,
    bps_p,
    flatten_summary,
    init_cash_p,
    lev_p,
    load_ohlcv,
    load_summary,
    mo,
    parse_tf_bps,
    pd,
    portfolio_table,
    run_btn,
    use_extra_p,
):
    """§5.2 — Replay walk-forward fold-par-fold pour chaque ligne cochée,
    puis VBT Portfolio mutualisé. Gate par `run_btn` (n'exécute que sur click)."""
    import sys as _sys
    _sys.path.insert(0, "/home/devbox/BTYZ/src")
    import warnings as _w
    _w.filterwarnings("ignore")
    import importlib as _importlib
    import traceback as _tb

    import vectorbtpro as _vbt

    # Normalisation de la sélection table
    _sel = portfolio_table.value if portfolio_table is not None else None
    _sel_df = None
    if isinstance(_sel, pd.DataFrame):
        _sel_df = _sel.reset_index(drop=True) if not _sel.empty else None
    elif isinstance(_sel, (list, tuple)) and len(_sel) > 0:
        _sel_df = pd.DataFrame(list(_sel)).reset_index(drop=True)

    _n_sel = 0 if _sel_df is None else len(_sel_df)

    if _sel_df is None or _sel_df.empty or _n_sel < 2:
        mo.output.replace(mo.callout(
            mo.md(f"Coche **au moins 2 lignes** dans le tableau §5, "
                  f"puis clique **▶ Lancer le backtest**.<br>"
                  f"_Actuellement cochées : **{_n_sel}**_"),
            kind="info",
        ))
    elif not run_btn.value:
        mo.output.replace(mo.callout(
            mo.md(f"**{_n_sel} lignes cochées.** Clique **▶ Lancer le backtest** "
                  f"pour démarrer le replay multi-paires."),
            kind="info",
        ))
    else:
        _alloc_pct = float(alloc_pair_p.value) / 100.0
        _init_cash = float(init_cash_p.value)
        _bps = float(bps_p.value)
        _fees = _bps * 1e-4
        _slippage = 0.0002
        _lev = float(lev_p.value)

        _per_pair = {}
        _errors = []

        from engine.approach_loader import (
            instantiate_strategy as _instantiate,
            load_strategy_module as _load_mod,
        )
        import pathlib as _pathlib
        import json as _json

        _RESULTS_ROOT = _pathlib.Path("/home/devbox/BTYZ/results")
        _reopti_status = []  # pour affichage live
        _n_sel_total = len(_sel_df)

        for _row_idx, _row in _sel_df.iterrows():
            _ap = _row["approach"]
            _rc = _row["run"]
            _pr = _row["pair"]
            _key = f"{_pr}"
            if _key in _per_pair:
                _key = f"{_pr}__{_ap}"
            _tf_str, _ = parse_tf_bps(_rc)

            try:
                _mod_check = _load_mod(_ap)
                _cls = getattr(_mod_check, "Strategy", None)
                _need_xe = bool(_cls is not None
                                and getattr(_cls, "DATA_SOURCE", None) == "cross_exchange")
            except Exception:
                _need_xe = False

            _ohlcv = load_ohlcv(_pr, _tf_str, with_cross_exchange=_need_xe)
            if _ohlcv is None:
                _errors.append(f"{_key}: OHLCV manquant")
                continue

            _summary = load_summary(_ap, _rc, _pr)
            if _summary is None:
                _errors.append(f"{_key}: summary.json manquant")
                continue
            _folds = flatten_summary(_summary)
            if _folds.empty:
                _errors.append(f"{_key}: pas de folds")
                continue

            _mod_name = f"approach.{_ap}.strategy"
            if _mod_name in _sys.modules:
                try:
                    _importlib.reload(_sys.modules[_mod_name])
                except Exception:
                    pass
            _strat = _instantiate(_ap)
            _mod = _sys.modules.get(_mod_name)
            if _mod is not None:
                if hasattr(_mod, "_target_fees"):
                    _mod._target_fees = _fees
                if hasattr(_mod, "_target_freq"):
                    _mod._target_freq = _tf_str
                if hasattr(_mod, "_target_slippage"):
                    _mod._target_slippage = _slippage
            _alloc_fixed = getattr(_mod, "ALLOC_FIXED", None) if _mod else None
            _alloc_eff = _alloc_fixed if (_alloc_fixed and _alloc_fixed > 0) else 1.0

            _size_chunks, _price_chunks = [], []
            for _fr in sorted(_folds.to_dict(orient="records"),
                              key=lambda r: r["fold"]):
                _params = {k[2:]: v for k, v in _fr.items() if k.startswith("p_")}
                _t0 = _fr.get("test_start")
                _t1 = _fr.get("test_end")
                if pd.isna(_t0) or pd.isna(_t1):
                    continue
                _si = int(_ohlcv.index.searchsorted(_t0))
                _te = int(_ohlcv.index.searchsorted(_t1, side="right"))
                if _te <= _si:
                    continue
                _wi = max(0, _si - 300)
                _slc = _ohlcv.iloc[_wi:_te]
                if len(_slc) < 50:
                    continue
                try:
                    with _w.catch_warnings():
                        _w.simplefilter("ignore")
                        _ts, _ep = _strat.compute_target_arrays(_slc, _params)
                except Exception as _ek:
                    _errors.append(
                        f"{_key} fold={_fr.get('fold','?')} compute_target_arrays: {_ek}"
                    )
                    continue
                if _ts is None or _ep is None:
                    continue
                _wl = _si - _wi
                _size_chunks.append(_ts.iloc[_wl:] * _alloc_eff)
                _price_chunks.append(_ep.iloc[_wl:])

            if not _size_chunks:
                _errors.append(f"{_key}: 0 folds valides")
                continue

            # Extension : couvre les bars APRÈS le dernier fold OOS via re-opti
            # Optuna 150 trials sur [anchor - 90j, anchor]. Cache JSON par paire.
            _wfa_end_ts = None
            _last_fr = sorted(_folds.to_dict(orient="records"),
                              key=lambda r: r["fold"])[-1]
            _last_t1 = _last_fr.get("test_end")
            if not pd.isna(_last_t1):
                _wfa_end_ts = pd.Timestamp(_last_t1)
            if use_extra_p.value and _wfa_end_ts is not None:
                _ext_start = int(_ohlcv.index.searchsorted(_wfa_end_ts, side="right"))
                _ext_end = len(_ohlcv)
                if _ext_end > _ext_start:
                    # Cache lookup
                    _cache_dir = _RESULTS_ROOT / _ap / "extension_reopti" / _rc
                    _cache_dir.mkdir(parents=True, exist_ok=True)
                    _anchor_iso = _wfa_end_ts.strftime("%Y%m%dT%H%M%S")
                    _cache_file = _cache_dir / f"{_pr}_anchor{_anchor_iso}_t90_tr150.json"
                    _reopti_params = None
                    if _cache_file.exists():
                        try:
                            _reopti_params = _json.loads(_cache_file.read_text())
                            _reopti_status.append(f"✓ {_pr} cache hit")
                        except Exception:
                            _reopti_params = None
                    if _reopti_params is None:
                        # Status live pendant la re-opti
                        _reopti_status.append(f"⏳ {_pr} Optuna 150 trials…")
                        mo.output.replace(mo.vstack([
                            mo.md(f"## §5 Replay paire {_row_idx + 1}/{_n_sel_total} : `{_pr}`"),
                            mo.md(f"_Re-opti extension (anchor {_anchor_iso}, train 90j, 150 trials)_"),
                            mo.md("\n".join(f"- {s}" for s in _reopti_status[-12:])),
                        ]))
                        # Train window [anchor - 90j, anchor]
                        _ts_anchor = _wfa_end_ts
                        _train_end_idx = int(_ohlcv.index.searchsorted(_ts_anchor, side="left"))
                        _train_start_idx = int(_ohlcv.index.searchsorted(
                            _ts_anchor - pd.Timedelta(days=90), side="left"))
                        if _train_end_idx - _train_start_idx >= 500:
                            _train_df = _ohlcv.iloc[_train_start_idx:_train_end_idx]
                            try:
                                from engine.tpe_search import run_tpe_fold as _run_tpe
                                with _w.catch_warnings():
                                    _w.simplefilter("ignore")
                                    _res = _run_tpe(
                                        train_data=_train_df, test_data=_train_df,
                                        param_space_fn=_strat.param_space,
                                        run_backtest_fn=_strat.run_backtest,
                                        score_fn=_strat.score,
                                        trials=150, min_trades_per_fold=10,
                                        seed=42, fold_idx=0, n_jobs=1,
                                    )
                            except Exception as _eopt:
                                _errors.append(f"{_key} reopti: {_eopt}")
                                _res = None
                            if _res is not None and _res.get("params"):
                                _reopti_params = _res["params"]
                                try:
                                    _cache_file.write_text(
                                        _json.dumps(_reopti_params, indent=2, default=str)
                                    )
                                    _reopti_status[-1] = (
                                        f"✓ {_pr} re-opti done "
                                        f"(score train={_res.get('train_metrics', {}).get('sharpe_ratio', 0):.2f})"
                                    )
                                except Exception:
                                    pass
                            else:
                                _reopti_status[-1] = f"✗ {_pr} reopti failed"
                        else:
                            _reopti_status[-1] = f"✗ {_pr} train trop court"
                    if _reopti_params is not None:
                        _wi = max(0, _ext_start - 300)
                        _slc_ext = _ohlcv.iloc[_wi:_ext_end]
                        if len(_slc_ext) >= 50:
                            try:
                                with _w.catch_warnings():
                                    _w.simplefilter("ignore")
                                    _ts_ext, _ep_ext = _strat.compute_target_arrays(
                                        _slc_ext, _reopti_params
                                    )
                            except Exception as _eext:
                                _errors.append(
                                    f"{_key} extension compute_target_arrays: {_eext}"
                                )
                                _ts_ext = _ep_ext = None
                            if _ts_ext is not None and _ep_ext is not None:
                                _wl_ext = _ext_start - _wi
                                _size_chunks.append(_ts_ext.iloc[_wl_ext:] * _alloc_eff)
                                _price_chunks.append(_ep_ext.iloc[_wl_ext:])

            _size_pair = pd.concat(_size_chunks).sort_index()
            _size_pair = _size_pair[~_size_pair.index.duplicated(keep="last")]
            _price_pair = pd.concat(_price_chunks).sort_index()
            _price_pair = _price_pair[~_price_pair.index.duplicated(keep="last")]
            _close_pair = _ohlcv["close"].loc[
                _size_pair.index[0]:_size_pair.index[-1]
            ]

            _per_pair[_key] = {
                "size":     _size_pair,
                "price":    _price_pair,
                "close":    _close_pair,
                "tf":       _tf_str,
                "wfa_end":  _wfa_end_ts,
            }

        if len(_per_pair) < 2:
            mo.output.replace(mo.callout(
                mo.md("Moins de 2 paires valides — vérifie les sélections.<br>"
                      f"Erreurs: {_errors}"),
                kind="danger",
            ))
        else:
            _tfs_used = {v["tf"] for v in _per_pair.values()}
            if len(_tfs_used) > 1:
                mo.output.replace(mo.callout(
                    mo.md(f"Timeframes mixés ({_tfs_used}) — choisis un seul TF."),
                    kind="danger",
                ))
            else:
                _tf_final = next(iter(_tfs_used))
                _union = sorted(set().union(
                    *[v["close"].index for v in _per_pair.values()]
                ))
                _close_wide = pd.DataFrame({
                    p: _per_pair[p]["close"].reindex(_union).ffill()
                    for p in _per_pair
                })
                _size_wide = pd.DataFrame({
                    p: _per_pair[p]["size"].reindex(_union)
                    for p in _per_pair
                })
                _price_wide = pd.DataFrame({
                    p: _per_pair[p]["price"].reindex(_union)
                    for p in _per_pair
                })
                _mask = _close_wide.notna().any(axis=1)
                _close_wide = _close_wide.loc[_mask]
                _size_wide = _size_wide.loc[_mask]
                _price_wide = _price_wide.loc[_mask]

                # Sémantique : `alloc_pair_p` = % capital TOTAL à diviser entre paires.
                # Avec 5 paires et slider=100% → chaque paire reçoit 20% (TargetPercent).
                # Quand toutes signalent en même temps → portfolio 100% investi.
                _n_pairs = max(len(_per_pair), 1)
                _per_pair_pct = _alloc_pct / _n_pairs
                _size_eff = _size_wide * _per_pair_pct * _lev

                import traceback as _tb
                _vbt_error = None
                _pf = None
                try:
                    with _w.catch_warnings():
                        _w.simplefilter("ignore")
                        _pf = _vbt.Portfolio.from_orders(
                            close=_close_wide,
                            size=_size_eff,
                            price=_price_wide,
                            size_type="TargetPercent",
                            init_cash=_init_cash,
                            fees=_fees,
                            slippage=_slippage,
                            freq=_tf_final,
                            leverage=_lev,
                            group_by=True,
                            cash_sharing=True,
                            call_seq="auto",
                        )
                except Exception as _evbt:
                    _vbt_error = f"{type(_evbt).__name__}: {_evbt}\n\n{_tb.format_exc()}"

                if _vbt_error is not None or _pf is None:
                    mo.output.replace(mo.callout(
                        mo.md(
                            f"**VBT Portfolio.from_orders a échoué :**\n\n```\n{_vbt_error}\n```\n\n"
                            f"_Shapes : close{_close_wide.shape}, size{_size_eff.shape}, "
                            f"price{_price_wide.shape}, paires={list(_per_pair.keys())}_"
                        ),
                        kind="danger",
                    ))
                    _ret = _max_dd = _sharpe = _sortino = _calmar = 0.0
                    _n_tr = 0
                    _val = None
                    _stats = pd.Series(dtype=object)
                    _allocs = pd.DataFrame()
                else:
                    try:
                        with _w.catch_warnings():
                            _w.simplefilter("ignore")
                            _val = _pf.value
                            _stats = _pf.stats()
                            _allocs = _pf.allocations
                        _ret = float((_val.iloc[-1] / _val.iloc[0] - 1) * 100)
                        _max_dd = float((_val / _val.cummax() - 1).min() * 100)
                        _sharpe = float(_stats.get("Sharpe Ratio", 0) or 0)
                        _sortino = float(_stats.get("Sortino Ratio", 0) or 0)
                        _calmar = float(_stats.get("Calmar Ratio", 0) or 0)
                        try:
                            _n_tr = int(_pf.trades.count().sum() or 0)
                        except Exception:
                            _n_tr = 0
                    except Exception as _emet:
                        mo.output.replace(mo.callout(
                            mo.md(f"**Erreur calcul métriques :** `{_emet}`\n\n```\n{_tb.format_exc()}\n```"),
                            kind="danger",
                        ))
                        _val = None

                if _val is not None:
                    # Equity + drawdown sur séries DAILY (downsampled — évite
                    # MARIMO_OUTPUT_MAX_BYTES sur portfolios multi-mois 5min).
                    import plotly.graph_objects as _go
                    try:
                        _val_d = _val.resample("1D").last().dropna()
                        _dd_d = (_val_d / _val_d.cummax() - 1) * 100
                        _fig_pf = _go.Figure()
                        _fig_pf.add_trace(_go.Scatter(
                            x=_val_d.index, y=_val_d.values,
                            mode="lines", name="Equity",
                            line=dict(color="#2ecc71", width=2),
                            yaxis="y",
                        ))
                        _fig_pf.add_trace(_go.Scatter(
                            x=_dd_d.index, y=_dd_d.values,
                            mode="lines", name="Drawdown %",
                            line=dict(color="#e74c3c", width=1),
                            fill="tozeroy", fillcolor="rgba(231,76,60,0.2)",
                            yaxis="y2",
                        ))
                        _fig_pf.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#0f0f1a", plot_bgcolor="#161625",
                            height=500,
                            title=(f"Portfolio mutualisé · {_n_pairs} paires · "
                                   f"total {_alloc_pct*100:.0f}% "
                                   f"({_per_pair_pct*100:.1f}%/paire) · "
                                   f"lev x{_lev:.1f} · fees {int(_bps)}bps"
                                   + (" · extension activée"
                                      if use_extra_p.value else "")),
                            xaxis=dict(title="Date"),
                            yaxis=dict(title="Equity ($)", side="left"),
                            yaxis2=dict(title="Drawdown %", overlaying="y",
                                        side="right", showgrid=False,
                                        range=[min(_dd_d.min() * 1.1, -1), 1]),
                            legend=dict(orientation="h", y=1.02, x=0),
                        )

                        # Trait vertical à la frontière WFA / extension
                        # On prend le min(wfa_end) = moment où AU MOINS une paire
                        # passe en mode extension.
                        if use_extra_p.value:
                            _wfa_ends = [v.get("wfa_end") for v in _per_pair.values()
                                         if v.get("wfa_end") is not None]
                            if _wfa_ends:
                                _earliest_cutoff = min(_wfa_ends)
                                _latest_cutoff = max(_wfa_ends)
                                _fig_pf.add_shape(
                                    type="line",
                                    x0=_earliest_cutoff, x1=_earliest_cutoff,
                                    y0=0, y1=1, xref="x", yref="paper",
                                    line=dict(color="#f39c12", width=2, dash="dash"),
                                )
                                _fig_pf.add_annotation(
                                    x=_earliest_cutoff, y=1.0, xref="x", yref="paper",
                                    text="⇤ WFA  |  extension ⇥",
                                    showarrow=False, yshift=10,
                                    font=dict(color="#f39c12", size=11),
                                    bgcolor="rgba(15,15,26,0.7)",
                                )
                                # 2ème ligne si gap entre paires > 30 jours
                                if (_latest_cutoff - _earliest_cutoff).days > 30:
                                    _fig_pf.add_shape(
                                        type="line",
                                        x0=_latest_cutoff, x1=_latest_cutoff,
                                        y0=0, y1=1, xref="x", yref="paper",
                                        line=dict(color="#f39c12", width=1, dash="dot"),
                                    )
                    except Exception:
                        _fig_pf = None

                    # Perf cumulative par paire en $ (asset_pnl par colonne, daily resamplé)
                    # group_by=False force le résultat per-asset (sinon = somme groupée = Series)
                    try:
                        with _w.catch_warnings():
                            _w.simplefilter("ignore")
                            _apnl = _pf.get_asset_pnl(group_by=False)
                        if isinstance(_apnl, pd.DataFrame) and not _apnl.empty:
                            _cum_d = _apnl.fillna(0).cumsum().resample("1D").last().ffill()
                            _fig_perf = _go.Figure()
                            _palette = [
                                "#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6",
                                "#1abc9c", "#e67e22", "#f1c40f", "#16a085", "#d35400",
                                "#8e44ad", "#27ae60", "#c0392b", "#2980b9", "#7f8c8d",
                            ]
                            for _idx, _p in enumerate(_per_pair):
                                if _p not in _cum_d.columns:
                                    continue
                                _y = _cum_d[_p]
                                _color = _palette[_idx % len(_palette)]
                                _fig_perf.add_trace(_go.Scatter(
                                    x=_y.index, y=_y.values,
                                    mode="lines", name=_p,
                                    line=dict(color=_color, width=1.5),
                                    hovertemplate=(f"<b>{_p}</b><br>"
                                                   "%{x|%Y-%m-%d}<br>"
                                                   "PnL cum: $%{y:.2f}<extra></extra>"),
                                ))
                            _fig_perf.add_hline(y=0, line=dict(
                                color="rgba(200,200,200,0.4)", width=1, dash="dot",
                            ))
                            # Reuse WFA cutoff line (cohérent avec equity plot)
                            if use_extra_p.value:
                                _wfa_ends2 = [v.get("wfa_end") for v in _per_pair.values()
                                              if v.get("wfa_end") is not None]
                                if _wfa_ends2:
                                    _fig_perf.add_shape(
                                        type="line",
                                        x0=min(_wfa_ends2), x1=min(_wfa_ends2),
                                        y0=0, y1=1, xref="x", yref="paper",
                                        line=dict(color="#f39c12", width=2, dash="dash"),
                                    )
                            _fig_perf.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="#0f0f1a", plot_bgcolor="#161625",
                                height=380,
                                title="PnL cumulatif par paire (contribution $ au portfolio)",
                                xaxis_title="Date", yaxis_title="PnL cumulé ($)",
                                hovermode="x unified",
                            )
                            _fig_alloc = _fig_perf
                        else:
                            _fig_alloc = None
                    except Exception:
                        _fig_alloc = None

                    _stats_df = pd.DataFrame({
                        "Métrique": _stats.index.astype(str),
                        "Valeur":   [str(v) for v in _stats.values],
                    })

                    _components = [
                        mo.md(f"## §5 Portfolio mutualisé — {len(_per_pair)} paires "
                              f"· `cash_sharing=True` · `TargetPercent`"),
                        mo.hstack([
                            mo.stat(label="Return total", value=f"{_ret:.1f}%"),
                            mo.stat(label="Max DD",       value=f"{_max_dd:.1f}%"),
                            mo.stat(label="Sharpe",       value=f"{_sharpe:.2f}"),
                            mo.stat(label="Sortino",      value=f"{_sortino:.2f}"),
                            mo.stat(label="Calmar",       value=f"{_calmar:.2f}"),
                            mo.stat(label="Trades total", value=str(_n_tr)),
                        ], gap=4, justify="start"),
                    ]
                    if _fig_pf is not None:
                        _components.append(mo.ui.plotly(_fig_pf))
                    if _fig_alloc is not None:
                        _components.append(mo.ui.plotly(_fig_alloc))
                    _components.extend([
                        mo.md("### pf.stats() — portefeuille groupé (VBT)"),
                        mo.ui.table(_stats_df.reset_index(drop=True),
                                    selection=None, page_size=60),
                    ])
                    if _errors:
                        _components.append(mo.callout(
                            mo.md(f"_Paires skippées_: {_errors}"), kind="warn",
                        ))
                    mo.output.replace(mo.vstack(_components))
    return


if __name__ == "__main__":
    app.run()
