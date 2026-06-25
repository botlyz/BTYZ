"""BTYZ engine — analyse MCCV (random-date OOS).

Charge les résultats produits par `python -m engine.cli mccv ...`
sous `results/<APPROACH>/mccv/<PAIR>_<TF>_<BPS>bps.json`.

§0 Selectors : approach / pair / configs (multi-select)
§1 Table résumée par config (mean/median oos_sh, %positive, max_dd)
§2 Box plot des OOS Sharpe par config
§3 Distribution des params optimaux choisis (= dispersion du regime)
§4 OOS return vs train sharpe (overfitting check par target date)
§5 Table détaillée des targets (1 ligne par random date)
"""
import marimo

__generated_with = "0.13.2"
app = marimo.App(width="full")


@app.cell
def _imports():
    import sys as _sys
    _sys.path.insert(0, "/home/botlyz-gpu/BTYZ/src")

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
    RESULTS_ROOT = pathlib.Path("/home/botlyz-gpu/BTYZ/results")
    return (RESULTS_ROOT,)


@app.cell
def _discover_mccv(RESULTS_ROOT):
    """Liste les approches avec un dossier mccv/ non vide."""
    approaches = []
    if RESULTS_ROOT.exists():
        for _d in sorted(RESULTS_ROOT.iterdir()):
            if _d.is_dir() and (_d / "mccv").exists() and any((_d / "mccv").iterdir()):
                approaches.append(_d.name)
    return (approaches,)


@app.cell
def _approach_selector(approaches, mo):
    mo.stop(not approaches, mo.callout(
        mo.md("Aucun résultat MCCV trouvé. Lance :\n\n"
              "```bash\npython -m engine.cli mccv --approach <ID> --pairs ... --tf ... --bps ... --n-targets 12\n```"),
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
def _load_all_mccv(RESULTS_ROOT, approach, json, mo, pd):
    """Charge tous les MCCV de l'approche choisie en DataFrame plat."""
    _files = sorted((RESULTS_ROOT / approach.value / "mccv").glob("*.json"))
    mo.stop(not _files, mo.callout(mo.md("Aucun fichier MCCV pour cette approche."), kind="warn"))

    _all_rows = []
    for _fp in _files:
        _d = json.loads(_fp.read_text())
        _pair = _d.get("pair")
        _tf = _d.get("tf")
        _bps = _d.get("bps")
        _config_tag = f"{_tf}_{_bps}bps"
        for _r in _d.get("rows", []):
            _row = {
                "pair": _pair, "tf": _tf, "bps": _bps, "config": _config_tag,
                **_r,
            }
            _all_rows.append(_row)
    mccv_df = pd.DataFrame(_all_rows)
    return (mccv_df,)


@app.cell
def _pair_selector(mccv_df, mo):
    pairs = sorted(mccv_df["pair"].unique().tolist()) if not mccv_df.empty else []
    mo.stop(not pairs, mo.callout(mo.md("Aucune paire dans les MCCV."), kind="warn"))
    pair = mo.ui.dropdown(
        options={p: p for p in pairs},
        value=pairs[0],
        label="Paire",
    )
    mo.output.replace(mo.hstack([pair], justify="start", gap=2))
    return (pair,)


@app.cell
def _filter(mccv_df, pair):
    sub = mccv_df[mccv_df["pair"] == pair.value].copy()
    return (sub,)


# ─────────────────────────────────────────────────────────────────────
# §1 Cross-config summary
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section1_summary(mo, np, pd, sub):
    if sub.empty:
        mo.output.replace(mo.md("## §1 Pas de données pour cette paire."))
    else:
        _agg_rows = []
        for _cfg, _g in sub.groupby("config"):
            _oos_sh = _g["oos_sh"].dropna()
            _oos_ret = _g["oos_ret"].dropna()
            _oos_dd = _g["oos_dd"].dropna()
            _train_sh = _g["train_sh"].dropna()
            _agg_rows.append({
                "config": _cfg,
                "n_targets": len(_g),
                "mean_oos_sh": round(_oos_sh.mean(), 2),
                "median_oos_sh": round(_oos_sh.median(), 2),
                "std_oos_sh": round(_oos_sh.std(), 2),
                "pct_pos_oos_sh": round((_oos_sh > 0).mean() * 100, 1),
                "mean_oos_ret_pct": round(_oos_ret.mean(), 2),
                "median_oos_ret_pct": round(_oos_ret.median(), 2),
                "max_oos_dd_pct": round(_oos_dd.max(), 2),
                "mean_train_sh": round(_train_sh.mean(), 2),
                "sh_degradation": round(_train_sh.mean() - _oos_sh.mean(), 2),
            })
        _agg = pd.DataFrame(_agg_rows).sort_values("mean_oos_sh", ascending=False).reset_index(drop=True)
        mo.output.replace(mo.vstack([
            mo.md(f"## §1 Résumé MCCV — paire `{sub['pair'].iloc[0]}`"),
            mo.md("_Trié par mean OOS Sharpe décroissant._"),
            mo.ui.table(_agg, selection=None),
        ]))


# ─────────────────────────────────────────────────────────────────────
# §2 OOS Sharpe distribution (box plot par config)
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section2_box(go, mo, sub):
    if sub.empty:
        mo.output.replace(mo.md("## §2 Box plot — pas de données."))
    else:
        _fig = go.Figure()
        for _cfg, _g in sub.groupby("config"):
            _fig.add_box(y=_g["oos_sh"], name=_cfg, boxpoints="all", jitter=0.3, marker=dict(size=5))
        _fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.4)
        _fig.update_layout(
            title=f"OOS Sharpe distribution par config — {sub['pair'].iloc[0]}",
            yaxis_title="OOS Sharpe", height=420,
        )
        mo.output.replace(mo.vstack([
            mo.md("## §2 Box plot OOS Sharpe par config"),
            mo.ui.plotly(_fig),
        ]))


# ─────────────────────────────────────────────────────────────────────
# §3 Param dispersion across targets (per config)
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section3_param_disp(go, mo, pd, sub):
    if sub.empty:
        mo.output.replace(mo.md("## §3 Pas de données."))
    else:
        # Common scalar params across runs (skip lists)
        _param_cols = [_c for _c in sub.columns if _c not in
                       ("target", "train_sh", "oos_sh", "oos_ret", "oos_dd", "oos_pf", "oos_wr",
                        "oos_n", "pair", "tf", "bps", "config")]
        _scalar_cols = []
        for _c in _param_cols:
            if sub[_c].apply(lambda x: not isinstance(x, (list, tuple))).all():
                try:
                    pd.to_numeric(sub[_c], errors="raise")
                    _scalar_cols.append(_c)
                except Exception:
                    pass

        if not _scalar_cols:
            mo.output.replace(mo.md("## §3 Aucun param scalaire à plotter."))
        else:
            _fig = go.Figure()
            for _col in _scalar_cols:
                for _cfg, _g in sub.groupby("config"):
                    _vals = pd.to_numeric(_g[_col], errors="coerce").dropna()
                    if len(_vals) == 0:
                        continue
                    _fig.add_box(y=_vals, name=f"{_col}/{_cfg}", showlegend=False)
            _fig.update_layout(
                title=f"Dispersion des paramètres scalaires across targets — {sub['pair'].iloc[0]}",
                yaxis_title="Valeur", height=400,
            )
            mo.output.replace(mo.vstack([
                mo.md("## §3 Dispersion des params optimaux par target (= régime instable si box large)"),
                mo.ui.plotly(_fig),
            ]))


# ─────────────────────────────────────────────────────────────────────
# §4 OOS return vs Train sharpe (overfitting check)
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section4_scatter(go, mo, np, sub):
    if sub.empty:
        mo.output.replace(mo.md("## §4 Pas de données."))
    else:
        _fig = go.Figure()
        for _cfg, _g in sub.groupby("config"):
            _fig.add_trace(go.Scatter(
                x=_g["train_sh"], y=_g["oos_sh"], mode="markers",
                name=_cfg, text=_g["target"],
                hovertemplate="target=%{text}<br>train_sh=%{x:.2f}<br>oos_sh=%{y:.2f}<extra></extra>",
                marker=dict(size=9, opacity=0.7),
            ))
        _all_train = sub["train_sh"].dropna()
        _all_oos = sub["oos_sh"].dropna()
        if len(_all_train) > 1 and len(_all_oos) > 1:
            _corr = np.corrcoef(_all_train, _all_oos)[0, 1]
        else:
            _corr = float("nan")
        _lim = max(abs(sub["train_sh"].abs().max()), abs(sub["oos_sh"].abs().max()), 1) * 1.1
        _fig.add_shape(type="line", x0=-_lim, y0=-_lim, x1=_lim, y1=_lim,
                       line=dict(dash="dot", color="gray"))
        _fig.update_layout(
            title=f"Train vs OOS Sharpe (corr={_corr:.2f}) — {sub['pair'].iloc[0]}",
            xaxis_title="Train Sharpe", yaxis_title="OOS Sharpe", height=420,
        )
        mo.output.replace(mo.vstack([
            mo.md(f"## §4 Train vs OOS Sharpe (corr={_corr:.2f})"),
            mo.ui.plotly(_fig),
        ]))


# ─────────────────────────────────────────────────────────────────────
# §5 Detailed targets table
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section5_detail(mo, pd, sub):
    if sub.empty:
        mo.output.replace(mo.md("## §5 Pas de données."))
    else:
        _df = sub.copy().sort_values(["config", "target"]).reset_index(drop=True)
        # Format list-valued cols (env_levels, allocations)
        for _c in _df.columns:
            if _df[_c].apply(lambda x: isinstance(x, (list, tuple))).any():
                _df[_c] = _df[_c].apply(lambda x: ("[" + ", ".join(f"{v:.4g}" for v in x) + "]")
                                        if isinstance(x, (list, tuple)) else x)
        for _c in _df.select_dtypes(include="float").columns:
            _df[_c] = _df[_c].round(4)
        mo.output.replace(mo.vstack([
            mo.md(f"## §5 Détail par target (random date)"),
            mo.ui.table(_df, selection=None, page_size=50),
        ]))


if __name__ == "__main__":
    app.run()
