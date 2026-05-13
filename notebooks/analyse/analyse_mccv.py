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
    files = sorted((RESULTS_ROOT / approach.value / "mccv").glob("*.json"))
    mo.stop(not files, mo.callout(mo.md("Aucun fichier MCCV pour cette approche."), kind="warn"))

    all_rows = []
    for fp in files:
        d = json.loads(fp.read_text())
        pair = d.get("pair")
        tf = d.get("tf")
        bps = d.get("bps")
        config_tag = f"{tf}_{bps}bps"
        for r in d.get("rows", []):
            row = {
                "pair": pair, "tf": tf, "bps": bps, "config": config_tag,
                **r,
            }
            all_rows.append(row)
    mccv_df = pd.DataFrame(all_rows)
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
        agg_rows = []
        for cfg, g in sub.groupby("config"):
            oos_sh = g["oos_sh"].dropna()
            oos_ret = g["oos_ret"].dropna()
            oos_dd = g["oos_dd"].dropna()
            train_sh = g["train_sh"].dropna()
            agg_rows.append({
                "config": cfg,
                "n_targets": len(g),
                "mean_oos_sh": round(oos_sh.mean(), 2),
                "median_oos_sh": round(oos_sh.median(), 2),
                "std_oos_sh": round(oos_sh.std(), 2),
                "pct_pos_oos_sh": round((oos_sh > 0).mean() * 100, 1),
                "mean_oos_ret_pct": round(oos_ret.mean(), 2),
                "median_oos_ret_pct": round(oos_ret.median(), 2),
                "max_oos_dd_pct": round(oos_dd.max(), 2),
                "mean_train_sh": round(train_sh.mean(), 2),
                "sh_degradation": round(train_sh.mean() - oos_sh.mean(), 2),
            })
        agg = pd.DataFrame(agg_rows).sort_values("mean_oos_sh", ascending=False).reset_index(drop=True)
        mo.output.replace(mo.vstack([
            mo.md(f"## §1 Résumé MCCV — paire `{sub['pair'].iloc[0]}`"),
            mo.md("_Trié par mean OOS Sharpe décroissant._"),
            mo.ui.table(agg, selection=None),
        ]))


# ─────────────────────────────────────────────────────────────────────
# §2 OOS Sharpe distribution (box plot par config)
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section2_box(go, mo, sub):
    if sub.empty:
        mo.output.replace(mo.md("## §2 Box plot — pas de données."))
    else:
        fig = go.Figure()
        for cfg, g in sub.groupby("config"):
            fig.add_box(y=g["oos_sh"], name=cfg, boxpoints="all", jitter=0.3, marker=dict(size=5))
        fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.4)
        fig.update_layout(
            title=f"OOS Sharpe distribution par config — {sub['pair'].iloc[0]}",
            yaxis_title="OOS Sharpe", height=420,
        )
        mo.output.replace(mo.vstack([
            mo.md("## §2 Box plot OOS Sharpe par config"),
            mo.ui.plotly(fig),
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
        param_cols = [c for c in sub.columns if c not in
                      ("target", "train_sh", "oos_sh", "oos_ret", "oos_dd", "oos_pf", "oos_wr",
                       "oos_n", "pair", "tf", "bps", "config")]
        scalar_cols = []
        for c in param_cols:
            if sub[c].apply(lambda x: not isinstance(x, (list, tuple))).all():
                try:
                    pd.to_numeric(sub[c], errors="raise")
                    scalar_cols.append(c)
                except Exception:
                    pass

        if not scalar_cols:
            mo.output.replace(mo.md("## §3 Aucun param scalaire à plotter."))
        else:
            fig = go.Figure()
            for col in scalar_cols:
                for cfg, g in sub.groupby("config"):
                    vals = pd.to_numeric(g[col], errors="coerce").dropna()
                    if len(vals) == 0:
                        continue
                    fig.add_box(y=vals, name=f"{col}/{cfg}", showlegend=False)
            fig.update_layout(
                title=f"Dispersion des paramètres scalaires across targets — {sub['pair'].iloc[0]}",
                yaxis_title="Valeur", height=400,
            )
            mo.output.replace(mo.vstack([
                mo.md("## §3 Dispersion des params optimaux par target (= régime instable si box large)"),
                mo.ui.plotly(fig),
            ]))


# ─────────────────────────────────────────────────────────────────────
# §4 OOS return vs Train sharpe (overfitting check)
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section4_scatter(go, mo, np, sub):
    if sub.empty:
        mo.output.replace(mo.md("## §4 Pas de données."))
    else:
        fig = go.Figure()
        for cfg, g in sub.groupby("config"):
            fig.add_trace(go.Scatter(
                x=g["train_sh"], y=g["oos_sh"], mode="markers",
                name=cfg, text=g["target"],
                hovertemplate="target=%{text}<br>train_sh=%{x:.2f}<br>oos_sh=%{y:.2f}<extra></extra>",
                marker=dict(size=9, opacity=0.7),
            ))
        all_train = sub["train_sh"].dropna()
        all_oos = sub["oos_sh"].dropna()
        if len(all_train) > 1 and len(all_oos) > 1:
            corr = np.corrcoef(all_train, all_oos)[0, 1]
        else:
            corr = float("nan")
        _lim = max(abs(sub["train_sh"].abs().max()), abs(sub["oos_sh"].abs().max()), 1) * 1.1
        fig.add_shape(type="line", x0=-_lim, y0=-_lim, x1=_lim, y1=_lim,
                      line=dict(dash="dot", color="gray"))
        fig.update_layout(
            title=f"Train vs OOS Sharpe (corr={corr:.2f}) — {sub['pair'].iloc[0]}",
            xaxis_title="Train Sharpe", yaxis_title="OOS Sharpe", height=420,
        )
        mo.output.replace(mo.vstack([
            mo.md(f"## §4 Train vs OOS Sharpe (corr={corr:.2f})"),
            mo.ui.plotly(fig),
        ]))


# ─────────────────────────────────────────────────────────────────────
# §5 Detailed targets table
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section5_detail(mo, pd, sub):
    if sub.empty:
        mo.output.replace(mo.md("## §5 Pas de données."))
    else:
        df = sub.copy().sort_values(["config", "target"]).reset_index(drop=True)
        # Format list-valued cols (env_levels, allocations)
        for c in df.columns:
            if df[c].apply(lambda x: isinstance(x, (list, tuple))).any():
                df[c] = df[c].apply(lambda x: ("[" + ", ".join(f"{v:.4g}" for v in x) + "]")
                                    if isinstance(x, (list, tuple)) else x)
        for c in df.select_dtypes(include="float").columns:
            df[c] = df[c].round(4)
        mo.output.replace(mo.vstack([
            mo.md(f"## §5 Détail par target (random date)"),
            mo.ui.table(df, selection=None, page_size=50),
        ]))


if __name__ == "__main__":
    app.run()
