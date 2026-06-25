"""WFA tuning analysis — visualise comparison of (train_days, test_days, step_days) configs.

Sélecteur d'approche, puis pour chaque config tuning testée:
- mean OOS Sharpe + WFE corr + % positive folds
- bar chart composite score
- scatter mean test return vs mean test DD
- heatmap par paire × config (mean test sharpe)
- recommandation finale

Charge: results/<APPROACH>/wfa_tuning/_summary.json + */<pair>/summary.json
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

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    return go, json, mo, np, pathlib, pd, px


@app.cell
def _constants(pathlib):
    RESULTS_ROOT = pathlib.Path("/home/botlyz-gpu/BTYZ/results")
    return (RESULTS_ROOT,)


@app.cell
def _discover(RESULTS_ROOT):
    """Liste les approaches avec _summary.json sous wfa_tuning."""
    approaches = []
    if RESULTS_ROOT.exists():
        for _d in sorted(RESULTS_ROOT.iterdir()):
            if not _d.is_dir():
                continue
            _s = _d / "wfa_tuning" / "_summary.json"
            if _s.exists():
                approaches.append(_d.name)
    return (approaches,)


@app.cell
def _approach_sel(approaches, mo):
    mo.stop(not approaches, mo.callout(
        mo.md("Aucun WFA tuning trouvé. Lance :\n\n"
              "```bash\npython -m engine.wfa_tuning --approach <ID> --pairs BTC ETH HYPE ...\n```"),
        kind="warn"
    ))
    approach = mo.ui.dropdown(options={a: a for a in approaches}, value=approaches[0], label="Approach")
    mo.output.replace(mo.hstack([approach], justify="start", gap=2))
    return (approach,)


@app.cell
def _load_summary(RESULTS_ROOT, approach, json, mo):
    sj = RESULTS_ROOT / approach.value / "wfa_tuning" / "_summary.json"
    mo.stop(not sj.exists(), mo.callout(mo.md(f"`{sj}` introuvable."), kind="warn"))
    summary = json.loads(sj.read_text())
    return (summary,)


# ─────────────────────────────────────────────────────────────────────
# §1 Ranking table
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section1(mo, pd, summary):
    ranking = summary.get("ranking", [])
    if not ranking:
        mo.output.replace(mo.md("## §1 Pas de ranking."))
    else:
        df = pd.DataFrame(ranking)
        cols = ["cfg_tag", "n_pairs_ok", "n_folds_total", "avg_folds_per_pair",
                "mean_train_sh", "mean_test_sh", "median_test_sh", "sh_degradation",
                "pct_positive_folds", "wfe_corr_sharpe",
                "mean_test_ret_pct", "mean_test_dd_pct", "composite_score"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        best = df.iloc[0]
        mo.output.replace(mo.vstack([
            mo.md(f"## §1 Ranking — {summary.get('approach')} "
                  f"(tf={summary.get('tf')}, bps={summary.get('bps')}, "
                  f"trials={summary.get('trials_per_fold')}, "
                  f"pairs={', '.join(summary.get('pairs', []))})"),
            mo.hstack([
                mo.stat(label="Best cfg", value=best["cfg_tag"]),
                mo.stat(label="Composite score", value=f"{best['composite_score']:.3f}"),
                mo.stat(label="Mean test Sharpe", value=f"{best['mean_test_sh']:.2f}"),
                mo.stat(label="% positive folds", value=f"{best['pct_positive_folds']:.0f}%"),
                mo.stat(label="WFE corr", value=f"{best.get('wfe_corr_sharpe', 0):.2f}"),
            ], justify="start", gap=4),
            mo.ui.table(df.reset_index(drop=True), selection=None),
        ]))


# ─────────────────────────────────────────────────────────────────────
# §2 Composite score bar chart
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section2(go, mo, pd, summary):
    ranking = summary.get("ranking", [])
    if not ranking:
        return
    df = pd.DataFrame(ranking)
    df = df.sort_values("composite_score", ascending=True)
    fig = go.Figure([go.Bar(
        x=df["composite_score"], y=df["cfg_tag"], orientation="h",
        marker_color=["#2ecc71" if s == df["composite_score"].max() else "#3498db" for s in df["composite_score"]],
        text=[f"{s:.2f}" for s in df["composite_score"]],
        textposition="auto",
    )])
    fig.update_layout(
        title="Composite score par config (plus haut = mieux)",
        xaxis_title="Composite score", height=350,
    )
    mo.output.replace(mo.vstack([
        mo.md("## §2 Composite score"),
        mo.ui.plotly(fig),
    ]))


# ─────────────────────────────────────────────────────────────────────
# §3 Per-pair × config heatmap
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section3(RESULTS_ROOT, approach, go, json, mo, np, pd, summary):
    base = RESULTS_ROOT / approach.value / "wfa_tuning"
    pairs = summary.get("pairs", [])
    cfg_tags = [r["cfg_tag"] for r in summary.get("ranking", [])]

    matrix = np.full((len(pairs), len(cfg_tags)), np.nan)
    for i, pair in enumerate(pairs):
        for j, cfg in enumerate(cfg_tags):
            sj = base / cfg / pair / "summary.json"
            if not sj.exists():
                continue
            d = json.loads(sj.read_text())
            ts = [f.get("test_metrics", {}).get("sharpe_ratio") for f in d.get("folds", [])]
            ts = [float(s) for s in ts if s is not None and not isinstance(s, str)]
            if ts:
                matrix[i, j] = float(np.mean(ts))

    if not pairs or not cfg_tags or np.isnan(matrix).all():
        return

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=cfg_tags, y=pairs,
        colorscale="RdYlGn", zmid=0, zmin=-3, zmax=3,
        text=[[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in matrix],
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title="Mean test Sharpe par paire × config",
        height=300 + 40 * len(pairs),
    )
    mo.output.replace(mo.vstack([
        mo.md("## §3 Heatmap par paire × config"),
        mo.ui.plotly(fig),
    ]))


# ─────────────────────────────────────────────────────────────────────
# §4 Test return vs DD scatter
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section4(go, mo, pd, summary):
    ranking = summary.get("ranking", [])
    if not ranking:
        return
    df = pd.DataFrame(ranking)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["mean_test_dd_pct"], y=df["mean_test_ret_pct"],
        mode="markers+text", text=df["cfg_tag"], textposition="top center",
        marker=dict(size=df["composite_score"] * 3 + 8,
                    color=df["wfe_corr_sharpe"], colorscale="RdYlGn",
                    showscale=True, colorbar=dict(title="WFE corr"),
                    cmid=0),
    ))
    fig.update_layout(
        title="Mean test return % vs mean test DD % (couleur = WFE corr, taille = score)",
        xaxis_title="Mean test DD %", yaxis_title="Mean test return %",
        height=420,
    )
    mo.output.replace(mo.vstack([
        mo.md("## §4 Return vs Drawdown trade-off"),
        mo.ui.plotly(fig),
    ]))


# ─────────────────────────────────────────────────────────────────────
# §5 Recommandation
# ─────────────────────────────────────────────────────────────────────

@app.cell
def _section5(mo, summary):
    ranking = summary.get("ranking", [])
    if not ranking:
        return
    best = ranking[0]
    cfg = best["cfg_tag"]
    # Parse train_d, test_d, step_d from tag like "90d_21d_21d"
    parts = cfg.replace("d", "").split("_")
    train_d = int(parts[0]) if len(parts) > 0 else None
    test_d  = int(parts[1]) if len(parts) > 1 else None
    step_d  = int(parts[2]) if len(parts) > 2 else None

    md = f"""## §5 Recommandation

**Config gagnante : `{cfg}`** (composite score = {best['composite_score']})

| Param | Valeur | Interprétation |
|---|---|---|
| `--train-days` | **{train_d}** | Fenêtre d'entraînement |
| `--test-days` | **{test_d}** | Période OOS pour valider chaque set de params |
| `--step-days` | **{step_d}** | {'Chevauchant (re-opti plus fréquente)' if step_d and test_d and step_d < test_d else 'Non chevauchant'} |

**Fréquence de re-optimisation en prod : tous les {step_d} jours**
(= step entre folds. Si tu re-optimise plus rarement, les params dérivent.)

**Métriques de cette config:**
- Mean OOS Sharpe : **{best['mean_test_sh']:.2f}**
- % folds rentables : **{best['pct_positive_folds']:.0f}%**
- WFE corr (train→test) : **{best.get('wfe_corr_sharpe', 0):.2f}** (positif = train Sharpe prédictif du OOS)
- Sharpe degradation : **{best['sh_degradation']:.2f}** (=train_sh - test_sh, plus bas = moins d'overfit)
- Mean test return : **{best['mean_test_ret_pct']:.2f}%** sur fenêtre {test_d}j
- Mean test DD : **{best['mean_test_dd_pct']:.2f}%**

**Lance la prod avec :**
```bash
python -m engine.cli wfa --approach <ID> --tf <tf> --bps <bps> \\
  --pairs ... --trials 500 \\
  --train-days {train_d} --test-days {test_d} --step-days {step_d}
```
"""
    mo.output.replace(mo.md(md))


if __name__ == "__main__":
    app.run()
