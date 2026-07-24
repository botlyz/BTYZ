import marimo

__generated_with = "0.21.0"
app = marimo.App(width="wide")


@app.cell
def _imports():
    import sys
    sys.path.insert(0, '.')
    import marimo as mo
    import glob as glob_mod
    import os
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from vectorbtpro import vbt
    return go, glob_mod, mo, np, os, pd, vbt


@app.cell
def _controls(mo, os, glob_mod):
    _files = sorted(glob_mod.glob('cache/*.pickle'))
    _options = {os.path.basename(f): f for f in _files}
    if not _options:
        mo.stop(True, mo.callout(mo.md("Aucun pickle trouvé dans `cache/`"), kind="warn"))
    cache_selector = mo.ui.dropdown(
        options=_options,
        value=list(_options.keys())[0],
        label="Fichier",
    )
    robust_threshold = mo.ui.slider(
        start=0.1, stop=1.0, step=0.05, value=0.5,
        label="Seuil robustesse",
        show_value=True,
    )
    mo.output.replace(mo.hstack([cache_selector, robust_threshold], justify="start", gap=2))
    return cache_selector, robust_threshold


@app.cell
def _load(cache_selector, mo, vbt):
    results = vbt.load(cache_selector.value)
    mo.output.replace(mo.md(f"**{len(results):,} résultats chargés** — `{cache_selector.value.split('/')[-1]}`"))
    return (results,)


@app.cell
def _split(results, mo):
    _names = list(results.index.names)
    _fixed = {"set", "split"}

    # Pickle multi-stats (pf.stats complet) : extraire le Sharpe Ratio
    if None in _names:
        _none_idx = _names.index(None)
        _r = results.xs("Sharpe Ratio", level=_none_idx)
    else:
        _r = results

    param_levels = [l for l in _r.index.names if l not in _fixed and l is not None]
    n_splits = int(_r.index.get_level_values("split").nunique())
    train_results = _r.xs("train", level="set").astype(float)
    test_results = _r.xs("test", level="set").astype(float)
    mo.output.replace(mo.md(
        f"Params: `{param_levels}` | Fenêtres: {n_splits} | "
        f"Train: {len(train_results):,} | Test: {len(test_results):,}"
    ))
    return n_splits, param_levels, test_results, train_results


@app.cell
def _stats(mo, n_splits, results, test_results, train_results):
    _corr = float(train_results.corr(test_results))
    _kind = "success" if _corr > 0.5 else ("warn" if _corr > 0 else "danger")
    _label = (
        "Bonne corrélation — résultats généralisent bien" if _corr > 0.5
        else "Corrélation faible — overfitting modéré" if _corr > 0
        else "Corrélation négative — overfitting probable"
    )
    return mo.vstack([
        mo.md("## 1. Vue d'ensemble"),
        mo.hstack([
            mo.stat(value=f"{len(results):,}", label="Résultats"),
            mo.stat(value=str(n_splits), label="Fenêtres"),
            mo.stat(value=f"{train_results.isna().sum()}", label="NaN train"),
            mo.stat(value=f"{_corr:.4f}", label="Corrélation train/test"),
        ]),
        mo.callout(mo.md(f"**{_label}**"), kind=_kind),
    ])


@app.cell
def _heatmap_controls(mo, param_levels):
    _clean = [l for l in param_levels if l is not None]
    _default_x = _clean[2] if len(_clean) > 2 else _clean[-1]
    _choices = {l: l for l in _clean}
    heatmap_x = mo.ui.dropdown(options=_choices, label="Axe X", value=_default_x)
    heatmap_y = mo.ui.dropdown(options=_choices, label="Axe Y", value=_clean[0])
    mo.output.replace(mo.hstack([heatmap_x, heatmap_y], justify="start", gap=2))
    return heatmap_x, heatmap_y


@app.cell
def _heatmaps(go, heatmap_x, heatmap_y, mo, test_results, train_results):
    if heatmap_x.value == heatmap_y.value:
        mo.stop(True, mo.callout(mo.md("Choisis deux axes différents."), kind="warn"))
    _axes = [heatmap_y.value, heatmap_x.value]

    # Heatmap sharpe médian train
    _mat_train = train_results.groupby(level=_axes).median().unstack(level=heatmap_x.value)
    _fig_train = go.Figure(go.Heatmap(
        z=_mat_train.values,
        x=[f"{x:.2g}" for x in _mat_train.columns],
        y=_mat_train.index,
        colorscale="RdYlGn",
        colorbar=dict(title="Sharpe médian"),
    ))
    _fig_train.update_layout(
        title=f"Sharpe médian (train) — {heatmap_y.value} vs {heatmap_x.value}",
        xaxis_title=heatmap_x.value, yaxis_title=heatmap_y.value, height=450,
    )

    # Heatmap delta test - train
    _mat_diff = (test_results - train_results).groupby(level=_axes).median().unstack(level=heatmap_x.value)
    _fig_diff = go.Figure(go.Heatmap(
        z=_mat_diff.values,
        x=[f"{x:.2g}" for x in _mat_diff.columns],
        y=_mat_diff.index,
        colorscale="RdBu", zmid=0,
        colorbar=dict(title="ΔSharpe"),
    ))
    _fig_diff.update_layout(
        title=f"Différence test − train — {heatmap_y.value} vs {heatmap_x.value}",
        xaxis_title=heatmap_x.value, yaxis_title=heatmap_y.value, height=450,
    )

    return mo.vstack([
        mo.md("## 2. Heatmaps"),
        mo.ui.plotly(_fig_train),
        mo.ui.plotly(_fig_diff),
    ])


@app.cell
def _robustness(mo, n_splits, param_levels, robust_threshold, test_results, train_results):
    _pos = train_results.groupby(level=param_levels).apply(
        lambda x: (x.dropna() > 0).sum()
    )
    _med_train = train_results.groupby(level=param_levels).median()
    _med_test = test_results.groupby(level=param_levels).median()
    _mask = _pos >= n_splits * robust_threshold.value
    n_robust = int(_mask.sum())

    if n_robust > 0:
        best_idx = _med_train.loc[_mask].idxmax()
        _label = f"Meilleure combo robuste (seuil {robust_threshold.value:.0%})"
        _kind = "success"
    else:
        best_idx = _med_train.idxmax()
        _label = "Aucune combo robuste — fallback meilleur sharpe médian train"
        _kind = "warn"

    best_params = dict(zip(param_levels, best_idx if isinstance(best_idx, tuple) else [best_idx]))
    _sharpe_train = float(_med_train.loc[best_idx])
    _sharpe_test = float(_med_test.loc[best_idx])

    mo.output.replace(mo.vstack([
        mo.md("## 3. Combos robustes"),
        mo.callout(mo.md(f"**{_label}**  \n`{best_params}`"), kind=_kind),
        mo.hstack([
            mo.stat(value=str(n_robust), label="Combos robustes"),
            mo.stat(value=f"{_sharpe_train:.3f}", label="Sharpe médian train"),
            mo.stat(value=f"{_sharpe_test:.3f}", label="Sharpe médian test"),
        ]),
    ]))
    return best_idx, best_params, n_robust


@app.cell
def _test_perf(best_idx, best_params, go, mo, n_splits, param_levels, test_results):
    _test_perf = test_results.xs(best_idx, level=param_levels)
    _colors = ["green" if v > 0 else "red" for v in _test_perf.values]
    _fig = go.Figure(go.Bar(
        x=[f"Split {i}" for i in _test_perf.index],
        y=_test_perf.values,
        marker_color=_colors,
    ))
    _fig.add_hline(y=0, line_dash="dash", line_color="gray")
    _fig.update_layout(
        title=f"Sharpe test par fenêtre — {best_params}",
        yaxis_title="Sharpe ratio", height=350,
    )
    return mo.vstack([
        mo.md("## 4. Performance out-of-sample"),
        mo.hstack([
            mo.stat(value=f"{_test_perf.mean():.3f}", label="Sharpe moyen"),
            mo.stat(value=f"{_test_perf.median():.3f}", label="Sharpe médian"),
            mo.stat(value=f"{int((_test_perf > 0).sum())}/{n_splits}", label="Fenêtres positives"),
        ]),
        mo.ui.plotly(_fig),
    ])


@app.cell
def _backtest(best_params, cache_selector, mo, os, pd):
    try:
        from src.strategies.keltner import run_backtest
        _fname = os.path.basename(cache_selector.value)
        _name = _fname.replace("kc_wfsl_", "").replace("kc_", "").replace(".pickle", "")
        _pair, _tf, _exchange = _name.split("_")
        _data = pd.read_csv(f"data/raw/{_exchange}/{_tf}/{_pair}.csv")
        _data["date"] = pd.to_datetime(_data["date"], unit="ms")
        _data = _data.set_index("date")
        _pf = run_backtest(_data, **best_params, size=1)
        _stats = _pf.stats()
        _df_stats = pd.DataFrame({"Métrique": _stats.index, "Valeur": _stats.values})
        _out = mo.vstack([
            mo.callout(mo.md("**In-sample** — les params viennent des mêmes données"), kind="warn"),
            mo.ui.table(_df_stats, selection=None),
        ])
    except Exception as _e:
        _out = mo.callout(mo.md(f"Backtest non disponible : `{_e}`"), kind="neutral")
    return mo.vstack([mo.md("## 5. Backtest final (in-sample)"), _out])


@app.cell
def _top10(mo, param_levels, pd, test_results, train_results):
    _med_train = train_results.groupby(level=param_levels).median()
    _med_test = test_results.groupby(level=param_levels).median()
    _top = _med_train.sort_values(ascending=False).head(10).reset_index()
    _top.columns = list(param_levels) + ["sharpe_train"]
    _top["sharpe_test"] = [float(_med_test.loc[tuple(row[param_levels])]) for _, row in _top.iterrows()]
    _top["delta"] = (_top["sharpe_test"] - _top["sharpe_train"]).round(3)
    _top["sharpe_train"] = _top["sharpe_train"].round(3)
    _top["sharpe_test"] = _top["sharpe_test"].round(3)
    return mo.vstack([
        mo.md("## 6. Top 10 combos"),
        mo.ui.table(_top, selection=None),
    ])


@app.cell
def _ranking(go, mo, os, pd, vbt, glob_mod):
    _files = sorted(glob_mod.glob("cache/*.pickle"))
    if len(_files) <= 1:
        mo.stop(True, mo.callout(
            mo.md("**Ranking multi-paires** — Lance `opti.py` en mode `full` pour comparer plusieurs paires."),
            kind="neutral",
        ))
    _rows = []
    for _f in _files:
        _res = vbt.load(_f)
        _tr = _res.xs("train", level="set")
        _te = _res.xs("test", level="set")
        _rnames = list(_res.index.names)
        if None in _rnames:
            _res = _res.xs("Sharpe Ratio", level=_rnames.index(None))
            _tr = _res.xs("train", level="set").astype(float)
            _te = _res.xs("test", level="set").astype(float)
        _pl = [l for l in _res.index.names if l not in {"set", "split"} and l is not None]
        _ns = int(_res.index.get_level_values("split").nunique())
        _pos = _tr.groupby(level=_pl).apply(lambda x: (x.dropna() > 0).sum(), include_groups=False)
        _mt = _tr.groupby(level=_pl).median()
        _mts = _te.groupby(level=_pl).median()
        _bi = _mt.idxmax()
        _rows.append({
            "fichier": os.path.basename(_f),
            "n_robust": int((_pos >= _ns * 0.5).sum()),
            "sharpe_train": round(float(_mt.loc[_bi]), 3),
            "sharpe_test": round(float(_mts.loc[_bi]), 3),
            "delta": round(float(_mts.loc[_bi] - _mt.loc[_bi]), 3),
            "corr": round(float(_tr.corr(_te)), 3),
            "best_params": str(_bi),
        })
    _df_rank = pd.DataFrame(_rows).sort_values("sharpe_test", ascending=False).reset_index(drop=True)
    _colors = ["green" if v > 0 else "red" for v in _df_rank["sharpe_test"]]
    _fig = go.Figure(go.Bar(
        x=_df_rank["fichier"],
        y=_df_rank["sharpe_test"],
        marker_color=_colors,
    ))
    _fig.add_hline(y=0, line_dash="dash", line_color="gray")
    _fig.update_layout(title="Sharpe test par paire (meilleure combo)", height=350, showlegend=False)
    return mo.vstack([
        mo.md("## 7. Ranking multi-paires"),
        mo.ui.table(_df_rank, selection=None),
        mo.ui.plotly(_fig),
    ])


if __name__ == "__main__":
    app.run()
