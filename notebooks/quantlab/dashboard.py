"""quantlab — dashboard marimo unique du pipeline (LECTURE SEULE).

Lit ledger.db + results/quantlab/<SID>/<stage>/ + le store parquet ; ne lance
JAMAIS de calcul de pipeline. Sections dans l'ordre du funnel :
header/sélecteur, ledger, data coverage, screening, optimisation, plateau,
stats, déploiement, holdout/incubation, fiche.

Lancer : marimo run notebooks/quantlab/dashboard.py

Règles marimo : noms exportés uniques entre cellules, locales préfixées `_`.
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import math
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # sys.path -> BTYZ/src (chemin relatif au notebook, worktree-compatible)
    _cands = []
    _nb = mo.notebook_dir()
    if _nb:
        _cands.append((Path(_nb) / ".." / ".." / "src").resolve())
    _cands.append((Path.cwd() / "src").resolve())
    _cands.append(Path.cwd().resolve())
    for _c in _cands:
        if (_c / "quantlab" / "config.py").exists():
            if str(_c) not in sys.path:
                sys.path.insert(0, str(_c))
            break
    return Path, go, json, math, mo, np, pd


@app.cell
def _(Path):
    from quantlab import config
    from quantlab.ledger import family_of

    # ---- overrides de TEST uniquement (laisser None : vrais chemins du repo)
    RESULTS_ROOT_OVERRIDE = None   # ex: "/tmp/qlab_fake/results/quantlab"
    LEDGER_DB_OVERRIDE = None      # ex: "/tmp/qlab_fake/ledger.db"

    RESULTS_ROOT = (Path(RESULTS_ROOT_OVERRIDE) if RESULTS_ROOT_OVERRIDE
                    else config.RESULTS_ROOT)
    LEDGER_DB = (Path(LEDGER_DB_OVERRIDE) if LEDGER_DB_OVERRIDE
                 else config.LEDGER_DB)
    STAGES = list(config.STAGES)
    return LEDGER_DB, RESULTS_ROOT, STAGES, config, family_of


@app.cell
def _(Path, STAGES, go, json, mo, np, pd):
    # ---------------- palette (référence dataviz — light, fond transparent)
    PAL = {
        "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
        "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
        "violet": "#4a3aa7", "red": "#e34948",
        "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
        "critical": "#d03b3b",
        "muted": "#898781",
        "grid": "rgba(137,135,129,0.22)",
        "axis": "rgba(137,135,129,0.45)",
        "neutral_mid": "#f0efec",
    }
    _CAT = [PAL["blue"], PAL["orange"], PAL["aqua"], PAL["yellow"],
            PAL["magenta"], PAL["green"], PAL["violet"], PAL["red"]]
    # ordre FIXE des tfs -> couleur stable quel que soit le sous-ensemble affiché
    TF_COLOR = {tf: _CAT[i % len(_CAT)]
                for i, tf in enumerate(["1m", "3m", "5m", "15m", "1h", "4h"])}
    SEQ_BLUES = [[0.0, "#cde2fb"], [0.25, "#9ec5f4"], [0.5, "#5598e7"],
                 [0.75, "#256abf"], [1.0, "#0d366b"]]
    DIV_SCALE = [[0.0, "#d03b3b"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]

    def read_json(fp):
        try:
            fp = Path(fp)
            if fp.exists():
                return json.loads(fp.read_text())
        except Exception:
            pass
        return None

    def fmt(v, nd=2, suffix=""):
        try:
            f = float(v)
            if not np.isfinite(f):
                return "n/a"
            return f"{f:.{nd}f}{suffix}"
        except (TypeError, ValueError):
            return "n/a"

    def style_fig(fig, height=340, title=None):
        fig.update_layout(
            template="none", height=height,
            margin=dict(l=55, r=20, t=42 if title else 22, b=42),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                      size=12, color=PAL["muted"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                        bgcolor="rgba(0,0,0,0)"),
            hoverlabel=dict(font_size=12),
        )
        if title:
            fig.update_layout(title=dict(text=title, x=0,
                                         font=dict(size=13, color=PAL["muted"])))
        fig.update_xaxes(gridcolor=PAL["grid"], zerolinecolor=PAL["axis"],
                         linecolor=PAL["axis"])
        fig.update_yaxes(gridcolor=PAL["grid"], zerolinecolor=PAL["axis"],
                         linecolor=PAL["axis"])
        return fig

    def gauge(value, title, zones, vrange, nd=2, suffix=""):
        """Jauge plotly ; zones = [(a, b, hex)] en couleurs statut translucides."""
        def _rgba(hx, a=0.30):
            return (f"rgba({int(hx[1:3], 16)},{int(hx[3:5], 16)},"
                    f"{int(hx[5:7], 16)},{a})")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value) if value is not None and np.isfinite(float(value))
            else 0.0,
            number={"valueformat": f".{nd}f", "suffix": suffix,
                    "font": {"size": 30}},
            title={"text": title, "font": {"size": 13, "color": PAL["muted"]}},
            gauge={"axis": {"range": list(vrange), "tickcolor": PAL["muted"],
                            "tickfont": {"size": 10}},
                   "bar": {"color": PAL["muted"], "thickness": 0.28},
                   "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                   "steps": [{"range": [a, b], "color": _rgba(c)}
                             for a, b, c in zones]}))
        fig.update_layout(template="none", height=210,
                          margin=dict(l=25, r=25, t=40, b=10),
                          paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=PAL["muted"]))
        return fig

    def stat_tile(label, value, sub="", accent=None, size=30):
        _color = accent or "inherit"
        return mo.Html(
            f'<div style="border:1px solid rgba(137,135,129,.35);'
            f'border-radius:10px;padding:12px 18px;min-width:160px;">'
            f'<div style="font-size:11px;text-transform:uppercase;'
            f'letter-spacing:.06em;color:#898781;">{label}</div>'
            f'<div style="font-size:{size}px;font-weight:700;color:{_color};'
            f'line-height:1.25;">{value}</div>'
            f'<div style="font-size:11px;color:#898781;">{sub}</div></div>')

    _CHIP = {
        "PASS": ("✅", "rgba(12,163,12,.14)"),
        "FIXED": ("✅", "rgba(12,163,12,.14)"),
        "ADAPTIVE": ("✅", "rgba(12,163,12,.14)"),
        "COHERENT": ("✅", "rgba(12,163,12,.14)"),
        "REJECT": ("❌", "rgba(208,59,59,.14)"),
        "EFFONDREMENT": ("❌", "rgba(208,59,59,.14)"),
        "ERROR": ("⚠️", "rgba(250,178,25,.16)"),
        "MARGINAL": ("⚠️", "rgba(250,178,25,.16)"),
    }

    def chips_html(states):
        """Rangée façon mermaid : chip par étape du funnel + flèches."""
        _parts = []
        for _i, _st in enumerate(STAGES):
            _v = states.get(_st)
            _emoji, _bg = _CHIP.get(str(_v), ("⚠️", "rgba(250,178,25,.16)")) \
                if _v else ("•", "rgba(137,135,129,.10)")
            _title = f"{_st}: {_v or 'non lancé'}"
            if _i:
                _parts.append('<span style="color:#898781;">→</span>')
            _parts.append(
                f'<span title="{_title}" style="background:{_bg};'
                f'border:1px solid rgba(137,135,129,.30);border-radius:999px;'
                f'padding:3px 10px;font-size:12px;white-space:nowrap;">'
                f'{_emoji} {_st}</span>')
        return mo.Html('<div style="display:flex;flex-wrap:wrap;gap:6px;'
                       'align-items:center;">' + "".join(_parts) + "</div>")

    def verdict_kind(v):
        v = str(v)
        if v in ("PASS", "FIXED", "ADAPTIVE", "COHERENT"):
            return "success"
        if v in ("REJECT", "EFFONDREMENT", "ERROR"):
            return "danger"
        if v in ("MARGINAL",):
            return "warn"
        return "neutral"

    def verdict_callout(vd, stage=""):
        if not vd:
            return None
        _v = str(vd.get("verdict", "?"))
        _kind = verdict_kind(_v)
        _emo = {"success": "✅", "danger": "❌", "warn": "⚠️"}.get(_kind, "•")
        _reason = vd.get("reason", "")
        return mo.callout(
            mo.md(f"**{_emo} {stage or vd.get('stage', '')} — {_v}** — {_reason}"),
            kind=_kind)

    def missing(msg, cmd=None):
        _txt = f"**{msg}**"
        if cmd:
            _txt += f"\n\n```\n{cmd}\n```"
        return mo.callout(mo.md(_txt), kind="info")

    def top_params(df, n=2, ycol="sharpe_concat"):
        """Les n params (colonnes p_*) les plus corrélés (|Spearman|) au Sharpe."""
        if df is None or ycol not in df:
            return []
        _scored = []
        _y = df[ycol].rank()
        for _c in df.columns:
            if not _c.startswith("p_"):
                continue
            if not pd.api.types.is_numeric_dtype(df[_c]) or df[_c].nunique() < 2:
                continue
            _rho = df[_c].rank().corr(_y)
            if pd.notna(_rho):
                _scored.append((abs(float(_rho)), _c))
        _scored.sort(reverse=True)
        return [c for _, c in _scored[:n]]

    def binned_surface(df, cx, cy, cz="sharpe_concat", bins=14):
        """Grille moyenne (bins x bins) du Sharpe sur 2 params -> contour."""
        _x = df[cx].to_numpy(dtype=float)
        _y = df[cy].to_numpy(dtype=float)
        _z = df[cz].to_numpy(dtype=float)
        _ok = np.isfinite(_x) & np.isfinite(_y) & np.isfinite(_z)
        _x, _y, _z = _x[_ok], _y[_ok], _z[_ok]
        if len(_x) < 10 or _x.min() == _x.max() or _y.min() == _y.max():
            return None
        _xe = np.linspace(_x.min(), _x.max(), bins + 1)
        _ye = np.linspace(_y.min(), _y.max(), bins + 1)
        _ix = np.clip(np.digitize(_x, _xe) - 1, 0, bins - 1)
        _iy = np.clip(np.digitize(_y, _ye) - 1, 0, bins - 1)
        _sum = np.zeros((bins, bins))
        _cnt = np.zeros((bins, bins))
        for _a, _b, _c in zip(_ix, _iy, _z):
            _sum[_b, _a] += _c
            _cnt[_b, _a] += 1
        with np.errstate(invalid="ignore"):
            _grid = np.where(_cnt > 0, _sum / np.maximum(_cnt, 1), np.nan)
        return ((_xe[:-1] + _xe[1:]) / 2, (_ye[:-1] + _ye[1:]) / 2, _grid)

    return (DIV_SCALE, PAL, SEQ_BLUES, TF_COLOR, binned_surface, chips_html,
            fmt, gauge, missing, read_json, stat_tile, style_fig, top_params,
            verdict_callout, verdict_kind)


@app.cell
def _(LEDGER_DB, STAGES):
    # ---------------- accès ledger STRICTEMENT lecture seule (URI mode=ro,
    # jamais d'instance Ledger : elle créerait le fichier/le schéma)
    def ro_query(sql, params=()):
        if not LEDGER_DB.exists():
            return []
        import sqlite3
        try:
            _con = sqlite3.connect(f"file:{LEDGER_DB}?mode=ro", uri=True,
                                   timeout=5)
            try:
                return _con.execute(sql, params).fetchall()
            finally:
                _con.close()
        except Exception:
            return []

    def get_debt(family):
        _rows = ro_query("SELECT research_debt FROM families WHERE family = ?",
                         (family,))
        return int(_rows[0][0]) if _rows else 0

    def ledger_states(sid):
        """stage -> dernier verdict du ledger (None si jamais couru)."""
        _states = {s: None for s in STAGES}
        for _st, _v in ro_query(
                "SELECT stage, verdict FROM runs WHERE strategy_id = ? AND "
                "verdict IS NOT NULL AND verdict != '' ORDER BY id", (sid,)):
            if _st in _states:
                _states[_st] = _v
        return _states

    def holdout_consumed(family):
        return bool(ro_query(
            "SELECT 1 FROM events WHERE family = ? AND kind = "
            "'holdout_consumed' LIMIT 1", (family,)))

    return get_debt, holdout_consumed, ledger_states, ro_query


@app.cell
def _(RESULTS_ROOT, mo, ro_query):
    _ids = set()
    if RESULTS_ROOT.is_dir():
        for _p in RESULTS_ROOT.iterdir():
            if _p.is_dir() and not _p.name.startswith((".", "_")):
                _ids.add(_p.name)
    for (_sid,) in ro_query(
            "SELECT DISTINCT strategy_id FROM runs WHERE strategy_id != ''"):
        _ids.add(_sid)
    try:
        from quantlab import registry as _registry
        _ids.update(_registry.list_strategy_dirs())
    except Exception:
        pass
    strategy_ids = sorted(_ids)
    strategy_dd = mo.ui.dropdown(
        options=strategy_ids,
        value=strategy_ids[0] if strategy_ids else None,
        label="**Stratégie**")
    return strategy_dd, strategy_ids


@app.cell
def _(RESULTS_ROOT, STAGES, family_of, ledger_states, read_json, strategy_dd):
    sel_sid = strategy_dd.value
    sel_family = family_of(sel_sid) if sel_sid else None
    sel_dir = RESULTS_ROOT / sel_sid if sel_sid else None

    def merge_states(sid):
        """Verdicts ledger, complétés par les verdict.json sur disque."""
        _states = ledger_states(sid)
        for _st in STAGES:
            if _states.get(_st) is None:
                _vd = read_json(RESULTS_ROOT / sid / _st / "verdict.json")
                if _vd:
                    _states[_st] = _vd.get("verdict")
        return _states

    sel_states = merge_states(sel_sid) if sel_sid else \
        {s: None for s in STAGES}
    return merge_states, sel_dir, sel_family, sel_sid, sel_states


@app.cell
def _(LEDGER_DB, chips_html, get_debt, math, mo, sel_family, sel_states,
      stat_tile, strategy_dd, strategy_ids):
    # ================================================================ 1 HEADER
    _out = [mo.md("# quantlab — funnel de validation")]
    if not strategy_ids:
        _out.append(mo.callout(mo.md(
            "**Aucune stratégie découverte** (ni `strategies/`, ni "
            "`results/quantlab/`, ni ledger). Dépose un dossier dans "
            "`strategies/<ID>/` puis lance `python -m quantlab.cli run <ID>`."),
            kind="warn"))
    else:
        _debt = get_debt(sel_family) if sel_family else 0
        _seuil = math.sqrt(2.0 * math.log(max(_debt, 2)))
        _tiles = mo.hstack([
            stat_tile("N — dette de recherche", f"{_debt:,}".replace(",", " "),
                      f"famille {sel_family} (tous backtests comptés)",
                      size=44),
            stat_tile("Seuil Sharpe crédible √(2·ln N)", f"{_seuil:.2f}",
                      "en dessous : indistinguable du bruit de la recherche",
                      accent="#eb6834", size=44),
        ], gap=1, justify="start")
        _out.append(mo.hstack([strategy_dd], justify="start"))
        _out.append(chips_html(sel_states))
        _out.append(_tiles)
        if not LEDGER_DB.exists():
            _out.append(mo.callout(mo.md(
                f"`{LEDGER_DB}` absent — aucun test loggé. Le seuil affiché "
                "est le plancher (N=2). La dette réelle apparaîtra au premier "
                "run du pipeline."), kind="info"))
    mo.vstack(_out, gap=1)
    return


@app.cell
def _(LEDGER_DB, STAGES, get_debt, go, json, math, merge_states, missing, mo,
      pd, ro_query, sel_family, style_fig):
    # ================================================================ 2 LEDGER
    _out = [mo.md("## 2 · Ledger — comptabilité de la recherche")]
    ledger_runs_table = None
    if not LEDGER_DB.exists():
        _out.append(missing(
            "ledger.db absent — rien n'a encore été loggé",
            "python -m quantlab.cli run <STRATEGY_ID>"))
    elif sel_family:
        # ---- runs de la famille
        _rows = ro_query(
            "SELECT id, strategy_id, stage, verdict, n_tests, started_at, "
            "finished_at FROM runs WHERE family = ? ORDER BY id DESC",
            (sel_family,))
        if _rows:
            _runs = pd.DataFrame(_rows, columns=[
                "run", "strategy_id", "stage", "verdict", "n_tests",
                "started_at", "finished_at"])
            for _c in ("started_at", "finished_at"):
                _runs[_c] = pd.to_datetime(
                    _runs[_c], errors="coerce", utc=True,
                    format="ISO8601").dt.strftime("%Y-%m-%d %H:%M")
            ledger_runs_table = mo.ui.table(_runs, page_size=10,
                                            selection=None)
            _out.append(mo.md(f"**Runs de la famille `{sel_family}`** "
                              f"({len(_runs)})"))
            _out.append(ledger_runs_table)
        else:
            _out.append(missing(f"aucun run loggé pour la famille "
                                f"`{sel_family}`"))

        # ---- N cumulé dans le temps (events kind='tests')
        _ev = ro_query(
            "SELECT at, payload_json FROM events WHERE family = ? AND "
            "kind = 'tests' ORDER BY id", (sel_family,))
        if _ev:
            _ts = pd.to_datetime([e[0] for e in _ev], errors="coerce",
                                 utc=True, format="ISO8601")
            _ns = pd.Series(
                [int(json.loads(e[1]).get("n", 0)) for e in _ev]).cumsum()
            _fig = go.Figure(go.Scatter(
                x=_ts, y=_ns, mode="lines", line_shape="hv",
                line=dict(color="#2a78d6", width=2), name="N cumulé",
                hovertemplate="%{x}<br>N=%{y}<extra></extra>"))
            _fig.update_yaxes(title_text="backtests comptés (N)")
            _out.append(style_fig(_fig, height=260,
                                  title="Dette de recherche cumulée "
                                        "(√(2·ln N) = seuil Sharpe)"))

        # ---- funnel global (toutes stratégies du ledger)
        _sids = sorted({r[0] for r in ro_query(
            "SELECT DISTINCT strategy_id FROM runs WHERE strategy_id != ''")})
        if _sids:
            _frows = []
            for _s in _sids:
                _st = merge_states(_s)
                from quantlab.ledger import family_of as _fam
                _d = get_debt(_fam(_s))
                _frows.append({"strategy_id": _s, **{k: (_st[k] or "·")
                                                     for k in STAGES},
                               "N": _d,
                               "seuil": round(math.sqrt(
                                   2 * math.log(max(_d, 2))), 2)})
            _out.append(mo.accordion({
                "Funnel global (toutes stratégies)":
                    mo.ui.table(pd.DataFrame(_frows), page_size=15,
                                selection=None)}))
    mo.vstack(_out, gap=1)
    return (ledger_runs_table,)


@app.cell
def _(SEQ_BLUES, config, go, missing, mo, pd, style_fig):
    # ========================================================= 3 DATA COVERAGE
    # lecture seule du store parquet (data.coverage lit les index, ne
    # déclenche aucun download/build)
    _out = [mo.md("## 3 · Data coverage — store parquet")]
    cov_df = None
    if config.STORE_ROOT.is_dir():
        try:
            from quantlab.data.sync import coverage as _coverage
            cov_df = _coverage()
        except Exception as _e:
            _out.append(mo.callout(
                mo.md(f"`data.coverage()` en échec : `{type(_e).__name__}: "
                      f"{_e}`"), kind="danger"))
    if cov_df is None or not len(cov_df):
        _out.append(missing(
            "store parquet vide — synchronise les données d'abord",
            "python -m quantlab.cli data sync"))
    else:
        _tabs = {}
        _tf_order = [t for t in config.TIMEFRAMES]
        for _src in sorted(cov_df["source"].unique()):
            _sub = cov_df[cov_df["source"] == _src]
            _piv = _sub.pivot_table(index="pair", columns="tf",
                                    values="n_bars", aggfunc="first")
            _piv = _piv.reindex(
                columns=[t for t in _tf_order if t in _piv.columns])
            _fig = go.Figure(go.Heatmap(
                z=_piv.to_numpy(dtype=float),
                x=list(_piv.columns), y=list(_piv.index),
                colorscale=SEQ_BLUES, xgap=2, ygap=2,
                colorbar=dict(title="n_bars", thickness=12),
                hovertemplate="%{y} · %{x}<br>%{z:,} barres<extra></extra>"))
            _fig.update_yaxes(autorange="reversed")
            _h = max(240, 26 * len(_piv.index) + 90)
            _cut0 = _sub["holdout_cutoff"].min()
            _cut1 = _sub["holdout_cutoff"].max()
            _tabs[_src] = mo.vstack([
                style_fig(_fig, height=_h,
                          title=f"{_src} — barres par paire × TF"),
                mo.md(f"Cutoff **holdout** (dev < cutoff ≤ holdout) : "
                      f"`{_cut0:%Y-%m-%d}` → `{_cut1:%Y-%m-%d}` selon la "
                      f"paire — verrouillé, lisible uniquement via le gate."),
            ])
        _out.append(mo.ui.tabs(_tabs))
        _cov_disp = cov_df.copy()
        for _c in ("start", "end", "holdout_cutoff"):
            _cov_disp[_c] = pd.to_datetime(
                _cov_disp[_c]).dt.strftime("%Y-%m-%d")
        _out.append(mo.accordion({
            "Table complète du store":
                mo.ui.table(_cov_disp, page_size=15, selection=None)}))
    mo.vstack(_out, gap=1)
    return (cov_df,)


@app.cell
def _(PAL, TF_COLOR, config, go, missing, mo, np, pd, read_json, sel_dir,
      sel_sid, style_fig, verdict_callout):
    # ============================================================ 4 SCREENING
    _out = [mo.md("## 4 · Screening — params par défaut, tout l'univers")]
    scr_cells = None
    _vd = None
    if sel_dir is not None:
        _fp = sel_dir / "screening" / "cells.parquet"
        if _fp.exists():
            try:
                scr_cells = pd.read_parquet(_fp)
            except Exception as _e:
                _out.append(mo.callout(mo.md(f"`{_fp}` illisible : `{_e}`"),
                                       kind="danger"))
        _vd = read_json(sel_dir / "screening" / "verdict.json")

    scr_table = None
    if scr_cells is None:
        _out.append(missing(
            f"screening pas encore lancé pour `{sel_sid or '—'}`",
            f"python -m quantlab.cli screen {sel_sid or '<SID>'}"))
    else:
        if _vd:
            _out.append(verdict_callout(_vd, "screening"))

        # ---- scatter sharpe vs qval (log x), couleur = tf
        _fig = go.Figure()
        for _tf in [t for t in config.TIMEFRAMES
                    if t in set(scr_cells["tf"])]:
            _sub = scr_cells[scr_cells["tf"] == _tf]
            _fig.add_trace(go.Scatter(
                x=np.clip(_sub["qval"].to_numpy(dtype=float), 1e-4, 1.0),
                y=_sub["sharpe"], mode="markers", name=_tf,
                marker=dict(size=9, color=TF_COLOR.get(_tf, PAL["muted"]),
                            line=dict(width=1, color="rgba(255,255,255,.7)")),
                customdata=_sub[["pair", "source"]],
                hovertemplate="%{customdata[0]} (%{customdata[1]}, " + _tf
                              + ")<br>sharpe=%{y:.2f} · q=%{x:.3f}"
                                "<extra></extra>"))
        _fig.add_vline(x=config.FDR_Q, line_dash="dash",
                       line_color=PAL["critical"],
                       annotation_text=f"FDR q={config.FDR_Q}",
                       annotation_font_color=PAL["critical"])
        _fig.add_hline(y=0, line_color=PAL["axis"])
        _fig.update_xaxes(type="log", title_text="q-value (BH)")
        _fig.update_yaxes(title_text="Sharpe annualisé")
        _out.append(style_fig(_fig, height=380,
                              title="Sharpe vs q-value — survivantes à "
                                    "gauche de la ligne"))

        # ---- fraction de paires positives par tf (seuil 45 %)
        _pos = (scr_cells.groupby(["tf", "pair"])["sharpe"].mean()
                .gt(0).groupby("tf").mean() * 100.0)
        _pos = _pos.reindex([t for t in config.TIMEFRAMES if t in _pos.index])
        _fig2 = go.Figure(go.Bar(
            x=list(_pos.index), y=_pos.to_numpy(),
            marker=dict(color=PAL["blue"],
                        cornerradius=4),
            width=0.55,
            hovertemplate="%{x}: %{y:.0f}% paires positives<extra></extra>"))
        _fig2.add_hline(y=config.MIN_POSITIVE_PAIR_FRAC * 100,
                        line_dash="dash", line_color=PAL["serious"],
                        annotation_text=f"seuil "
                        f"{config.MIN_POSITIVE_PAIR_FRAC:.0%}",
                        annotation_font_color=PAL["serious"])
        _fig2.update_yaxes(title_text="% paires Sharpe > 0", range=[0, 100])
        _out.append(style_fig(_fig2, height=260,
                              title="Généralisation inter-paires par TF"))

        # ---- table triée par q-value
        _disp = scr_cells.sort_values("qval").copy()
        for _c, _nd in (("sharpe", 2), ("pval", 4), ("qval", 4),
                        ("null_q95", 2), ("total_return_pct", 1),
                        ("fees_bps", 1)):
            if _c in _disp:
                _disp[_c] = _disp[_c].astype(float).round(_nd)
        for _c in ("data_start", "data_end"):
            if _c in _disp:
                _disp[_c] = pd.to_datetime(
                    _disp[_c], errors="coerce").dt.strftime("%Y-%m-%d")
        scr_table = mo.ui.table(_disp, page_size=12, selection=None)
        _out.append(scr_table)
    mo.vstack(_out, gap=1)
    return scr_cells, scr_table


@app.cell
def _(pd, read_json, sel_dir):
    # ------------------------------------------------ chargement optimize
    # (partagé entre les sections 5 optimisation et 6 plateau)
    opt_summary = None
    opt_trials_df = None
    opt_error = None
    opt_dir = sel_dir / "optimize" if sel_dir is not None else None
    if opt_dir is not None:
        opt_summary = read_json(opt_dir / "summary.json")
        _db = opt_dir / "optuna.db"
        if _db.exists():
            try:
                import optuna as _optuna
                _optuna.logging.set_verbosity(_optuna.logging.WARNING)
                _storage = f"sqlite:///{_db}"
                _name = (opt_summary or {}).get("study_name")
                if not _name:
                    _sums = _optuna.study.get_all_study_summaries(
                        storage=_storage)
                    _name = _sums[-1].study_name if _sums else None
                if _name:
                    _study = _optuna.load_study(study_name=_name,
                                                storage=_storage)
                    _rows = []
                    for _t in _study.get_trials(deepcopy=False):
                        if _t.state.name != "COMPLETE":
                            continue
                        _ua = _t.user_attrs
                        _sc = _ua.get("sharpe_concat")
                        _params = _ua.get("params") or _t.params
                        _rows.append({
                            "trial": _t.number,
                            "sharpe_concat": (float(_sc) if _sc is not None
                                              else float("nan")),
                            "hard_reject": bool(_ua.get("hard_reject", False)),
                            "worst_fold": _ua.get("worst_fold"),
                            **{f"p_{k}": v for k, v in _params.items()}})
                    if _rows:
                        opt_trials_df = pd.DataFrame(_rows).sort_values(
                            "trial").reset_index(drop=True)
            except Exception as _e:
                opt_error = f"{type(_e).__name__}: {_e}"
    return opt_dir, opt_error, opt_summary, opt_trials_df


@app.cell
def _(PAL, fmt, go, missing, mo, np, opt_dir, opt_error, opt_summary,
      opt_trials_df, read_json, sel_sid, stat_tile, style_fig, top_params,
      verdict_callout):
    # ========================================================= 5 OPTIMISATION
    _out = [mo.md("## 5 · Optimisation — Optuna, walk-forward purgé")]
    _vd = read_json(opt_dir / "verdict.json") if opt_dir else None
    if opt_summary is None and opt_trials_df is None and _vd is None:
        _out.append(missing(
            f"optimisation pas encore lancée pour `{sel_sid or '—'}`",
            f"python -m quantlab.cli optimize {sel_sid or '<SID>'}"))
    else:
        if _vd:
            _out.append(verdict_callout(_vd, "optimize"))
        if opt_error:
            _out.append(mo.callout(
                mo.md(f"étude Optuna illisible : `{opt_error}`"),
                kind="warn"))

        _best = (opt_summary or {}).get("best")
        _cfg = (opt_summary or {}).get("config") or {}
        _tiles = []
        if opt_summary:
            _tiles = [
                stat_tile("Trials complétés",
                          str(opt_summary.get("n_trials_completed", "?")),
                          f"dont {opt_summary.get('n_trials_valid', '?')} "
                          "valides"),
                stat_tile("Best Sharpe concaténé",
                          fmt((_best or {}).get("sharpe_concat")),
                          f"trial #{(_best or {}).get('trial', '?')}"),
                stat_tile("Config",
                          f"{_cfg.get('k_folds', '?')} folds",
                          f"embargo {_cfg.get('embargo', '?')} barres · "
                          f"{len(_cfg.get('pairs') or [])} paires · "
                          f"{_cfg.get('tf', '?')}"),
            ]
            _out.append(mo.hstack(_tiles, gap=1, justify="start"))

        if opt_trials_df is not None:
            # finite = évaluations réelles (hard_reject = trades insuffisants,
            # le Sharpe reste informatif pour la forme de la surface)
            _fin = opt_trials_df[np.isfinite(opt_trials_df["sharpe_concat"])]
            _ok = _fin[~_fin["hard_reject"]]
            _rej = _fin[_fin["hard_reject"]]
            # ---- historique des trials + best courant
            _fig = go.Figure()
            if len(_rej):
                _fig.add_trace(go.Scatter(
                    x=_rej["trial"], y=_rej["sharpe_concat"], mode="markers",
                    name="hard-reject (trades/param)",
                    marker=dict(size=6, color="rgba(137,135,129,.45)"),
                    hovertemplate="trial %{x}<br>sharpe=%{y:.2f} "
                                  "(hard-reject)<extra></extra>"))
            _fig.add_trace(go.Scatter(
                x=_ok["trial"], y=_ok["sharpe_concat"], mode="markers",
                name="trial valide",
                marker=dict(size=8, color=PAL["blue"], opacity=0.75),
                hovertemplate="trial %{x}<br>sharpe=%{y:.2f}"
                              "<extra></extra>"))
            if len(_ok):
                _run_best = _ok.set_index("trial")["sharpe_concat"].cummax()
                _fig.add_trace(go.Scatter(
                    x=_run_best.index, y=_run_best.to_numpy(),
                    mode="lines", name="best courant", line_shape="hv",
                    line=dict(color=PAL["orange"], width=2)))
            _fig.update_xaxes(title_text="trial")
            _fig.update_yaxes(title_text="Sharpe concaténé (validation)")
            _out.append(style_fig(
                _fig, height=320,
                title=f"Historique des trials ({len(_ok)} valides, "
                      f"{len(_rej)} hard-rejects)"))

            # ---- 2 params principaux vs sharpe
            _tops = top_params(_fin, n=2)
            if _tops:
                _pfigs = []
                for _pc in _tops:
                    _pf = go.Figure(go.Scatter(
                        x=_fin[_pc], y=_fin["sharpe_concat"], mode="markers",
                        marker=dict(size=7, color=PAL["blue"], opacity=0.6),
                        customdata=_fin["hard_reject"],
                        hovertemplate=_pc[2:] + "=%{x}<br>sharpe=%{y:.2f}"
                                      "<extra></extra>"))
                    _pf.update_xaxes(title_text=_pc[2:])
                    _pf.update_yaxes(title_text="Sharpe concaténé")
                    _pfigs.append(style_fig(
                        _pf, height=280,
                        title=f"Sharpe vs {_pc[2:]} (param principal)"))
                _out.append(mo.hstack(_pfigs, widths="equal", gap=1))

        # ---- diagnostics par fold du best
        if _best and _best.get("fold_sharpes"):
            _fs = [float(s) if s is not None else float("nan")
                   for s in _best["fold_sharpes"]]
            _worst = _best.get("worst_fold", -1)
            _cols = [PAL["serious"] if _i == _worst else PAL["blue"]
                     for _i in range(len(_fs))]
            _fb = go.Figure(go.Bar(
                x=[f"fold {_i}" for _i in range(len(_fs))], y=_fs,
                marker=dict(color=_cols, cornerradius=4), width=0.55,
                customdata=_best.get("fold_trades", [None] * len(_fs)),
                hovertemplate="%{x}: sharpe=%{y:.2f} · trades=%{customdata}"
                              "<extra></extra>"))
            _fb.add_hline(y=0, line_color=PAL["axis"])
            _fb.update_yaxes(title_text="Sharpe validation")
            _out.append(style_fig(
                _fb, height=260,
                title=f"Best trial par fold — pire fold : "
                      f"{_worst if _worst is not None and _worst >= 0 else 'n/a'} "
                      f"(orange)"))
    mo.vstack(_out, gap=1)
    return


@app.cell
def _(DIV_SCALE, PAL, binned_surface, config, fmt, go, missing, mo, np,
      opt_trials_df, pd, read_json, sel_dir, sel_sid, stat_tile, style_fig,
      top_params, verdict_callout):
    # ============================================================== 6 PLATEAU
    _out = [mo.md("## 6 · Plateau — centre robuste, pas le best trial")]
    _pdir = sel_dir / "plateau" if sel_dir is not None else None
    _vd = read_json(_pdir / "verdict.json") if _pdir else None
    _sel = read_json(_pdir / "selected_params.json") if _pdir else None
    plat_perts = None
    if _pdir is not None and (_pdir / "perturbations.parquet").exists():
        try:
            plat_perts = pd.read_parquet(_pdir / "perturbations.parquet")
        except Exception as _e:
            _out.append(mo.callout(mo.md(f"perturbations.parquet illisible : "
                                         f"`{_e}`"), kind="danger"))
    if _vd is None and _sel is None and plat_perts is None:
        _out.append(missing(
            f"étape plateau pas encore lancée pour `{sel_sid or '—'}`",
            f"python -m quantlab.cli plateau {sel_sid or '<SID>'}"))
    else:
        if _vd:
            _out.append(verdict_callout(_vd, "plateau"))

        # ---- params retenus
        if _sel:
            _params = _sel.get("params", {})
            _cell = _sel.get("cell", {})
            _tiles = [stat_tile(_k, fmt(_v, 4) if isinstance(_v, float)
                                else str(_v), "param retenu")
                      for _k, _v in _params.items()]
            _tiles.append(stat_tile(
                "Sharpe centre", fmt(_sel.get("sharpe_concat")),
                f"cellule: {_cell.get('n_trials', '?')} trials · "
                f"médiane {fmt(_cell.get('median_sharpe'))} ± "
                f"{fmt(_cell.get('std_sharpe'))}"))
            _out.append(mo.hstack(_tiles, gap=1, justify="start", wrap=True))

        # ---- perturbations ±20 % (chute %) — ligne rouge à 40 %
        if plat_perts is not None and len(plat_perts):
            _pp = plat_perts.copy()
            _pp["label"] = _pp["param"] + _pp["direction"]
            _drop = np.clip(_pp["drop"].to_numpy(dtype=float) * 100.0,
                            -100.0, 200.0)
            _cols = [PAL["critical"] if _d > config.MAX_SHARPE_DROP * 100
                     else PAL["blue"] for _d in _drop]
            _fig = go.Figure(go.Bar(
                x=_pp["label"], y=_drop,
                marker=dict(color=_cols, cornerradius=4), width=0.55,
                customdata=np.stack([_pp["value"].astype(str),
                                     _pp["sharpe_concat"].round(2)], axis=1),
                hovertemplate="%{x} → %{customdata[0]}<br>chute %{y:.0f}% · "
                              "sharpe=%{customdata[1]}<extra></extra>"))
            _fig.add_hline(y=config.MAX_SHARPE_DROP * 100, line_dash="dash",
                           line_color=PAL["critical"],
                           annotation_text=f"rejet > "
                           f"{config.MAX_SHARPE_DROP:.0%}",
                           annotation_font_color=PAL["critical"])
            _fig.add_hline(y=0, line_color=PAL["axis"])
            _fig.update_yaxes(title_text="chute du Sharpe (%)")
            _out.append(style_fig(
                _fig, height=300,
                title=f"Perturbation ±{config.PERTURBATION_PCT:.0%} de chaque "
                      "param (pic étroit = rejet)"))

        # ---- surface 2D du Sharpe sur les 2 params les plus importants
        if opt_trials_df is not None:
            _fin = opt_trials_df[np.isfinite(opt_trials_df["sharpe_concat"])]
            _tops = top_params(_fin, n=2)
            if len(_tops) == 2 and len(_fin) >= 10:
                _surf = binned_surface(_fin, _tops[0], _tops[1])
                if _surf is not None:
                    _xc, _yc, _grid = _surf
                    _fig2 = go.Figure()
                    _fig2.add_trace(go.Contour(
                        x=_xc, y=_yc, z=_grid, colorscale=DIV_SCALE,
                        zmid=0.0, connectgaps=True,
                        contours=dict(showlines=True),
                        line=dict(width=0.5),
                        colorbar=dict(title="Sharpe", thickness=12),
                        hovertemplate=_tops[0][2:] + "=%{x:.4g}<br>"
                                      + _tops[1][2:] + "=%{y:.4g}<br>"
                                      "sharpe=%{z:.2f}<extra></extra>"))
                    _fig2.add_trace(go.Scatter(
                        x=_fin[_tops[0]], y=_fin[_tops[1]], mode="markers",
                        name="trials",
                        marker=dict(size=4, color="rgba(11,11,11,.35)")))
                    if _sel:
                        _px = _sel.get("params", {}).get(_tops[0][2:])
                        _py = _sel.get("params", {}).get(_tops[1][2:])
                        if _px is not None and _py is not None:
                            _fig2.add_trace(go.Scatter(
                                x=[_px], y=[_py], mode="markers",
                                name="centre retenu",
                                marker=dict(size=16, symbol="star",
                                            color=PAL["yellow"],
                                            line=dict(width=1.5,
                                                      color="#0b0b0b"))))
                    _fig2.update_xaxes(title_text=_tops[0][2:])
                    _fig2.update_yaxes(title_text=_tops[1][2:])
                    _out.append(style_fig(
                        _fig2, height=420,
                        title="Surface du Sharpe (moyenne binnée des trials) "
                              "— le centre doit être sur un plateau"))
    mo.vstack(_out, gap=1)
    return (plat_perts,)


@app.cell
def _(PAL, config, fmt, gauge, go, missing, mo, read_json, sel_dir, sel_sid,
      stat_tile, style_fig, verdict_callout):
    # ======================================================== 7 STATS
    _out = [mo.md("## 7 · Batterie statistique — PBO, DSR, permutation")]
    _sdir = sel_dir / "stats" if sel_dir is not None else None
    _vd = read_json(_sdir / "verdict.json") if _sdir else None
    _pbo = read_json(_sdir / "pbo.json") if _sdir else None
    _dsr = read_json(_sdir / "dsr.json") if _sdir else None
    _perm = read_json(_sdir / "permutation.json") if _sdir else None
    if _vd is None and _pbo is None and _dsr is None:
        _out.append(missing(
            f"batterie stats pas encore lancée pour `{sel_sid or '—'}`",
            f"python -m quantlab.cli validate {sel_sid or '<SID>'}"))
    else:
        if _vd:
            _out.append(verdict_callout(_vd, "stats"))

        _gauges = []
        if _pbo and _pbo.get("pbo") is not None:
            _gauges.append(gauge(
                _pbo["pbo"], "PBO (CSCV)",
                zones=[(0, config.PBO_PASS, PAL["good"]),
                       (config.PBO_PASS, config.PBO_REJECT, PAL["warning"]),
                       (config.PBO_REJECT, 1.0, PAL["critical"])],
                vrange=(0, 1)))
        if _dsr and _dsr.get("dsr") is not None:
            _gauges.append(gauge(
                _dsr["dsr"], "DSR — Prob(SR vrai > SR0)",
                zones=[(0, 0.5, PAL["critical"]),
                       (0.5, config.DSR_CONFIDENCE, PAL["warning"]),
                       (config.DSR_CONFIDENCE, 1.0, PAL["good"])],
                vrange=(0, 1)))
        if _gauges:
            _out.append(mo.hstack(_gauges, widths="equal", gap=1))

        if _dsr:
            _out.append(mo.hstack([
                stat_tile("N effectif", str(_dsr.get("n_effective", "?")),
                          f"dette famille {_dsr.get('research_debt', '?')} "
                          "corrigée de la corrélation inter-trials"),
                stat_tile("SR0 (E[max] sous H0)", fmt(_dsr.get("sr0"), 4),
                          f"annualisé {fmt(_dsr.get('sr0_ann'))}"),
                stat_tile("SR candidat", fmt(_dsr.get("sr_hat"), 4),
                          f"annualisé {fmt(_dsr.get('sr_hat_ann'))} · "
                          f"T={_dsr.get('t_obs', '?')}"),
            ], gap=1, justify="start"))

        # ---- histogramme des logits CSCV (PBO = fraction λ <= 0)
        if _pbo and _pbo.get("logits"):
            _fig = go.Figure(go.Histogram(
                x=_pbo["logits"], nbinsx=40,
                marker=dict(color=PAL["blue"],
                            line=dict(width=2, color="rgba(0,0,0,0)")),
                hovertemplate="λ=%{x:.2f} : %{y}<extra></extra>"))
            _fig.add_vline(x=0, line_dash="dash", line_color=PAL["critical"],
                           annotation_text="λ=0 (PBO = masse à gauche)",
                           annotation_font_color=PAL["critical"])
            _fig.update_xaxes(title_text="logit λ du rang OOS du best IS")
            _fig.update_yaxes(title_text="combinaisons CSCV")
            _out.append(style_fig(
                _fig, height=280,
                title=f"CSCV — {_pbo.get('n_combos', '?')} combinaisons, "
                      f"{_pbo.get('s_blocks', '?')} blocs, médiane λ = "
                      f"{fmt(_pbo.get('logit_median'))}"))

        # ---- permutation Monte-Carlo (opt-in)
        if _perm:
            _m = _perm.get("metrics", {})
            _real = _m.get("real_best_sharpe")
            _out.append(verdict_callout(_perm, "permutation"))
            _fig2 = go.Figure()
            # la distribution complète n'est pas persistée : mean + q95 + réel
            if _m.get("perm_best_mean") is not None:
                _fig2.add_trace(go.Scatter(
                    x=[_m["perm_best_mean"]], y=[0], mode="markers+text",
                    name="permuté (moyenne)", text=["moyenne H0"],
                    textposition="top center", textfont=dict(size=11),
                    marker=dict(size=13, color=PAL["muted"])))
            if _m.get("perm_best_q95") is not None:
                _fig2.add_trace(go.Scatter(
                    x=[_m["perm_best_q95"]], y=[0], mode="markers+text",
                    name="permuté (q95)", text=["q95 H0"],
                    textposition="top center", textfont=dict(size=11),
                    marker=dict(size=13, symbol="diamond",
                                color=PAL["orange"])))
            if _real is not None:
                _fig2.add_vline(x=_real, line_color=PAL["blue"], line_width=2)
                _fig2.add_trace(go.Scatter(
                    x=[_real], y=[0], mode="markers+text", name="Sharpe réel",
                    text=["réel"], textposition="bottom center",
                    textfont=dict(size=11),
                    marker=dict(size=16, symbol="star", color=PAL["blue"])))
            _fig2.update_yaxes(visible=False, range=[-1, 1])
            _fig2.update_xaxes(title_text="best Sharpe")
            _out.append(style_fig(
                _fig2, height=200,
                title=f"Permutation MC — p = {fmt(_m.get('p_value'), 4)} "
                      f"({_m.get('n_valid_runs', '?')}/"
                      f"{_m.get('n_runs', '?')} runs, seuil "
                      f"{config.PERMUTATION_PVALUE})"))
        else:
            _out.append(mo.callout(mo.md(
                "Permutation Monte-Carlo non lancée (opt-in) : "
                f"`python -m quantlab.cli validate {sel_sid or '<SID>'} "
                "--permutation`"), kind="neutral"))
    mo.vstack(_out, gap=1)
    return


@app.cell
def _(PAL, config, fmt, gauge, go, json, missing, mo, np, pd, read_json,
      sel_dir, sel_sid, stat_tile, style_fig, verdict_callout):
    # ==================================================== 8 DÉPLOIEMENT
    _out = [mo.md("## 8 · Règle de déploiement — FIXED / ADAPTIVE / REJECT")]
    _ddir = sel_dir / "deploy_rule" if sel_dir is not None else None
    _vd = read_json(_ddir / "verdict.json") if _ddir else None
    dep_folds = None
    _fp = (sel_dir / "optimize" / "folds_is_oos.parquet"
           if sel_dir is not None else None)
    if _fp is not None and _fp.exists():
        try:
            dep_folds = pd.read_parquet(_fp)
            if not len(dep_folds):
                dep_folds = None
        except Exception:
            dep_folds = None
    if _vd is None and dep_folds is None:
        _out.append(missing(
            f"deploy-rule pas encore lancé pour `{sel_sid or '—'}`",
            f"python -m quantlab.cli deploy-rule {sel_sid or '<SID>'}"))
    else:
        _metrics = (_vd or {}).get("metrics", {})
        if _vd:
            _out.append(verdict_callout(_vd, "deploy_rule"))

        # ---- jauge WFE (seuils 0.3 / 0.5)
        _wfe = _metrics.get("wfe")
        if _wfe is not None:
            _hi = max(1.2, float(_wfe) * 1.15)
            _out.append(mo.hstack([
                gauge(_wfe, "WFE = mean(OOS) / mean(IS)",
                      zones=[(0, config.WFE_REJECT, PAL["critical"]),
                             (config.WFE_REJECT, config.WFE_FIXED,
                              PAL["warning"]),
                             (config.WFE_FIXED, _hi, PAL["good"])],
                      vrange=(0, _hi)),
                mo.vstack([
                    stat_tile("mean Sharpe IS",
                              fmt((_metrics.get("wfe_detail") or {})
                                  .get("mean_is")), "optima par fold"),
                    stat_tile("mean Sharpe OOS",
                              fmt((_metrics.get("wfe_detail") or {})
                                  .get("mean_oos")), "mêmes optima, fenêtres "
                              "validation"),
                ], gap=0.5),
            ], widths=[2, 1], gap=1))

        # ---- scatter IS vs OOS par fold (diagonale = zéro dégradation)
        if dep_folds is not None:
            _ok = dep_folds[np.isfinite(dep_folds["sharpe_is"])
                            & np.isfinite(dep_folds["sharpe_oos"])]
            _fig = go.Figure()
            if len(_ok):
                _fig.add_trace(go.Scatter(
                    x=_ok["sharpe_is"], y=_ok["sharpe_oos"], mode="markers",
                    name="trials (top 20 %)",
                    marker=dict(size=6, color="rgba(137,135,129,.4)"),
                    customdata=_ok[["trial", "fold"]],
                    hovertemplate="trial %{customdata[0]} · fold "
                                  "%{customdata[1]}<br>IS=%{x:.2f} · "
                                  "OOS=%{y:.2f}<extra></extra>"))
                _bidx = _ok.groupby("fold")["sharpe_is"].idxmax()
                _bst = _ok.loc[_bidx]
                _fig.add_trace(go.Scatter(
                    x=_bst["sharpe_is"], y=_bst["sharpe_oos"],
                    mode="markers+text", name="optimum du fold",
                    text=[f"f{int(_f)}" for _f in _bst["fold"]],
                    textposition="top center", textfont=dict(size=11),
                    marker=dict(size=12, color=PAL["blue"],
                                line=dict(width=1.5,
                                          color="rgba(255,255,255,.8)"))))
                _lim = float(max(abs(_ok["sharpe_is"]).max(),
                                 abs(_ok["sharpe_oos"]).max(), 1.0)) * 1.1
                _fig.add_trace(go.Scatter(
                    x=[-_lim, _lim], y=[-_lim, _lim], mode="lines",
                    name="IS = OOS", line=dict(color=PAL["axis"],
                                               dash="dash", width=1)))
                _fig.add_hline(y=0, line_color=PAL["grid"])
                _fig.add_vline(x=0, line_color=PAL["grid"])
            _fig.update_xaxes(title_text="Sharpe IS (train)")
            _fig.update_yaxes(title_text="Sharpe OOS (validation)",
                              scaleanchor="x")
            _out.append(style_fig(
                _fig, height=420,
                title="IS vs OOS — sous la diagonale = surapprentissage"))

            # ---- dérive des optima entre folds (params normalisés 0-1)
            _stab = _metrics.get("stability") or {}
            _plist = _stab.get("params_by_fold")
            if not _plist and "params_json" in dep_folds.columns and len(_ok):
                _plist = [json.loads(s) for s in _bst.sort_values(
                    "fold")["params_json"]]
            if _plist and len(_plist) >= 2:
                _pmat = pd.DataFrame(_plist).select_dtypes(include=[np.number])
                _fig2 = go.Figure()
                _cat = [PAL["blue"], PAL["orange"], PAL["aqua"],
                        PAL["yellow"], PAL["magenta"], PAL["green"]]
                for _i, _cname in enumerate(_pmat.columns):
                    _col = _pmat[_cname].astype(float)
                    _span = _col.max() - _col.min()
                    _norm = ((_col - _col.min()) / _span if _span > 0
                             else _col * 0 + 0.5)
                    _fig2.add_trace(go.Scatter(
                        x=list(range(len(_norm))), y=_norm, name=_cname,
                        mode="lines+markers",
                        line=dict(color=_cat[_i % len(_cat)], width=2),
                        marker=dict(size=8),
                        customdata=_col,
                        hovertemplate=_cname + "=%{customdata:.4g} "
                                      "(fold %{x})<extra></extra>"))
                _fig2.update_xaxes(title_text="fold", dtick=1)
                _fig2.update_yaxes(title_text="valeur normalisée [0-1]",
                                   range=[-0.05, 1.05])
                _l2 = _stab.get("l2_mean")
                _out.append(style_fig(
                    _fig2, height=300,
                    title=f"Dérive des optima entre folds — L2 moyen = "
                          f"{fmt(_l2, 3)} (stable ≤ 0.5)"))

        # ---- méta-backtest : fixe vs adaptatif
        _fixed = _metrics.get("fixed")
        _adapt = _metrics.get("adaptive")
        if _fixed and _adapt:
            _labels = ["fixe"] + [f"adaptatif M={m}" for m in _adapt]
            _rets = [_fixed.get("total_return")] + \
                [_adapt[m].get("total_return") for m in _adapt]
            _shs = [_fixed.get("sharpe")] + \
                [_adapt[m].get("sharpe") for m in _adapt]
            _rets = [float(r) * 100 if r is not None and
                     np.isfinite(float(r)) else float("nan") for r in _rets]
            _cols = [PAL["blue"]] + [PAL["orange"]] * (len(_labels) - 1)
            _fig3 = go.Figure(go.Bar(
                x=_labels, y=_rets, marker=dict(color=_cols, cornerradius=4),
                width=0.5, customdata=_shs,
                hovertemplate="%{x}: %{y:.1f}% · sharpe="
                              "%{customdata:.2f}<extra></extra>"))
            _fig3.add_hline(y=0, line_color=PAL["axis"])
            _fig3.update_yaxes(title_text="return total (%)")
            _out.append(style_fig(
                _fig3, height=280,
                title="Méta-backtest — jeu fixe du plateau vs ré-optimisation "
                      f"périodique (best M={_metrics.get('best_reopt_months')})"))
    mo.vstack(_out, gap=1)
    return (dep_folds,)


@app.cell
def _(fmt, holdout_consumed, missing, mo, read_json, sel_dir, sel_family,
      sel_sid, stat_tile, verdict_callout):
    # ============================================== 9 HOLDOUT / INCUBATION
    _out = [mo.md("## 9 · Gates humains — holdout (one-shot) & incubation")]
    _hvd = read_json(sel_dir / "holdout" / "verdict.json") if sel_dir else None
    _inc = read_json(sel_dir / "incubation" / "report.json") if sel_dir \
        else None

    if _hvd is None:
        _consumed = holdout_consumed(sel_family) if sel_family else False
        if _consumed:
            _out.append(mo.callout(mo.md(
                f"⚠️ Le holdout de la famille `{sel_family}` a **déjà été "
                "consommé** (événement au ledger) mais aucun verdict n'est "
                "sur disque pour cette stratégie — vérifier l'autre version "
                "de la famille. AUCUN second passage possible."),
                kind="warn"))
        else:
            _out.append(mo.callout(mo.md(
                "🔒 **Holdout non consommé** — les derniers "
                "min(6 mois, 20 %) de chaque série restent verrouillés. "
                "UN SEUL passage par famille, irréversible, succès ou échec "
                f"compris :\n\n```\npython -m quantlab.cli holdout "
                f"{sel_sid or '<SID>'}\n```\n(exige de taper "
                f"`BURN {sel_family or '<FAMILY>'}`)"), kind="info"))
    else:
        _out.append(verdict_callout(_hvd, "holdout"))
        _m = _hvd.get("metrics", {})
        _deg = _m.get("degradation")
        _out.append(mo.hstack([
            stat_tile("Sharpe holdout", fmt(_m.get("sharpe_holdout")),
                      f"{len(_m.get('pairs') or [])} paires · "
                      f"{_m.get('tf', '?')}"),
            stat_tile("Sharpe OOS (référence)", fmt(_m.get("sharpe_oos")),
                      "best concaténé d'optimize"),
            stat_tile("Dégradation vs OOS",
                      fmt(float(_deg) * 100, 0, " %") if _deg is not None
                      else "n/a",
                      "attendu 30-50 % · effondrement > 80 %",
                      accent="#d03b3b" if (_deg or 0) > 0.8 else None),
            stat_tile("Trades", str(_m.get("n_trades", "?")),
                      f"résultat : {_hvd.get('result', '?')}"),
        ], gap=1, justify="start", wrap=True))

    # ---- incubation
    if _inc is None:
        _out.append(missing(
            "incubation : pas de rapport (étape live, après holdout)",
            "quantlab.gates.incubation_report(strategy, fills_csv=...)"))
    elif _inc.get("status") == "WAITING_LIVE_DATA":
        _out.append(mo.callout(mo.md(
            "⏳ **Incubation en attente des fills live** — "
            f"{_inc.get('instructions', '')}"), kind="neutral"))
    else:
        _out.append(verdict_callout(_inc, "incubation"))
        _im = _inc.get("metrics", {})
        _out.append(mo.hstack([
            stat_tile("Coût réalisé", fmt(_im.get("realized_cost_bps_mean"),
                                          2, " bps"),
                      f"modèle {fmt(_im.get('model_cost_bps'), 2)} bps"),
            stat_tile("Fills", str(_im.get("n_fills", "?")),
                      " · ".join(_im.get("pairs", [])[:6])),
        ], gap=1, justify="start"))
    mo.vstack(_out, gap=1)
    return


@app.cell
def _(missing, mo, sel_dir, sel_sid):
    # ================================================== 10 FICHE (REPORT.md)
    _out = [mo.md("## 10 · Fiche stratégie")]
    _fp = sel_dir / "REPORT.md" if sel_dir is not None else None
    if _fp is not None and _fp.exists():
        try:
            _out.append(mo.callout(mo.md(_fp.read_text()), kind="neutral"))
        except Exception as _e:
            _out.append(mo.callout(mo.md(f"REPORT.md illisible : `{_e}`"),
                                   kind="danger"))
    else:
        _out.append(missing(
            f"pas de REPORT.md pour `{sel_sid or '—'}`",
            f"python -m quantlab.cli report {sel_sid or '<SID>'}"))
    mo.vstack(_out, gap=1)
    return


if __name__ == "__main__":
    app.run()
