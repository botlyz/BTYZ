"""Fiche stratégie — agrégation des verdicts du pipeline -> REPORT.md.

build() relit tous les verdict.json écrits par les étapes (aucun recalcul),
y ajoute params retenus, research_debt, seuil Sharpe crédible et hash code.
render_markdown() écrit RESULTS_ROOT/<sid>/REPORT.md et retourne le markdown.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from quantlab import config

_SYMBOL = {"PASS": "✅", "FIXED": "✅", "ADAPTIVE": "✅", "COHERENT": "✅",
           "REJECT": "❌", "ERROR": "❌", "EFFONDREMENT": "❌"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sid_of(strategy) -> str:
    return strategy if isinstance(strategy, str) else strategy.strategy_id()


def _read_json(path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "oui" if v else "non"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _key_metrics(stage: str, verdict: dict) -> list[tuple[str, str]]:
    """Métriques scalaires notables d'une étape (ordre stable, max 6)."""
    prefer = {
        "contract": ["pair", "tf", "source", "n_bars", "n_prefixes"],
        "screening": ["n_cells", "n_fdr_survivors", "positive_pair_frac_native",
                      "null_beaten_frac_native", "best_sharpe", "best_cell"],
        "optimize": ["best_sharpe_concat", "n_trials_valid", "n_free_params",
                     "embargo"],
        "plateau": ["center_sharpe", "max_drop", "worst_param",
                    "n_perturbations", "cell_median_sharpe"],
        "stats": ["pbo", "pbo_verdict", "dsr", "n_effective", "research_debt"],
        "deploy_rule": ["wfe", "mode"],
        "holdout": ["sharpe_holdout", "sharpe_oos", "degradation", "n_trades"],
        "incubation": ["n_fills", "realized_cost_bps_mean", "model_cost_bps"],
    }
    metrics = verdict.get("metrics") or {}
    rows: list[tuple[str, str]] = []
    for k in prefer.get(stage, []):
        v = metrics.get(k, verdict.get(k) if k == "mode" else None)
        if v is not None and isinstance(v, (int, float, str, bool)):
            rows.append((k, _fmt(v)))
    if not rows:  # fallback: premiers scalaires trouvés
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                rows.append((k, _fmt(v)))
            if len(rows) >= 6:
                break
    return rows[:6]


def build(strategy, *, ledger=None) -> dict:
    """Fiche complète : verdicts par étape + params + dette + hash code."""
    from quantlab.ledger import family_of, ledger as default_ledger

    led = ledger if ledger is not None else default_ledger
    sid = _sid_of(strategy)
    family = family_of(sid)
    root = config.RESULTS_ROOT / sid

    stages: dict[str, dict | None] = {}
    for stage in config.STAGES:
        v = _read_json(root / stage / "verdict.json")
        if v is None and stage == "incubation":
            v = _read_json(root / stage / "report.json")
        stages[stage] = v

    sel = _read_json(root / "plateau" / "selected_params.json")
    params = sel.get("params", sel) if isinstance(sel, dict) else None

    code_hash = ""
    try:
        if isinstance(strategy, str):
            from quantlab import registry
            strategy = registry.load(sid)
        code_hash = strategy.code_hash()
    except Exception:
        pass

    return {
        "strategy_id": sid,
        "family": family,
        "research_debt": led.research_debt(family),
        "credible_sharpe": led.credible_sharpe(family),
        "holdout_available": led.holdout_available(family),
        "stages": stages,
        "permutation": _read_json(root / "stats" / "permutation.json"),
        "params": params,
        "code_hash": code_hash,
        "generated_at": _now(),
    }


def render_markdown(strategy, *, ledger=None) -> str:
    """Rend la fiche en markdown, l'écrit dans RESULTS_ROOT/<sid>/REPORT.md."""
    rep = build(strategy, ledger=ledger)
    sid = rep["strategy_id"]

    lines = [
        f"# {sid} — fiche pipeline quantlab",
        "",
        f"- **Famille** : {rep['family']}",
        f"- **Research debt (N, famille)** : {rep['research_debt']}",
        f"- **Seuil Sharpe crédible √(2·ln N)** : {rep['credible_sharpe']:.3f}",
        f"- **Hash code** : `{rep['code_hash'] or 'n/a'}`",
        f"- **Généré** : {rep['generated_at']}",
        "",
        "## Funnel",
        "",
        "| Étape | Verdict | Raison | Métriques clés |",
        "|---|---|---|---|",
    ]
    for stage in config.STAGES:
        v = rep["stages"].get(stage)
        if v is None:
            lines.append(f"| {stage} | — | | |")
            continue
        verdict = str(v.get("verdict", v.get("status", "?")))
        sym = _SYMBOL.get(verdict, "•")
        reason = str(v.get("reason", "")).replace("|", "\\|").replace("\n", " ")
        metrics = " ; ".join(f"{k}={val}" for k, val in _key_metrics(stage, v))
        lines.append(f"| {stage} | {sym} {verdict} | {reason} | {metrics} |")

    perm = rep.get("permutation")
    if perm:
        verdict = str(perm.get("verdict", "?"))
        lines += ["",
                  f"**Permutation MC (opt-in)** : {_SYMBOL.get(verdict, '•')} "
                  f"{verdict} — {perm.get('reason', '')}"]

    lines += ["", "## Paramètres production", ""]
    plateau_v = rep["stages"].get("plateau") or {}
    if rep["params"]:
        lines += ["| Paramètre | Valeur |", "|---|---|"]
        lines += [f"| {k} | {_fmt(v)} |" for k, v in rep["params"].items()]
        note = "(centre du plateau — `plateau/selected_params.json`)"
        if plateau_v.get("verdict") != "PASS":
            note += " — **plateau REJETÉ : jeu indicatif, ne pas déployer**"
        lines += ["", note]
    else:
        lines.append("_Aucun jeu retenu (plateau non passé)._")

    hold = rep["stages"].get("holdout")
    if hold is None or hold.get("verdict") != "PASS":
        avail = ("holdout encore disponible (non consommé)"
                 if rep["holdout_available"] else
                 "holdout CONSOMMÉ — famille brûlée, aucun retry possible")
        lines += ["", "## ⚠️ Avertissement",
                  "",
                  f"Le gate **holdout n'est pas passé** ({avail}). "
                  "Cette stratégie n'est PAS validée pour la production."]

    md = "\n".join(lines) + "\n"
    out_dir = config.RESULTS_ROOT / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REPORT.md").write_text(md)
    return md
