"""Étape 3 — centre du plateau + perturbation (jamais le best trial).

Depuis l'étude Optuna de l'étape 2 : les trials sont regroupés en régions
(binning par paramètre, 5 quantiles pour les numériques), la meilleure cellule
est celle de médiane haute + faible dispersion, et les params retenus sont les
MÉDIANES des trials de cette cellule. Chaque param est ensuite perturbé de
±PERTURBATION_PCT et ré-évalué sur l'objectif walk-forward complet
(optimize.eval_params_worker via pool_map) : une chute relative de Sharpe
> MAX_SHARPE_DROP signe un pic étroit -> REJECT.
"""
from __future__ import annotations

import json
import math
from contextlib import ExitStack
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantlab import config, parallel
from quantlab import optimize as opt
from quantlab.ledger import ledger as default_ledger
from quantlab.progress import PipelineProgress

_MIN_CELL_TRIALS = 3
_N_QUANTILE_BINS = 5
_ENSEMBLE_SIZE = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- étude
def _load_study_and_summary(sid: str):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fp = config.RESULTS_ROOT / sid / "optimize" / "summary.json"
    if not fp.exists():
        raise FileNotFoundError(f"{fp} absent : lance optimize d'abord")
    summary = json.loads(fp.read_text())
    study = optuna.load_study(study_name=summary["study_name"],
                              storage=opt._rdb_storage(summary["storage"]))
    return study, summary


def _valid_trials(study) -> list:
    import optuna
    return [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and not t.user_attrs.get("hard_reject", True)
            and t.user_attrs.get("sharpe_concat") is not None]


# ---------------------------------------------------------------- régions
def _bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.quantile(values, qs))


def _numeric(dist) -> bool:
    from optuna.distributions import FloatDistribution, IntDistribution
    return isinstance(dist, (FloatDistribution, IntDistribution))


def find_plateau(trials: list, distributions: dict) -> dict:
    """Cellule = tuple de bins (quantiles par param numérique, 5 max, adapté
    au nombre de trials ; valeur pour les catégoriels). Score = médiane des
    sharpe_concat (cellules < 3 trials ignorées) ; départage par dispersion
    basse. Centre = params médians de la meilleure cellule."""
    names = sorted(distributions)
    n_bins = max(2, min(_N_QUANTILE_BINS, int(len(trials) ** 0.5 / 1.5)))
    edges = {}
    for name in names:
        if _numeric(distributions[name]):
            vals = np.array([float(t.user_attrs["params"][name]) for t in trials])
            edges[name] = _bin_edges(vals, n_bins)

    cells: dict[tuple, list] = {}
    for t in trials:
        key = []
        for name in names:
            v = t.user_attrs["params"][name]
            if name in edges:
                key.append(int(np.searchsorted(edges[name], float(v),
                                               side="right")))
            else:
                key.append(v)
        cells.setdefault(tuple(key), []).append(t)

    scored = []
    for key, ts in cells.items():
        if len(ts) < _MIN_CELL_TRIALS:
            continue
        sharpes = np.array([t.user_attrs["sharpe_concat"] for t in ts])
        scored.append((float(np.median(sharpes)), -float(np.std(sharpes)),
                       key, ts))
    if not scored:
        return {}
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    med, neg_std, key, ts = scored[0]

    center = {}
    for name in names:
        vals = [t.user_attrs["params"][name] for t in ts]
        if _numeric(distributions[name]):
            center[name] = _snap(float(np.median([float(v) for v in vals])),
                                 distributions[name])
        else:  # catégoriel : valeur modale de la cellule
            center[name] = max(set(vals), key=vals.count)
    return {"center": center, "cell_key": key, "cell_trials": ts,
            "cell_median_sharpe": med, "cell_std_sharpe": -neg_std,
            "n_cells_scored": len(scored)}


# ---------------------------------------------------------------- perturbations
def _snap(v: float, dist):
    """Projette v sur la grille du param_space (step + bornes)."""
    from optuna.distributions import FloatDistribution, IntDistribution
    if isinstance(dist, IntDistribution):
        step = dist.step or 1
        x = round((v - dist.low) / step) * step + dist.low
        return int(min(max(x, dist.low), dist.high))
    if isinstance(dist, FloatDistribution):
        x = float(v)
        if dist.step:
            x = round((x - dist.low) / dist.step) * dist.step + dist.low
            x = round(x, 10)
        return float(min(max(x, dist.low), dist.high))
    return v


def perturbations_of(center: dict, distributions: dict,
                     pct: float = config.PERTURBATION_PCT) -> list[dict]:
    """[{param, direction, value}] — numériques ±pct (snap grille, éloigné
    d'au moins 1 pas), catégoriels : valeurs voisines dans la liste."""
    from optuna.distributions import CategoricalDistribution
    out = []
    for name, dist in sorted(distributions.items()):
        v = center[name]
        if isinstance(dist, CategoricalDistribution):
            choices = list(dist.choices)
            i = choices.index(v)
            for direction, j in (("-", i - 1), ("+", i + 1)):
                if 0 <= j < len(choices):
                    out.append({"param": name, "direction": direction,
                                "value": choices[j]})
            continue
        if not _numeric(dist):
            continue
        step = dist.step or 0
        for sign, direction in ((1, "+"), (-1, "-")):
            x = _snap(float(v) * (1 + sign * pct), dist)
            if x == v and step:   # éloigné d'au moins 1 pas
                x = _snap(float(v) + sign * step, dist)
            if x != v:
                out.append({"param": name, "direction": direction, "value": x})
    return out


# ---------------------------------------------------------------- run
def run(strategy, *, top_frac: float = 0.2, progress=None, ledger=None,
        workers=None) -> dict:
    """Étape 3 complète. Retourne le verdict dict (écrit + ledger)."""
    ledger = ledger or default_ledger
    sid = strategy.strategy_id()
    out_dir = config.RESULTS_ROOT / sid / "plateau"
    out_dir.mkdir(parents=True, exist_ok=True)

    study, summary = _load_study_and_summary(sid)
    cfg = {k: summary["config"][k] for k in opt._CFG_KEYS}
    k_folds = int(cfg["k_folds"])

    trials = _valid_trials(study)
    if not trials:
        verdict = {"stage": "plateau", "verdict": "REJECT",
                   "reason": "aucun trial valide dans l'étude optimize",
                   "metrics": {}, "n_tests": 0, "at": _now()}
        (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
        ledger.record_verdict(strategy, "plateau", verdict)
        return verdict

    # top_frac : régions cherchées dans les meilleurs trials uniquement
    trials.sort(key=lambda t: t.user_attrs["sharpe_concat"], reverse=True)
    top = trials[:max(_MIN_CELL_TRIALS, math.ceil(top_frac * len(trials)))]
    distributions = dict(top[0].distributions)

    plat = find_plateau(top, distributions)
    if not plat:   # cellules trop clairsemées -> régions sur tous les trials
        plat = find_plateau(trials, distributions)
    if not plat:   # dernier recours : le top entier = une seule cellule
        sharpes = np.array([t.user_attrs["sharpe_concat"] for t in top])
        center = {}
        for name, dist in distributions.items():
            vals = [t.user_attrs["params"][name] for t in top]
            center[name] = _snap(float(np.median([float(v) for v in vals])),
                                 dist) if _numeric(dist) \
                else max(set(vals), key=vals.count)
        plat = {"center": center, "cell_key": ("top_frac",),
                "cell_trials": top,
                "cell_median_sharpe": float(np.median(sharpes)),
                "cell_std_sharpe": float(np.std(sharpes)),
                "n_cells_scored": 0}

    center = plat["center"]
    perts = perturbations_of(center, distributions)

    run_id = ledger.start_run(strategy, "plateau", meta={
        "seed": cfg["seed"], "center": center,
        "n_perturbations": len(perts), "k_folds": k_folds,
        "cell_median_sharpe": plat["cell_median_sharpe"]})

    # ---- ré-évaluations WF (centre + perturbations) via pool_map
    payloads = [{"sid": sid, "cfg": cfg, "params": center, "tag": "center"}]
    for p in perts:
        params = {**center, p["param"]: p["value"]}
        payloads.append({"sid": sid, "cfg": cfg, "params": params,
                         "tag": f"{p['param']}{p['direction']}"})

    with ExitStack() as stack:
        if progress is None:
            progress = stack.enter_context(
                PipelineProgress(sid, "plateau", ledger=ledger))
        task = progress.task("perturbations", total=len(payloads))
        results = parallel.pool_map(
            opt.eval_params_worker, payloads,
            workers=min(workers or config.N_WORKERS, len(payloads)),
            ordered=True, progress_handle=task)
        task.done()

    n_tests = len(payloads) * k_folds
    ledger.record_tests(sid, n_tests, "plateau",
                        meta={"n_perturbations": len(perts)})

    base = results[0]
    base_sharpe = base["sharpe_concat"] if not base["hard_reject"] \
        else opt.HARD_REJECT

    rows, worst = [], None
    for p, res in zip(perts, results[1:]):
        s = res["sharpe_concat"] if not res["hard_reject"] else opt.HARD_REJECT
        drop = ((base_sharpe - s) / abs(base_sharpe)
                if base_sharpe not in (0.0,) and math.isfinite(base_sharpe)
                else float("inf"))
        rows.append({"param": p["param"], "direction": p["direction"],
                     "value": p["value"], "params_json": json.dumps(
                         {**center, p["param"]: p["value"]}),
                     "sharpe_concat": s, "drop": drop,
                     "hard_reject": res["hard_reject"],
                     "min_fold_trades": (min(res["fold_trades"])
                                         if res["fold_trades"] else 0)})
        if worst is None or drop > worst[1]:
            worst = (p["param"], drop)
    pert_df = pd.DataFrame(rows)
    pert_df.to_parquet(out_dir / "perturbations.parquet")

    # ---- verdict
    if base["hard_reject"] or not math.isfinite(base_sharpe) or base_sharpe <= 0:
        verdict_str = "REJECT"
        reason = (f"centre du plateau invalide (sharpe={base_sharpe:.3f}, "
                  f"hard_reject={base['hard_reject']})")
    else:
        bad = pert_df[pert_df["drop"] > config.MAX_SHARPE_DROP]
        if len(bad):
            p0 = bad.sort_values("drop", ascending=False).iloc[0]
            verdict_str = "REJECT"
            reason = (f"pic étroit: {p0['param']} (chute "
                      f"{p0['drop'] * 100:.0f}% > "
                      f"{config.MAX_SHARPE_DROP * 100:.0f}%)")
        else:
            verdict_str = "PASS"
            reason = (f"plateau stable: chute max "
                      f"{(worst[1] * 100 if worst else 0):.0f}% "
                      f"({worst[0] if worst else '-'}) sous ±"
                      f"{config.PERTURBATION_PCT * 100:.0f}%")

    # ---- selected_params.json (+ ensemble : 3 jeux tirés de la cellule)
    rng = np.random.default_rng(int(cfg["seed"]))
    cell_trials = plat["cell_trials"]
    picks = rng.choice(len(cell_trials),
                       size=min(_ENSEMBLE_SIZE, len(cell_trials)),
                       replace=False)
    selected = {
        "params": center,
        "sharpe_concat": base_sharpe,
        "fold_sharpes": base["fold_sharpes"],
        "fold_trades": base["fold_trades"],
        "ensemble": [cell_trials[int(i)].user_attrs["params"] for i in picks],
        "cell": {"key": [str(x) for x in plat["cell_key"]],
                 "n_trials": len(cell_trials),
                 "median_sharpe": plat["cell_median_sharpe"],
                 "std_sharpe": plat["cell_std_sharpe"]},
        "at": _now(),
    }
    (out_dir / "selected_params.json").write_text(json.dumps(selected, indent=2))

    verdict = {"stage": "plateau", "verdict": verdict_str, "reason": reason,
               "metrics": {
                   "center_sharpe": base_sharpe,
                   "max_drop": worst[1] if worst else None,
                   "worst_param": worst[0] if worst else None,
                   "n_perturbations": len(perts),
                   "cell_median_sharpe": plat["cell_median_sharpe"],
                   "cell_n_trials": len(cell_trials)},
               "n_tests": n_tests, "at": _now()}
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    ledger.record_verdict(strategy, "plateau", verdict)
    ledger.finish_run(run_id, verdict_str, n_tests)
    return verdict
