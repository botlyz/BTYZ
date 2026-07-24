"""Étape 5 — règle de déploiement : FIXED | ADAPTIVE | REJECT.

Trois diagnostics depuis les sorties d'optimize/plateau :
1. WFE = mean(Sharpe OOS) / mean(Sharpe IS) des optima par fold
   (optimize/folds_is_oos.parquet). WFE < WFE_REJECT -> REJECT (la
   ré-optimisation ne sauve PAS ce cas). Si mean IS <= 0, WFE non défini
   -> REJECT motivé.
2. Stabilité des optima : distance L2 des params normalisés (bornes de
   l'espace Optuna) entre folds successifs + Spearman des rangs.
3. Méta-backtest ADAPTATIF : "tous les M mois, ré-opti mini-budget sur les
   T=3 derniers mois puis application le mois suivant" sur le dev set,
   equity concaténée vs jeu fixe du plateau — celui qui gagne net de frais.

Verdict fichier deploy_rule/verdict.json ; au ledger le mode FIXED/ADAPTIVE
est enregistré comme PASS (prérequis du gate holdout).
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantlab import config

REOPT_TRAIN_MONTHS = 3                       # T mois de train par ré-opti
MINI_BUDGET = max(8, config.DEFAULT_TRIALS // 30)   # trials par ré-opti
L2_STABLE_MAX = 0.5                          # distance L2 normalisée moyenne


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _freq_per_year(tf: str) -> float:
    minutes = pd.Timedelta(config.FREQ_MAP[tf]).total_seconds() / 60.0
    return 365.0 * 24.0 * 60.0 / minutes


def _ann_sharpe(r: pd.Series, tf: str) -> float:
    """Sharpe annualisé d'une série de returns CONCATÉNÉE (méta-backtest —
    pas un portefeuille vbt, d'où le calcul direct)."""
    r = r.dropna()
    if len(r) < 3 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * math.sqrt(_freq_per_year(tf)))


# ------------------------------------------------------------------ WFE

def _pstart(period) -> "pd.Timestamp":
    """Début de période en tz-aware UTC (les index de données sont UTC)."""
    return period.start_time.tz_localize("UTC")


def _pend(period) -> "pd.Timestamp":
    return period.end_time.tz_localize("UTC")

def _load_folds(sid: str) -> pd.DataFrame:
    path = config.RESULTS_ROOT / sid / "optimize" / "folds_is_oos.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} manquant — lancer optimize avant deploy-rule")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"{path} vide (aucun trial valide à l'optimize)")
    return df


def _best_per_fold(folds: pd.DataFrame) -> pd.DataFrame:
    """Par fold : la ligne au meilleur Sharpe IS (l'optimum in-sample)."""
    ok = folds[np.isfinite(folds["sharpe_is"])]
    if ok.empty:
        return ok
    idx = ok.groupby("fold")["sharpe_is"].idxmax()
    return ok.loc[idx].sort_values("fold")


def _wfe(folds: pd.DataFrame) -> dict:
    best = _best_per_fold(folds)
    if best.empty:
        return {"wfe": None, "reason": "aucun Sharpe IS fini"}
    mean_is = float(best["sharpe_is"].mean())
    oos = best["sharpe_oos"][np.isfinite(best["sharpe_oos"])]
    mean_oos = float(oos.mean()) if len(oos) else float("nan")
    if mean_is <= 0 or not np.isfinite(mean_oos):
        return {"wfe": None, "mean_is": mean_is, "mean_oos": mean_oos,
                "reason": f"WFE non défini (mean IS = {mean_is:.3f})"}
    return {"wfe": mean_oos / mean_is, "mean_is": mean_is,
            "mean_oos": mean_oos, "n_folds": int(len(best))}


# ------------------------------------------------------------------ stabilité
def _space_bounds(strategy) -> dict:
    """Bornes de l'espace Optuna {param: (low, high)} via un trial sondé."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(seed=0))
        trial = study.ask()
        strategy.param_space(trial)
        bounds = {}
        for name, dist in trial.distributions.items():
            low = getattr(dist, "low", None)
            high = getattr(dist, "high", None)
            if low is not None and high is not None:
                bounds[name] = (float(low), float(high))
        return bounds
    except Exception:
        return {}


def _stability(folds: pd.DataFrame, strategy) -> dict:
    """Dérive des optima entre folds successifs (params normalisés)."""
    from scipy.stats import spearmanr

    best = _best_per_fold(folds)
    if best.empty or "params_json" not in best.columns:
        return {"l2_mean": None, "spearman_mean": None,
                "reason": "params par fold indisponibles"}
    plist = [json.loads(s) for s in best["params_json"]]
    pmat = pd.DataFrame(plist)
    num = pmat.select_dtypes(include=[np.number])
    if num.empty or len(num) < 2:
        return {"l2_mean": None, "spearman_mean": None,
                "params_by_fold": plist,
                "reason": "moins de 2 folds ou aucun param numérique"}
    bounds = _space_bounds(strategy) if strategy is not None else {}
    norm = pd.DataFrame(index=num.index)
    for cname in num.columns:
        col = num[cname].astype(float)
        lo, hi = bounds.get(cname, (col.min(), col.max()))
        span = hi - lo
        norm[cname] = (col - lo) / span if span > 0 else 0.0
    diffs = norm.diff().dropna()
    l2 = np.sqrt((diffs ** 2).sum(axis=1) / norm.shape[1])
    spearmans = []
    if norm.shape[1] >= 2:
        vals = num.to_numpy()
        for i in range(1, len(vals)):
            rho = spearmanr(vals[i - 1], vals[i]).statistic
            if np.isfinite(rho):
                spearmans.append(float(rho))
    return {
        "l2_mean": float(l2.mean()),
        "l2_max": float(l2.max()),
        "spearman_mean": float(np.mean(spearmans)) if spearmans else None,
        "params_by_fold": plist,
        "bounds_source": "param_space" if bounds else "observed",
        "stable": bool(l2.mean() <= L2_STABLE_MAX),
    }


# ------------------------------------------------------------------ méta-backtest
def _slice_warm(df: pd.DataFrame, start, end, warmup: int) -> pd.DataFrame:
    """Fenêtre [start, end) précédée de `warmup` barres de contexte."""
    upto = df[df.index < end]
    i0 = int(upto.index.searchsorted(start))
    return upto.iloc[max(i0 - warmup, 0):]


def _eval_returns(strategy, datas: dict, params: dict, *, tf: str,
                  fees: float, start, end) -> pd.Series:
    """Returns du backtest poolé, restreints à [start, end)."""
    from quantlab.backtest import pooled_backtest, portfolio_returns

    warm = int(strategy.WARMUP_BARS)
    sliced = {}
    for p, df in datas.items():
        sl = _slice_warm(df, start, end, warm)
        if len(sl) > warm:
            sliced[p] = sl
    if not sliced:
        return pd.Series(dtype=float)
    pf = pooled_backtest(strategy, sliced, params, fees=fees, tf=tf)
    r = portfolio_returns(pf)
    return r[(r.index >= start) & (r.index < end)]


def _random_search(strategy, datas: dict, *, tf: str, fees: float,
                   start, end, budget: int, seed: int) -> tuple[dict | None, int]:
    """Mini random-search sur [start, end) -> (best params, n backtests)."""
    import optuna

    from quantlab.backtest import pooled_backtest, portfolio_returns

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warm = int(strategy.WARMUP_BARS)
    sliced = {}
    for p, df in datas.items():
        sl = _slice_warm(df, start, end, warm)
        if len(sl) > warm:
            sliced[p] = sl
    if not sliced:
        return None, 0
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=seed), direction="maximize")
    best_val, best_params, n_bt = float("-inf"), None, 0
    for _ in range(budget):
        trial = study.ask()
        try:
            params = strategy.param_space(trial)
            pf = pooled_backtest(strategy, sliced, params, fees=fees, tf=tf)
            r = portfolio_returns(pf)
            r = r[r.index >= start]
            val = _ann_sharpe(r, tf)
            n_bt += 1
        except Exception:
            val, params = float("nan"), None
        val = val if np.isfinite(val) else -1e9
        study.tell(trial, val)
        if params is not None and val > best_val:
            best_val, best_params = val, params
    return best_params, n_bt


def _adaptive_curve(strategy, datas: dict, *, tf: str, fees: float,
                    months: pd.PeriodIndex, reopt_m: int, fixed_params: dict,
                    budget: int, seed: int, task=None) -> tuple[pd.Series, int]:
    """Ré-opti tous les `reopt_m` mois sur T=3 mois, application ensuite."""
    parts, n_bt = [], 0
    k = REOPT_TRAIN_MONTHS
    step = 0
    while k < len(months):
        train_start = _pstart(months[k - REOPT_TRAIN_MONTHS])
        train_end = _pstart(months[k])
        apply_end_idx = min(k + reopt_m, len(months))
        apply_start = train_end
        apply_end = (_pend(months[apply_end_idx - 1])
                     if apply_end_idx == len(months)
                     else _pstart(months[apply_end_idx]))
        params, nb = _random_search(
            strategy, datas, tf=tf, fees=fees, start=train_start,
            end=train_end, budget=budget, seed=seed + 1000 * step + k)
        n_bt += nb
        if params is None:
            params = fixed_params      # pas de ré-opti possible -> on garde
        r = _eval_returns(strategy, datas, params, tf=tf, fees=fees,
                          start=apply_start, end=apply_end)
        n_bt += 1
        if len(r):
            parts.append(r)
        if task is not None:
            task.advance(1, M=reopt_m)
        k += reopt_m
        step += 1
    curve = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    return curve, n_bt


def _n_windows(n_months: int, reopt_m: int) -> int:
    return max(0, math.ceil((n_months - REOPT_TRAIN_MONTHS) / reopt_m))


# ------------------------------------------------------------------ run
def run(strategy, *, reopt_months=(1, 2, 3), progress=None, ledger=None,
        adaptive: bool = True, mini_budget: int = MINI_BUDGET) -> dict:
    """Étape 5 complète. `adaptive=False` saute le méta-backtest (tests)."""
    from quantlab import registry
    from quantlab.ledger import family_of, ledger as _default_ledger
    from quantlab.progress import PipelineProgress

    led = ledger if ledger is not None else _default_ledger
    if isinstance(strategy, str):
        strategy = registry.load(strategy)
    sid = strategy.strategy_id()
    family = family_of(sid)
    out_dir = config.RESULTS_ROOT / sid / "deploy_rule"
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = _load_folds(sid)
    wfe_res = _wfe(folds)
    stab = _stability(folds, strategy)

    run_id = led.start_run(strategy, "deploy_rule",
                           {"reopt_months": list(reopt_months)})

    def _finish(mode: str, reason: str, metrics: dict, n_tests: int) -> dict:
        out = {"stage": "deploy_rule", "verdict": mode, "reason": reason,
               "metrics": metrics, "n_tests": n_tests, "at": _now()}
        (out_dir / "verdict.json").write_text(json.dumps(out, indent=2))
        ledger_verdict = "REJECT" if mode == "REJECT" else "PASS"
        led.record_verdict(strategy, "deploy_rule",
                           {**out, "verdict": ledger_verdict, "mode": mode})
        led.finish_run(run_id, ledger_verdict, n_tests, {"mode": mode})
        if n_tests:
            led.record_tests(family, n_tests, "deploy_rule",
                             {"reopt_months": list(reopt_months)})
        return out

    base_metrics = {"wfe": wfe_res.get("wfe"),
                    "wfe_detail": wfe_res, "stability": stab}

    # ---------------------------------------------------------- gardes WFE
    if wfe_res.get("wfe") is None:
        return _finish("REJECT", wfe_res.get("reason", "WFE non défini"),
                       base_metrics, 0)
    wfe = float(wfe_res["wfe"])
    if wfe < config.WFE_REJECT:
        return _finish(
            "REJECT",
            f"WFE={wfe:.3f} < {config.WFE_REJECT} — ne généralise pas "
            "(la ré-optimisation ne sauve pas ce cas)", base_metrics, 0)

    # ---------------------------------------------------------- jeu fixe
    sel_path = config.RESULTS_ROOT / sid / "plateau" / "selected_params.json"
    if not sel_path.exists():
        raise FileNotFoundError(f"{sel_path} manquant — lancer plateau d'abord")
    selected = json.loads(sel_path.read_text())
    fixed_params = selected.get("params", selected)

    if not adaptive:
        return _finish("FIXED",
                       f"WFE={wfe:.3f}, stabilité L2={stab.get('l2_mean')} "
                       "(méta-backtest adaptatif sauté)", base_metrics, 0)

    # ---------------------------------------------------------- méta-backtest
    opt_summary = json.loads(
        (config.RESULTS_ROOT / sid / "optimize" / "summary.json").read_text())
    cfg = opt_summary.get("config") or {}
    tf = cfg.get("tf") or strategy.TF
    source = cfg.get("source") or strategy.DATA_SOURCE
    fees = float(cfg.get("fees_bps", config.DEFAULT_FEES_BPS)) / 1e4
    pairs = cfg.get("pairs") or []

    from quantlab.data import splits
    datas = {}
    for p in pairs:
        df = splits.load_dev(source, p, tf)
        if df is not None and len(df) >= config.MIN_BARS:
            datas[p] = df
    if not datas:
        raise ValueError(f"aucune donnée dev pour le méta-backtest ({pairs})")

    t0 = min(df.index[0] for df in datas.values())
    t1 = max(df.index[-1] for df in datas.values())
    months = pd.period_range(t0, t1, freq="M")
    if len(months) <= REOPT_TRAIN_MONTHS + 1:
        return _finish("FIXED",
                       f"dev set trop court ({len(months)} mois) pour le "
                       f"méta-backtest — WFE={wfe:.3f}, jeu fixe par défaut",
                       base_metrics, 0)
    eval_start = _pstart(months[REOPT_TRAIN_MONTHS])

    own = progress is None
    if own:
        progress = PipelineProgress(sid, "deploy_rule", ledger=led)
        progress.__enter__()
    n_tests = 0
    try:
        total_windows = sum(_n_windows(len(months), m) for m in reopt_months)
        task = progress.task("méta-backtest adaptatif", total=total_windows)

        r_fixed = _eval_returns(strategy, datas, fixed_params, tf=tf,
                                fees=fees, start=eval_start, end=t1)
        n_tests += 1
        fixed_stats = {"total_return": float((1 + r_fixed).prod() - 1)
                       if len(r_fixed) else float("nan"),
                       "sharpe": _ann_sharpe(r_fixed, tf)}

        adaptive_stats = {}
        for m in reopt_months:
            curve, nb = _adaptive_curve(
                strategy, datas, tf=tf, fees=fees, months=months,
                reopt_m=int(m), fixed_params=fixed_params,
                budget=int(mini_budget), seed=42 + int(m), task=task)
            n_tests += nb
            adaptive_stats[int(m)] = {
                "total_return": float((1 + curve).prod() - 1)
                if len(curve) else float("nan"),
                "sharpe": _ann_sharpe(curve, tf),
                "n_bars": int(len(curve)),
            }
        task.done()
    finally:
        if own:
            progress.__exit__(None, None, None)

    valid = {m: s for m, s in adaptive_stats.items()
             if np.isfinite(s["total_return"])}
    best_m = (max(valid, key=lambda m: valid[m]["total_return"])
              if valid else None)
    best_adapt = valid.get(best_m, {"total_return": float("-inf"),
                                    "sharpe": float("nan")})
    stable = bool(stab.get("stable"))
    fixed_tr = fixed_stats["total_return"]

    adaptive_wins = (best_m is not None
                     and np.isfinite(best_adapt["total_return"])
                     and best_adapt["total_return"] > (
                         fixed_tr if np.isfinite(fixed_tr) else float("-inf"))
                     and best_adapt["total_return"] > 0)
    if adaptive_wins and (not stable or wfe < config.WFE_FIXED):
        mode = "ADAPTIVE"
        reason = (f"WFE={wfe:.3f}, optima {'stables' if stable else 'dérivants'}, "
                  f"ré-opti M={best_m} bat le jeu fixe net de frais "
                  f"({best_adapt['total_return']:+.2%} vs {fixed_tr:+.2%})")
    else:
        mode = "FIXED"
        reason = (f"WFE={wfe:.3f} ({'>=':s} {config.WFE_FIXED} requis: "
                  f"{wfe >= config.WFE_FIXED}), L2={stab.get('l2_mean')}, "
                  f"jeu fixe {fixed_tr:+.2%} vs adaptatif "
                  f"{best_adapt['total_return']:+.2%} (M={best_m})")

    metrics = {**base_metrics,
               "fixed": fixed_stats,
               "adaptive": adaptive_stats,
               "best_reopt_months": best_m,
               "eval_start": str(eval_start),
               "mini_budget": int(mini_budget),
               "pairs": sorted(datas), "tf": tf, "fees_bps": fees * 1e4}
    return _finish(mode, reason, metrics, n_tests)
