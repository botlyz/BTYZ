"""Étape 2 — Optuna avec walk-forward purgé DANS la fonction objectif.

K folds temporels sur l'index commun du dev set. L'objectif d'un trial =
backtest POOLÉ multi-paires sur chaque fenêtre de validation (warmup fourni,
returns comptés uniquement dans la fenêtre), returns par barre concaténés sur
les K folds -> Sharpe annualisé de la série concaténée. Purge : embargo E
barres entre fin de train et début de validation. Le train ne sert qu'au
diagnostic IS (WFE, étape 5), recalculé après l'étude pour le top 20 %.

Parallélisme : les trials tournent multi-process (pattern Optuna standard —
N workers font chacun study.optimize(n) sur la même storage sqlite), via
quantlab.parallel.pool_map. Le backtest pooled vbt reste vectorisé 1 process.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantlab import config, parallel
from quantlab.backtest import pooled_backtest, portfolio_returns
from quantlab.data import splits, store
from quantlab.ledger import ledger as default_ledger
from quantlab.progress import PipelineProgress

HARD_REJECT = -10.0
_N_PERIODS = config.PBO_BLOCKS * 2   # périodes pour trial_returns (PBO)
_IS_TOP_FRAC = 0.20                  # fraction des trials re-runs IS

# clés de config sérialisables passées aux workers -> make_evaluator
_CFG_KEYS = ("pairs", "tf", "source", "fees_bps", "k_folds", "embargo",
             "seed", "anchored", "min_trades_per_param")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quiet_optuna():
    """Logs WARNING + warnings expérimentaux (multivariate...) silencieux."""
    import warnings

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore",
                            category=optuna.exceptions.ExperimentalWarning)


# ---------------------------------------------------------------- données
def load_dev_data(source: str, pair: str, tf: str) -> pd.DataFrame | None:
    """Dev set d'une paire. store pour lighter/binance ; engine.data_loader
    pour les sources spéciales (oi, gapfill...), tronquées au cutoff dev.

    Garde-fou sources spéciales : cutoff = min entre celui de la série chargée
    et celui de la série lighter du store (le holdout partagé de la paire ne
    doit jamais être visible, même via une source dérivée).
    """
    if source in ("lighter", "binance"):
        return splits.load_dev(source, pair, tf)
    from engine.data_loader import load_ohlcv
    df = load_ohlcv(pair, tf, source=source)
    if df is None or not len(df):
        return None
    cut = splits.dev_holdout_cut(df.index)
    try:
        base = store.load("lighter", pair, tf)
        if base is not None and len(base):
            cut = min(cut, splits.dev_holdout_cut(base.index))
    except Exception:
        pass
    dev = df[df.index < cut]
    return dev if len(dev) else None


def screening_survivors(strategy) -> list[str]:
    """Paires FDR-survivantes du screening (tf native, source native)."""
    sid = strategy.strategy_id()
    fp = config.RESULTS_ROOT / sid / "screening" / "cells.parquet"
    if not fp.exists():
        raise FileNotFoundError(
            f"{fp} absent : lance le screening d'abord ou passe pairs= explicitement")
    cells = pd.read_parquet(fp)
    mask = cells["qval"] < config.FDR_Q
    if "tf" in cells.columns:
        mask &= cells["tf"] == strategy.TF
    if "source" in cells.columns:
        mask &= cells["source"] == strategy.DATA_SOURCE
    pairs = sorted(cells.loc[mask, "pair"].unique().tolist())
    if not pairs:
        raise ValueError(f"aucune paire FDR-survivante dans {fp} "
                         f"(tf={strategy.TF}, source={strategy.DATA_SOURCE})")
    return pairs


# ---------------------------------------------------------------- folds
@dataclass
class Fold:
    """Fenêtres d'un fold (timestamps sur l'index commun, [start, end))."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp      # = valid_start - E barres (purge)
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    warm_start: pd.Timestamp     # valid_start - WARMUP - E (contexte signaux)
    train_warm_start: pd.Timestamp


def build_folds(index: pd.DatetimeIndex, k: int, embargo: int, warmup: int,
                *, anchored: bool = config.DEFAULT_ANCHORED) -> list[Fold]:
    """K folds glissants : index coupé en K+1 segments égaux ; fold i valide
    sur le segment i+1, train = segment(s) précédent(s) moins l'embargo E."""
    n = len(index)
    seg = n // (k + 1)
    if seg <= embargo + 10:
        raise ValueError(f"dev set trop court: {n} barres pour k={k} folds "
                         f"avec embargo={embargo}")
    step = index[1] - index[0]
    folds = []
    for i in range(k):
        v0 = (i + 1) * seg
        v1 = (i + 2) * seg if i < k - 1 else n
        t0 = 0 if anchored else i * seg
        t1 = max(v0 - embargo, t0 + 1)
        folds.append(Fold(
            train_start=index[t0], train_end=index[t1 - 1] + step,
            valid_start=index[v0], valid_end=index[v1 - 1] + step,
            warm_start=index[max(v0 - warmup - embargo, 0)],
            train_warm_start=index[max(t0 - warmup, 0)]))
    return folds


def estimate_embargo(strategy, sample: pd.DataFrame, seed: int = 0) -> int:
    """E auto = WARMUP_BARS + td_stop max estimé (3 trials random du
    param_space) ; 2 × WARMUP_BARS si aucun td_stop observable."""
    import optuna
    _quiet_optuna()
    warm = int(strategy.WARMUP_BARS)
    tds: list[int] = []
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    for _ in range(3):
        trial = study.ask()
        try:
            params = strategy.param_space(trial)
            sig = strategy.signals(sample, strategy.full_params(params))
            if sig.td_stop is not None:
                tds.append(int(sig.td_stop))
        except Exception:
            continue
    return warm + max(tds) if tds else 2 * warm


# ---------------------------------------------------------------- objectif WF
class WFEvaluator:
    """Évalue un jeu de params sur les K folds (validation poolée multi-paires).

    `evaluate(params)` -> {sharpe_concat, fold_sharpes, fold_trades,
    worst_fold, hard_reject, returns}. `evaluate_train(params)` -> Sharpe IS
    par fold (diagnostic WFE, fenêtres train)."""

    def __init__(self, strategy, datas: dict[str, pd.DataFrame], folds: list[Fold],
                 *, tf: str, fees: float, min_trades_per_param: int, n_free: int):
        self.strategy = strategy
        self.datas = datas
        self.folds = folds
        self.tf = tf
        self.fees = fees
        self.freq = config.FREQ_MAP[tf]
        self.min_trades = int(min_trades_per_param) * max(int(n_free), 1)

    def _window_backtest(self, params: dict, warm_start, start, end):
        """Backtest poolé sur [warm_start, end) ; returns/trades sur [start, end)."""
        datas = {}
        for p, df in self.datas.items():
            sl = df[(df.index >= warm_start) & (df.index < end)]
            if len(sl):
                datas[p] = sl
        if not datas:
            return None, 0
        pf = pooled_backtest(self.strategy, datas, params,
                             fees=self.fees, tf=self.tf)
        r = portfolio_returns(pf)
        r = r[(r.index >= start) & (r.index < end)]
        rec = pf.trades.records_arr
        n_trades = 0
        if len(rec):
            ts = pf.wrapper.index[rec["entry_idx"]]
            n_trades = int(((ts >= start) & (ts < end)).sum())
        return r, n_trades

    def _sharpe(self, r: pd.Series | None) -> float:
        if r is None or len(r) < 2:
            return float("nan")
        return float(r.vbt.returns(freq=self.freq).sharpe_ratio())

    def evaluate(self, params: dict) -> dict:
        fold_sharpes, fold_trades, chunks = [], [], []
        for f in self.folds:
            r, n = self._window_backtest(params, f.warm_start,
                                         f.valid_start, f.valid_end)
            fold_sharpes.append(self._sharpe(r))
            fold_trades.append(n)
            if r is not None:
                chunks.append(r)
        concat = pd.concat(chunks) if chunks else pd.Series(dtype=float)
        sharpe_concat = self._sharpe(concat)
        hard = (min(fold_trades) < self.min_trades if fold_trades else True) \
            or not math.isfinite(sharpe_concat)
        worst = int(np.nanargmin(fold_sharpes)) if fold_sharpes and \
            not all(math.isnan(s) for s in fold_sharpes) else -1
        return {"sharpe_concat": sharpe_concat, "fold_sharpes": fold_sharpes,
                "fold_trades": fold_trades, "worst_fold": worst,
                "hard_reject": bool(hard), "returns": concat}

    def evaluate_train(self, params: dict) -> dict:
        sharpes, trades = [], []
        for f in self.folds:
            r, n = self._window_backtest(params, f.train_warm_start,
                                         f.train_start, f.train_end)
            sharpes.append(self._sharpe(r))
            trades.append(n)
        return {"fold_sharpes": sharpes, "fold_trades": trades}


def _resolve(strategy, pairs, tf, source, fees_bps, embargo, k_folds, seed,
             overrides) -> dict:
    """Résout la config effective (défauts + overrides) + charge les données."""
    allowed = {"anchored", "min_trades_per_param"}
    unknown = set(overrides) - allowed
    if unknown:
        raise TypeError(f"overrides inconnus: {sorted(unknown)} "
                        f"(choix: {sorted(allowed)})")
    tf = tf or strategy.TF
    source = source or strategy.DATA_SOURCE
    fees_bps = config.DEFAULT_FEES_BPS if fees_bps is None else float(fees_bps)
    if pairs is None:
        pairs = screening_survivors(strategy)
    datas = {}
    for p in pairs:
        df = load_dev_data(source, p, tf)
        if df is not None and len(df) >= config.MIN_BARS:
            datas[p] = df
    if not datas:
        raise ValueError(f"aucune donnée dev pour {pairs} ({source}/{tf})")
    common = None
    for df in datas.values():
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) < config.MIN_BARS:
        raise ValueError("index commun trop court entre les paires")
    if embargo is None:
        sample = next(iter(datas.values())).iloc[:max(2 * strategy.WARMUP_BARS, 600)]
        embargo = estimate_embargo(strategy, sample, seed=seed)
    anchored = bool(overrides.get("anchored", config.DEFAULT_ANCHORED))
    folds = build_folds(common, int(k_folds), int(embargo),
                        int(strategy.WARMUP_BARS), anchored=anchored)
    return {"pairs": sorted(datas), "tf": tf, "source": source,
            "fees_bps": fees_bps, "embargo": int(embargo),
            "k_folds": int(k_folds), "seed": int(seed), "anchored": anchored,
            "min_trades_per_param": int(overrides.get(
                "min_trades_per_param", config.MIN_TRADES_PER_PARAM)),
            "datas": datas, "folds": folds, "common_index": common}


def make_evaluator(strategy, *, pairs=None, tf=None, source=None, fees_bps=None,
                   k_folds=config.DEFAULT_K_FOLDS, embargo=None, seed=42,
                   **overrides) -> tuple[WFEvaluator, dict]:
    """Construit l'évaluateur WF (partagé avec plateau.py + workers).
    Retourne (evaluator, cfg) — cfg sérialisable (sans données/folds)."""
    cfg = _resolve(strategy, pairs, tf, source, fees_bps, embargo,
                   k_folds, seed, overrides)
    ev = WFEvaluator(strategy, cfg["datas"], cfg["folds"], tf=cfg["tf"],
                     fees=cfg["fees_bps"] / 1e4,
                     min_trades_per_param=cfg["min_trades_per_param"],
                     n_free=strategy.n_free_params())
    slim = {k: cfg[k] for k in _CFG_KEYS}
    return ev, slim


def make_objective(strategy, **kwargs):
    """callable(params: dict) -> float (Sharpe concaténé, HARD_REJECT si rejet).
    Réutilisé par plateau.py pour les perturbations."""
    ev, _ = make_evaluator(strategy, **kwargs)

    def objective(params: dict) -> float:
        res = ev.evaluate(params)
        return HARD_REJECT if res["hard_reject"] else res["sharpe_concat"]

    objective.evaluator = ev
    return objective


# ---------------------------------------------------------------- workers
_EV_CACHE: dict = {}


def _cfg_evaluator(sid: str, cfg: dict) -> WFEvaluator:
    """Évaluateur (re)construit côté worker, mis en cache par process."""
    key = (sid, json.dumps(cfg, sort_keys=True))
    ev = _EV_CACHE.get(key)
    if ev is None:
        _quiet_optuna()
        from quantlab import registry
        strategy = registry.load(sid)
        kw = dict(cfg)
        overrides = {"anchored": kw.pop("anchored"),
                     "min_trades_per_param": kw.pop("min_trades_per_param")}
        ev, _ = make_evaluator(strategy, **kw, **overrides)
        _EV_CACHE.clear()          # une seule config vivante par worker (RAM)
        _EV_CACHE[key] = ev
    return ev


def _make_trial_objective(ev: WFEvaluator, multi_objective: bool):
    def objective(trial):
        params = ev.strategy.param_space(trial)
        res = ev.evaluate(params)
        trial.set_user_attr("params", params)
        sc = res["sharpe_concat"]
        trial.set_user_attr("sharpe_concat", sc if math.isfinite(sc) else None)
        trial.set_user_attr("fold_sharpes", [s if math.isfinite(s) else None
                                             for s in res["fold_sharpes"]])
        trial.set_user_attr("fold_trades", res["fold_trades"])
        trial.set_user_attr("worst_fold", res["worst_fold"])
        trial.set_user_attr("hard_reject", res["hard_reject"])
        trial.set_user_attr("period_returns", _period_returns(res["returns"]))
        if res["hard_reject"]:
            return (HARD_REJECT, abs(HARD_REJECT)) if multi_objective \
                else HARD_REJECT
        if multi_objective:
            finite = [s for s in res["fold_sharpes"] if math.isfinite(s)]
            return sc, (float(np.std(finite)) if finite else abs(HARD_REJECT))
        return sc
    return objective


def _rdb_storage(url: str):
    import optuna
    return optuna.storages.RDBStorage(
        url, engine_kwargs={"connect_args": {"timeout": 120}})


def _optuna_batch_worker(payload: dict) -> dict:
    """Un lot de trials sur la storage sqlite partagée (pattern Optuna
    multi-process standard). Sampler re-seedé par lot."""
    import optuna
    _quiet_optuna()
    ev = _cfg_evaluator(payload["sid"], payload["cfg"])
    mo = payload["multi_objective"]
    if mo:
        sampler = optuna.samplers.NSGAIISampler(seed=payload["seed"])
    else:
        sampler = optuna.samplers.TPESampler(
            multivariate=True, seed=payload["seed"], constant_liar=True,
            warn_independent_sampling=False)
    study = optuna.load_study(study_name=payload["study_name"],
                              storage=_rdb_storage(payload["storage"]),
                              sampler=sampler)
    study.optimize(_make_trial_objective(ev, mo), n_trials=payload["n_trials"],
                   n_jobs=1, catch=(Exception,), gc_after_trial=True)
    if mo:
        vals = [t.values[0] for t in study.get_trials(deepcopy=False)
                if t.values is not None]
    else:
        vals = [t.value for t in study.get_trials(deepcopy=False)
                if t.value is not None]
    gc.collect()
    return {"n": payload["n_trials"], "best": max(vals) if vals else None}


def _is_rerun_worker(payload: dict) -> dict:
    """Re-run IS (fenêtres train) d'un trial du top 20 % — pour le WFE."""
    ev = _cfg_evaluator(payload["sid"], payload["cfg"])
    res = ev.evaluate_train(payload["params"])
    gc.collect()
    return {"trial": payload["trial"], **res}


def eval_params_worker(payload: dict) -> dict:
    """Évalue un jeu de params sur l'objectif WF complet (utilisé par
    plateau.py via pool_map). payload: {sid, cfg, params, tag}."""
    ev = _cfg_evaluator(payload["sid"], payload["cfg"])
    res = ev.evaluate(payload["params"])
    res.pop("returns", None)
    gc.collect()
    return {"tag": payload.get("tag"), "params": payload["params"], **res}


class _BatchAdvance:
    """Proxy TaskHandle : chaque lot terminé avance de sa taille de lot."""

    def __init__(self, task, sizes: list[int]):
        self._task = task
        self._sizes = list(sizes)

    def advance(self, n: int = 1, **postfix):
        size = self._sizes.pop(0) if self._sizes else 1
        self._task.advance(size, **postfix)


def _space_hash(strategy) -> str:
    """Hash de l'espace de paramètres (distributions d'un trial random figé)."""
    import optuna
    _quiet_optuna()
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    try:
        strategy.param_space(trial)
    except Exception:
        pass
    frozen = study.tell(trial, 0.0)
    txt = json.dumps({k: repr(v) for k, v in
                      sorted(frozen.distributions.items())})
    return hashlib.sha256(txt.encode()).hexdigest()[:16]


def _period_returns(r: pd.Series, n_periods: int = _N_PERIODS) -> list[float]:
    """Return composé de chacune des n_periods tranches égales de la série."""
    if r is None or not len(r):
        return [float("nan")] * n_periods
    out = []
    for chunk in np.array_split(r.to_numpy(dtype=float), n_periods):
        out.append(float(np.prod(1.0 + chunk) - 1.0) if len(chunk)
                   else float("nan"))
    return out


# ---------------------------------------------------------------- run
def run(strategy, *, pairs=None, tf=None, source=None, fees_bps=None,
        k_folds=config.DEFAULT_K_FOLDS, trials=config.DEFAULT_TRIALS,
        embargo=None, seed=42, multi_objective=False, progress=None,
        ledger=None, workers=None, **overrides) -> dict:
    """Étape 2 complète. Retourne le verdict dict (écrit + ledger)."""
    import optuna
    _quiet_optuna()

    ledger = ledger or default_ledger
    sid = strategy.strategy_id()
    out_dir = config.RESULTS_ROOT / sid / "optimize"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- refus AVANT run : trop de degrés de liberté
    n_free = strategy.n_free_params()
    if n_free > config.MAX_FREE_PARAMS:
        verdict = {"stage": "optimize", "verdict": "REJECT",
                   "reason": f"{n_free} paramètres libres > "
                             f"MAX_FREE_PARAMS={config.MAX_FREE_PARAMS}",
                   "metrics": {"n_free_params": n_free}, "n_tests": 0,
                   "at": _now()}
        (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
        ledger.record_verdict(strategy, "optimize", verdict)
        return verdict

    # cfg résolue une fois dans le parent (embargo auto figé -> workers idem)
    _, cfg = make_evaluator(strategy, pairs=pairs, tf=tf, source=source,
                            fees_bps=fees_bps, k_folds=k_folds,
                            embargo=embargo, seed=seed, **overrides)
    k = cfg["k_folds"]

    # ---- budget loggé AVANT le run
    run_id = ledger.start_run(strategy, "optimize", meta={
        "trials": int(trials), "k_folds": k, "seed": seed,
        "space_hash": _space_hash(strategy), "embargo": cfg["embargo"],
        "pairs": cfg["pairs"], "tf": cfg["tf"], "source": cfg["source"],
        "fees_bps": cfg["fees_bps"], "multi_objective": bool(multi_objective)})

    storage_url = f"sqlite:///{out_dir / 'optuna.db'}"
    study_name = f"run{run_id}"
    if multi_objective:
        optuna.create_study(study_name=study_name,
                            storage=_rdb_storage(storage_url),
                            directions=["maximize", "minimize"])
    else:
        optuna.create_study(study_name=study_name,
                            storage=_rdb_storage(storage_url),
                            direction="maximize")

    # ---- lots de trials multi-process (workers Optuna sur storage partagée)
    n_workers = workers if workers is not None else config.N_WORKERS
    n_workers = max(1, min(n_workers, int(trials)))
    n_batches = min(int(trials), n_workers * 3)   # lots courts -> TPE informé
    sizes = [len(c) for c in parallel.chunked(list(range(int(trials))),
                                              n_batches)]
    payloads = [{"sid": sid, "cfg": cfg, "storage": storage_url,
                 "study_name": study_name, "n_trials": s,
                 "seed": seed + 1000 * (i + 1),
                 "multi_objective": bool(multi_objective)}
                for i, s in enumerate(sizes)]

    with ExitStack() as stack:
        if progress is None:
            progress = stack.enter_context(
                PipelineProgress(sid, "optimize", ledger=ledger))
        task = progress.task("trials", total=int(trials))
        parallel.pool_map(_optuna_batch_worker, payloads, workers=n_workers,
                          ordered=False,
                          progress_handle=_BatchAdvance(task, sizes))
        task.done()

        study = optuna.load_study(study_name=study_name,
                                  storage=_rdb_storage(storage_url))
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]

        # ---- trial_returns.parquet (trials × périodes, pour le PBO)
        rows = {t.number: t.user_attrs.get(
                    "period_returns", [float("nan")] * _N_PERIODS)
                for t in completed}
        tr = pd.DataFrame.from_dict(
            rows, orient="index", columns=[f"p{i}" for i in range(_N_PERIODS)])
        tr.index.name = "trial"
        tr.sort_index().to_parquet(out_dir / "trial_returns.parquet")

        # ---- re-runs IS (top 20 %) pour folds_is_oos.parquet (WFE étape 5)
        done = [t for t in completed
                if not t.user_attrs.get("hard_reject", True)
                and t.user_attrs.get("sharpe_concat") is not None]
        done.sort(key=lambda t: t.user_attrs["sharpe_concat"], reverse=True)
        top = done[:max(1, math.ceil(_IS_TOP_FRAC * len(done)))] if done else []
        is_rows = []
        if top:
            task_is = progress.task("IS re-runs (WFE)", total=len(top))
            is_payloads = [{"sid": sid, "cfg": cfg, "trial": t.number,
                            "params": t.user_attrs["params"]} for t in top]
            is_results = parallel.pool_map(
                _is_rerun_worker, is_payloads, workers=n_workers,
                ordered=True, progress_handle=task_is)
            task_is.done()
            by_num = {t.number: t for t in top}
            for res in is_results:
                t = by_num[res["trial"]]
                params = t.user_attrs["params"]
                for i in range(k):
                    oos = t.user_attrs["fold_sharpes"][i]
                    is_rows.append({
                        "trial": t.number, "fold": i,
                        "params_json": json.dumps(params),
                        "sharpe_is": res["fold_sharpes"][i],
                        "sharpe_oos": oos if oos is not None else float("nan"),
                        "n_trades_is": res["fold_trades"][i],
                        "n_trades": t.user_attrs["fold_trades"][i]})
        pd.DataFrame(is_rows).to_parquet(out_dir / "folds_is_oos.parquet")

    n_tests = int(trials) * k + len(top) * k
    ledger.record_tests(sid, n_tests, "optimize",
                        meta={"trials": int(trials), "k_folds": k,
                              "is_reruns": len(top)})

    # ---- verdict + summary
    if not done:
        verdict_str = "REJECT"
        reason = ("aucun trial valide (hard-reject trades/param sur tous les "
                  "folds ou échec backtest)")
        best = None
    else:
        best = done[0]
        sc = best.user_attrs["sharpe_concat"]
        if sc <= 0:
            verdict_str, reason = "REJECT", \
                f"meilleur Sharpe concaténé <= 0 ({sc:.3f})"
        else:
            verdict_str, reason = "PASS", (
                f"Sharpe concaténé {sc:.3f}, trades/param >= "
                f"{cfg['min_trades_per_param']} sur les {k} folds")

    summary = {
        "study_name": study_name, "storage": storage_url, "run_id": run_id,
        "config": {**cfg, "trials": int(trials),
                   "multi_objective": bool(multi_objective),
                   "n_free_params": n_free,
                   "warmup": int(strategy.WARMUP_BARS)},
        "n_trials_completed": len(completed),
        "n_trials_valid": len(done),
        "best": None if best is None else {
            "trial": best.number, "params": best.user_attrs["params"],
            "sharpe_concat": best.user_attrs["sharpe_concat"],
            "fold_sharpes": best.user_attrs["fold_sharpes"],
            "fold_trades": best.user_attrs["fold_trades"],
            "worst_fold": best.user_attrs["worst_fold"]},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    verdict = {"stage": "optimize", "verdict": verdict_str, "reason": reason,
               "metrics": {
                   "best_sharpe_concat": (best.user_attrs["sharpe_concat"]
                                          if best else None),
                   "n_trials_valid": len(done), "n_free_params": n_free,
                   "embargo": cfg["embargo"], "pairs": cfg["pairs"]},
               "n_tests": n_tests, "at": _now()}
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    ledger.record_verdict(strategy, "optimize", verdict)
    ledger.finish_run(run_id, verdict_str, n_tests)
    return verdict
