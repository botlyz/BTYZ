"""Test de permutation Monte-Carlo (opt-in — le plus coûteux de la batterie).

H0 : l'edge du candidat est un artefact de la recherche. Pour chaque run on
block-permute les RETURNS du dev set de chaque paire (l'autocorrélation
intra-bloc est préservée, la structure exploitable est détruite), on
reconstruit un OHLC synthétique cohérent, puis on relance une
mini-optimisation walk-forward (budget_frac du budget réel) sur ces données
permutées. p-value = position du best Sharpe réel dans la distribution des
best Sharpe permutés.

Fan-out via quantlab.parallel.pool_map (config.N_WORKERS, recyclage
max_tasks_per_child anti-fuite RAM). quantlab.optimize est importé
PARESSEUSEMENT (dans les fonctions) — jamais au top du module.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantlab import config


# ------------------------------------------------------------------ permutation
def block_permute_positions(n: int, block_bars: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Positions des barres après permutation des blocs (ordre intra-bloc
    conservé)."""
    n_blocks = max(1, math.ceil(n / block_bars))
    order = rng.permutation(n_blocks)
    return np.concatenate([np.arange(b * block_bars, min((b + 1) * block_bars, n))
                           for b in order])


def block_permute_ohlc(df: pd.DataFrame, block_bars: int,
                       rng: np.random.Generator) -> pd.DataFrame:
    """OHLCV synthétique : blocs de returns close-close permutés, close
    reconstruit par cumprod, open/high/low reconstruits en préservant les
    ratios o/h/l vs close de la barre d'origine du bloc. Index conservé."""
    c = df["close"].to_numpy(dtype="float64")
    r = np.zeros(len(c))
    r[1:] = c[1:] / c[:-1] - 1.0
    pos = block_permute_positions(len(c), block_bars, rng)
    new_close = c[0] * np.cumprod(1.0 + r[pos])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_o = df["open"].to_numpy(dtype="float64") / c
        ratio_h = df["high"].to_numpy(dtype="float64") / c
        ratio_l = df["low"].to_numpy(dtype="float64") / c
    out = pd.DataFrame({
        "open": ratio_o[pos] * new_close,
        "high": ratio_h[pos] * new_close,
        "low": ratio_l[pos] * new_close,
        "close": new_close,
    }, index=df.index)
    if "volume" in df.columns:
        out["volume"] = df["volume"].to_numpy()[pos]
    return out


# ------------------------------------------------------------------ mini-opti
def _best_sharpe_permuted(strategy, datas: dict, cfg: dict, *,
                          seed: int, n_trials: int) -> float:
    """Random-search réduit sur données permutées -> best Sharpe.

    Chemin nominal : WFEvaluator d'optimize (mêmes folds/purge/hard-reject
    que la vraie optimisation, sur les données PERMUTÉES). Fallback autonome
    si optimize indisponible : Sharpe du backtest poolé complet.
    NB : optimize.make_objective charge lui-même le dev set réel — il ne peut
    pas consommer des données permutées, d'où l'usage direct de WFEvaluator.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    evaluate = None
    try:  # import paresseux — optimize écrit par un autre agent
        from quantlab.optimize import WFEvaluator, build_folds

        common = None
        for df in datas.values():
            common = df.index if common is None else common.intersection(df.index)
        folds = build_folds(common, int(cfg["k_folds"]), int(cfg["embargo"]),
                            int(strategy.WARMUP_BARS),
                            anchored=bool(cfg.get("anchored", False)))
        ev = WFEvaluator(
            strategy, datas, folds, tf=cfg["tf"], fees=cfg["fees_bps"] / 1e4,
            min_trades_per_param=int(cfg.get("min_trades_per_param",
                                             config.MIN_TRADES_PER_PARAM)),
            n_free=strategy.n_free_params())

        def evaluate(params: dict) -> float:
            res = ev.evaluate(params)
            return float("-inf") if res["hard_reject"] \
                else float(res["sharpe_concat"])
    except Exception:
        from engine.metrics import extract_from_portfolio
        from quantlab.backtest import pooled_backtest

        def evaluate(params: dict) -> float:
            pf = pooled_backtest(strategy, datas, params,
                                 fees=cfg["fees_bps"] / 1e4, tf=cfg["tf"])
            sr = extract_from_portfolio(pf).get("sharpe_ratio")
            return float(sr) if sr is not None and np.isfinite(sr) \
                else float("-inf")

    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=seed), direction="maximize")
    best = float("-inf")
    for _ in range(n_trials):
        trial = study.ask()
        try:
            val = float(evaluate(strategy.param_space(trial)))
        except Exception:
            val = float("-inf")
        study.tell(trial, val if np.isfinite(val) else -1e9)
        best = max(best, val)
    return best


def _one_run(args: tuple) -> float:
    """Worker pool_map : un run de permutation complet -> best Sharpe permuté."""
    sid, cfg, seed_i, n_trials = args
    from quantlab import registry
    from quantlab.data import splits

    strategy = registry.load(sid)
    rng = np.random.default_rng(seed_i)
    datas = {}
    for p in cfg["pairs"]:
        df = splits.load_dev(cfg["source"], p, cfg["tf"])
        if df is not None and len(df) > strategy.WARMUP_BARS:
            datas[p] = block_permute_ohlc(df, int(cfg["block_bars"]), rng)
    if not datas:
        return float("-inf")
    return _best_sharpe_permuted(strategy, datas, cfg,
                                 seed=seed_i, n_trials=n_trials)


# ------------------------------------------------------------------ lecture état
def _optimize_state(sid: str) -> tuple[dict, float]:
    """(config effective d'optimize, best Sharpe réel) depuis summary.json."""
    path = config.RESULTS_ROOT / sid / "optimize" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} manquant — lancer optimize avant la permutation")
    summary = json.loads(path.read_text())
    cfg = dict(summary.get("config") or {})
    best = summary.get("best") or {}
    real = best.get("sharpe_concat", summary.get("best_sharpe"))
    if real is None or not np.isfinite(float(real)):
        raise ValueError(
            f"pas de best Sharpe valide dans {path} (optimize REJECT ?)")
    for key, default in (("tf", None), ("source", None), ("pairs", None),
                         ("fees_bps", config.DEFAULT_FEES_BPS),
                         ("k_folds", config.DEFAULT_K_FOLDS),
                         ("embargo", None)):
        cfg.setdefault(key, default)
    return cfg, float(real)


# ------------------------------------------------------------------ run
def run(strategy, *, n_runs: int = config.PERMUTATION_RUNS,
        budget_frac: float = 0.1, seed: int = 42, progress=None,
        ledger=None) -> dict:
    """Test de permutation complet. Écrit stats/permutation.json.

    `n_runs` réduit (ex. 50) accepté : la p-value plancher 1/(n_runs+1) et le
    verdict en tiennent compte (même estimateur sans biais).
    """
    from quantlab import parallel, registry
    from quantlab.ledger import family_of, ledger as _default_ledger
    from quantlab.progress import PipelineProgress

    led = ledger if ledger is not None else _default_ledger
    if isinstance(strategy, str):
        strategy = registry.load(strategy)
    sid = strategy.strategy_id()
    family = family_of(sid)

    cfg, real_best = _optimize_state(sid)
    cfg["tf"] = cfg["tf"] or strategy.TF
    cfg["source"] = cfg["source"] or strategy.DATA_SOURCE
    if not cfg["pairs"]:
        raise ValueError("paires retenues absentes d'optimize/summary.json")
    if cfg["embargo"] is None:
        cfg["embargo"] = 2 * int(strategy.WARMUP_BARS)
    cfg["block_bars"] = config.PERMUTATION_BLOCK_BARS
    n_runs = int(n_runs)
    n_trials = max(1, int(config.DEFAULT_TRIALS * budget_frac))
    k_folds = int(cfg["k_folds"])

    run_id = led.start_run(strategy, "stats", {
        "seed": seed, "test": "permutation", "n_runs": n_runs,
        "n_trials": n_trials, "pairs": cfg["pairs"]})

    own = progress is None
    if own:
        progress = PipelineProgress(sid, "stats:permutation", ledger=led)
        progress.__enter__()
    try:
        task = progress.task("permutations", total=n_runs)
        items = [(sid, cfg, seed + i, n_trials) for i in range(n_runs)]
        results = parallel.pool_map(_one_run, items, progress_handle=task,
                                    ordered=False)
        task.done()
    finally:
        if own:
            progress.__exit__(None, None, None)

    perm_best = np.asarray([r for r in results if np.isfinite(r)])
    n_ok = len(perm_best)
    p_value = (1.0 + float((perm_best >= real_best).sum())) / (n_runs + 1.0)
    verdict = "PASS" if p_value < config.PERMUTATION_PVALUE else "REJECT"

    n_tests = n_runs * n_trials * k_folds
    led.record_tests(family, n_tests, "stats",
                     {"test": "permutation", "n_runs": n_runs})

    out = {
        "stage": "stats",
        "test": "permutation",
        "verdict": verdict,
        "reason": (f"p={p_value:.4f} vs seuil {config.PERMUTATION_PVALUE} "
                   f"({n_ok}/{n_runs} runs valides)"),
        "metrics": {
            "p_value": p_value,
            "real_best_sharpe": real_best,
            "perm_best_mean": float(perm_best.mean()) if n_ok else None,
            "perm_best_q95": (float(np.quantile(perm_best, 0.95))
                              if n_ok else None),
            "n_runs": n_runs, "n_valid_runs": n_ok,
            "n_trials_per_run": n_trials, "budget_frac": budget_frac,
            "block_bars": config.PERMUTATION_BLOCK_BARS,
            "pairs": cfg["pairs"], "tf": cfg["tf"], "source": cfg["source"],
            "seed": seed,
        },
        "n_tests": n_tests,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = config.RESULTS_ROOT / sid / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "permutation.json").write_text(json.dumps(out, indent=2))
    led.finish_run(run_id, verdict, n_tests, {"p_value": p_value})
    return out
