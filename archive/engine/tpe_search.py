"""Optuna TPE per-fold with aggressive memory cleanup."""
import gc

import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

from .metrics import extract_from_portfolio


def _trade_count(metrics) -> float:
    return metrics.get("total_trades", 0) or metrics.get("trades_count", 0) or 0


def _absolute_min_trades(target_trades: int) -> int:
    """Hard floor below which a trial is auto-rejected."""
    return max(3, min(5, int(target_trades) // 4))


def _trade_density_adjustment(trades: float, target_trades: int) -> float:
    """Penalty below target, capped bonus above."""
    target = max(float(target_trades), 1.0)
    if trades < target:
        return -3.0 * (1.0 - trades / target)
    return min(0.75, 0.25 * ((trades - target) / target))


def run_tpe_fold(
    train_data,
    test_data,
    param_space_fn,
    run_backtest_fn,
    score_fn,
    trials: int = 500,
    min_trades_per_fold: int = 30,
    seed: int = 42,
    fold_idx: int = 0,
    n_jobs: int = 1,
):
    """Run TPE on one fold and return best params + train/test metrics.

    Returns:
        dict {fold, params, train_metrics, test_metrics} or None on failure.
    """

    def objective(trial):
        params = param_space_fn(trial)
        trial.set_user_attr("full_params", params)
        try:
            result = run_backtest_fn(train_data, params)
            if result is None:
                return -10.0
            metrics = extract_from_portfolio(result) if hasattr(result, "stats") else result
            tr = _trade_count(metrics)
            abs_min = _absolute_min_trades(min_trades_per_fold)
            if tr < abs_min:
                return -10.0 + (tr / abs_min)
            return score_fn(metrics) + _trade_density_adjustment(tr, min_trades_per_fold)
        except Exception:
            return -10.0

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(
            seed=seed + fold_idx,
            multivariate=True,
            warn_independent_sampling=False,
        ),
    )
    study.optimize(
        objective,
        n_trials=trials,
        show_progress_bar=False,
        catch=(Exception,),
        gc_after_trial=True,
        n_jobs=n_jobs,
    )

    completed = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
    if not completed:
        del study, completed
        gc.collect()
        return None

    best = max(completed, key=lambda t: t.value)
    best_params = best.user_attrs.get("full_params", best.params)

    try:
        train_result = run_backtest_fn(train_data, best_params)
        test_result = run_backtest_fn(test_data, best_params)
    except Exception:
        del study, completed, best
        gc.collect()
        return None

    if train_result is None or test_result is None:
        del study, completed, best, train_result, test_result
        gc.collect()
        return None

    train_metrics = extract_from_portfolio(train_result) if hasattr(train_result, "stats") else train_result
    test_metrics = extract_from_portfolio(test_result) if hasattr(test_result, "stats") else test_result

    train_trades = _trade_count(train_metrics)
    if train_trades < _absolute_min_trades(min_trades_per_fold):
        del study, completed, best, train_result, test_result, train_metrics, test_metrics
        gc.collect()
        return None

    result = {
        "fold": fold_idx,
        "params": best_params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    del study, completed, best, train_result, test_result, train_metrics, test_metrics
    gc.collect()
    return result
