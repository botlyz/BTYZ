"""Composite scoring functions for Optuna objectives."""
from .config import MIN_TRADES


def _trade_count(metrics) -> float:
    return metrics.get("total_trades", 0) or metrics.get("trades_count", 0) or 0


def _trade_density_score(trades: float, target: int = MIN_TRADES) -> float:
    """Reward sufficient observations, plateau above 3× target."""
    if trades <= 0:
        return -8.0
    target = max(float(target), 1.0)
    if trades < target:
        return -2.0 * (1.0 - trades / target)
    if trades <= target * 3:
        return min(0.75, 0.25 + 0.25 * ((trades - target) / target))
    return 0.75


def score_default(metrics) -> float:
    """Default composite: sharpe-dominant + PF + return - DD penalty - WR penalty."""
    if metrics is None:
        return -10.0

    sh = metrics.get("sharpe_ratio", 0) or 0
    dd = abs(metrics.get("max_drawdown_pct", 0) or 0)
    tr = _trade_count(metrics)
    pf = metrics.get("profit_factor", 0) or metrics.get("trades_profit_factor", 0) or 0
    ret = metrics.get("total_return_pct", 0) or 0
    wr = (metrics.get("win_rate_pct", 0) or 0) / 100.0

    if tr <= 0:
        return -8.0
    if dd > 50:
        return -3.0 + sh * 0.1

    sharpe_score = min(max(sh, 0), 7) * 2.0
    pf_score = min(max(pf - 1.0, 0), 2.0) * 0.4
    ret_score = min(max(ret, 0), 40.0) * 0.01
    dd_penalty = max(dd - 20.0, 0) * 0.05
    wr_penalty = max(0.50 - wr, 0) * 1.0
    trade_score = _trade_density_score(tr)

    return sharpe_score + pf_score + ret_score + trade_score - dd_penalty - wr_penalty


def score_high_frequency(metrics) -> float:
    """For HF strategies (target 500–1500 trades)."""
    if metrics is None:
        return -10.0

    sh = metrics.get("sharpe_ratio", 0) or 0
    dd = abs(metrics.get("max_drawdown_pct", 0) or 0)
    tr = _trade_count(metrics)
    pf = metrics.get("profit_factor", 0) or metrics.get("trades_profit_factor", 0) or 0
    ret = metrics.get("total_return_pct", 0) or 0

    if tr < 200:
        return -5.0 + sh * 0.1
    if dd > 50:
        return -3.0 + sh * 0.1
    if ret < 0 and sh > 0:
        return sh - 1

    TRADE_TARGET_MIN, TRADE_TARGET_MAX = 500, 1500
    if TRADE_TARGET_MIN <= tr <= TRADE_TARGET_MAX:
        trade_b = 0.5
    elif tr < TRADE_TARGET_MIN:
        trade_b = min(tr / TRADE_TARGET_MIN, 1.0) * 0.3
    else:
        trade_b = max(0, 1 - (tr - TRADE_TARGET_MAX) / 2000) * 0.3

    dd_pen = max(0, dd - 20) / 30.0 * 0.5
    pf_bonus = min(max((pf - 1) / 0.3, 0), 1.0) * 0.3
    return sh + trade_b - dd_pen + pf_bonus


def score_trend(metrics) -> float:
    """For trend-following (favors Calmar + annualized return)."""
    if metrics is None:
        return -10.0
    calmar = metrics.get("calmar_ratio", 0) or 0
    ann_ret = metrics.get("annualized_return", 0) or 0
    dd = abs(metrics.get("max_drawdown_pct", 0) or 0)
    tr = _trade_count(metrics)
    if tr < MIN_TRADES:
        return -5.0 + calmar * 0.1
    ret_score = min(max(ann_ret, 0), 1.0) * 3.0
    dd_penalty = max(dd - 25.0, 0) * 0.05
    return calmar * 2.0 + ret_score - dd_penalty


SCORING_REGISTRY = {
    "default": score_default,
    "high_frequency": score_high_frequency,
    "trend": score_trend,
}
