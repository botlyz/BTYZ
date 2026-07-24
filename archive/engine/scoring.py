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


def score_robust(metrics) -> float:
    """Robust composite — version 2 (2026-05-16) optimisée pour la régularité.

    Changements vs v1:
      - HARD reject `tr == 0` à -15 (vs -8) → force TPE à éviter no-trade
      - HARD reject `tr < 5` à -8 → folds 1-4 trades = bruit pur, inutilisables
      - Ajout Sortino (smoothness downside : pertes contrôlées vs gains)
      - Ajout Calmar (return/DD efficacité capital)
      - Sharpe gardé cap 5 ×1.5 (poids principal)
      - Pénalités tail / DD-dur / worst trade gardées
    """
    if metrics is None:
        return -10.0

    sh      = metrics.get("sharpe_ratio", 0) or 0
    sortino = metrics.get("sortino_ratio", 0) or 0
    calmar  = metrics.get("calmar_ratio", 0) or 0
    dd      = abs(metrics.get("max_drawdown_pct", 0) or 0)
    dd_dur  = float(metrics.get("max_drawdown_duration", 0) or 0)  # en jours
    tr      = _trade_count(metrics)
    pf      = metrics.get("profit_factor", 0) or metrics.get("trades_profit_factor", 0) or 0
    ret     = metrics.get("total_return_pct", 0) or 0
    wr      = (metrics.get("win_rate_pct", 0) or 0) / 100.0
    worst   = abs(metrics.get("worst_trade_pct", 0) or 0)

    # HARD rejects
    if tr <= 0:
        return -15.0                       # pénalise no-signal (vs -8 v1)
    if tr < 5:
        return -8.0 + sh * 0.05            # 1-4 trades = bruit
    if tr < 10:
        return -5.0 + sh * 0.1
    if dd > 50:
        return -3.0 + sh * 0.1

    # Sharpe : poids principal (cap 5 ×1.5)
    sharpe_score   = min(max(sh, 0), 5) * 1.5            # max 7.5
    # Sortino : récompense smoothness (pertes < gains)
    sortino_score  = min(max(sortino, 0), 5) * 0.5       # max 2.5
    # Calmar : récompense return/DD (efficacité)
    calmar_score   = min(max(calmar, 0), 10) * 0.2       # max 2.0
    # PF, return : bonus mineurs
    pf_score       = min(max(pf - 1.0, 0), 2.0) * 0.4    # max 0.8
    ret_score      = min(max(ret, 0), 30.0) * 0.01       # max 0.3
    trade_score    = _trade_density_score(tr)            # max 0.75

    # Pénalités structurelles
    dd_penalty     = max(dd - 5.0, 0) * 0.10             # kick-in à 5%
    dd_dur_penalty = max(dd_dur - 7.0, 0) * 0.15         # >7 jours
    worst_penalty  = max(worst - 5.0, 0) * 0.10          # tail risk
    wr_penalty     = max(0.50 - wr, 0) * 1.0

    return (sharpe_score + sortino_score + calmar_score + pf_score + ret_score + trade_score
            - dd_penalty - dd_dur_penalty - worst_penalty - wr_penalty)


def score_robust_v2(metrics) -> float:
    """Trade-density-forcing variant of score_robust (2026-05-19).

    Motivation: en WFA 90j train / 30j test, score_robust v1 sélectionne des
    params qui ne génèrent que 5-15 trades/fold → Sharpe statistiquement bruité,
    OOS médiocre, peu d'observations. v2 rend le trade count DOMINANT dans le
    score : à Sharpe comparable, optuna est forcé vers des params qui tradent
    beaucoup plus fréquemment.

    Diffs vs v1 :
      - HARD reject sous 10 trades à -10 (v1 : 5 trades)
      - HARD reject sous 30 trades = rampe linéaire (v1 : 10 trades à -5 plat)
      - Trade density bonus max **5.0** (v1 : 0.75) → comparable au Sharpe
      - Sharpe cap baissé : 4 × 1.0 = max 4.0 (v1 : 5 × 1.5 = max 7.5)
      - Conséquence : un combo Sharpe 3 / 60 trades bat Sharpe 5 / 15 trades.
    """
    if metrics is None:
        return -10.0

    sh      = metrics.get("sharpe_ratio", 0) or 0
    sortino = metrics.get("sortino_ratio", 0) or 0
    calmar  = metrics.get("calmar_ratio", 0) or 0
    dd      = abs(metrics.get("max_drawdown_pct", 0) or 0)
    dd_dur  = float(metrics.get("max_drawdown_duration", 0) or 0)
    tr      = _trade_count(metrics)
    pf      = metrics.get("profit_factor", 0) or metrics.get("trades_profit_factor", 0) or 0
    ret     = metrics.get("total_return_pct", 0) or 0
    wr      = (metrics.get("win_rate_pct", 0) or 0) / 100.0
    worst   = abs(metrics.get("worst_trade_pct", 0) or 0)

    if tr <= 0:
        return -20.0
    if tr < 10:
        return -10.0 + sh * 0.05
    if tr < 30:
        return -5.0 + (tr - 10) * 0.1 + sh * 0.1
    if dd > 50:
        return -3.0 + sh * 0.1

    # Trade density : composante dominante (max 5.0)
    #   tr=30  → 0.0   |   tr=60 → 1.71   |   tr=100 → 4.0
    #   tr=200 → 5.0 (plateau)
    if tr <= 100:
        trade_score = (tr - 30) * (4.0 / 70.0)
    else:
        trade_score = 4.0 + min((tr - 100) / 100.0, 1.0)

    sharpe_score   = min(max(sh, 0), 4) * 1.0
    sortino_score  = min(max(sortino, 0), 5) * 0.4
    calmar_score   = min(max(calmar, 0), 10) * 0.15
    pf_score       = min(max(pf - 1.0, 0), 2.0) * 0.4
    ret_score      = min(max(ret, 0), 30.0) * 0.01

    dd_penalty     = max(dd - 5.0, 0) * 0.10
    dd_dur_penalty = max(dd_dur - 7.0, 0) * 0.15
    worst_penalty  = max(worst - 5.0, 0) * 0.10
    wr_penalty     = max(0.50 - wr, 0) * 0.5

    return (trade_score + sharpe_score + sortino_score + calmar_score + pf_score + ret_score
            - dd_penalty - dd_dur_penalty - worst_penalty - wr_penalty)


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
    "robust": score_robust,
    "robust_v2": score_robust_v2,
    "high_frequency": score_high_frequency,
    "trend": score_trend,
}
