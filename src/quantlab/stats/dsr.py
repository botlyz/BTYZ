"""Deflated Sharpe Ratio — Bailey & López de Prado (2014).

Adapté de archive/notebooks/analyse_full.py::compute_dsr. Deux différences :
- N est le N EFFECTIF dérivé du research_debt du ledger (recherche TOTALE de
  la famille) corrigé de la corrélation moyenne inter-trials ;
- le PSR du candidat est calculé sur SA série de returns (skew/kurtosis de la
  série, pas de la distribution des Sharpes).

Toutes les grandeurs (SR candidat, sharpe_variance) sont en unités PAR
PÉRIODE — la même période pour les deux (la probabilité DSR est invariante
d'échelle tant que les unités sont cohérentes).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from quantlab import config

_GAMMA_EM = 0.5772156649  # Euler-Mascheroni


def expected_max_sharpe(n_effective: int, sharpe_variance: float) -> float:
    """E[max SR] sous H0 (approximation EV) : SR0 du benchmark à déflater."""
    n = max(int(n_effective), 2)
    e_max = ((1.0 - _GAMMA_EM) * sp_stats.norm.ppf(1.0 - 1.0 / n)
             + _GAMMA_EM * sp_stats.norm.ppf(1.0 - 1.0 / (n * math.e)))
    return float(math.sqrt(max(sharpe_variance, 0.0)) * e_max)


def deflated_sharpe(candidate_returns: pd.Series, *, n_effective: int,
                    sharpe_variance: float,
                    freq_per_year: float | None = None) -> dict:
    """DSR = PSR(SR0) : Prob(SR vrai > SR0) avec correction skew/kurtosis.

    candidate_returns : returns PAR PÉRIODE du candidat.
    sharpe_variance   : variance des Sharpes des trials (mêmes unités période).
    freq_per_year     : optionnel, uniquement pour reporter le Sharpe annualisé.
    """
    r = pd.Series(candidate_returns).astype(float).dropna()
    t = len(r)
    if t < 4:
        return {"dsr": float("nan"), "pass": False,
                "reason": f"série trop courte (T={t})"}
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    if sd <= 0:
        return {"dsr": float("nan"), "pass": False, "reason": "std nulle"}
    sr_hat = mu / sd
    sr0 = expected_max_sharpe(n_effective, sharpe_variance)

    skew = float(sp_stats.skew(r))
    kurt = float(sp_stats.kurtosis(r, fisher=False))  # kurtosis totale
    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2
    denom = max(denom, 1e-12)
    z = (sr_hat - sr0) * math.sqrt(t - 1.0) / math.sqrt(denom)
    dsr = float(sp_stats.norm.cdf(z))

    out = {
        "dsr": dsr,
        "pass": bool(dsr >= config.DSR_CONFIDENCE),
        "sr_hat": sr_hat,
        "sr0": sr0,
        "n_effective": int(n_effective),
        "sharpe_variance": float(sharpe_variance),
        "t_obs": int(t),
        "skew": skew,
        "kurtosis": kurt,
        "confidence": config.DSR_CONFIDENCE,
    }
    if freq_per_year:
        out["sr_hat_ann"] = sr_hat * math.sqrt(freq_per_year)
        out["sr0_ann"] = sr0 * math.sqrt(freq_per_year)
    return out


def effective_n(trial_returns: pd.DataFrame, research_debt: int,
                *, max_pairs: int = 200, seed: int = 42) -> int:
    """N effectif : research_debt corrigé de la corrélation moyenne rho des
    trials. N_eff = max(2, round(debt·(1-rho) + rho)).

    rho estimé sur au plus `max_pairs` paires de trials tirées au hasard.
    """
    from quantlab.stats.pbo import _as_trials_by_periods

    df = _as_trials_by_periods(trial_returns)
    mat = df.to_numpy(dtype="float64")
    mat = mat[np.isfinite(mat).any(axis=1)]
    n = len(mat)
    debt = max(int(research_debt), 2)
    if n < 2:
        return debt
    rng = np.random.default_rng(seed)
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    corrs = []
    for _ in range(n_pairs):
        i, j = rng.choice(n, 2, replace=False)
        a, b = mat[i], mat[j]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 4:
            continue
        sa, sb = a[ok].std(), b[ok].std()
        if sa <= 0 or sb <= 0:
            continue
        corrs.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
    if not corrs:
        return debt
    rho = float(np.clip(np.nanmean(corrs), 0.0, 1.0))
    return max(2, int(round(debt * (1.0 - rho) + rho)))


def freq_per_year(tf: str) -> float:
    """Périodes par an pour une tf crypto 24/7."""
    minutes = pd.Timedelta(config.FREQ_MAP[tf]).total_seconds() / 60.0
    return 365.0 * 24.0 * 60.0 / minutes
