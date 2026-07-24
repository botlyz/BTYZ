"""PBO par CSCV — Bailey, Borwein, López de Prado, Zhu,
"The Probability of Backtest Overfitting" (2015).

Entrée : la matrice des returns de validation par trial produite par
optimize (`optimize/trial_returns.parquet`, lignes = trials, colonnes =
périodes). On découpe les périodes en S blocs, on énumère les combinaisons
C(S, S/2) train/test : le meilleur trial in-sample est classé out-of-sample,
logit λ = ln(w/(1-w)) de son rang relatif w. PBO = frac(λ <= 0).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from quantlab import config

MAX_COMBOS = 3000


def _as_trials_by_periods(trial_returns: pd.DataFrame) -> pd.DataFrame:
    """Force l'orientation lignes=trials, colonnes=périodes.

    Heuristique : un axe datetime = axe des périodes. Sinon on fait confiance
    à l'orientation documentée par la SPEC (trials × périodes).
    """
    def _is_dt(axis) -> bool:
        if isinstance(axis, pd.DatetimeIndex):
            return True
        try:
            pd.to_datetime(axis[:3])
            return not pd.api.types.is_numeric_dtype(np.asarray(axis))
        except Exception:
            return False

    if _is_dt(trial_returns.index) and not _is_dt(trial_returns.columns):
        return trial_returns.T
    return trial_returns


def _block_moments(mat: np.ndarray, s_blocks: int):
    """Par (trial, bloc) : somme, somme des carrés, effectif (NaN ignorés)."""
    n_trials, n_periods = mat.shape
    edges = np.linspace(0, n_periods, s_blocks + 1).astype(int)
    sums = np.empty((n_trials, s_blocks))
    sq = np.empty((n_trials, s_blocks))
    cnt = np.empty((n_trials, s_blocks))
    for b in range(s_blocks):
        seg = mat[:, edges[b]:edges[b + 1]]
        valid = np.isfinite(seg)
        segz = np.where(valid, seg, 0.0)
        sums[:, b] = segz.sum(axis=1)
        sq[:, b] = (segz ** 2).sum(axis=1)
        cnt[:, b] = valid.sum(axis=1)
    return sums, sq, cnt


def _sharpe_from_moments(sums, sq, cnt, cols) -> np.ndarray:
    """Sharpe (par période, non annualisé — seul le rang compte) sur un
    sous-ensemble de blocs `cols`."""
    s = sums[:, cols].sum(axis=1)
    q = sq[:, cols].sum(axis=1)
    n = cnt[:, cols].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
        var = q / n - mean ** 2
        var = np.where(var > 0, var, np.nan)
        sr = mean / np.sqrt(var)
    return np.where(np.isfinite(sr), sr, -np.inf)


def pbo_cscv(trial_returns: pd.DataFrame, s_blocks: int = config.PBO_BLOCKS,
             *, max_combos: int = MAX_COMBOS, seed: int = 42) -> dict:
    """CSCV complet. Retourne {"pbo", "n_combos", "logits", "verdict", ...}."""
    df = _as_trials_by_periods(trial_returns)
    mat = df.to_numpy(dtype="float64")
    # trials sans aucune donnée -> exclus
    keep = np.isfinite(mat).any(axis=1)
    mat = mat[keep]
    n_trials, n_periods = mat.shape
    if n_trials < 3:
        return {"pbo": float("nan"), "n_combos": 0, "logits": [],
                "verdict": "REJECT", "reason": f"trop peu de trials ({n_trials})",
                "n_trials": n_trials, "s_blocks": s_blocks}
    s_blocks = min(s_blocks, n_periods)
    if s_blocks % 2:
        s_blocks -= 1
    if s_blocks < 4:
        return {"pbo": float("nan"), "n_combos": 0, "logits": [],
                "verdict": "REJECT",
                "reason": f"trop peu de périodes ({n_periods}) pour le CSCV",
                "n_trials": n_trials, "s_blocks": s_blocks}

    sums, sq, cnt = _block_moments(mat, s_blocks)

    n_total = math.comb(s_blocks, s_blocks // 2)
    rng = np.random.default_rng(seed)
    if n_total > max_combos:
        chosen = set()
        while len(chosen) < max_combos:
            pick = tuple(sorted(rng.choice(s_blocks, s_blocks // 2,
                                           replace=False).tolist()))
            chosen.add(pick)
        combos = sorted(chosen)
    else:
        combos = list(itertools.combinations(range(s_blocks), s_blocks // 2))

    all_blocks = set(range(s_blocks))
    logits = []
    for train in combos:
        test = sorted(all_blocks - set(train))
        sr_is = _sharpe_from_moments(sums, sq, cnt, list(train))
        sr_oos = _sharpe_from_moments(sums, sq, cnt, test)
        best = int(np.argmax(sr_is))
        # rang relatif OOS du best IS : w in (0,1)
        rank = float((sr_oos < sr_oos[best]).sum()
                     + 0.5 * (sr_oos == sr_oos[best]).sum())
        w = rank / (len(sr_oos) + 1.0)
        w = min(max(w, 1e-9), 1 - 1e-9)
        logits.append(math.log(w / (1.0 - w)))

    logits = np.asarray(logits)
    pbo = float((logits <= 0).mean())
    if pbo > config.PBO_REJECT:
        verdict = "REJECT"
    elif pbo < config.PBO_PASS:
        verdict = "PASS"
    else:
        verdict = "MARGINAL"
    sample = logits if len(logits) <= 500 else \
        logits[np.linspace(0, len(logits) - 1, 500).astype(int)]
    return {
        "pbo": pbo,
        "n_combos": len(combos),
        "logits": [round(float(x), 4) for x in sample],
        "verdict": verdict,
        "n_trials": int(n_trials),
        "n_periods": int(n_periods),
        "s_blocks": int(s_blocks),
        "logit_median": float(np.median(logits)),
        "thresholds": {"pass": config.PBO_PASS, "reject": config.PBO_REJECT},
    }
