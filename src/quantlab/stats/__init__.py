"""Étape 4 — batterie statistique : PBO (CSCV) puis DSR.

run_battery écrit stats/pbo.json, stats/dsr.json, stats/verdict.json.
La permutation Monte-Carlo (la plus coûteuse) est séparée : run_permutation,
opt-in via le CLI (--permutation), écrit stats/permutation.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantlab import config
from quantlab.stats.dsr import deflated_sharpe, effective_n
from quantlab.stats.pbo import _as_trials_by_periods, pbo_cscv

__all__ = ["run_battery", "run_permutation", "pbo_cscv", "deflated_sharpe",
           "effective_n"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _freq_per_year_from_columns(cols) -> float | None:
    """Périodes/an si les colonnes de la matrice sont des timestamps."""
    if not (isinstance(cols, pd.DatetimeIndex)
            or (len(cols) and isinstance(cols[0], (pd.Timestamp,))) ):
        return None
    try:
        ts = pd.to_datetime(cols)
        if len(ts) < 3:
            return None
        sec = float(pd.Series(ts).diff().dropna().median().total_seconds())
        return 365.0 * 86400.0 / sec if sec > 0 else None
    except Exception:
        return None


def run_battery(strategy, progress=None, *, ledger=None) -> dict:
    """PBO d'abord (REJECT court-circuite), puis DSR avec N effectif dérivé
    du research_debt TOTAL de la famille (ledger)."""
    from quantlab.ledger import family_of, ledger as _default_ledger
    from quantlab.progress import PipelineProgress

    led = ledger if ledger is not None else _default_ledger
    sid = strategy if isinstance(strategy, str) else strategy.strategy_id()
    family = family_of(sid)
    out_dir = config.RESULTS_ROOT / sid / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_path = config.RESULTS_ROOT / sid / "optimize" / "trial_returns.parquet"
    if not tr_path.exists():
        raise FileNotFoundError(
            f"{tr_path} manquant — lancer optimize avant la batterie stats")
    trial_returns = pd.read_parquet(tr_path)

    run_id = led.start_run(strategy, "stats", {"test": "battery"})
    own = progress is None
    if own:
        progress = PipelineProgress(sid, "stats", ledger=led)
        progress.__enter__()
    try:
        task = progress.task("batterie PBO+DSR", total=2)

        # ---------------------------------------------------------- PBO
        pbo_res = pbo_cscv(trial_returns)
        (out_dir / "pbo.json").write_text(json.dumps(pbo_res, indent=2))
        task.advance(1, pbo=pbo_res.get("pbo"))

        # ---------------------------------------------------------- DSR
        df = _as_trials_by_periods(trial_returns)
        mat = df.to_numpy(dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nanmean(mat, axis=1)
            std = np.nanstd(mat, axis=1, ddof=1)
            trial_sharpes = np.where(std > 0, mean / std, np.nan)
        finite = trial_sharpes[np.isfinite(trial_sharpes)]
        sharpe_variance = float(np.var(finite, ddof=1)) if len(finite) > 2 \
            else float("nan")
        best_idx = int(np.nanargmax(trial_sharpes))
        candidate = df.iloc[best_idx].dropna()

        debt = max(led.research_debt(family), len(df))
        n_eff = effective_n(trial_returns, debt)
        dsr_res = deflated_sharpe(
            candidate, n_effective=n_eff, sharpe_variance=sharpe_variance,
            freq_per_year=_freq_per_year_from_columns(df.columns))
        dsr_res.update({"research_debt": int(debt), "best_trial": best_idx})
        (out_dir / "dsr.json").write_text(json.dumps(dsr_res, indent=2))
        task.advance(1, dsr=dsr_res.get("dsr"))
        task.done()
    finally:
        if own:
            progress.__exit__(None, None, None)

    # -------------------------------------------------------------- verdict
    pbo_v = pbo_res["verdict"]
    if pbo_v == "REJECT":
        verdict, reason = "REJECT", (
            f"PBO={pbo_res['pbo']:.3f} > {config.PBO_REJECT} — sélection de bruit")
    elif not dsr_res.get("pass", False):
        verdict, reason = "REJECT", (
            f"DSR={dsr_res.get('dsr', float('nan')):.3f} < "
            f"{config.DSR_CONFIDENCE} (N_eff={n_eff}, debt={debt})")
    else:
        verdict = "PASS"
        reason = (f"PBO={pbo_res['pbo']:.3f}, DSR={dsr_res['dsr']:.3f} "
                  f"(N_eff={n_eff})")
        if pbo_v == "MARGINAL":
            reason += (f" — WARNING: PBO marginal ({config.PBO_PASS} <= "
                       f"{pbo_res['pbo']:.3f} <= {config.PBO_REJECT})")

    out = {
        "stage": "stats",
        "verdict": verdict,
        "reason": reason,
        "metrics": {
            "pbo": pbo_res.get("pbo"),
            "pbo_verdict": pbo_v,
            "dsr": dsr_res.get("dsr"),
            "n_effective": int(n_eff),
            "research_debt": int(debt),
            "sharpe_variance": sharpe_variance,
            "best_trial": best_idx,
        },
        "n_tests": 0,   # analyses de backtests déjà comptés par optimize
        "at": _now(),
    }
    (out_dir / "verdict.json").write_text(json.dumps(out, indent=2))
    led.record_verdict(strategy, "stats", out)
    led.finish_run(run_id, verdict, 0, {"pbo": pbo_res.get("pbo"),
                                        "dsr": dsr_res.get("dsr")})
    return out


def run_permutation(strategy, **kwargs) -> dict:
    """Permutation Monte-Carlo (opt-in). Voir stats/permutation.py."""
    from quantlab.stats import permutation
    return permutation.run(strategy, **kwargs)
