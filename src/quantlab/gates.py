"""Étapes 6-7 — gates humains : holdout one-shot + rapport d'incubation.

run_holdout est le SEUL point du pipeline autorisé à lire le holdout :
il consomme le token one-shot du ledger (famille brûlée définitivement,
succès OU échec), backteste les params du plateau sur le segment gelé et
écrit le résultat QUEL QU'IL SOIT (holdout/verdict.json). Aucun retry.

incubation_report compare les coûts réalisés live (fills CSV) au modèle
fees+slippage de la config — squelette en attendant les données du serveur.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantlab import config

HOLDOUT_COLLAPSE_DEGRADATION = 0.80    # dégradation > 80 % -> effondrement
INCUBATION_COST_TOLERANCE = 1.25       # coût réalisé <= 125 % du modèle
FILLS_COLUMNS = ("ts", "pair", "side", "qty", "px", "fee", "funding")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _out_dir(sid: str, stage: str):
    d = config.RESULTS_ROOT / sid / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def _retained_pairs(sid: str, strategy) -> list[str]:
    """Paires retenues : config d'optimize, sinon survivantes du screening
    (TF native)."""
    opt = config.RESULTS_ROOT / sid / "optimize" / "summary.json"
    if opt.exists():
        pairs = (json.loads(opt.read_text()).get("config") or {}).get("pairs")
        if pairs:
            return list(pairs)
    cells_fp = config.RESULTS_ROOT / sid / "screening" / "cells.parquet"
    if cells_fp.exists():
        cells = pd.read_parquet(cells_fp)
        mask = cells["qval"] < config.FDR_Q
        if "tf" in cells.columns:
            mask &= cells["tf"] == strategy.TF
        if "source" in cells.columns:
            mask &= cells["source"] == strategy.DATA_SOURCE
        pairs = sorted(cells.loc[mask, "pair"].unique().tolist())
        if pairs:
            return pairs
    raise FileNotFoundError(
        f"paires retenues introuvables pour {sid} (optimize/summary.json "
        "ou screening/cells.parquet requis)")


def _oos_sharpe(sid: str) -> float | None:
    """Sharpe OOS de référence (best concaténé d'optimize)."""
    opt = config.RESULTS_ROOT / sid / "optimize" / "summary.json"
    if not opt.exists():
        return None
    best = json.loads(opt.read_text()).get("best") or {}
    v = best.get("sharpe_concat")
    return float(v) if v is not None and np.isfinite(float(v)) else None


# ------------------------------------------------------------------ étape 6
def run_holdout(strategy, *, progress=None, ledger=None) -> dict:
    """Gate holdout one-shot. RuntimeError si la famille est déjà brûlée.

    Le résultat (COHERENT ou EFFONDREMENT — dégradation > 80 % ou Sharpe
    holdout < 0) est écrit quel qu'il soit. AUCUN retry possible.
    """
    from quantlab import registry
    from quantlab.ledger import family_of, ledger as _default_ledger
    from quantlab.progress import PipelineProgress

    led = ledger if ledger is not None else _default_ledger
    if isinstance(strategy, str):
        strategy = registry.load(strategy)
    sid = strategy.strategy_id()
    family = family_of(sid)

    # --- paramètres et périmètre AVANT de brûler le token
    sel_path = config.RESULTS_ROOT / sid / "plateau" / "selected_params.json"
    if not sel_path.exists():
        raise FileNotFoundError(f"{sel_path} manquant — lancer plateau d'abord")
    selected = json.loads(sel_path.read_text())
    params = selected.get("params", selected)
    pairs = _retained_pairs(sid, strategy)
    tf, source = strategy.TF, strategy.DATA_SOURCE
    opt = config.RESULTS_ROOT / sid / "optimize" / "summary.json"
    fees = config.DEFAULT_FEES_BPS / 1e4
    if opt.exists():
        cfg = json.loads(opt.read_text()).get("config") or {}
        tf = cfg.get("tf") or tf
        source = cfg.get("source") or source
        fees = float(cfg.get("fees_bps", config.DEFAULT_FEES_BPS)) / 1e4

    # --- token one-shot : la famille est brûlée ICI, quoi qu'il arrive après
    token = led.consume_holdout(family)
    if token is None:
        raise RuntimeError(
            f"REFUS définitif : le holdout de la famille '{family}' a déjà "
            "été consommé (one-shot). Aucun retry — la famille est brûlée.")

    run_id = led.start_run(strategy, "holdout",
                           {"pairs": pairs, "tf": tf, "source": source})
    out_dir = _out_dir(sid, "holdout")

    def _write(verdict: str, result: str, reason: str, metrics: dict,
               n_tests: int) -> dict:
        out = {"stage": "holdout", "verdict": verdict, "result": result,
               "reason": reason, "metrics": metrics, "n_tests": n_tests,
               "at": _now()}
        (out_dir / "verdict.json").write_text(json.dumps(out, indent=2))
        led.record_verdict(strategy, "holdout", out)
        led.finish_run(run_id, verdict, n_tests, {"result": result})
        if n_tests:
            led.record_tests(family, n_tests, "holdout", {"pairs": pairs})
        return out

    own = progress is None
    if own:
        progress = PipelineProgress(sid, "holdout", ledger=led)
        progress.__enter__()
    try:
        task = progress.task("backtest holdout", total=len(pairs) + 1)
        from quantlab.backtest import pooled_backtest
        from quantlab.data import splits

        datas = {}
        for p in pairs:
            try:
                df = splits.load_holdout(source, p, tf, _token=token)
                if df is not None and len(df) > strategy.WARMUP_BARS:
                    datas[p] = df
            except Exception:
                pass
            task.advance(1, pair=p)
        if not datas:
            return _write("REJECT", "EFFONDREMENT",
                          "aucune donnée holdout chargeable pour les paires "
                          f"retenues {pairs}", {"pairs": pairs}, 0)

        from engine.metrics import extract_from_portfolio
        pf = pooled_backtest(strategy, datas, params, fees=fees, tf=tf)
        m = extract_from_portfolio(pf)
        task.advance(1)
        task.done()
    except Exception as exc:   # résultat écrit même en cas d'erreur
        _write("REJECT", "ERROR",
               f"échec du backtest holdout ({type(exc).__name__}: {exc}) — "
               "token consommé, famille brûlée", {"pairs": pairs}, 0)
        raise
    finally:
        if own:
            progress.__exit__(None, None, None)

    sharpe_h = m.get("sharpe_ratio")
    sharpe_h = float(sharpe_h) if sharpe_h is not None and \
        np.isfinite(sharpe_h) else float("nan")
    oos = _oos_sharpe(sid)
    degradation = None
    if oos is not None and oos > 0 and np.isfinite(sharpe_h):
        degradation = 1.0 - sharpe_h / oos

    collapse = (not np.isfinite(sharpe_h)) or sharpe_h < 0 or (
        degradation is not None and degradation > HOLDOUT_COLLAPSE_DEGRADATION)
    result = "EFFONDREMENT" if collapse else "COHERENT"
    verdict = "REJECT" if collapse else "PASS"
    reason = (f"Sharpe holdout {sharpe_h:.3f} vs OOS "
              f"{oos if oos is not None else 'n/a'}"
              + (f", dégradation {degradation:.0%}" if degradation is not None
                 else "") + f" -> {result}")

    metrics = {
        "sharpe_holdout": sharpe_h,
        "sharpe_oos": oos,
        "degradation": degradation,
        "total_return": m.get("total_return_pct", m.get("total_return")),
        "max_drawdown": m.get("max_drawdown_pct", m.get("max_drawdown")),
        "n_trades": m.get("total_trades"),
        "pairs": sorted(datas), "tf": tf, "source": source,
        "fees_bps": fees * 1e4, "params": params,
    }
    return _write(verdict, result, reason, metrics, len(datas))


# ------------------------------------------------------------------ étape 7
def incubation_report(strategy, fills_csv=None) -> dict:
    """Rapport d'incubation : coûts réalisés live vs modèle.

    fills_csv : CSV avec colonnes ts (ISO8601), pair, side (buy/sell),
    qty (base), px (prix), fee (quote, >0 = payé), funding (quote, signé).
    Sans fichier : squelette WAITING_LIVE_DATA avec les instructions.
    """
    from quantlab import registry

    if isinstance(strategy, str):
        strategy = registry.load(strategy)
    sid = strategy.strategy_id()
    out_dir = _out_dir(sid, "incubation")

    model_bps = config.DEFAULT_FEES_BPS + config.DEFAULT_SLIPPAGE * 1e4

    if fills_csv is None:
        out = {
            "stage": "incubation",
            "status": "WAITING_LIVE_DATA",
            "instructions": (
                "Exporter les fills live du serveur en CSV (colonnes: "
                f"{', '.join(FILLS_COLUMNS)}) puis relancer "
                "incubation_report(strategy, fills_csv=<path>). Verdict en "
                "nombre de trades, taille réduite."),
            "expected_columns": list(FILLS_COLUMNS),
            "cost_model_bps": model_bps,
            "at": _now(),
        }
        (out_dir / "report.json").write_text(json.dumps(out, indent=2))
        return out

    fills = pd.read_csv(fills_csv)
    missing = [c for c in FILLS_COLUMNS if c not in fills.columns]
    if missing:
        raise ValueError(f"colonnes manquantes dans {fills_csv}: {missing} "
                         f"(attendues: {list(FILLS_COLUMNS)})")
    notional = (fills["qty"].abs() * fills["px"]).replace(0, np.nan)
    fee_bps = (fills["fee"] / notional * 1e4)
    funding_bps = (fills["funding"] / notional * 1e4)
    realized_bps = float((fee_bps + funding_bps).mean())

    within = realized_bps <= model_bps * INCUBATION_COST_TOLERANCE
    verdict = "PASS" if within else "REJECT"
    out = {
        "stage": "incubation",
        "status": "REPORT",
        "verdict": verdict,
        "reason": (f"coût réalisé moyen {realized_bps:.2f} bps vs modèle "
                   f"{model_bps:.2f} bps (tolérance x"
                   f"{INCUBATION_COST_TOLERANCE})"),
        "metrics": {
            "n_fills": int(len(fills)),
            "realized_cost_bps_mean": realized_bps,
            "fee_bps_mean": float(fee_bps.mean()),
            "funding_bps_mean": float(funding_bps.mean()),
            "model_cost_bps": model_bps,
            "pairs": sorted(fills["pair"].astype(str).unique().tolist()),
            "span": [str(fills["ts"].min()), str(fills["ts"].max())],
        },
        "n_tests": 0,
        "at": _now(),
    }
    (out_dir / "report.json").write_text(json.dumps(out, indent=2))
    return out
