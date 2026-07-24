"""Étape 1 — screening large : sources × timeframes × paires, DEFAULT_PARAMS.

Par cellule : backtest (quantlab.backtest.run_signals_backtest) -> Sharpe
annualisé (engine.metrics.extract_from_portfolio) + p-value du Sharpe par
bootstrap stationnaire des returns par barre (H0 : Sharpe <= 0). Puis BH-FDR
sur toutes les p-values, et nulle empirique appariée (permutations circulaires
des signaux, mêmes sl/td) sur les cellules candidates.

Données :
  - DATA_SOURCE "lighter"/"binance" -> store canonique, dev set via
    quantlab.data.splits.load_dev (jamais le holdout).
  - DATA_SOURCE spéciaux ("oi", "gapfill", ...) -> engine.data_loader.load_ohlcv
    (source imposée par la stratégie, pas de produit multi-exchange) ; le dev
    set est tronqué ICI au cutoff splits.dev_holdout_cut(index) puisque ces
    loaders ne passent pas par le store.

Fees : liquidity.json — paires "tres_liquide"/"liquide"/"moyen" -> fees_bps
(défaut config.DEFAULT_FEES_BPS) ; "poubelle" ou absentes -> ILLIQUID_FEES_BPS.

Sorties : RESULTS_ROOT/<sid>/screening/cells.parquet + verdict.json.
n_tests comptés au ledger = cellules + backtests nuls.
"""
from __future__ import annotations

import json
import zlib
from contextlib import ExitStack
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantlab import config, parallel
from quantlab.backtest import _stop_kwargs, run_signals_backtest, portfolio_returns
from quantlab.ledger import family_of, ledger as default_ledger

BOOTSTRAP_BLOCK_BARS = 24     # longueur moyenne des blocs (géométrique)
MIN_TRADES_FOR_PVALUE = 10    # < 10 trades -> p = 1 (pas de puissance)

# ------------------------------------------------------------------ helpers

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bh_qvalues(pvals) -> np.ndarray:
    """Benjamini-Hochberg : q-values (p ajustées, monotones) — implémentation locale."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _sharpe_pvalue(returns, n_boot: int, rng: np.random.Generator,
                   mean_block: int = BOOTSTRAP_BLOCK_BARS) -> float:
    """p-value du Sharpe par bootstrap stationnaire (Politis-Romano).

    Returns par barre centrés pour imposer H0 (Sharpe = 0), blocs de longueur
    géométrique moyenne `mean_block`, p = frac(sharpe_boot >= sharpe_obs).
    L'annualisation est omise : elle est monotone, la comparaison est invariante.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 100:
        return 1.0
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return 1.0
    obs = r.mean() / sd
    rc = r - r.mean()
    t = np.arange(n)
    p_restart = 1.0 / float(mean_block)
    count, done = 0, 0
    batch = max(1, min(n_boot, int(2e7 // n)))   # borne mémoire ~160 Mo/lot
    while done < n_boot:
        b = min(batch, n_boot - done)
        restart = rng.random((b, n)) < p_restart
        restart[:, 0] = True
        starts = rng.integers(0, n, size=(b, n))
        last = np.maximum.accumulate(np.where(restart, t, 0), axis=1)
        idx = (np.take_along_axis(starts, last, axis=1) + (t - last)) % n
        s = rc[idx]
        stat = s.mean(axis=1) / s.std(axis=1, ddof=1)
        count += int(np.sum(stat >= obs))
        done += b
    return (1 + count) / (n_boot + 1)


# ------------------------------------------------------------------ fees
_LIQ_CACHE: dict | None = None


def _liquid_pairs() -> set[str]:
    """Paires classées tradables dans liquidity.json (hors 'poubelle')."""
    global _LIQ_CACHE
    if _LIQ_CACHE is None:
        try:
            with open(config.LIQUIDITY_JSON) as fh:
                _LIQ_CACHE = json.load(fh)
        except FileNotFoundError:
            _LIQ_CACHE = {}
    out: set[str] = set()
    for level in ("tres_liquide", "liquide", "moyen"):
        out.update(_LIQ_CACHE.get(level, []))
    return out


def _fees_bps_for(pair: str, base_bps: float) -> float:
    base = pair.replace("USDT", "")
    return base_bps if base in _liquid_pairs() else config.ILLIQUID_FEES_BPS


# ------------------------------------------------------------------ données
def _is_store_source(source: str) -> bool:
    from quantlab.data.sources import SOURCES
    return source in SOURCES


def _load_cell_dev(source: str, pair: str, tf: str):
    """Dev set de la cellule. Store -> splits.load_dev ; source spéciale ->
    engine.data_loader.load_ohlcv tronqué au cutoff holdout (voir docstring module)."""
    from quantlab.data import splits
    if _is_store_source(source):
        return splits.load_dev(source, pair, tf)
    from engine.data_loader import load_ohlcv
    df = load_ohlcv(pair, tf=config.FREQ_MAP[tf], source=source)
    if df is None or not len(df):
        return None
    cut = splits.dev_holdout_cut(df.index)
    dev = df[df.index < cut]
    return dev if len(dev) else None


def _oi_pairs() -> list[str]:
    """Paires avec métriques OI Binance ET données Lighter (le loader croise les 2)."""
    metrics_dir = config.DATA_ROOT / "raw" / "binance" / "metrics"
    if not metrics_dir.is_dir():
        return []
    from quantlab.data.sources import SOURCES
    lighter = set(SOURCES["lighter"].pairs())
    syms = sorted(p.stem.replace("USDT", "") for p in metrics_dir.glob("*.parquet"))
    return [s for s in syms if s in lighter]


# ------------------------------------------------------------------ workers
_WORKER_STRATS: dict[str, object] = {}


def _worker_strategy(strategy_id: str):
    if strategy_id not in _WORKER_STRATS:
        from quantlab import registry
        _WORKER_STRATS[strategy_id] = registry.load(strategy_id)
    return _WORKER_STRATS[strategy_id]


def _n_trades_of(metrics: dict) -> int:
    for key in ("total_trades", "trades_count"):
        v = metrics.get(key)
        if v is not None and np.isfinite(v):
            return int(v)
    return 0


def _cell_seed(seed: int, source: str, pair: str, tf: str) -> int:
    return (seed + zlib.crc32(f"{source}/{pair}/{tf}".encode())) % (2**31)


def _cell_task(args: dict) -> dict:
    """Phase 1 — backtest d'une cellule + p-value bootstrap. Tourne en worker."""
    strat = _worker_strategy(args["strategy_id"])
    source, pair, tf = args["source"], args["pair"], args["tf"]
    row = {"pair": pair, "tf": tf, "source": source,
           "fees_bps": args["fees_bps"], "sharpe": np.nan, "pval": 1.0,
           "n_trades": 0, "total_return_pct": np.nan,
           "data_start": pd.NaT, "data_end": pd.NaT, "error": ""}
    try:
        df = _load_cell_dev(source, pair, tf)
        if df is None or len(df) < config.MIN_BARS:
            row["error"] = "no_data"
            return row
        row["data_start"], row["data_end"] = df.index[0], df.index[-1]
        pf = run_signals_backtest(strat, df, strat.DEFAULT_PARAMS,
                                  fees=args["fees_bps"] * 1e-4, tf=tf)
        from engine.metrics import extract_from_portfolio
        metrics = extract_from_portfolio(pf)
        row["sharpe"] = float(metrics.get("sharpe_ratio", np.nan))
        row["total_return_pct"] = float(metrics.get("total_return_pct", np.nan))
        row["n_trades"] = _n_trades_of(metrics)
        if row["n_trades"] >= MIN_TRADES_FOR_PVALUE and np.isfinite(row["sharpe"]):
            rng = np.random.default_rng(
                _cell_seed(args["seed"], source, pair, tf))
            row["pval"] = _sharpe_pvalue(portfolio_returns(pf).to_numpy(),
                                         args["bootstrap"], rng)
    except Exception as e:  # cellule cassée -> visible dans le parquet, p=1
        row["error"] = f"{type(e).__name__}: {e}"[:200]
    return row


def _roll_signals(sig, k: int):
    """Permutation circulaire (roll k barres) des 4 séries, mêmes sl/tp/td."""
    from quantlab.contract import Signals

    def _r(s):
        if s is None:
            return None
        return pd.Series(np.roll(s.to_numpy(), k), index=s.index)
    return Signals(long_entries=_r(sig.long_entries),
                   long_exits=_r(sig.long_exits),
                   short_entries=_r(sig.short_entries),
                   short_exits=_r(sig.short_exits),
                   sl_stop=sig.sl_stop, tp_stop=sig.tp_stop, td_stop=sig.td_stop)


def _null_task(args: dict) -> dict:
    """Phase 2 — nulle empirique appariée d'une cellule candidate (worker).

    n_null rolls circulaires des signaux du candidat (mêmes stops, mêmes fees)
    -> distribution de Sharpe nul -> quantile NULL_QUANTILE.
    """
    import vectorbtpro as vbt

    strat = _worker_strategy(args["strategy_id"])
    source, pair, tf = args["source"], args["pair"], args["tf"]
    out = {"pair": pair, "tf": tf, "source": source,
           "null_q95": np.nan, "n_null": 0}
    df = _load_cell_dev(source, pair, tf)
    if df is None or len(df) < config.MIN_BARS:
        return out
    sig = strat.signals(df, strat.full_params(strat.DEFAULT_PARAMS)).align(df.index)
    stop_kw = _stop_kwargs(sig, tf)
    rng = np.random.default_rng(_cell_seed(args["seed"], source, pair, tf) + 1)
    n = len(df)
    sharpes = []
    for _ in range(args["n_null"]):
        rolled = _roll_signals(sig, int(rng.integers(1, n)))
        pf = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=rolled.long_entries, exits=rolled.long_exits,
            short_entries=rolled.short_entries, short_exits=rolled.short_exits,
            fees=args["fees_bps"] * 1e-4, slippage=config.DEFAULT_SLIPPAGE,
            init_cash=config.DEFAULT_INIT_CASH, freq=config.FREQ_MAP[tf],
            **stop_kw)
        s = pf.sharpe_ratio
        try:
            s = float(s)
        except TypeError:
            s = float(np.asarray(s).item())
        if np.isfinite(s):
            sharpes.append(s)
    out["n_null"] = args["n_null"]
    if sharpes:
        out["null_q95"] = float(np.quantile(sharpes, config.NULL_QUANTILE))
    return out


# ------------------------------------------------------------------ run
def run(strategy, *, sources=None, tfs=None, fees_bps=None, seed=42,
        progress=None, pairs=None, bootstrap=None, n_null=None,
        max_workers=None, fdr_q=None, ledger=None) -> dict:
    """Screening complet -> dict verdict (écrit aussi cells.parquet + verdict.json).

    Overrides (smoke / debug) : `pairs` (liste imposée pour toutes les sources),
    `bootstrap` (resamples p-value), `n_null` (strats nulles par cellule),
    `max_workers`, `fdr_q`. DEV SET UNIQUEMENT — le holdout est inaccessible ici.
    """
    led = ledger if ledger is not None else default_ledger
    sid = strategy.strategy_id()
    family = family_of(sid)
    bootstrap = int(bootstrap or config.SCREEN_BOOTSTRAP)
    n_null = int(n_null or config.NULL_STRATEGIES_PER_CELL)
    fdr_q = float(fdr_q if fdr_q is not None else config.FDR_Q)
    base_bps = float(fees_bps if fees_bps is not None else config.DEFAULT_FEES_BPS)
    tfs = list(tfs or strategy.SCREEN_TFS or [strategy.TF])

    special = not _is_store_source(strategy.DATA_SOURCE)
    if special:
        # source de données imposée par la stratégie (oi, gapfill...) : pas de
        # produit multi-exchange, la donnée vient d'engine.data_loader.
        sources = [strategy.DATA_SOURCE]
    else:
        sources = list(sources or config.SOURCES)

    # ---------------- cellules
    from quantlab.data import store
    cells: list[dict] = []
    for source in sources:
        for tf in tfs:
            if pairs is not None:
                plist = list(pairs)
            elif not special:
                plist = store.universe(source, tf)
            elif strategy.DATA_SOURCE == "oi":
                plist = _oi_pairs()
            else:
                raise ValueError(
                    f"DATA_SOURCE spécial '{strategy.DATA_SOURCE}' : univers "
                    "inconnu, passe pairs=[...] explicitement")
            for pair in plist:
                cells.append({"strategy_id": sid, "source": source, "pair": pair,
                              "tf": tf, "fees_bps": _fees_bps_for(pair, base_bps),
                              "seed": int(seed), "bootstrap": bootstrap,
                              "n_null": n_null})
    if not cells:
        raise ValueError("screening: aucune cellule (univers vide ?)")

    run_id = led.start_run(strategy, "screening", {
        "seed": int(seed), "sources": sources, "tfs": tfs,
        "fees_bps": base_bps, "bootstrap": bootstrap, "n_null": n_null,
        "n_cells": len(cells)})

    workers = max_workers or config.N_WORKERS

    with ExitStack() as stack:
        if progress is None:
            from quantlab.progress import PipelineProgress
            progress = stack.enter_context(
                PipelineProgress(sid, "screening", ledger=led))

        # ------------ phase 1 : backtests + p-values (fan-out parallel.pool_map)
        t_cells = progress.task("cells", total=len(cells))
        rows: list[dict] = parallel.pool_map(
            _cell_task, cells, workers=workers,
            progress_handle=t_cells, ordered=False)
        t_cells.done()

        df = pd.DataFrame(rows)
        df["qval"] = bh_qvalues(df["pval"].to_numpy())
        df["positive"] = df["sharpe"] > 0

        # ------------ phase 2 : nulle empirique appariée (candidats FDR)
        cand = df[df["qval"] < fdr_q]
        df["null_q95"] = np.nan
        n_null_tests = len(cand) * n_null
        t_null = progress.task("null", total=max(len(cand), 1))
        if len(cand):
            # une tâche = une cellule candidate (n_null backtests groupés — le
            # spawn est amorti, pas besoin de parallel.chunked ici)
            null_args = [{"strategy_id": sid, "source": r.source, "pair": r.pair,
                          "tf": r.tf, "fees_bps": r.fees_bps, "seed": int(seed),
                          "n_null": n_null}
                         for r in cand.itertuples()]
            null_res = parallel.pool_map(
                _null_task, null_args, workers=workers,
                progress_handle=t_null, ordered=False)
            for res in null_res:
                mask = ((df["pair"] == res["pair"]) & (df["tf"] == res["tf"])
                        & (df["source"] == res["source"]))
                df.loc[mask, "null_q95"] = res["null_q95"]
        t_null.done()

    df["beats_null"] = df["sharpe"] > df["null_q95"]

    # ---------------- verdict
    fdr_ok = bool((df["qval"] < fdr_q).any())
    native = df[df["tf"] == strategy.TF]
    # fraction de paires positives dans la tf native : une paire compte positive
    # si son Sharpe moyen (sur les sources screenées) est > 0
    if len(native):
        pair_sharpe = native.groupby("pair")["sharpe"].mean()
        pos_frac = float((pair_sharpe > 0).mean())
    else:
        pos_frac = 0.0
    pos_ok = pos_frac >= config.MIN_POSITIVE_PAIR_FRAC
    native_surv = df[(df["tf"] == strategy.TF) & (df["qval"] < fdr_q)]
    if len(native_surv):
        null_frac = float(native_surv["beats_null"].mean())
        null_ok = null_frac > 0.5
    else:
        null_frac = float("nan")
        null_ok = False

    verdict = "PASS" if (fdr_ok and pos_ok and null_ok) else "REJECT"
    reason = (
        f"cellules={len(df)} (sources={sources}, tfs={tfs}) ; "
        f"FDR q<{fdr_q}: {int((df['qval'] < fdr_q).sum())} survivante(s) "
        f"[{'ok' if fdr_ok else 'ECHEC'}] ; "
        f"paires Sharpe>0 en tf native {strategy.TF}: {pos_frac:.0%} "
        f"(seuil {config.MIN_POSITIVE_PAIR_FRAC:.0%}) "
        f"[{'ok' if pos_ok else 'ECHEC'}] ; "
        f"nulle appariée (q{config.NULL_QUANTILE:.2f}, {n_null}/cellule) battue "
        f"sur {null_frac:.0%} des survivantes natives"
        if len(native_surv) else
        f"cellules={len(df)} (sources={sources}, tfs={tfs}) ; "
        f"FDR q<{fdr_q}: {int((df['qval'] < fdr_q).sum())} survivante(s) "
        f"[{'ok' if fdr_ok else 'ECHEC'}] ; "
        f"paires Sharpe>0 en tf native {strategy.TF}: {pos_frac:.0%} "
        f"(seuil {config.MIN_POSITIVE_PAIR_FRAC:.0%}) "
        f"[{'ok' if pos_ok else 'ECHEC'}] ; "
        f"aucune survivante FDR en tf native -> nulle non battue [ECHEC]")
    if len(native_surv):
        reason += f" [{'ok' if null_ok else 'ECHEC'}]"

    n_tests = len(df) + n_null_tests
    best = df.loc[df["sharpe"].idxmax()] if df["sharpe"].notna().any() else None
    verdict_dict = {
        "stage": "screening", "verdict": verdict, "reason": reason,
        "metrics": {
            "n_cells": int(len(df)),
            "n_fdr_survivors": int((df["qval"] < fdr_q).sum()),
            "positive_pair_frac_native": pos_frac,
            "null_beaten_frac_native": None if np.isnan(null_frac) else null_frac,
            "best_sharpe": None if best is None else float(best["sharpe"]),
            "best_cell": None if best is None else
                f"{best['source']}/{best['pair']}/{best['tf']}",
            "fees_bps_base": base_bps, "bootstrap": bootstrap, "n_null": n_null,
            "seed": int(seed),
        },
        "n_tests": int(n_tests), "at": _now(),
    }

    # ---------------- sorties + ledger
    out_dir = config.RESULTS_ROOT / sid / "screening"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["pair", "tf", "source", "fees_bps", "sharpe", "pval", "qval",
            "null_q95", "n_trades", "total_return_pct", "positive",
            "beats_null", "data_start", "data_end", "error"]
    df[cols].to_parquet(out_dir / "cells.parquet")
    with open(out_dir / "verdict.json", "w") as fh:
        json.dump(verdict_dict, fh, indent=2)

    led.record_tests(family, n_tests, "screening",
                     {"cells": len(df), "null": n_null_tests})
    led.finish_run(run_id, verdict, n_tests)
    led.record_verdict(strategy, "screening", verdict_dict)
    return verdict_dict
