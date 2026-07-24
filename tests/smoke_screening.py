"""Smoke test du screening (étape 1) sur données réelles locales.

Exécution : cd BTYZ/src && python ../tests/smoke_screening.py
Univers réduit + bootstrap/nulles réduits pour tenir en < 2 min.

NB: le fan-out passe par quantlab.parallel.pool_map (contexte spawn imposé par
max_tasks_per_child) -> tout le corps du smoke est sous __main__.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd


def check(label, cond):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        raise SystemExit(f"smoke failed: {label}")


def _pid_task(_):
    time.sleep(0.3)          # laisse le temps aux autres workers de spawner
    return os.getpid()


def main():
    from quantlab import config, parallel, registry, screening
    from quantlab.data import splits, store
    from quantlab.data.sources import SOURCES
    from quantlab.ledger import Ledger

    # --------------------------------------------------------- 1. BH-FDR
    print("== 1. Benjamini-Hochberg (cas connu à la main) ==")
    p = [0.01, 0.04, 0.03, 0.005]
    q = screening.bh_qvalues(p)
    # à la main : tri [.005,.01,.03,.04] -> p*n/i = [.02,.02,.04,.04] -> cummin droite
    check("q-values exactes", np.allclose(q, [0.02, 0.04, 0.04, 0.02]))
    from scipy.stats import false_discovery_control
    rand = np.random.default_rng(0).random(200)
    check("BH == scipy.false_discovery_control",
          np.allclose(screening.bh_qvalues(rand), false_discovery_control(rand)))
    order = np.argsort(rand)
    check("q croissantes le long des p triées",
          bool(np.all(np.diff(screening.bh_qvalues(rand)[order]) >= -1e-12)))

    # --------------------------------------------------------- 2. pool_map
    print("== 2. fan-out via parallel.pool_map (multi-process) ==")
    pids = parallel.pool_map(_pid_task, range(8), workers=4)
    check("pool_map tourne hors du process parent",
          all(pid != os.getpid() for pid in pids))
    check("pool_map utilise plusieurs workers", len(set(pids)) > 1)

    calls = []
    _real_pool_map = parallel.pool_map

    def _spy_pool_map(fn, items, **kw):
        items = list(items)
        calls.append((getattr(fn, "__name__", "?"), len(items)))
        return _real_pool_map(fn, items, **kw)

    screening.parallel.pool_map = _spy_pool_map

    # --------------------------------------------------------- 3. SMA_CROSS_v1
    print("== 3. screening SMA_CROSS_v1 (lighter/1h, 8 paires, réduit) ==")
    tmp = tempfile.mkdtemp(prefix="smoke_screening_")
    config.RESULTS_ROOT = Path(tmp) / "results"      # pas de pollution results/
    led = Ledger(Path(tmp) / "ledger.db")

    strat = registry.load("SMA_CROSS_v1")
    wanted = ["BTC", "ETH", "SOL", "DOGE", "LINK", "AVAX", "ADA", "AAVE"]
    avail = set(SOURCES["lighter"].pairs())
    pairs = [p for p in wanted if p in avail][:8]
    check(f"paires dispo ({len(pairs)})", len(pairs) >= 6)

    t0 = time.time()
    verdict = screening.run(strat, sources=["lighter"], tfs=["1h"], pairs=pairs,
                            bootstrap=200, n_null=20, seed=42, ledger=led)
    dt = time.time() - t0
    print(f"  verdict={verdict['verdict']}  n_tests={verdict['n_tests']}  ({dt:.1f}s)")
    print(f"  reason: {verdict['reason']}")

    out_dir = config.RESULTS_ROOT / "SMA_CROSS_v1" / "screening"
    check("cells.parquet écrit", (out_dir / "cells.parquet").exists())
    check("verdict.json écrit", (out_dir / "verdict.json").exists())
    vj = json.loads((out_dir / "verdict.json").read_text())
    check("verdict.json == retour", vj["verdict"] == verdict["verdict"])
    check("champs verdict", all(k in vj for k in
                                ("stage", "verdict", "reason", "metrics",
                                 "n_tests", "at")))

    cells = pd.read_parquet(out_dir / "cells.parquet")
    check(f"1 ligne par cellule ({len(cells)})", len(cells) == len(pairs))
    check("colonnes SPEC", all(c in cells.columns for c in
                               ("pair", "tf", "source", "fees_bps", "sharpe",
                                "pval", "qval", "null_q95", "n_trades",
                                "total_return_pct", "positive")))
    check("aucune cellule en erreur", (cells["error"] == "").all())
    check("pvals dans [0,1]", cells["pval"].between(0, 1).all())
    check("q >= p (BH)", (cells["qval"] >= cells["pval"] - 1e-12).all())

    # jamais le holdout : max(index utilisé) < cutoff calculé sur la série complète
    for r in cells.itertuples():
        full = store.load(r.source, r.pair, r.tf)
        cut = splits.dev_holdout_cut(full.index)
        check(f"holdout intact {r.pair} (fin dev < cut {cut:%Y-%m-%d})",
              pd.Timestamp(r.data_end) < cut)

    # ledger : n_tests = cellules + nulles, débité sur la famille
    n_cand = int((cells["qval"] < config.FDR_Q).sum())
    check("n_tests = cellules + nulles",
          verdict["n_tests"] == len(cells) + n_cand * 20)
    check("research_debt famille == n_tests",
          led.research_debt("SMA_CROSS") == verdict["n_tests"])
    check("verdict screening au ledger",
          led.pipeline_state("SMA_CROSS_v1")["screening"] == verdict["verdict"])
    check("fan-out cellules passé par pool_map",
          any(name == "_cell_task" and n == len(pairs) for name, n in calls))

    # nulle appariée : exercée directement (peut n'avoir aucun candidat FDR)
    res = screening._null_task({"strategy_id": "SMA_CROSS_v1", "source": "lighter",
                                "pair": pairs[0], "tf": "1h", "fees_bps": 3.0,
                                "seed": 42, "n_null": 5})
    check("_null_task retourne null_q95 fini", np.isfinite(res["null_q95"]))

    # --------------------------------------------------------- 4. OI_FADE_v1
    print("== 4. screening OI_FADE_v1 (source oi, 3 paires) ==")
    strat_oi = registry.load("OI_FADE_v1")
    t0 = time.time()
    v_oi = screening.run(strat_oi, pairs=["BTC", "ETH", "SOL"],
                         bootstrap=200, n_null=20, seed=42, ledger=led)
    dt = time.time() - t0
    print(f"  verdict={v_oi['verdict']}  n_tests={v_oi['n_tests']}  ({dt:.1f}s)")
    print(f"  reason: {v_oi['reason']}")

    cells_oi = pd.read_parquet(config.RESULTS_ROOT / "OI_FADE_v1" / "screening"
                               / "cells.parquet")
    check("source forcée 'oi'", (cells_oi["source"] == "oi").all())
    check("aucune cellule en erreur", (cells_oi["error"] == "").all())
    from engine.data_loader import load_ohlcv
    for r in cells_oi.itertuples():
        full = load_ohlcv(r.pair, tf="1h", source="oi")
        cut = splits.dev_holdout_cut(full.index)
        check(f"holdout intact {r.pair} (source oi)", pd.Timestamp(r.data_end) < cut)
    check("ledger OI_FADE incrémenté",
          led.research_debt("OI_FADE") == v_oi["n_tests"])

    print("\nSMOKE SCREENING: TOUT OK")


if __name__ == "__main__":
    main()
