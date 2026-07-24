"""Smoke test optimize.py (étape 2) + plateau.py (étape 3).

Exécution : cd BTYZ/src && python ../tests/smoke_optimize.py
Données réelles Lighter (dev set), résultats redirigés vers un tmpdir,
ledger temporaire. Doit tourner en < ~3 min.

NB: tout est sous `if __name__ == "__main__"` — pool_map utilise le contexte
spawn (max_tasks_per_child), les workers ré-importent ce module.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantlab.contract import BaseStrategy, Signals


class Toy5(BaseStrategy):
    """5 paramètres libres -> doit être refusée avant tout backtest."""
    FAMILY = "SMOKE_TOY5"
    TF = "1h"
    DEFAULT_PARAMS = {"a": 1}

    def signals(self, data, params):
        return Signals()

    def param_space(self, trial):
        return {k: trial.suggest_int(k, 1, 10) for k in "abcde"}


def check(label, cond):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        raise SystemExit(f"smoke failed: {label}")


def main():
    import pandas as pd

    from quantlab import config
    from quantlab.ledger import Ledger

    tmp = tempfile.mkdtemp(prefix="smoke_optimize_")
    config.RESULTS_ROOT = Path(tmp)          # jamais les résultats réels
    led = Ledger(Path(tmp) / "ledger.db")
    print(f"(résultats smoke: {tmp})")

    from quantlab import optimize, plateau, registry
    from quantlab.data import splits, store

    PAIRS = ["BTC", "ETH", "SOL", "HYPE"]
    strat = registry.load("SMA_CROSS_v1")

    # -------------------------------------------- 1. refus > MAX_FREE_PARAMS
    print("== 1. refus si n_free_params > MAX_FREE_PARAMS ==")
    v = optimize.run(Toy5(), pairs=PAIRS[:2], trials=5, k_folds=3, ledger=led)
    check("verdict REJECT immédiat (5 params libres)",
          v["verdict"] == "REJECT" and "MAX_FREE_PARAMS" in v["reason"])
    check("aucun test compté", v["n_tests"] == 0)

    # -------------------------------------------- 2. folds / embargo / holdout
    print("== 2. structure des folds (purge + holdout jamais lu) ==")
    K = 3
    ev, cfg = optimize.make_evaluator(strat, pairs=PAIRS, k_folds=K, seed=42)
    common = None
    for df in ev.datas.values():
        common = df.index if common is None else common.intersection(df.index)
    E = cfg["embargo"]
    check(f"embargo auto > 0 (E={E} = 2×WARMUP, pas de td_stop)",
          E == 2 * strat.WARMUP_BARS and E > 0)

    for p, df in ev.datas.items():
        full = store.load("lighter", p, "1h")
        cut = splits.dev_holdout_cut(full.index)
        check(f"{p}: index max {df.index.max()} < cutoff holdout {cut}",
              df.index.max() < cut)

    folds = ev.folds
    check(f"{K} folds", len(folds) == K)
    for i, f in enumerate(folds):
        n_gap = int(((common >= f.train_end) & (common < f.valid_start)).sum())
        check(f"fold {i}: purge de {n_gap} barres == embargo {E} "
              f"entre fin train et début validation", n_gap == E)
        check(f"fold {i}: train_end <= valid_start", f.train_end <= f.valid_start)
    for a, b in zip(folds, folds[1:]):
        check("fenêtres de validation disjointes et ordonnées",
              a.valid_end <= b.valid_start)
    seg = len(common) // (K + 1)
    check("la concaténation couvre K fenêtres disjointes jusqu'à la fin du dev",
          folds[0].valid_start == common[seg]
          and folds[-1].valid_end > common[-1])

    # -------------------------------------------- 3. hard-reject trades/param
    print("== 3. hard-reject si MIN_TRADES_PER_PARAM forcé haut ==")
    t0 = time.time()
    v = optimize.run(strat, pairs=PAIRS[:2], trials=3, k_folds=K, seed=42,
                     ledger=led, workers=3, min_trades_per_param=100000)
    check("verdict REJECT (aucun trial valide)", v["verdict"] == "REJECT")
    out = config.RESULTS_ROOT / "SMA_CROSS_v1" / "optimize"
    summary_hr = json.loads((out / "summary.json").read_text())
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_hr = optuna.load_study(
        study_name=summary_hr["study_name"],
        storage=optimize._rdb_storage(summary_hr["storage"]))
    hard = [t for t in study_hr.trials if t.user_attrs.get("hard_reject")]
    check(f"hard_reject visible dans {len(hard)}/3 trials "
          f"(valeur {optimize.HARD_REJECT})",
          len(hard) == 3 and all(t.value == optimize.HARD_REJECT for t in hard))
    print(f"  (mini-run hard-reject: {time.time() - t0:.1f}s)")

    # -------------------------------------------- 4. étude complète 25 trials
    print("== 4. optimize.run SMA_CROSS_v1, 4 paires 1h, 25 trials, k=3 ==")
    t0 = time.time()
    v = optimize.run(strat, pairs=PAIRS, trials=25, k_folds=K, seed=42,
                     ledger=led)
    dt = time.time() - t0
    print(f"  verdict={v['verdict']} ({v['reason']}) en {dt:.1f}s "
          f"({dt / 25:.2f}s/trial)")
    check("étude terminée < 3 min", dt < 180)
    check("n_tests = trials×k + re-runs IS ×k",
          v["n_tests"] >= 25 * K and v["n_tests"] % K == 0)

    tr = pd.read_parquet(out / "trial_returns.parquet")
    check(f"trial_returns.parquet {tr.shape} == (25, {config.PBO_BLOCKS * 2})",
          tr.shape == (25, config.PBO_BLOCKS * 2))
    iso = pd.read_parquet(out / "folds_is_oos.parquet")
    check(f"folds_is_oos.parquet non vide ({len(iso)} lignes, "
          f"{iso['trial'].nunique()} trials × {K} folds)",
          len(iso) > 0 and set(iso["fold"]) == set(range(K))
          and {"params_json", "sharpe_is", "sharpe_oos",
               "n_trades"} <= set(iso.columns))
    summary = json.loads((out / "summary.json").read_text())
    check("summary.json: best params + sharpe_concat",
          summary["best"] is not None and "fast" in summary["best"]["params"]
          and isinstance(summary["best"]["sharpe_concat"], float))
    print(f"  best: {summary['best']['params']} "
          f"sharpe_concat={summary['best']['sharpe_concat']:.3f} "
          f"fold_trades={summary['best']['fold_trades']}")
    check("tests comptés au ledger (debt famille)",
          led.research_debt("SMA_CROSS") >= 25 * K)

    # -------------------------------------------- 5. plateau
    print("== 5. plateau.run ==")
    t0 = time.time()
    vp = plateau.run(strat, top_frac=0.6, ledger=led)
    dtp = time.time() - t0
    print(f"  verdict={vp['verdict']} ({vp['reason']}) en {dtp:.1f}s")
    pdir = config.RESULTS_ROOT / "SMA_CROSS_v1" / "plateau"
    sel = json.loads((pdir / "selected_params.json").read_text())
    check("selected_params.json: centre + ensemble",
          {"fast", "slow"} <= set(sel["params"]) and len(sel["ensemble"]) >= 1)
    perts = pd.read_parquet(pdir / "perturbations.parquet")
    check(f"perturbations.parquet ({len(perts)} perturbations)",
          len(perts) >= 2 and {"param", "direction", "sharpe_concat",
                               "drop"} <= set(perts.columns))
    print(f"  centre plateau: {sel['params']} "
          f"(best trial: {summary['best']['params']})")
    print(perts[["param", "direction", "value", "sharpe_concat", "drop"]]
          .to_string(index=False))

    # -------------------------------------------- 6. OI_FADE make_objective
    print("== 6. OI_FADE_v1 make_objective (source oi, 2 paires) ==")
    oi = registry.load("OI_FADE_v1")
    obj = optimize.make_objective(oi, pairs=["BTC", "ETH"], k_folds=3, seed=42)
    for params in ({"doi_w": 6, "pctl": 0.93, "hold": 36, "sl_pct": 0.08},
                   {"doi_w": 4, "pctl": 0.89, "hold": 24, "sl_pct": 0.12},
                   {"doi_w": 8, "pctl": 0.95, "hold": 48, "sl_pct": 0.04}):
        val = obj(params)
        check(f"objective({params}) = {val:.3f} (float, pas d'erreur)",
              isinstance(val, float))
    for p, df in obj.evaluator.datas.items():
        full = store.load("lighter", p, "1h")
        cut = splits.dev_holdout_cut(full.index)
        check(f"oi/{p}: dev max {df.index.max()} < cutoff lighter {cut}",
              df.index.max() < cut)

    print("\nSMOKE OPTIMIZE: tout est passé.")


if __name__ == "__main__":
    main()
