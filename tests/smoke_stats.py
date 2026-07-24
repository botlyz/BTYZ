"""Smoke test étapes 4-7 (stats/, deploy_rule, gates).

Exécution : cd BTYZ/src && python ../tests/smoke_stats.py
Tout est synthétique (RESULTS_ROOT + ledger temporaires monkeypatchés) —
aucune écriture dans les résultats réels, aucun holdout réel touché.
"""
import json
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from quantlab import config
from quantlab.contract import BaseStrategy, Signals
from quantlab.ledger import Ledger


def check(label, cond):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        raise SystemExit(f"smoke failed: {label}")


# ================================================================== 1. PBO
print("== 1. pbo_cscv ==")
from quantlab.stats.pbo import pbo_cscv

rng = np.random.default_rng(0)
noise = pd.DataFrame(rng.normal(0.0, 0.01, size=(80, 64)),
                     columns=[f"p{i}" for i in range(64)])
res_noise = pbo_cscv(noise, seed=1)
# le PBO d'UN dataset de bruit est très dispersé (0.27-0.74 selon le tirage) :
# on teste la moyenne sur 5 tirages (~0.5) + le verdict sur un tirage haut
pbo_noise = [pbo_cscv(pd.DataFrame(
    np.random.default_rng(s).normal(0.0, 0.01, size=(80, 64))),
    seed=1)["pbo"] for s in range(5)]
check(f"bruit pur: PBO moyen={np.mean(pbo_noise):.3f} ~0.5 (> 0.3)",
      float(np.mean(pbo_noise)) > 0.3)
res_high = pbo_cscv(pd.DataFrame(
    np.random.default_rng(5).normal(0.0, 0.01, size=(80, 64))), seed=1)
check(f"bruit (tirage haut, PBO={res_high['pbo']:.3f}): verdict REJECT",
      res_high["verdict"] == "REJECT")

signal = noise.copy()
signal.iloc[0] = rng.normal(0.006, 0.01, 64)   # un vrai edge persistant
res_sig = pbo_cscv(signal, seed=1)
check(f"signal fort: PBO={res_sig['pbo']:.3f} bas (< 0.3)",
      res_sig["pbo"] < 0.3)
check("n_combos plafonné", res_noise["n_combos"] <= 3000)
check("logits échantillonnés", 0 < len(res_noise["logits"]) <= 500)

# orientation périodes × trials (index datetime) -> transposée automatiquement
dt_idx = pd.date_range("2025-01-01", periods=64, freq="1h")
res_t = pbo_cscv(signal.T.set_axis(dt_idx, axis=0), seed=1)
check("orientation auto (matrice transposée, index datetime)",
      abs(res_t["pbo"] - res_sig["pbo"]) < 1e-12)

# ================================================================== 2. DSR
print("== 2. deflated_sharpe / effective_n ==")
from quantlab.stats.dsr import deflated_sharpe, effective_n

fpy = 24 * 365
sr_bar = 2.0 / np.sqrt(fpy)                    # Sharpe annualisé ~2, barres 1h
r = pd.Series(rng.normal(sr_bar * 0.01, 0.01, 5000))
d10 = deflated_sharpe(r, n_effective=10, sharpe_variance=1e-4,
                      freq_per_year=fpy)
d100k = deflated_sharpe(r, n_effective=100_000, sharpe_variance=1e-4,
                        freq_per_year=fpy)
check(f"DSR(N=10)={d10['dsr']:.3f} élevé (> 0.5)", d10["dsr"] > 0.5)
check(f"DSR(N=1e5)={d100k['dsr']:.3f} s'effondre (< 0.3)", d100k["dsr"] < 0.3)
check("monotonie en N_eff", d10["dsr"] > d100k["dsr"])
check("sr annualisé reporté ~2 (bruit d'échantillonnage ±1.3 à T=5000)",
      abs(d10["sr_hat_ann"] - 2.0) < 1.5)

common = rng.normal(0, 0.01, 256)
trials = pd.DataFrame(
    [np.sqrt(0.9) * common + np.sqrt(0.1) * rng.normal(0, 0.01, 256)
     for _ in range(50)])
n_eff = effective_n(trials, research_debt=10_000)
check(f"trials corrélés 0.9: N_eff={n_eff} << debt=10000", 2 <= n_eff < 3000)
n_eff_iid = effective_n(
    pd.DataFrame(rng.normal(0, 0.01, size=(50, 256))), research_debt=10_000)
check(f"trials iid: N_eff={n_eff_iid} ~ debt", n_eff_iid > 8000)

# ================================================================== 3. permutation (unitaire)
print("== 3. block_permute_ohlc ==")
from quantlab.stats.permutation import block_permute_ohlc

n = 2400
ar = np.zeros(n)
eps = rng.normal(0, 0.004, n)
for i in range(1, n):                          # AR(1) rho=0.6
    ar[i] = 0.6 * ar[i - 1] + eps[i]
close = 100 * np.cumprod(1 + ar)
openp = np.roll(close, 1); openp[0] = close[0]
df = pd.DataFrame({
    "open": openp,
    "high": np.maximum(openp, close) * 1.001,
    "low": np.minimum(openp, close) * 0.999,
    "close": close,
    "volume": rng.uniform(1, 10, n),
}, index=pd.date_range("2025-01-01", periods=n, freq="1h"))

perm = block_permute_ohlc(df, config.PERMUTATION_BLOCK_BARS,
                          np.random.default_rng(7))
r_orig = df["close"].to_numpy()
r0 = np.concatenate([[0.0], r_orig[1:] / r_orig[:-1] - 1.0])
pc = perm["close"].to_numpy()
r_perm = np.concatenate([[pc[0] / r_orig[0] - 1.0], pc[1:] / pc[:-1] - 1.0])
check("mêmes stats marginales (multiset des returns identique)",
      np.allclose(np.sort(r_perm), np.sort(r0), atol=1e-12))
check("index d'origine conservé", perm.index.equals(df.index))
check("OHLC cohérent (h >= max(o,c), l <= min(o,c))",
      bool((perm["high"] >= np.maximum(perm["open"], perm["close"]) - 1e-9).all()
           and (perm["low"] <= np.minimum(perm["open"], perm["close"]) + 1e-9).all()))
check("close > 0 partout", bool((perm["close"] > 0).all()))


def _ac1(x):
    x = np.asarray(x[1:])
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


ac_o, ac_p = _ac1(r0), _ac1(r_perm)
check(f"autocorrélation intra-bloc préservée grossièrement "
      f"({ac_p:.2f} vs {ac_o:.2f})", ac_p > 0.6 * ac_o)
check("permutation non triviale (ordre changé)",
      not np.allclose(r_perm, r0))


# ================================================================== stratégie factice
class _DummyStrategy(BaseStrategy):
    FAMILY = "SMOKESTAT"
    VERSION = 1
    DATA_SOURCE = "lighter"
    TF = "1h"
    WARMUP_BARS = 20
    DEFAULT_PARAMS = {"win": 12}

    def signals(self, data, params):
        win = int(params.get("win", 12))
        ma = data["close"].rolling(win).mean()
        return Signals(long_entries=(data["close"] > ma).fillna(False),
                       long_exits=(data["close"] < ma).fillna(False))

    def param_space(self, trial):
        return {"win": trial.suggest_int("win", 6, 48)}


# ============================================================ 4. permutation -> optimize (WFEvaluator)
print("== 4. permutation -> optimize.WFEvaluator (mini-opti in-process) ==")
from quantlab.stats.permutation import _best_sharpe_permuted

strat = _DummyStrategy()
rng2 = np.random.default_rng(11)
datas = {}
for pair in ("AAA", "BBB"):
    permuted = block_permute_ohlc(df, config.PERMUTATION_BLOCK_BARS, rng2)
    datas[pair] = permuted
cfg = {"tf": "1h", "fees_bps": 3.0, "k_folds": 2, "embargo": 30,
       "anchored": False, "min_trades_per_param": 1}
best = _best_sharpe_permuted(strat, datas, cfg, seed=3, n_trials=3)
check(f"mini-opti WF sur données permutées -> best sharpe fini ({best:.2f})",
      np.isfinite(best))

# ============================================================ 5. deploy_rule (fichiers factices)
print("== 5. deploy_rule ==")
import quantlab.deploy_rule as deploy_rule

with tempfile.TemporaryDirectory() as tmp:
    old_root = config.RESULTS_ROOT
    config.RESULTS_ROOT = Path(tmp) / "results"
    led = Ledger(Path(tmp) / "ledger.db")
    try:
        sid = strat.strategy_id()
        opt_dir = config.RESULTS_ROOT / sid / "optimize"
        opt_dir.mkdir(parents=True)
        rows = []
        # 2 trials × 4 folds ; trial 1 meilleur IS partout
        for fold in range(4):
            rows.append({"trial": 0, "fold": fold,
                         "params_json": json.dumps({"win": 10 + fold}),
                         "sharpe_is": 1.0, "sharpe_oos": 0.4,
                         "n_trades_is": 50, "n_trades": 40})
            rows.append({"trial": 1, "fold": fold,
                         "params_json": json.dumps({"win": 20 + 2 * fold}),
                         "sharpe_is": 2.0, "sharpe_oos": 1.2,
                         "n_trades_is": 50, "n_trades": 40})
        pd.DataFrame(rows).to_parquet(opt_dir / "folds_is_oos.parquet")
        plat_dir = config.RESULTS_ROOT / sid / "plateau"
        plat_dir.mkdir(parents=True)
        (plat_dir / "selected_params.json").write_text(
            json.dumps({"params": {"win": 21}}))

        res = deploy_rule.run(strat, ledger=led, adaptive=False)
        # best IS par fold = trial 1 -> WFE = 1.2 / 2.0 = 0.6
        check(f"WFE calculé juste ({res['metrics']['wfe']:.3f} == 0.6)",
              abs(res["metrics"]["wfe"] - 0.6) < 1e-9)
        check("verdict FIXED (WFE > 0.5, adaptatif sauté)",
              res["verdict"] == "FIXED")
        check("stabilité calculée (L2 + params par fold)",
              res["metrics"]["stability"]["l2_mean"] is not None)
        check("deploy_rule/verdict.json écrit",
              (config.RESULTS_ROOT / sid / "deploy_rule" / "verdict.json").exists())
        check("ledger: deploy_rule enregistré PASS",
              led.pipeline_state(sid)["deploy_rule"] == "PASS")

        # WFE < WFE_REJECT -> REJECT
        low = pd.DataFrame(rows)
        low["sharpe_oos"] = 0.1
        low.to_parquet(opt_dir / "folds_is_oos.parquet")
        res2 = deploy_rule.run(strat, ledger=led, adaptive=False)
        check("WFE=0.05 -> REJECT", res2["verdict"] == "REJECT")

        # mean IS <= 0 -> WFE non défini -> REJECT motivé
        neg = pd.DataFrame(rows)
        neg["sharpe_is"] = -0.5
        neg.to_parquet(opt_dir / "folds_is_oos.parquet")
        res3 = deploy_rule.run(strat, ledger=led, adaptive=False)
        check("mean IS <= 0 -> REJECT 'WFE non défini'",
              res3["verdict"] == "REJECT" and "non défini" in res3["reason"])

        # ==================================================== 6. run_battery
        print("== 6. stats.run_battery (fichiers factices) ==")
        import quantlab.stats as qstats

        tr = signal.copy()                     # 80 trials, un vrai signal
        tr.index.name = "trial"
        tr.to_parquet(opt_dir / "trial_returns.parquet")
        bat = qstats.run_battery(strat, ledger=led)
        sdir = config.RESULTS_ROOT / sid / "stats"
        check("stats/pbo.json + dsr.json + verdict.json écrits",
              all((sdir / f).exists()
                  for f in ("pbo.json", "dsr.json", "verdict.json")))
        check(f"PBO bas sur matrice à signal ({bat['metrics']['pbo']:.3f})",
              bat["metrics"]["pbo"] < 0.3)
        check("verdict combiné PASS|REJECT",
              bat["verdict"] in ("PASS", "REJECT"))

        # ==================================================== 7. gates
        print("== 7. gates.run_holdout (one-shot) ==")
        import quantlab.data.splits as splits
        import quantlab.gates as gates

        (opt_dir / "summary.json").write_text(json.dumps({
            "config": {"pairs": ["AAA", "BBB"], "tf": "1h",
                       "source": "lighter", "fees_bps": 3.0, "k_folds": 2},
            "best": {"sharpe_concat": 2.0}}))

        def _fake_holdout(source, pair, tf, *, _token):
            if not isinstance(_token, str) or not _token:
                raise PermissionError("token requis")
            return df.iloc[-800:]

        old_lh = splits.load_holdout
        splits.load_holdout = _fake_holdout
        try:
            hres = gates.run_holdout(strat, ledger=led)
            check("résultat écrit quel qu'il soit (holdout/verdict.json)",
                  (config.RESULTS_ROOT / sid / "holdout" / "verdict.json").exists())
            check(f"result COHERENT|EFFONDREMENT ({hres['result']})",
                  hres["result"] in ("COHERENT", "EFFONDREMENT"))
            check("sharpe holdout extrait",
                  "sharpe_holdout" in hres["metrics"])
            try:
                gates.run_holdout(strat, ledger=led)
                check("2e appel holdout -> RuntimeError", False)
            except RuntimeError as e:
                check(f"2e appel holdout -> RuntimeError (famille brûlée)",
                      "brûlée" in str(e) or "consommé" in str(e))
            check("ledger: holdout_available False",
                  not led.holdout_available(strat.FAMILY))
        finally:
            splits.load_holdout = old_lh

        # ==================================================== 8. incubation
        print("== 8. gates.incubation_report ==")
        inc = gates.incubation_report(strat)
        check("squelette WAITING_LIVE_DATA",
              inc["status"] == "WAITING_LIVE_DATA")
        fills = pd.DataFrame({
            "ts": ["2026-07-01T00:00:00Z"] * 4,
            "pair": ["AAA"] * 4, "side": ["buy", "sell"] * 2,
            "qty": [1.0] * 4, "px": [100.0] * 4,
            "fee": [0.03] * 4, "funding": [0.0] * 4})   # 3 bps
        fp = Path(tmp) / "fills.csv"
        fills.to_csv(fp, index=False)
        inc2 = gates.incubation_report(strat, fills_csv=str(fp))
        check(f"coût réalisé vs modèle -> verdict ({inc2['verdict']})",
              inc2["verdict"] in ("PASS", "REJECT"))
        check("coût moyen ~3 bps",
              abs(inc2["metrics"]["realized_cost_bps_mean"] - 3.0) < 0.01)
    finally:
        config.RESULTS_ROOT = old_root

print("\nsmoke_stats: TOUT OK")
