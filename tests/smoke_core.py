"""Smoke test du noyau quantlab (ledger, registry, lookahead, backtest, progress).

Exécution : cd BTYZ/src && python ../tests/smoke_core.py
"""
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from quantlab import config
from quantlab.contract import BaseStrategy, Signals
from quantlab.ledger import Ledger, family_of


def check(label, cond):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        raise SystemExit(f"smoke failed: {label}")


# ------------------------------------------------------------------ 1. ledger
print("== 1. ledger ==")
with tempfile.TemporaryDirectory() as tmp:
    led = Ledger(Path(tmp) / "ledger.db")
    check("family_of", family_of("OI_FADE_v2") == "OI_FADE")

    led.record_tests("SMOKE_v1", 100, "screening", {"src": "smoke"})
    led.record_tests("SMOKE_v2", 50, "optimize")   # même famille
    check("research_debt par famille (100+50)", led.research_debt("SMOKE_v1") == 150)
    exp = float(np.sqrt(2 * np.log(150)))
    check("credible_sharpe = sqrt(2 ln N)",
          abs(led.credible_sharpe("SMOKE") - exp) < 1e-9)

    ok, why = led.can_run("SMOKE_v1", "contract")
    check("can_run contract sans prérequis", ok)
    ok, why = led.can_run("SMOKE_v1", "screening")
    check("can_run screening refusé sans contract PASS", not ok)

    led.record_verdict("SMOKE_v1", "contract",
                       {"stage": "contract", "verdict": "PASS", "reason": "ok",
                        "metrics": {}, "n_tests": 1, "at": "2026-07-24T00:00:00Z"})
    ok, why = led.can_run("SMOKE_v1", "screening")
    check("can_run screening après contract PASS", ok)
    ok, why = led.can_run("SMOKE_v1", "holdout")
    check("can_run holdout refusé (chaîne incomplète)", not ok)
    for st in ["screening", "optimize", "plateau", "stats", "deploy_rule"]:
        led.record_verdict("SMOKE_v1", st, {"stage": st, "verdict": "PASS",
                                            "reason": "ok", "n_tests": 10})
    ok, why = led.can_run("SMOKE_v1", "holdout")
    check("can_run holdout après chaîne PASS", ok)

    rid = led.start_run("SMOKE_v1", "screening", {"seed": 42})
    led.finish_run(rid, "PASS", 33, {"note": "smoke"})
    check("start/finish_run", isinstance(rid, int))

    check("holdout_available avant", led.holdout_available("SMOKE"))
    tok = led.consume_holdout("SMOKE_v1")
    check("consume_holdout retourne un token", isinstance(tok, str) and len(tok) == 36)
    check("consume_holdout one-shot -> None", led.consume_holdout("SMOKE_v2") is None)
    check("holdout_available après", not led.holdout_available("SMOKE"))

    tbl = led.status_table()
    check("status_table DataFrame", isinstance(tbl, pd.DataFrame) and len(tbl) >= 1)
    row = tbl.loc["SMOKE_v1"]
    check("status_table verdicts + debt",
          row["contract"] == "PASS" and row["research_debt"] == 150
          and abs(row["credible_sharpe"] - round(exp, 3)) < 1e-9)
    print(tbl.to_string())


# ------------------------------------------------------------------ 2. lookahead
print("== 2. lookahead ==")
from quantlab import lookahead


class CleanSMA(BaseStrategy):
    FAMILY = "SMOKE_SMA"
    TF = "1h"
    DEFAULT_PARAMS = {"fast": 10, "slow": 30}

    def signals(self, data, params):
        fast = data["close"].rolling(params["fast"]).mean()
        slow = data["close"].rolling(params["slow"]).mean()
        up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        return Signals(long_entries=up.fillna(False), long_exits=dn.fillna(False))

    def param_space(self, trial):
        return {}


class PeekShift(CleanSMA):
    FAMILY = "SMOKE_PEEK"

    def signals(self, data, params):
        nxt = data["close"].shift(-1)               # regarde la barre suivante
        up = (nxt > data["close"]).fillna(False)
        return Signals(long_entries=up, long_exits=~up)


class FullZScore(CleanSMA):
    FAMILY = "SMOKE_Z"

    def signals(self, data, params):
        z = (data["close"] - data["close"].mean()) / data["close"].std()  # full-série
        return Signals(long_entries=(z < -1.0), long_exits=(z > 0.0))


rng = np.random.default_rng(42)
idx = pd.date_range("2024-01-01", periods=3000, freq="1h", tz="UTC")
toy = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))},
                   index=idx)
toy["open"] = toy["high"] = toy["low"] = toy["close"]
toy["volume"] = 1.0

r = lookahead.check(CleanSMA(), toy)
check(f"SMA causale passed=True ({r['detail']['n_prefixes_checked']} préfixes)",
      r["passed"] and r["first_divergence"] is None)
r = lookahead.check(PeekShift(), toy)
check(f"shift(-1) attrapé -> {r['first_divergence']}",
      not r["passed"] and r["first_divergence"]["timestamp"] is not None)
r = lookahead.check(FullZScore(), toy)
check(f"z-score full-série attrapé -> série {r['first_divergence']['series']}",
      not r["passed"] and r["first_divergence"] is not None)


# ------------------------------------------------------------------ 3. backtest
print("== 3. backtest (données réelles Lighter) ==")
from engine.data_loader import load_lighter
from engine.metrics import extract_from_portfolio
from quantlab.backtest import pooled_backtest, portfolio_returns, run_signals_backtest


class SMAWithStops(CleanSMA):
    FAMILY = "SMOKE_BT"

    def signals(self, data, params):
        sig = super().signals(data, params)
        sig.sl_stop, sig.tp_stop, sig.td_stop = 0.05, 0.10, 48  # 48 barres
        return sig


strat = SMAWithStops()
btc = load_lighter("BTC", "1h")
check("BTC 1h chargé", btc is not None and len(btc) > 2000)
btc = btc.iloc[-8000:]

pf = run_signals_backtest(strat, btc, {}, fees=0.0003, tf="1h")
m = extract_from_portfolio(pf)
print(f"  BTC: sharpe={m.get('sharpe_ratio')}, trades={m.get('trades_count')}, "
      f"total_return={m.get('total_return_pct', m.get('total_return'))}")
check("pf.stats cohérent (trades > 0, sharpe fini)",
      isinstance(m.get("sharpe_ratio"), float) and m.get("trades_count", 0) > 0)
r1 = portfolio_returns(pf)
check("portfolio_returns Series mono-paire",
      isinstance(r1, pd.Series) and len(r1) == len(btc))

datas = {}
for pair in ["BTC", "ETH", "SOL"]:
    d = load_lighter(pair, "1h")
    check(f"{pair} 1h chargé", d is not None)
    datas[pair] = d.iloc[-8000:]
pfp = pooled_backtest(strat, datas, {}, fees=0.0003, tf="1h")
check("pooled: portefeuille groupé (1 seul)", pfp.wrapper.grouper.is_grouped())
rp = portfolio_returns(pfp)
check("pooled: returns agrégés en Series", isinstance(rp, pd.Series))
mp = extract_from_portfolio(pfp)
print(f"  pooled 3 paires: sharpe={mp.get('sharpe_ratio')}, "
      f"trades={mp.get('trades_count')}")
check("pooled: trades sur le pool", mp.get("trades_count", 0) > 0)


# ------------------------------------------------------------------ 4. registry
print("== 4. registry ==")
from quantlab import registry

with tempfile.TemporaryDirectory() as tmp:
    sdir = Path(tmp) / "SMOKE_REG_v1"
    sdir.mkdir()
    (sdir / "strategy.py").write_text(
        "from quantlab.contract import BaseStrategy, Signals\n"
        "class Strategy(BaseStrategy):\n"
        "    FAMILY = 'SMOKE_REG'\n"
        "    TF = '1h'\n"
        "    DEFAULT_PARAMS = {'n': 5}\n"
        "    def signals(self, data, params):\n"
        "        return Signals()\n"
        "    def param_space(self, trial):\n"
        "        return {}\n")
    (sdir / "manifest.yaml").write_text(
        "tfs: [1h, 4h]\nmax_pairs: 12\nlive: false\nnote: 'smoke'\n")
    old_root = config.STRATEGIES_ROOT
    config.STRATEGIES_ROOT = Path(tmp)
    try:
        found = registry.discover()
        check("discover trouve SMOKE_REG_v1", "SMOKE_REG_v1" in found)
        try:
            registry.load("SMOKE_REG_v1")
            check("load refuse sans RATIONALE.md", False)
        except ValueError as e:
            check("load refuse sans RATIONALE.md", "RATIONALE" in str(e))
        (sdir / "RATIONALE.md").write_text("# Thèse\nsmoke test\n")
        inst = registry.load("SMOKE_REG_v1")
        check("load instancie BaseStrategy", isinstance(inst, BaseStrategy))
        check("module picklable enregistré",
              "strategies.SMOKE_REG_v1.strategy" in sys.modules)
        man = registry.read_manifest("SMOKE_REG_v1")
        check(f"manifest parsé {man}",
              man.get("max_pairs") == 12 and man.get("live") is False
              and man.get("tfs") == ["1h", "4h"])
    finally:
        config.STRATEGIES_ROOT = old_root


# ------------------------------------------------------------------ 5. progress
print("== 5. progress (non-TTY) ==")
from quantlab.progress import PipelineProgress

with tempfile.TemporaryDirectory() as tmp:
    led = Ledger(Path(tmp) / "l.db")
    led.record_tests("SMOKE_v1", 500, "screening")
    with PipelineProgress("SMOKE_v1", "screening", ledger=led) as prog:
        t1 = prog.task("cellules", total=40)
        for i in range(40):
            t1.advance(1, pair=f"P{i % 3}", sharpe=1.0 + i / 40)
        t1.done()
        t2 = prog.task("nulles", total=25)
        for i in range(25):
            t2.advance(1)
        t2.done()
check("progress non-TTY sans crash", True)

print("\nSMOKE CORE: tout est passé.")
