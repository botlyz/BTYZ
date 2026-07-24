"""Smoke test quantlab.data — à lancer depuis src/ : python ../tests/smoke_data.py

Vérifie sur données réelles : store (natif + resample 4h), universe, split
dev/holdout (coupure exacte + verrou PermissionError), coverage, et un download
binance réel (1 paire × 1 mois, dans un dossier temporaire).
"""
import logging
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
warnings.simplefilter("error", FutureWarning)   # aucun warning pandas bruyant

import pandas as pd  # noqa: E402

from quantlab import config  # noqa: E402
from quantlab.data import (SOURCES, coverage, holdout_hash, load, load_dev,  # noqa: E402
                           load_holdout, universe)
from quantlab.data.splits import dev_holdout_cut  # noqa: E402

t0 = time.time()


def check(label, cond):
    status = "OK " if cond else "FAIL"
    print(f"[{status}] {label}  (t={time.time()-t0:.1f}s)")
    if not cond:
        sys.exit(1)


# ---------------------------------------------------------------- store natif
btc_1h = load("lighter", "BTC", "1h")
check("store.load lighter BTC 1h", btc_1h is not None and len(btc_1h) > 1000)
check("index UTC croissant", str(btc_1h.index.tz) == "UTC"
      and btc_1h.index.is_monotonic_increasing)
check("colonnes OHLCV", all(c in btc_1h.columns
                            for c in ("open", "high", "low", "close", "volume")))

# ---------------------------------------------------------------- resample 4h
btc_4h = load("lighter", "BTC", "4h")
check("store.load lighter BTC 4h (resample)", btc_4h is not None and len(btc_4h) > 250)
check("4h ~= 1h/4", abs(len(btc_4h) - len(btc_1h) / 4) < 10)
check("4h aligné sur la grille", (btc_4h.index.minute == 0).all()
      and (btc_4h.index.hour % 4 == 0).all())
# resample correct : high 4h >= high 1h max sur la période
h1 = btc_1h["high"].resample("4h", label="left", closed="left").max().dropna()
common = btc_4h.index.intersection(h1.index)
check("resample high == max(high 1h)",
      (btc_4h.loc[common, "high"] >= h1.loc[common] - 1e-9).all())

# cache hit (2e appel instantané, même objet lru)
t = time.time()
load("lighter", "BTC", "4h")
check("lru cache store.load", time.time() - t < 0.1)

# ---------------------------------------------------------------- universe
t = time.time()
uni = universe("lighter", "5m")
print(f"    universe lighter 5m: {len(uni)} paires en {time.time()-t:.1f}s "
      f"(ex: {uni[:8]})")
check("universe lighter 5m non vide", len(uni) > 20 and "BTC" in uni)

# ---------------------------------------------------------------- splits
cut = dev_holdout_cut(btc_1h.index)
dev = load_dev("lighter", "BTC", "1h")
check("load_dev strictement avant cutoff", dev.index.max() < cut)

try:
    load_holdout("lighter", "BTC", "1h", _token="")
    check("load_holdout sans token -> PermissionError", False)
except PermissionError:
    check("load_holdout sans token -> PermissionError", True)
try:
    load_holdout("lighter", "BTC", "1h", _token=None)
    check("load_holdout token None -> PermissionError", False)
except PermissionError:
    check("load_holdout token None -> PermissionError", True)

ho = load_holdout("lighter", "BTC", "1h", _token="smoke-test-token")
check("holdout commence au cutoff", ho.index.min() >= cut)
check("dev + holdout = série complète", len(dev) + len(ho) == len(btc_1h)
      and dev.index.max() < ho.index.min())
span = btc_1h.index[-1] - btc_1h.index[0]
expected = min(pd.Timedelta(days=config.HOLDOUT_MAX_DAYS),
               span * config.HOLDOUT_FRACTION)
check("cutoff = fin - min(183j, 20% span)", cut == btc_1h.index[-1] - expected)

hh = holdout_hash("lighter", "BTC", "1h")
check("holdout_hash sha256 stable", len(hh) == 64
      and hh == holdout_hash("lighter", "BTC", "1h"))

# ---------------------------------------------------------------- coverage
cov = coverage()
sub = cov[(cov["pair"] == "BTC") & (cov["source"] == "lighter")]
print(sub.to_string(index=False))
check("coverage colonnes", list(cov.columns) == ["source", "pair", "tf", "n_bars",
                                                 "start", "end", "holdout_cutoff"])
check("coverage contient BTC lighter 1h/4h", {"1h", "4h"} <= set(sub["tf"]))

# ------------------------------------------------- download binance réel (1 mois)
from quantlab.data.sources import binance as bmod  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    orig = bmod.RAW_1M
    bmod.RAW_1M = Path(tmp)
    try:
        src = bmod.BinanceSource()
        ok = src.download("SOL", "1m", start_month="2026-06")
        check("binance download SOL 2026-06", ok)
        df = src.load_raw("SOL", "1m")
        check("download relisible (mois complet ~43200 barres 1m)",
              df is not None and 40000 < len(df) <= 44641
              and str(df.index[0])[:7] == "2026-06")
        # append idempotent : re-download du même mois ne duplique rien
        n1 = len(df)
        src.download("SOL", "1m", start_month="2026-06")
        check("re-download -> pas de doublon",
              len(src.load_raw("SOL", "1m")) == n1)
    finally:
        bmod.RAW_1M = orig

# ---------------------------------------------------------------- binance store
sol_1h = load("binance", "SOL", "1h")
check("store.load binance SOL 1h (resample depuis um/1m)",
      sol_1h is not None and len(sol_1h) > 10000)

print(f"\nSMOKE DATA: tout est passé en {time.time()-t0:.1f}s")
