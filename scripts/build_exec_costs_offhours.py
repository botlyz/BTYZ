#!/usr/bin/env python3
"""Coût d'exécution OFF-HOURS par paire TradFi (gap-fill) depuis les ticks Lighter.

Demi-spread effectif reconstruit via is_maker_ask (trades ask vs bid proches),
calculé UNIQUEMENT hors heures de bourse US (13:30-21:00 UTC, lun-ven) = la fenêtre
où la stratégie gap-fill trade. Validé sur fills réels : AAPL est. 2.1 vs réel 1.87 bps ;
AMD est. 7.5 vs réel 10.4 (résidu ~3 bps = prime de staleness, couverte par le
_target_slippage 2 bps de la stratégie).

Sortie : data/exec_costs_offhours.json  {pair: {half_spread_offhours_bps, median_gap_offhours_s, n_ticks_offhours}}
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICKS = os.path.join(ROOT, "data", "lighter_ticks")
OUT = os.path.join(ROOT, "data", "exec_costs_offhours.json")

# univers gap-fill = paires des résultats GAPFILL_v1 + extension commodités/indices/actions
UNIVERSE = sorted(
    p.rstrip("/").split("/")[-1]
    for p in glob.glob(os.path.join(ROOT, "results/GAPFILL_v1/full/*/*/"))
)
EXTRA = ["XAU", "XAG", "XPT", "XPD", "WTI", "BRENTOIL", "NATGAS", "XCU",
         "US500", "US100", "EWY", "MU", "URA", "BOTZ",
         # v3 : nouveaux candidats actions + tout le forex
         "RKLB", "BABA", "STRC",
         "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD", "USDKRW"]
UNIVERSE = sorted(set(UNIVERSE) | set(EXTRA))

# Sous-jacent qui ferme le WEEK-END uniquement (pas les nuits) -> coût mesuré sur
# ven 22:00 -> dim. Inclut futures/indices ET forex (FX = 24/5, ferme le week-end).
# Actions/ETF : hors RTH US (13:30-21:00 lun-ven).
FOREX = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD", "USDKRW"}
WEEKEND_CLOSE = {"XAU", "XAG", "XPT", "XPD", "WTI", "BRENTOIL", "NATGAS", "XCU",
                 "US500", "US100", "SPX", "PAXG"} | FOREX


def offhours_cost(sym: str):
    f = os.path.join(TICKS, f"{sym}.parquet")
    if not os.path.exists(f):
        return None
    t = pq.read_table(f, columns=["timestamp", "px", "is_maker_ask"]).to_pandas()
    dt = pd.to_datetime(t["timestamp"], unit="ms", utc=True)
    h = dt.dt.hour + dt.dt.minute / 60
    wd = dt.dt.weekday
    if sym in WEEKEND_CLOSE:
        off = ((wd == 5) | (wd == 6) | ((wd == 4) & (h >= 22))).to_numpy()
    else:
        off = ~((wd < 5) & (h >= 13.5) & (h <= 21)).to_numpy()
    t = t[off]
    if len(t) < 300:
        return None
    ts = t["timestamp"].to_numpy()
    px = t["px"].to_numpy()
    ask = t["is_maker_ask"].to_numpy().astype(bool)
    a_ts, a_px = ts[ask], px[ask]
    b_ts, b_px = ts[~ask], px[~ask]
    if len(a_ts) < 50 or len(b_ts) < 50:
        return None
    i = np.searchsorted(b_ts, a_ts)
    i0 = np.clip(i - 1, 0, len(b_ts) - 1)
    i1 = np.clip(i, 0, len(b_ts) - 1)
    d0 = np.abs(b_ts[i0] - a_ts)
    d1 = np.abs(b_ts[i1] - a_ts)
    use = np.where(d0 < d1, i0, i1)
    dist = np.minimum(d0, d1)
    bp = b_px[use]
    mid = (a_px + bp) / 2
    sp = (a_px - bp) / mid * 1e4
    v = (dist < 120_000) & (sp > 0) & np.isfinite(sp)
    s = sp[v]
    if len(s) < 30:
        return None
    gaps = np.diff(np.sort(ts)) / 1000.0
    return {
        "half_spread_offhours_bps": round(float(np.median(s)) / 2, 2),
        "median_gap_offhours_s": round(float(np.median(gaps)), 1),
        "n_ticks_offhours": int(len(t)),
    }


def main():
    out = {}
    for sym in UNIVERSE:
        r = offhours_cost(sym)
        if r:
            out[sym] = r
            print(f"  {sym:8} {r['half_spread_offhours_bps']:>6.2f} bps off-hours | gap {r['median_gap_offhours_s']:>7.0f}s | {r['n_ticks_offhours']} ticks")
        else:
            print(f"  {sym:8} — pas assez de données tick")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n{len(out)} paires -> {OUT}")


if __name__ == "__main__":
    main()
