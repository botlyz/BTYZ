"""Weekend / off-hours gap-fill backtest for tokenized TradFi assets on Lighter.

Thesis: when the real market is CLOSED, the Lighter 24/7 perp drifts from the last
real close (noise/speculation). We fade that drift, betting on mean-reversion toward
the close, and exit when the market reopens (or the gap fills).

Data:
  - Lighter 5m  : data/raw/lighter/5m/<SYM>.csv      (24/7 perp price)
  - yfinance 1h : data/raw/yfinance/1h/<SYM>.csv      (real underlying, has gaps when closed)

Logic (vectorized):
  1. Align both on a 5min UTC grid over the common span.
  2. ffill the real close -> "fair value" = last known real-market price.
  3. market_open[t] = a real yfinance bar exists within the last RECENT_MIN minutes.
  4. deviation[t] = (lighter_close - fair_value) / fair_value.
  5. While market CLOSED and |deviation| > ENTRY_THRESH -> open a fade position
     (short if Lighter above fair value, long if below).
  6. Exit when market REOPENS (closed->open transition) or |deviation| < EXIT_THRESH.

Run single: python src/approach/gapfill/backtest_gapfill.py NVDA
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import vectorbtpro as vbt

LGT_DIR = Path("data/raw/lighter/5m")
YF_DIR = Path("data/raw/yfinance/1h")

# --- Strategy params (defaults; swept later) ---
ENTRY_THRESH = 0.005   # 0.5% drift from fair value to enter
EXIT_THRESH = 0.001    # gap considered filled below 0.1%
RECENT_MIN = 90        # real bar within 90min => market considered open
FEES = 0.001           # 10 bps per fill (Lighter integrator fee)
SLIPPAGE = 0.0003
INIT_CASH = 1000.0
YF_INTERVAL = "1h"     # yfinance underlying resolution


def load_aligned(sym: str):
    """Return a 5min-aligned DataFrame with lighter close + ffilled real fair value + market_open flag."""
    lgt_fp = LGT_DIR / f"{sym}.csv"
    yf_fp = YF_DIR / f"{sym}.csv"
    if not lgt_fp.exists() or not yf_fp.exists():
        return None

    lgt = pd.read_csv(lgt_fp)
    lgt["dt"] = pd.to_datetime(lgt["date"], unit="ms", utc=True)
    lgt = lgt.set_index("dt").sort_index()[["open", "high", "low", "close"]]

    yf = pd.read_csv(yf_fp)
    yf["dt"] = pd.to_datetime(yf["date"], unit="ms", utc=True)
    yf = yf.set_index("dt").sort_index()[["close"]].rename(columns={"close": "real_close"})
    yf = yf[~yf.index.duplicated(keep="last")]

    # --- ANTI-LOOK-AHEAD (donnée) ---
    # Un bar yfinance intraday est timestampé au DÉBUT de l'intervalle mais son `close`
    # est le prix de FIN d'intervalle. Un ffill brut associerait à 19:35 le close de
    # 20:00 (futur). On décale l'index de +YF_INTERVAL → le close n'est connu qu'après
    # complétion de la barre. `age_min` (et donc la détection marché-fermé) est mesuré
    # depuis cette fin de barre : conservateur, jamais optimiste.
    yf.index = yf.index + pd.Timedelta(YF_INTERVAL)

    # Common 5min grid over overlapping span
    start = max(lgt.index[0], yf.index[0])
    end = min(lgt.index[-1], yf.index[-1])
    if start >= end:
        return None
    grid = pd.date_range(start.ceil("5min"), end.floor("5min"), freq="5min", tz="UTC")
    df = pd.DataFrame(index=grid)

    # Lighter close on the grid (ffill within 5min, it's native 5min)
    df["lgt_close"] = lgt["close"].reindex(grid, method="ffill")

    # Real market: a bar present in the last RECENT_MIN => market open
    # Mark grid points that have a real bar within RECENT_MIN
    yf_reidx = yf["real_close"].reindex(grid, method="ffill")
    df["fair_value"] = yf_reidx  # last known real close
    # last real bar timestamp at/under each grid point (keep tz-aware int64 ns)
    yf_ts_ns = pd.Series(yf.index.view("int64"), index=yf.index).reindex(grid, method="ffill")
    grid_ns = pd.Series(grid.view("int64"), index=grid)
    age_min = (grid_ns - yf_ts_ns) / 1e9 / 60.0  # ns -> minutes
    df["market_open"] = (age_min <= RECENT_MIN).values

    df = df.dropna(subset=["lgt_close", "fair_value"])
    return df


def run_backtest(sym, entry_thresh=ENTRY_THRESH, exit_thresh=EXIT_THRESH,
                 fees=FEES, slippage=SLIPPAGE):
    df = load_aligned(sym)
    if df is None or len(df) < 500:
        return None

    fair = df["fair_value"].values
    px = df["lgt_close"].values
    closed = ~df["market_open"].values
    dev = (px - fair) / fair  # positive = Lighter above fair value

    # --- ANTI-LOOK-AHEAD (exécution) ---
    # Décision calculée sur la donnée du bar i, exécutée au close du bar i+1
    # (lag de 1 bar). Convention identique au kernel FUNDING_ARB_v1 (décide [i-1],
    # fill close[i]). Évite le biais intra-bar « je vois le close ET je remplis au
    # même close ».
    def _lag(a):
        return np.r_[False, a[:-1]]

    # Signals: enter fade only while market closed.
    # short_entry: closed & dev > +thresh (Lighter too high -> short, bet on drop)
    # long_entry : closed & dev < -thresh (Lighter too low  -> long, bet on rise)
    short_entries = _lag(closed & (dev > entry_thresh))
    long_entries = _lag(closed & (dev < -entry_thresh))

    # Exit when: market reopens (closed->open transition) OR gap filled (|dev|<exit_thresh)
    reopened = (~closed) & np.r_[False, closed[:-1]]  # transition closed->open
    gap_filled = np.abs(dev) < exit_thresh
    exits = _lag(reopened | gap_filled | (~closed))  # force-flat quand marché ouvert

    close_s = pd.Series(px, index=df.index)
    pf = vbt.Portfolio.from_signals(
        close=close_s,
        entries=pd.Series(long_entries, index=df.index),
        exits=pd.Series(exits, index=df.index),
        short_entries=pd.Series(short_entries, index=df.index),
        short_exits=pd.Series(exits, index=df.index),
        init_cash=INIT_CASH,
        fees=fees,
        slippage=slippage,
        freq="5min",
    )
    return pf


def main():
    sym = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    print(f"=== Gap-fill backtest: {sym} ===")
    df = load_aligned(sym)
    if df is None:
        print("No aligned data."); return
    closed_frac = (~df["market_open"]).mean()
    print(f"Aligned bars: {len(df)}  | span {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Market closed fraction: {closed_frac:.1%}")
    dev = (df['lgt_close'] - df['fair_value']) / df['fair_value']
    print(f"Deviation stats (closed only): mean={dev[~df['market_open']].mean()*100:+.3f}% "
          f"std={dev[~df['market_open']].std()*100:.3f}% "
          f"max={dev[~df['market_open']].abs().max()*100:.2f}%")

    pf = run_backtest(sym)
    if pf is None:
        print("Backtest failed."); return
    st = pf.stats()
    def g(k, d=0):
        try:
            v = st[k]
            return float(v) if not pd.isna(v) else d
        except Exception:
            return d
    print(f"\n=== Results ===")
    print(f"  Total Return  : {g('Total Return [%]'):+.2f}%")
    print(f"  Sharpe        : {g('Sharpe Ratio'):.2f}")
    print(f"  Max Drawdown  : {g('Max Drawdown [%]'):.2f}%")
    print(f"  Total Trades  : {int(g('Total Trades'))}")
    print(f"  Win Rate      : {g('Win Rate [%]'):.1f}%")
    print(f"  Profit Factor : {g('Profit Factor'):.2f}")
    print(f"  Avg Trade     : {g('Avg Winning Trade [%]'):.3f}% W / {g('Avg Losing Trade [%]'):.3f}% L")


if __name__ == "__main__":
    main()
