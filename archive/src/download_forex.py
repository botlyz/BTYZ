"""Download real underlying prices for Lighter tokenized TradFi assets via yfinance.

Maps each Lighter token (NVDA, WTI, XAU, SPY, US500, EURUSD, ...) to its real
yfinance ticker, then downloads OHLCV. Used for the weekend gap-fill alpha:
compare Lighter 24/7 price vs real market close/open to detect mean-reversion.

yfinance interval limits:
  - 1m  : last 7 days
  - 5m  : last 60 days
  - 1h  : last 730 days   <- default (best history/resolution tradeoff)
  - 1d  : full history

Output: data/raw/yfinance/<interval>/<LIGHTER_SYMBOL>.csv
        columns: date(ms),open,high,low,close,volume  (aligned with Lighter format)

Usage:
  python src/download_forex.py                  # all candidates, 1h, 730d
  python src/download_forex.py --interval 1d    # daily, full history
  python src/download_forex.py NVDA WTI XAU     # specific symbols
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ---- Mapping: Lighter token -> (yfinance ticker, category, trading window) ----
# category: equity / etf / index / commodity / forex
# Note: SPACEX/BMNR/CRCL/etc are pre-IPO or private -> no public underlying -> skip
MAPPING = {
    # --- Equities (NASDAQ/NYSE, 9:30-16:00 ET) ---
    "NVDA":  ("NVDA", "equity"),
    "TSLA":  ("TSLA", "equity"),
    "AAPL":  ("AAPL", "equity"),
    "MSFT":  ("MSFT", "equity"),
    "GOOGL": ("GOOGL", "equity"),
    "AMZN":  ("AMZN", "equity"),
    "META":  ("META", "equity"),
    "AMD":   ("AMD", "equity"),
    "INTC":  ("INTC", "equity"),
    "MU":    ("MU", "equity"),
    "ORCL":  ("ORCL", "equity"),
    "PLTR":  ("PLTR", "equity"),
    "COIN":  ("COIN", "equity"),
    "MSTR":  ("MSTR", "equity"),
    "HOOD":  ("HOOD", "equity"),
    "CRWV":  ("CRWV", "equity"),
    "MRVL":  ("MRVL", "equity"),
    "SNDK":  ("SNDK", "equity"),
    "TSM":   ("TSM", "equity"),       # Taiwan Semi ADR (US-listed)
    "ASML":  ("ASML", "equity"),      # ASML ADR (US-listed)
    "GME":   ("GME", "equity"),
    "RKLB":  ("RKLB", "equity"),      # Rocket Lab
    "BABA":  ("BABA", "equity"),      # Alibaba ADR
    "CRCL":  ("CRCL", "equity"),      # Circle (may be recent IPO)
    # --- Sector ETFs ---
    "SOXX":  ("SOXX", "etf"),         # semiconductors
    "BOTZ":  ("BOTZ", "etf"),         # robotics/AI
    "ROBO":  ("ROBO", "etf"),         # robotics
    "URA":   ("URA", "etf"),          # uranium
    "EWY":   ("EWY", "etf"),          # South Korea
    "MAGS":  ("MAGS", "etf"),         # Magnificent 7
    # --- Broad ETFs / Index ---
    "SPY":   ("SPY", "etf"),          # S&P 500 ETF (~1/10 of index)
    "QQQ":   ("QQQ", "etf"),          # NASDAQ-100 ETF
    "IWM":   ("IWM", "etf"),          # Russell 2000 ETF
    "US500": ("ES=F", "index"),       # S&P 500 FUTURES (quasi 24/5 -> fermé week-end seulement ; ^GSPC cash = piège price discovery nocturne)
    "US100": ("NQ=F", "index"),       # NASDAQ-100 FUTURES (idem)
    "SPX":   ("^GSPC", "index"),      # S&P 500 index
    # --- Commodities (CME futures front-month, ~24/5) ---
    "WTI":      ("CL=F", "commodity"),   # crude oil West Texas
    "BRENTOIL": ("BZ=F", "commodity"),   # crude oil Brent
    "NATGAS":   ("NG=F", "commodity"),   # natural gas
    "XAU":      ("GC=F", "commodity"),   # gold
    "XAG":      ("SI=F", "commodity"),   # silver
    "XPT":      ("PL=F", "commodity"),   # platinum
    "XPD":      ("PA=F", "commodity"),   # palladium
    "XCU":      ("HG=F", "commodity"),   # copper
    "WHEAT":    ("ZW=F", "commodity"),   # wheat
    "CC":       ("CC=F", "commodity"),   # cocoa
    # --- Forex (24/5) ---
    "EURUSD": ("EURUSD=X", "forex"),
    "GBPUSD": ("GBPUSD=X", "forex"),
    "USDJPY": ("USDJPY=X", "forex"),
    "USDCHF": ("USDCHF=X", "forex"),
    "AUDUSD": ("AUDUSD=X", "forex"),
    "NZDUSD": ("NZDUSD=X", "forex"),
    "USDCAD": ("USDCAD=X", "forex"),
    "USDKRW": ("USDKRW=X", "forex"),
}

PERIOD_BY_INTERVAL = {
    "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
    "1h": "730d", "1d": "max",
}


def download_one(lighter_sym, yf_ticker, interval, out_dir):
    period = PERIOD_BY_INTERVAL.get(interval, "730d")
    try:
        df = yf.Ticker(yf_ticker).history(period=period, interval=interval, auto_adjust=False)
    except Exception as e:
        return None, f"ERR {e}"
    if df is None or df.empty:
        return None, "empty"
    df = df.reset_index()
    # The datetime column is named 'Datetime' (intraday) or 'Date' (daily)
    dt_col = "Datetime" if "Datetime" in df.columns else "Date"
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True)
    out = pd.DataFrame({
        "date": (df[dt_col].astype("int64") // 1_000_000),  # ns -> ms
        "open": df["Open"].astype(float),
        "high": df["High"].astype(float),
        "low": df["Low"].astype(float),
        "close": df["Close"].astype(float),
        "volume": df["Volume"].astype(float),
    }).dropna(subset=["close"])
    fp = out_dir / f"{lighter_sym}.csv"
    out.to_csv(fp, index=False)
    return len(out), str(fp.name)


def main():
    args = sys.argv[1:]
    interval = "1h"
    symbols = []
    i = 0
    while i < len(args):
        if args[i] == "--interval":
            interval = args[i + 1]; i += 2
        else:
            symbols.append(args[i].upper()); i += 1

    targets = {s: MAPPING[s] for s in symbols if s in MAPPING} if symbols else MAPPING
    if symbols and not targets:
        print(f"No valid symbols in {symbols}. Available: {sorted(MAPPING)}")
        return

    out_dir = Path("data/raw/yfinance") / interval
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(targets)} TradFi underlyings @ {interval} -> {out_dir}\n")

    results = []
    for lighter_sym, (yf_ticker, cat) in targets.items():
        n, msg = download_one(lighter_sym, yf_ticker, interval, out_dir)
        status = f"{n} rows" if n else f"FAIL ({msg})"
        print(f"  {lighter_sym:10} -> {yf_ticker:10} [{cat:9}] {status}")
        results.append((lighter_sym, cat, n))
        time.sleep(0.3)  # be polite to yahoo

    ok = [r for r in results if r[2]]
    print(f"\nDone: {len(ok)}/{len(results)} downloaded to {out_dir}")
    failed = [r[0] for r in results if not r[2]]
    if failed:
        print(f"Failed (no data / delisted / pre-IPO): {failed}")


if __name__ == "__main__":
    main()
