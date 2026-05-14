"""Data loaders for BTYZ. Supports Lighter 1m and Binance UM."""
import gc
from functools import lru_cache

import pandas as pd

from .config import DATA_ROOT, FREQ_MAP


@lru_cache(maxsize=32)
def load_lighter(pair: str, tf: str = "15min") -> pd.DataFrame | None:
    """Load OHLCV from data/raw/lighter/<tf>/<PAIR>.csv (direct if exists, else resample from 1m)."""
    base = pair.replace("USDT", "")
    tf_folder = FREQ_MAP.get(tf, tf).replace("min", "m")

    # Try direct pre-built folder first (faster, less RAM)
    fp_direct = DATA_ROOT / "lighter" / tf_folder / f"{base}.csv"
    if fp_direct.exists():
        df = pd.read_csv(fp_direct, low_memory=False,
                         usecols=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        df = df.set_index("date").sort_index()
        return df if len(df) >= 1000 else None

    # Fallback: resample from 1m
    for sub in ("1m", "1min"):
        fp = DATA_ROOT / "lighter" / sub / f"{base}.csv"
        if fp.exists():
            break
    else:
        return None

    df = pd.read_csv(fp, low_memory=False, usecols=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    df = df.set_index("date").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ohlc = df.resample(FREQ_MAP.get(tf, tf), label="left", closed="left").agg(agg).dropna()
    return ohlc if len(ohlc) >= 1000 else None


@lru_cache(maxsize=32)
def load_binance(pair: str, tf: str = "15min") -> pd.DataFrame | None:
    """Load OHLCV from data/raw/binance/{tf}/<PAIR>USDT.csv (already resampled)."""
    pair_usdt = pair if pair.endswith("USDT") else f"{pair}USDT"
    tf_folder = FREQ_MAP.get(tf, tf).replace("min", "m")
    fp = DATA_ROOT / "binance" / tf_folder / f"{pair_usdt}.csv"
    if not fp.exists():
        # fallback on 1m + resample
        fp_1m = DATA_ROOT / "binance" / "1m" / f"{pair_usdt}.csv"
        if not fp_1m.exists():
            return None
        df = pd.read_csv(fp_1m, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        df = df.set_index("date").sort_index()
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ohlc = df.resample(FREQ_MAP.get(tf, tf), label="left", closed="left").agg(agg).dropna()
        return ohlc if len(ohlc) >= 1000 else None

    df = pd.read_csv(fp, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True) if df["date"].dtype.kind in "iu" else pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep] if len(df) >= 1000 else None


def load_ohlcv(pair: str, tf: str = "15min", source: str = "auto") -> pd.DataFrame | None:
    """Try Lighter then Binance (or force one)."""
    if source == "lighter":
        return load_lighter(pair, tf)
    if source == "binance":
        return load_binance(pair, tf)
    lt = load_lighter(pair, tf)
    if lt is not None:
        return lt
    return load_binance(pair, tf)


def clear_cache():
    load_lighter.cache_clear()
    load_binance.cache_clear()
    gc.collect()
