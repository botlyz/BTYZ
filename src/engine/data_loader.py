"""Data loaders for BTYZ. Supports Lighter 1m and Binance UM."""
import gc
from functools import lru_cache

import pandas as pd

from .config import DATA_ROOT, FREQ_MAP


def _maybe_attach_funding(df: pd.DataFrame, pair: str, tf: str) -> pd.DataFrame:
    """Si data/raw/lighter/funding/<pair>.csv existe, merge signed_rate + apr (1h native).
    Le funding est aligné par forward-fill sur l'index OHLC. Si tf > 1h, on prend la
    dernière valeur funding de chaque période. Si tf < 1h (rare), on propage la même
    valeur sur chaque sous-bar.
    """
    base = pair.replace("USDT", "")
    fp_fund = DATA_ROOT / "lighter" / "funding" / f"{base}.csv"
    if not fp_fund.exists():
        return df
    try:
        fdg = pd.read_csv(fp_fund, low_memory=False,
                          usecols=["timestamp", "signed_rate", "apr"])
        fdg["ts"] = pd.to_datetime(fdg["timestamp"].astype(int), unit="s", utc=True)
        fdg = fdg.set_index("ts").sort_index().drop(columns=["timestamp"])
    except Exception:
        return df
    # Align funding (1h) to df's index. For 1h tf the index matches exactly.
    # For other tf we reindex with ffill (a funding value is constant over its hour).
    merged = df.join(fdg, how="left")
    merged["signed_rate"] = merged["signed_rate"].ffill()
    merged["apr"] = merged["apr"].ffill()
    return merged


def _hl_symbol_for(pair: str) -> str | None:
    """Résout le symbole Hyperliquid pour une paire normalisée (clé common_pairs.json).

    Ex: 'BTC' → 'BTC', '1000PEPE' → 'kPEPE', 'HYPE' → 'HYPE'.
    Renvoie None si la paire n'existe pas sur HL.
    """
    import json
    cp_path = DATA_ROOT / "common_pairs.json"
    if not cp_path.exists():
        return None
    try:
        with open(cp_path) as f:
            common = json.load(f)
    except Exception:
        return None
    base = pair.replace("USDT", "")
    info = common.get(base)
    if info is None:
        return None
    return info.get("hyperliquid")


def _attach_hl_funding(df: pd.DataFrame, pair: str) -> pd.DataFrame | None:
    """Merge le funding Hyperliquid à df (qui contient déjà signed_rate Lighter).

    Renvoie un DF avec colonnes:
      - signed_rate_lighter (renommée depuis signed_rate)
      - signed_rate_hl
      - spread = signed_rate_lighter - signed_rate_hl
    Drop les lignes où l'un des deux est NaN (intersection des historiques).
    None si HL funding indisponible.
    """
    coin = _hl_symbol_for(pair)
    if coin is None:
        return None
    fp_hl = DATA_ROOT / "hyperliquid" / "funding" / f"{coin}.csv"
    if not fp_hl.exists():
        return None
    try:
        hl = pd.read_csv(fp_hl, low_memory=False, usecols=["timestamp", "signed_rate"])
        hl["ts"] = pd.to_datetime(hl["timestamp"].astype(int), unit="s", utc=True)
        hl = hl.set_index("ts").sort_index().drop(columns=["timestamp"])
        hl = hl.rename(columns={"signed_rate": "signed_rate_hl"})
    except Exception:
        return None

    if "signed_rate" not in df.columns:
        return None
    out = df.rename(columns={"signed_rate": "signed_rate_lighter"}).copy()
    out = out.join(hl, how="left")
    # Intersection stricte : on a besoin des 2 funding pour calculer spread
    out = out.dropna(subset=["signed_rate_lighter", "signed_rate_hl"])
    out["spread"] = out["signed_rate_lighter"] - out["signed_rate_hl"]
    # apr resté en colonne mais devient le Lighter only — drop pour éviter confusion
    if "apr" in out.columns:
        out = out.drop(columns=["apr"])
    return out


@lru_cache(maxsize=32)
def load_cross_exchange(pair: str, tf: str = "1h") -> pd.DataFrame | None:
    """Loader cross-exchange Lighter + Hyperliquid funding pour FUNDING_ARB_v1.

    Charge OHLCV Lighter, attache funding Lighter, puis merge funding HL.
    Le résultat ne contient que les bars où les 2 venues ont du funding (intersection).
    Colonnes: open, high, low, close, volume, signed_rate_lighter, signed_rate_hl, spread.
    """
    df = load_lighter(pair, tf)
    if df is None or "signed_rate" not in df.columns:
        return None
    return _attach_hl_funding(df, pair)


@lru_cache(maxsize=32)
def load_lighter(pair: str, tf: str = "15min") -> pd.DataFrame | None:
    """Load OHLCV from data/raw/lighter/<tf>/<PAIR>.csv (direct if exists, else resample from 1m).

    Si un fichier funding existe (data/raw/lighter/funding/<PAIR>.csv), 2 colonnes
    supplémentaires sont mergées : signed_rate et apr (utilisées par FUNDING_HARVEST_v1).
    Backward-compat : les autres strats ignorent ces colonnes."""
    base = pair.replace("USDT", "")
    tf_folder = FREQ_MAP.get(tf, tf).replace("min", "m")

    # Try direct pre-built folder first (faster, less RAM)
    fp_direct = DATA_ROOT / "lighter" / tf_folder / f"{base}.csv"
    if fp_direct.exists():
        df = pd.read_csv(fp_direct, low_memory=False,
                         usecols=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        df = df.set_index("date").sort_index()
        if len(df) < 1000:
            return None
        return _maybe_attach_funding(df, pair, tf)

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
    if len(ohlc) < 1000:
        return None
    return _maybe_attach_funding(ohlc, pair, tf)


@lru_cache(maxsize=32)
def load_gapfill(pair: str, tf: str = "5min", yf_interval: str = "1h") -> pd.DataFrame | None:
    """Loader weekend/off-hours gap-fill : Lighter perp 24/7 vs sous-jacent réel (yfinance).

    Aligne sur une grille `tf` (5min par défaut) :
      - open/high/low/close  : prix Lighter (le perp qu'on TRADE), ffill
      - fair_value           : dernier close réel CONNU (yfinance ffill)
      - real_age_min         : âge en minutes du dernier bar réel COMPLÉTÉ

    ANTI-LOOK-AHEAD (donnée) : un bar yfinance intraday est timestampé au DÉBUT
    de l'intervalle mais son `close` est le prix de FIN d'intervalle. Un ffill brut
    associerait donc à 19:35 le close de 20:00 (futur). On décale l'index yfinance
    de +`yf_interval` (fin de barre) → le close n'est « connu » qu'une fois la barre
    terminée. `real_age_min` est mesuré depuis cette fin de barre. Conséquence : la
    détection « marché fermé » est légèrement conservatrice (on attend la complétion),
    jamais optimiste. Le filtre marché-ouvert/fermé est appliqué dans la stratégie via
    `real_age_min > recent_min`.

    Colonnes toutes numériques (compat WFA/MCCV qui sérialise via DataFrame.values).
    """
    base = pair.replace("USDT", "")
    vbt_freq = FREQ_MAP.get(tf, tf)

    lgt = load_lighter(pair, tf)
    if lgt is None:
        return None

    yf_fp = DATA_ROOT / "yfinance" / yf_interval / f"{base}.csv"
    if not yf_fp.exists():
        return None
    try:
        yf = pd.read_csv(yf_fp, low_memory=False, usecols=["date", "close"])
    except Exception:
        return None
    yf["dt"] = pd.to_datetime(yf["date"], unit="ms", utc=True)
    yf = yf.set_index("dt").sort_index()[["close"]].rename(columns={"close": "real_close"})
    yf = yf[~yf.index.duplicated(keep="last")]
    if len(yf) < 50:
        return None

    # --- ANTI-LOOK-AHEAD : décale le bar à sa FIN (close connu après complétion) ---
    yf.index = yf.index + pd.Timedelta(yf_interval)

    start = max(lgt.index[0], yf.index[0])
    end = min(lgt.index[-1], yf.index[-1])
    if start >= end:
        return None
    grid = pd.date_range(start.ceil(vbt_freq), end.floor(vbt_freq), freq=vbt_freq, tz="UTC")
    if len(grid) < 1000:
        return None

    out = pd.DataFrame(index=grid)
    for col in ("open", "high", "low", "close"):
        out[col] = lgt[col].reindex(grid, method="ffill")
    out["fair_value"] = yf["real_close"].reindex(grid, method="ffill")
    # Âge du dernier bar réel complété (depuis sa FIN, cf. shift ci-dessus)
    yf_ts_ns = pd.Series(yf.index.view("int64"), index=yf.index).reindex(grid, method="ffill")
    grid_ns = pd.Series(grid.view("int64"), index=grid)
    out["real_age_min"] = ((grid_ns - yf_ts_ns) / 1e9 / 60.0).values

    out = out.dropna(subset=["open", "high", "low", "close", "fair_value", "real_age_min"])
    if len(out) < 1000:
        return None
    return out


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
    if source == "cross_exchange":
        return load_cross_exchange(pair, tf)
    if source == "gapfill":
        return load_gapfill(pair, tf)
    lt = load_lighter(pair, tf)
    if lt is not None:
        return lt
    return load_binance(pair, tf)


def clear_cache():
    load_lighter.cache_clear()
    load_binance.cache_clear()
    load_cross_exchange.cache_clear()
    load_gapfill.cache_clear()
    gc.collect()
