"""
Téléchargement des données Binance Public Data (S3, sans API key, sans rate limit)
Sources : UM Futures, Spot, CM Futures — granularité 1m
Données supplémentaires UM : funding rate (8h), métriques OI + ratios L/S (5min)

Incrémental : lit le dernier timestamp de chaque CSV et ne télécharge que la suite.

Structure :
    data/raw/binance/um/1m/{symbol}.csv
    data/raw/binance/spot/1m/{symbol}.csv
    data/raw/binance/cm/1m/{symbol}.csv
    data/raw/binance/um/funding/{symbol}.csv
    data/raw/binance/um/metrics/{symbol}.csv

Usage :
    python src/download_binance.py                        # tout (incrémental)
    python src/download_binance.py --type klines          # klines seulement
    python src/download_binance.py --type funding         # funding seulement
    python src/download_binance.py --type metrics         # métriques seulement
    python src/download_binance.py --pairs BTC ETH SOL   # sélection de paires
    python src/download_binance.py --source um            # source klines uniquement
    python src/download_binance.py --verify               # vérification seule
    python src/download_binance.py --no-skip              # force re-téléchargement complet
"""

import asyncio
import aiohttp
import io
import zipfile
import csv
import os
import json
import argparse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────

BASE_URL    = "https://data.binance.vision/data"
OUT_DIR     = "./data/raw/binance"
CONCURRENCY = 200
RETRY_MAX   = 3
RETRY_DELAY = 2

# Klines
KLINE_IDX  = [0, 1, 2, 3, 4, 5, 9]
KLINE_COLS = ["date", "open", "high", "low", "close", "volume", "taker_buy_volume"]

# Funding rate
FUNDING_COLS = ["date", "funding_rate"]

# Métriques OI + ratios (colonnes S3)
# 0:create_time 1:symbol 2:sum_oi 3:sum_oi_value 4:count_top_ls 5:sum_top_ls
# 6:count_ls 7:sum_taker_ls_vol
METRICS_IDX  = [2, 3, 5, 6, 7]   # indices après conversion de la date
METRICS_COLS = ["date", "oi", "oi_value", "top_trader_ls", "ls_ratio", "taker_ls_vol"]

# ─── PAIRES ──────────────────────────────────────────────────────────────────

def get_live_um_pairs():
    url  = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    data = json.loads(urllib.request.urlopen(url, timeout=10).read())
    return sorted(s["symbol"] for s in data["symbols"]
                  if s["status"] == "TRADING" and s["symbol"].endswith("USDT"))

def get_live_cm_pairs():
    url  = "https://dapi.binance.com/dapi/v1/exchangeInfo"
    data = json.loads(urllib.request.urlopen(url, timeout=10).read())
    return sorted(set(
        s["pair"] + "_PERP" for s in data["symbols"]
        if s.get("contractType") == "PERPETUAL"
        and s.get("contractStatus") == "TRADING"
    ))

def get_spot_pairs():
    url = ("https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
           "?delimiter=/&prefix=data/spot/monthly/klines/&max-keys=2000")
    from xml.etree import ElementTree as ET
    root = ET.fromstring(urllib.request.urlopen(url, timeout=15).read())
    ns   = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    return sorted(
        p.find(f"{ns}Prefix").text.split("/")[-2]
        for p in root.findall(f"{ns}CommonPrefixes")
        if p.find(f"{ns}Prefix").text.split("/")[-2].endswith("USDT")
    )

def _get_s3_pairs(prefix):
    url = (f"https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
           f"?delimiter=/&prefix={prefix}&max-keys=2000")
    from xml.etree import ElementTree as ET
    root = ET.fromstring(urllib.request.urlopen(url, timeout=15).read())
    ns   = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    return sorted(
        p.find(f"{ns}Prefix").text.split("/")[-2]
        for p in root.findall(f"{ns}CommonPrefixes")
    )

def get_funding_pairs():
    return _get_s3_pairs("data/futures/um/monthly/fundingRate/")

def get_metrics_pairs():
    return _get_s3_pairs("data/futures/um/daily/metrics/")

# ─── TIMESTAMP HELPERS ───────────────────────────────────────────────────────

def to_ms(ts_str):
    """Normalise un timestamp Binance en ms. Spot daily = µs (16 ch) → ÷1000."""
    n = int(ts_str)
    return n // 1000 if len(ts_str) == 16 else n

def get_last_ts(path):
    """Retourne le dernier timestamp ms du CSV (normalisé), ou 0 si absent/vide."""
    if not path.exists() or path.stat().st_size < 50:
        return 0
    with open(path, "rb") as f:
        try:
            f.seek(-512, 2)
        except OSError:
            f.seek(0)
        lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")
    for line in reversed(lines):
        parts = line.strip().split(",")
        if parts and parts[0].isdigit() and len(parts[0]) in (13, 16):
            return to_ms(parts[0])
    return 0

def month_end_ms(year, month):
    """Ms timestamp de la dernière candle du mois (dernier minute = début mois suivant - 1min)."""
    if month == 12:
        nxt = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        nxt = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return int(nxt.timestamp() * 1000) - 60_000

def next_month(year, month):
    return (year + 1, 1) if month == 12 else (year, month + 1)

# ─── CONSTRUCTEURS D'URLS ────────────────────────────────────────────────────

def _kline_base(source, symbol, freq):
    if source == "um":
        return f"{BASE_URL}/futures/um/{freq}/klines/{symbol}/1m"
    if source == "spot":
        return f"{BASE_URL}/spot/{freq}/klines/{symbol}/1m"
    if source == "cm":
        return f"{BASE_URL}/futures/cm/{freq}/klines/{symbol}/1m"

def build_kline_urls(source, symbol, last_ts):
    today            = date.today()
    last_month_start = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    urls             = []

    # Fichiers mensuels
    if last_ts == 0:
        d = date(2020, 1, 1)
    else:
        last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).date()
        d = last_dt.replace(day=1)  # re-télécharger le mois partiel

    while d <= last_month_start:
        if month_end_ms(d.year, d.month) > last_ts:
            fname = f"{symbol}-1m-{d.year}-{d.month:02d}.zip"
            urls.append(f"{_kline_base(source, symbol, 'monthly')}/{fname}")
        y, m = next_month(d.year, d.month)
        d = date(y, m, 1)

    # Fichiers journaliers (mois courant uniquement)
    if last_ts == 0:
        d = today.replace(day=1)
    else:
        last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).date()
        d = max(today.replace(day=1), last_dt + timedelta(days=1))

    end = today - timedelta(days=1)
    while d <= end:
        fname = f"{symbol}-1m-{d.strftime('%Y-%m-%d')}.zip"
        urls.append(f"{_kline_base(source, symbol, 'daily')}/{fname}")
        d += timedelta(days=1)

    return urls

def build_funding_urls(symbol, last_ts):
    today            = date.today()
    last_month_start = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    base             = f"{BASE_URL}/futures/um/monthly/fundingRate/{symbol}"
    urls             = []

    if last_ts == 0:
        d = date(2020, 1, 1)
    else:
        last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).date()
        d = last_dt.replace(day=1)

    while d <= last_month_start:
        if month_end_ms(d.year, d.month) > last_ts:
            fname = f"{symbol}-fundingRate-{d.year}-{d.month:02d}.zip"
            urls.append(f"{base}/{fname}")
        y, m = next_month(d.year, d.month)
        d = date(y, m, 1)

    return urls

def build_metrics_urls(symbol, last_ts):
    today = date.today()
    base  = f"{BASE_URL}/futures/um/daily/metrics/{symbol}"
    urls  = []

    if last_ts == 0:
        d = date(2020, 9, 1)  # métriques disponibles depuis sept 2020
    else:
        last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).date()
        d = last_dt + timedelta(days=1)

    end = today - timedelta(days=1)
    while d <= end:
        fname = f"{symbol}-metrics-{d.strftime('%Y-%m-%d')}.zip"
        urls.append(f"{base}/{fname}")
        d += timedelta(days=1)

    return urls

# ─── PARSERS ─────────────────────────────────────────────────────────────────

def parse_kline_zip(zip_bytes):
    rows = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                parts = line.rstrip("\n").split(",")
                if not parts[0].isdigit():
                    continue
                row = [parts[i] for i in KLINE_IDX]
                row[0] = str(to_ms(row[0]))  # normalise µs→ms (spot daily)
                rows.append(row)
    return rows

def parse_funding_zip(zip_bytes):
    # format: calc_time, funding_interval_hours, last_funding_rate
    rows = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                parts = line.rstrip("\n").split(",")
                if not parts[0].isdigit():
                    continue
                rows.append([parts[0], parts[2]])
    return rows

def parse_metrics_zip(zip_bytes):
    # format: create_time, symbol, sum_oi, sum_oi_value, count_top_ls,
    #         sum_top_ls, count_ls, sum_taker_ls_vol
    rows = []
    seen = set()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                parts = line.rstrip("\n").split(",")
                if not parts or not parts[0][0:4].isdigit():
                    continue
                try:
                    dt    = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    ts_ms = str(int(dt.timestamp() * 1000))
                except ValueError:
                    continue
                if ts_ms in seen:
                    continue
                seen.add(ts_ms)
                rows.append([ts_ms] + [parts[i] for i in METRICS_IDX])
    return rows

# ─── FETCH ───────────────────────────────────────────────────────────────────

async def fetch(session, url, semaphore):
    async with semaphore:
        for attempt in range(RETRY_MAX):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as r:
                    if r.status == 404:
                        return None
                    if r.status == 200:
                        return await r.read()
            except Exception:
                if attempt < RETRY_MAX - 1:
                    await asyncio.sleep(RETRY_DELAY)
    return None

# ─── ÉCRITURE INCRÉMENTALE ───────────────────────────────────────────────────

def append_rows(out_path, rows, cols, last_ts):
    """Filtre les lignes après last_ts, déduplique, trie, et append au CSV."""
    new_rows = [r for r in rows if int(r[0]) > last_ts]
    if not new_rows:
        return 0

    seen, unique = set(), []
    for row in new_rows:
        if row[0] not in seen:
            seen.add(row[0])
            unique.append(row)
    unique.sort(key=lambda x: int(x[0]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(cols)
        w.writerows(unique)
    return len(unique)

# ─── DOWNLOAD PAR SYMBOLE ────────────────────────────────────────────────────

async def dl_klines(session, sem, source, symbol, out_path, pbar):
    last_ts = get_last_ts(out_path)
    urls    = build_kline_urls(source, symbol, last_ts)

    results  = await asyncio.gather(*[fetch(session, u, sem) for u in urls])
    all_rows = []
    for data in results:
        if data:
            try:
                all_rows.extend(parse_kline_zip(data))
            except Exception:
                pass

    n = append_rows(out_path, all_rows, KLINE_COLS, last_ts)
    pbar.update(1)
    return symbol, n

async def dl_funding(session, sem, symbol, out_path, pbar):
    last_ts = get_last_ts(out_path)
    urls    = build_funding_urls(symbol, last_ts)

    results  = await asyncio.gather(*[fetch(session, u, sem) for u in urls])
    all_rows = []
    for data in results:
        if data:
            try:
                all_rows.extend(parse_funding_zip(data))
            except Exception:
                pass

    # Mois courant non disponible sur S3 → API REST (limit 1000 = ~333 jours)
    try:
        start_ms = last_ts if last_ts > 0 else 1577836800000
        api_url  = (f"https://fapi.binance.com/fapi/v1/fundingRate"
                    f"?symbol={symbol}&startTime={start_ms}&limit=1000")
        api_data = json.loads(urllib.request.urlopen(api_url, timeout=10).read())
        for item in api_data:
            all_rows.append([str(item["fundingTime"]), str(item["fundingRate"])])
    except Exception:
        pass

    n = append_rows(out_path, all_rows, FUNDING_COLS, last_ts)
    pbar.update(1)
    return symbol, n

async def dl_metrics(session, sem, symbol, out_path, pbar):
    last_ts = get_last_ts(out_path)
    urls    = build_metrics_urls(symbol, last_ts)

    results  = await asyncio.gather(*[fetch(session, u, sem) for u in urls])
    all_rows = []
    for data in results:
        if data:
            try:
                all_rows.extend(parse_metrics_zip(data))
            except Exception:
                pass

    n = append_rows(out_path, all_rows, METRICS_COLS, last_ts)
    pbar.update(1)
    return symbol, n

# ─── ORCHESTRATION ───────────────────────────────────────────────────────────

def repair_spot_csvs(symbols):
    """Corrige les timestamps µs (16 chiffres) → ms dans les CSV spot existants."""
    fixed = 0
    for sym in tqdm(symbols, desc="Réparation spot timestamps", unit="paire"):
        path = Path(OUT_DIR) / "spot" / "1m" / f"{sym}.csv"
        if not path.exists():
            continue
        # Vérifie si réparation nécessaire (lit les dernières lignes)
        with open(path, "rb") as f:
            try:
                f.seek(-256, 2)
            except OSError:
                f.seek(0)
            tail = f.read().decode("utf-8", errors="ignore")
        if not any(len(p.split(",")[0]) == 16 for p in tail.strip().split("\n")
                   if p and p[0].isdigit()):
            continue
        # Lecture et normalisation complète
        rows, header = [], None
        with open(path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if row and len(row[0]) == 16 and row[0].isdigit():
                    row[0] = str(int(row[0]) // 1000)
                rows.append(row)
        # Dedup + tri
        seen, unique = set(), []
        for row in rows:
            if row[0] not in seen:
                seen.add(row[0])
                unique.append(row)
        unique.sort(key=lambda x: int(x[0]))
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(unique)
        fixed += 1
    if fixed:
        print(f"  {fixed} fichier(s) spot réparé(s).")

async def run_klines(pairs_by_source, force):
    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        for source, symbols in pairs_by_source.items():
            if not symbols:
                print(f"[{source}] Aucune paire.")
                continue
            print(f"\n{'='*60}")
            print(f"  KLINES {source.upper()} — {len(symbols)} paires")
            print(f"{'='*60}")
            if force:
                for sym in symbols:
                    p = Path(OUT_DIR) / source / "1m" / f"{sym}.csv"
                    p.unlink(missing_ok=True)
            elif source == "spot":
                repair_spot_csvs(symbols)
            with tqdm(total=len(symbols), desc=source.upper(), unit="paire",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                tasks = [
                    dl_klines(session, semaphore, source, sym,
                              Path(OUT_DIR) / source / "1m" / f"{sym}.csv", pbar)
                    for sym in symbols
                ]
                await asyncio.gather(*tasks)

async def run_funding(symbols, force):
    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"\n{'='*60}")
        print(f"  FUNDING RATE UM — {len(symbols)} paires")
        print(f"{'='*60}")
        if force:
            for sym in symbols:
                Path(OUT_DIR, "um", "funding", f"{sym}.csv").unlink(missing_ok=True)
        with tqdm(total=len(symbols), desc="FUNDING", unit="paire",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            tasks = [
                dl_funding(session, semaphore, sym,
                           Path(OUT_DIR) / "um" / "funding" / f"{sym}.csv", pbar)
                for sym in symbols
            ]
            await asyncio.gather(*tasks)

async def run_metrics(symbols, force):
    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"\n{'='*60}")
        print(f"  MÉTRIQUES UM (OI + L/S ratios) — {len(symbols)} paires")
        print(f"{'='*60}")
        if force:
            for sym in symbols:
                Path(OUT_DIR, "um", "metrics", f"{sym}.csv").unlink(missing_ok=True)
        with tqdm(total=len(symbols), desc="METRICS", unit="paire",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            tasks = [
                dl_metrics(session, semaphore, sym,
                           Path(OUT_DIR) / "um" / "metrics" / f"{sym}.csv", pbar)
                for sym in symbols
            ]
            await asyncio.gather(*tasks)

# ─── VÉRIFICATION ────────────────────────────────────────────────────────────

def verify_dir(label, path, expected_gap_ms, min_rows):
    if not path.exists():
        return []
    files  = sorted(path.glob("*.csv"))
    issues = []
    print(f"\n[{label}] {len(files)} fichiers")
    for fpath in tqdm(files, desc=f"Vérif {label}", unit="fichier"):
        rows, prev_ts, max_gap, gap_count = 0, None, 0, 0
        with open(fpath) as f:
            reader = csv.reader(f)
            next(reader, None)
            for line in reader:
                if not line or not line[0].isdigit():
                    continue
                rows += 1
                ts = int(line[0])
                if prev_ts is not None:
                    gap = ts - prev_ts
                    if gap > expected_gap_ms * 2:
                        gap_count += 1
                        max_gap = max(max_gap, gap)
                prev_ts = ts
        if rows < min_rows:
            issues.append(f"⚠ {fpath.name}: seulement {rows} lignes")
        if max_gap > expected_gap_ms * 60:  # gap > 60× l'intervalle attendu
            issues.append(f"⚠ {fpath.name}: gap max {max_gap//expected_gap_ms} intervalles ({gap_count} gaps)")
    return issues

def verify(sources):
    print("\n=== Vérification des données ===")
    issues = []
    for src in sources:
        issues += verify_dir(f"{src}/1m",      Path(OUT_DIR) / src / "1m",      60_000,      1_000)
    issues += verify_dir("um/funding",  Path(OUT_DIR) / "um" / "funding",  28_800_000,  100)
    issues += verify_dir("um/metrics",  Path(OUT_DIR) / "um" / "metrics",  300_000,     100)
    if issues:
        print(f"\n{len(issues)} anomalie(s) :")
        for issue in issues:
            print(" ", issue)
    else:
        print("\n✅ Aucune anomalie détectée.")

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs",   nargs="*", help="Paires ex: BTC ETH → BTCUSDT")
    parser.add_argument("--source",  choices=["um", "spot", "cm", "all"], default="all")
    parser.add_argument("--type",    choices=["klines", "funding", "metrics", "all"],
                        default="all", help="Type de données à télécharger")
    parser.add_argument("--no-skip", action="store_true",
                        help="Force re-téléchargement complet (supprime les fichiers existants)")
    parser.add_argument("--verify",  action="store_true", help="Vérification seule")
    args = parser.parse_args()

    if args.verify:
        verify(["um", "spot", "cm"])
        return

    force      = args.no_skip
    do_klines  = args.type in ("klines",  "all")
    do_funding = args.type in ("funding", "all")
    do_metrics = args.type in ("metrics", "all")

    # Filtre paires
    wanted = None
    if args.pairs:
        wanted = {p.upper() + "USDT" if not p.upper().endswith("USDT") else p.upper()
                  for p in args.pairs}

    def filter_pairs(pairs, src):
        if wanted is None:
            return pairs
        if src == "cm":
            return [p for p in pairs if any(p.startswith(w.replace("USDT", "")) for w in wanted)]
        return [p for p in pairs if p in wanted]

    # ── Klines ──
    if do_klines:
        print("Récupération des paires actives (klines)...")
        pairs_by_source = {}
        if args.source in ("um",   "all"):
            pairs_by_source["um"]   = filter_pairs(get_live_um_pairs(), "um")
        if args.source in ("spot", "all"):
            pairs_by_source["spot"] = filter_pairs(get_spot_pairs(),    "spot")
        if args.source in ("cm",   "all"):
            pairs_by_source["cm"]   = filter_pairs(get_live_cm_pairs(), "cm")
        asyncio.run(run_klines(pairs_by_source, force))

    # ── Funding ──
    if do_funding:
        print("\nRécupération des paires avec funding rate (S3)...")
        funding_pairs = filter_pairs(get_funding_pairs(), "um")
        asyncio.run(run_funding(funding_pairs, force))

    # ── Métriques ──
    if do_metrics:
        print("\nRécupération des paires avec métriques OI/L/S (S3)...")
        metrics_pairs = filter_pairs(get_metrics_pairs(), "um")
        asyncio.run(run_metrics(metrics_pairs, force))

    verify(["um", "spot", "cm"])
    print(f"\nDonnées dans : {OUT_DIR}/")

if __name__ == "__main__":
    main()
