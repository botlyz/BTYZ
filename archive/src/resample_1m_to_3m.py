#!/usr/bin/env python3
"""Resample toutes les paires 1m → 3m en parallèle.

Écrase les fichiers existants dans data/raw/lighter/3m/.
"""
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

LIGHTER = Path("/home/devbox/BTYZ/data/raw/lighter")
SRC_DIR = LIGHTER / "1m"
DST_DIR = LIGHTER / "3m"


def resample_pair(src_path: Path) -> tuple[str, int, str]:
    pair = src_path.stem
    dst_path = DST_DIR / f"{pair}.csv"

    df = pd.read_csv(src_path)
    if "date" not in df.columns:
        df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    df = df.set_index("date").sort_index()

    df3 = df.resample("3min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()

    df3.index = df3.index.astype("int64") // 10**6
    df3.index.name = "date"
    df3.to_csv(dst_path)
    last_ts = pd.to_datetime(int(df3.index[-1]), unit="ms", utc=True)
    return pair, len(df3), str(last_ts)


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    src_files = sorted(SRC_DIR.glob("*.csv"))
    if not src_files:
        print(f"Aucun fichier dans {SRC_DIR}")
        sys.exit(1)

    print(f"Resampling {len(src_files)} paires 1m → 3m...")
    print(f"  Source : {SRC_DIR}")
    print(f"  Dest   : {DST_DIR}")

    n_workers = 12
    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(resample_pair, p): p for p in src_files}
        with tqdm(total=len(futs), unit="pair", ncols=80) as pb:
            for fut in as_completed(futs):
                src = futs[fut]
                try:
                    pair, n, last = fut.result()
                    results.append((pair, n, last))
                except Exception as e:
                    errors.append((src.stem, str(e)))
                pb.update(1)

    print(f"\n✓ {len(results)} paires resamplées")
    if errors:
        print(f"✗ {len(errors)} erreurs :")
        for pair, err in errors[:10]:
            print(f"  {pair}: {err}")

    # Aperçu dernières dates
    results.sort(key=lambda x: x[2])
    print(f"\nPlage des dernières barres : {results[0][2]} → {results[-1][2]}")


if __name__ == "__main__":
    main()
