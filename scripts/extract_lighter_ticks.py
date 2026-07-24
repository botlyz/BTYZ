#!/usr/bin/env python3
"""Extrait TOUT l'historique tick Lighter du bucket S3 -> 1 parquet trié par paire.
data/lighter_ticks/<SYMBOL>.parquet  (timestamp trié, px/sz décodés réels).
Pass 1 : scan multithread du bucket, partition par market_id (streaming, RAM-safe).
Pass 2 : par marché, tri par timestamp + décodage prix/size + nommage symbole.
"""
import pyarrow as pa, pyarrow.fs as fs, pyarrow.dataset as ds, pyarrow.parquet as pq, pyarrow.compute as pc
import time, json, os, shutil, glob, gc

pa.set_io_thread_count(48); pa.set_cpu_count(24)
ak, sk = open("/home/botlyz-gpu/.lighter_s3_creds").read().split()[:2]
s3 = fs.S3FileSystem(access_key=ak, secret_key=sk, region="ap-northeast-1")
BASE = "lighter-historical-data/trade/db-2026-06-02/indexer/public.trade/1"
COLS = ["market_id", "timestamp", "price", "size", "is_maker_ask",
        "ask_account_id", "bid_account_id", "trade_type"]
TMP = "/home/botlyz-gpu/BTYZ/data/_ticks_tmp"
OUT = "/home/botlyz-gpu/BTYZ/data/lighter_ticks"
MM = json.load(open("/home/botlyz-gpu/BTYZ/data/market_map.json"))

allf = sorted(f.path for f in s3.get_file_info(fs.FileSelector(BASE, recursive=False)) if f.path.endswith(".parquet"))
print(f"PASS 1 : {len(allf)} fichiers -> partition par marché (48 io threads)", flush=True)
t0 = time.time()
scanner = ds.dataset(allf, filesystem=s3, format="parquet").scanner(columns=COLS)
if os.path.exists(TMP):
    shutil.rmtree(TMP)
ds.write_dataset(scanner, TMP, format="parquet",
                 partitioning=ds.partitioning(pa.schema([("market_id", pa.int16())]), flavor="hive"),
                 existing_data_behavior="overwrite_or_ignore", max_rows_per_file=40_000_000)
print(f"PASS 1 fini en {time.time()-t0:.0f}s", flush=True)

print("PASS 2 : tri + décodage + nommage par symbole", flush=True)
t1 = time.time(); os.makedirs(OUT, exist_ok=True); idx = {}
for p in sorted(glob.glob(f"{TMP}/market_id=*")):
    mid = int(p.split("=")[-1]); info = MM.get(str(mid), {})
    sym = (info.get("symbol") or f"MKT{mid}").replace("/", "_")
    pdec = int(info.get("price_dec") or 0); sdec = int(info.get("size_dec") or 0)
    t = ds.dataset(p, format="parquet").to_table()
    n = t.num_rows
    srt = n < 200_000_000  # géants (BTC) non triés pour rester RAM-safe (sort-on-use)
    if srt:
        t = t.sort_by("timestamp")
    t = t.append_column("px", pc.divide(pc.cast(t["price"], pa.float64()), float(10 ** pdec)))
    t = t.append_column("sz", pc.divide(pc.cast(t["size"], pa.float64()), float(10 ** sdec)))
    pq.write_table(t, f"{OUT}/{sym}.parquet")
    idx[sym] = {"market_id": mid, "n_ticks": n, "price_dec": pdec, "size_dec": sdec, "sorted": srt}
    print(f"  {sym:12} mid {mid:4} : {n:>10} ticks {'' if srt else '(non trié - géant)'}", flush=True)
    del t; gc.collect()
json.dump(idx, open(f"{OUT}/_index.json", "w"), indent=1)
shutil.rmtree(TMP)
print(f"PASS 2 fini en {time.time()-t1:.0f}s. {len(idx)} paires dans {OUT}", flush=True)
