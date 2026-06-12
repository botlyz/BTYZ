#!/bin/bash
# Push fresh OHLCV des 5 paires SIGMA AWF vers le serveur prod
# Cron GPU : */5 * * * * /home/botlyz-gpu/BTYZ/scripts/push_ohlcv_to_prod.sh

set -e
PAIRS=(HYPE SKY PENDLE XMR ASTER)
SRC_DIR="/home/botlyz-gpu/BTYZ/data/raw/lighter/5m"
DST="botlyz@88.151.197.77:/root/Botlyz_TG/Botlyz_client_clean/sigma_awf_bt/ohlcv_cache/"
LOG="/home/botlyz-gpu/BTYZ/scripts/push_ohlcv.log"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] Push OHLCV → prod" >> "$LOG"
for p in "${PAIRS[@]}"; do
  src="$SRC_DIR/$p.csv"
  if [ -f "$src" ]; then
    rsync -a --timeout=30 "$src" "$DST" 2>>"$LOG"
    echo "  $p: ok" >> "$LOG"
  else
    echo "  $p: MISSING $src" >> "$LOG"
  fi
done
