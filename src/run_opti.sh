#!/bin/bash
# Wrapper auto-restart pour opti.py
# Usage: bash src/run_opti.sh

MAX_RETRIES=50
RETRY=0
COOLDOWN=10
CACHE_DIR=""

TG_TOKEN="8045706367:AAF9MV280K9NitKUiQhcjwiR7uUUWWe02g8"
TG_CHAT="1069067907"

tg_send() {
    curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
        -d chat_id="$TG_CHAT" -d parse_mode="HTML" -d text="$1" > /dev/null 2>&1
}

while [ $RETRY -lt $MAX_RETRIES ]; do
    echo ""
    echo "══════════════════════════════════════════"

    if [ $RETRY -eq 0 ]; then
        echo "  Lancement interactif..."
        echo "══════════════════════════════════════════"
        python3 src/opti.py
        EXIT_CODE=$?
    else
        echo "  RELANCE #$RETRY (après crash)"
        tg_send "🔄 <b>Opti relance #${RETRY}</b> après crash (exit code $EXIT_CODE)"
        echo "  Pause ${COOLDOWN}s..."
        sleep $COOLDOWN
        echo "══════════════════════════════════════════"

        # Trouver le dernier cache_dir créé
        CACHE_DIR=$(ls -td cache/full_*/ram_* cache/full_*/kc_* 2>/dev/null | head -1)

        if [ -n "$CACHE_DIR" ] && [ -d "$CACHE_DIR" ]; then
            echo "  Resume → $CACHE_DIR"
            python3 src/opti.py --resume "$CACHE_DIR"
            EXIT_CODE=$?
        else
            echo "  Pas de cache_dir, relance interactive..."
            python3 src/opti.py
            EXIT_CODE=$?
        fi
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Opti terminée avec succès."
        exit 0
    fi

    RETRY=$((RETRY + 1))
    echo ""
    echo ">>> Crash (exit code $EXIT_CODE) — relance dans ${COOLDOWN}s... ($RETRY/$MAX_RETRIES)"
done

tg_send "❌ <b>Opti abandonnée</b> après $MAX_RETRIES tentatives"
echo "Abandonné après $MAX_RETRIES tentatives."
exit 1
