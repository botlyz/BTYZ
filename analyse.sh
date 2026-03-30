#!/bin/bash
PORT=2718

# Kill toutes les instances existantes
echo "Nettoyage des instances précédentes..."
pkill -f "cloudflared tunnel" 2>/dev/null
pkill -f "marimo edit" 2>/dev/null
pkill -f "opti.py" 2>/dev/null
sleep 1

# ── Menu ────────────────────────────────────────────────────────────────────
echo ""
echo "┌─────────────────────────────────────────┐"
echo "│              BTYZ Launcher              │"
echo "├─────────────────────────────────────────┤"
echo "│  1) Analyse (walk-forward results)      │"
echo "│  2) CVD Explorer                        │"
echo "│  3) Optimisation (opti.py)              │"
echo "│  4) Analyse (ancienne version)          │"
echo "└─────────────────────────────────────────┘"
echo ""
read -p "  Choix [1-4] : " CHOICE

case "$CHOICE" in
    1) MODE="analyse"  ;;
    2) MODE="cvd"      ;;
    3) MODE="opti"     ;;
    4) MODE="old"      ;;
    *) echo "Choix invalide. Lancement de l'analyse par défaut."
       MODE="analyse"  ;;
esac

# ── Mode opti : pas de marimo ────────────────────────────────────────────────
if [ "$MODE" = "opti" ]; then
    echo ""
    echo "Lancement de l'optimisation..."
    echo "(Ctrl+C pour arrêter)"
    echo ""
    .venv/bin/python src/opti.py
    exit 0
fi

# ── Sélection du notebook ────────────────────────────────────────────────────
if [ "$MODE" = "old" ]; then
    NB="notebooks/analyse.py"
elif [ "$MODE" = "cvd" ]; then
    NB="notebooks/cvd_explorer.py"
else
    NB="notebooks/analyse_full.py"
fi

# ── 1. Lancer marimo ─────────────────────────────────────────────────────────
export MARIMO_OUTPUT_MAX_BYTES=200000000
echo ""
echo "Démarrage de marimo → $NB"
.venv/bin/marimo edit "$NB" --host 0.0.0.0 --port $PORT --headless --no-token &
MARIMO_PID=$!

# ── 2. Attendre que le serveur HTTP réponde (max 60s) ────────────────────────
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$PORT" -o /dev/null 2>/dev/null; then
        break
    fi
    sleep 1
done

if ! curl -sf "http://localhost:$PORT" -o /dev/null 2>/dev/null; then
    echo "Erreur : marimo n'a pas démarré sur le port $PORT"
    kill $MARIMO_PID 2>/dev/null
    exit 1
fi

# ── 3. Lancer cloudflared ────────────────────────────────────────────────────
LOGFILE=$(mktemp /tmp/cloudflared.XXXXXX.log)
cloudflared tunnel --url http://localhost:$PORT > "$LOGFILE" 2>&1 &
CF_PID=$!

# ── 4. Attendre l'URL (max 30s) ──────────────────────────────────────────────
URL=""
for i in $(seq 1 30); do
    URL=$(grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' "$LOGFILE" 2>/dev/null | head -1)
    [ -n "$URL" ] && break
    sleep 1
done

if [ -z "$URL" ]; then
    echo "Timeout: URL Cloudflare non trouvée. Accès local : http://localhost:$PORT"
else
    echo "Vérification du tunnel..."
    for i in $(seq 1 20); do
        if curl -sf --max-time 3 "$URL" -o /dev/null 2>/dev/null; then
            break
        fi
        sleep 1
    done
    echo ""
    echo "┌──────────────────────────────────────────────────────────┐"
    echo "│  $NB"
    echo "│  Lien : $URL"
    echo "└──────────────────────────────────────────────────────────┘"
    echo ""
fi

# ── 5. Attendre Ctrl+C ───────────────────────────────────────────────────────
wait $MARIMO_PID

# Cleanup
kill $CF_PID 2>/dev/null
rm -f "$LOGFILE"
