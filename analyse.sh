#!/bin/bash
PORT=2718

# Kill toutes les instances existantes
echo "Nettoyage des instances précédentes..."
pkill -f "cloudflared tunnel" 2>/dev/null
pkill -f "marimo edit" 2>/dev/null
pkill -f "marimo run" 2>/dev/null
# pkill -f "opti.py" 2>/dev/null  # désactivé pour ne pas tuer l'opti en cours
sleep 1

# ── Menu ────────────────────────────────────────────────────────────────────
echo ""
echo "┌─────────────────────────────────────────┐"
echo "│              BTYZ Launcher              │"
echo "├─────────────────────────────────────────┤"
echo "│  1) BTYZ Engine — Visualisation WFA     │"
echo "│  2) BTYZ Engine — Visualisation MCCV    │"
echo "│  3) BTYZ Engine — WFA Tuning comparator │"
echo "│  4) Optimisation Keltner (opti.py)      │"
echo "│  5) Optimisation RAM DCA (opti_ram.py)  │"
echo "│  6) RAM DCA — HYPE Lighter (marimo)     │"
echo "│  7) CVD Explorer                        │"
echo "│  8) Evaluate ML signals (backtest OOS)  │"
echo "│  9) Analyse (ancienne version pickle)   │"
echo "│ 10) Analyse (legacy)                    │"
echo "└─────────────────────────────────────────┘"
echo ""
read -p "  Choix [1-10] : " CHOICE

case "$CHOICE" in
    1) MODE="engine_wfa"   ;;
    2) MODE="engine_mccv"  ;;
    3) MODE="engine_tuning";;
    4) MODE="opti"      ;;
    5) MODE="opti_ram"  ;;
    6) MODE="ram"       ;;
    7) MODE="cvd"       ;;
    8) MODE="evaluate"  ;;
    9) MODE="analyse"   ;;
    10) MODE="old"      ;;
    *) echo "Choix invalide. Lancement de la visualisation WFA par défaut."
       MODE="engine_wfa"   ;;
esac

# ── Mode opti : pas de marimo ────────────────────────────────────────────────
if [ "$MODE" = "opti" ]; then
    echo ""
    echo "Lancement de l'optimisation Keltner..."
    echo "(Ctrl+C pour arrêter)"
    echo ""
    .venv/bin/python src/opti.py
    exit 0
fi

if [ "$MODE" = "opti_ram" ]; then
    echo ""
    echo "Lancement de l'optimisation RAM DCA..."
    echo "(Ctrl+C pour arrêter)"
    echo ""
    .venv/bin/python src/opti_ram.py
    exit 0
fi

# ── Sélection du notebook ────────────────────────────────────────────────────
if [ "$MODE" = "old" ]; then
    NB="notebooks/analyse.py"
elif [ "$MODE" = "cvd" ]; then
    NB="notebooks/cvd_explorer.py"
elif [ "$MODE" = "ram" ]; then
    NB="notebooks/ram_dca_lighter.py"
elif [ "$MODE" = "evaluate" ]; then
    NB="notebooks/evaluate.py"
elif [ "$MODE" = "engine_wfa" ]; then
    NB="notebooks/analyse/analyse_engine.py"
elif [ "$MODE" = "engine_mccv" ]; then
    NB="notebooks/analyse/analyse_mccv.py"
elif [ "$MODE" = "engine_tuning" ]; then
    NB="notebooks/analyse/analyse_wfa_tuning.py"
elif [ "$MODE" = "analyse" ]; then
    NB="notebooks/analyse_full.py"
else
    NB="notebooks/analyse/analyse_engine.py"
fi

# ── 1. Lancer marimo ─────────────────────────────────────────────────────────
export MARIMO_OUTPUT_MAX_BYTES=200000000
echo ""
MARIMO_CMD="run"
echo "Démarrage de marimo → $NB"
.venv/bin/marimo $MARIMO_CMD "$NB" --host 0.0.0.0 --port $PORT --headless --no-token &
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

# ── 3. URL Tailscale (direct, pas de tunnel) ─────────────────────────────────
TAILSCALE_IP="$(tailscale ip -4 2>/dev/null | head -1)"
TAILSCALE_URL="http://$TAILSCALE_IP:$PORT"

echo ""
echo "┌──────────────────────────────────────────────────────────┐"
echo "│  $NB"
echo "│  Tailscale : $TAILSCALE_URL"
echo "│  Local     : http://localhost:$PORT"
echo "└──────────────────────────────────────────────────────────┘"
echo ""

# ── 4. Attendre Ctrl+C ───────────────────────────────────────────────────────
wait $MARIMO_PID
