#!/usr/bin/env bash
# Lanzador DAW — Linux / Mac
# Uso: ./run_gui.sh  (desde cualquier lugar)

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/.venv"

if [ ! -f "$VENV/bin/python3" ]; then
    echo "⚠ Entorno virtual no encontrado. Creando..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet PyQt6 sounddevice numpy
    echo "✓ Dependencias instaladas."
fi

cd "$DIR"
exec "$VENV/bin/python3" gui.py "$@"
