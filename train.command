#!/bin/sh
# Train on N more self-play games: ./train.command 1000
# Every other argument goes to "python -m heroes train"; see --help.
set -eu
cd "$(dirname "$0")"
PYTHON=${HEROES_PYTHON:-"$HOME/Projects/Heroes-artifacts/python/mlx-v1/venv/bin/python"}
if [ ! -x "$PYTHON" ]; then
    echo "Set HEROES_PYTHON to the Python that has torch and mlx installed." >&2
    exit 2
fi
# Keep the Mac awake while training; the display may still sleep.
if command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -is "$PYTHON" -m heroes train "$@"
fi
exec "$PYTHON" -m heroes train "$@"
