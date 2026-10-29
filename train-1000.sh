#!/bin/sh
# Manual continuation for 1,000 new admitted games, with total wall-clock timing.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TRAINER_PYTHON=${HEROES_PYTHON:-"$HOME/Projects/Heroes/.venv/bin/python"}
if [ ! -x "$TRAINER_PYTHON" ]; then
    echo "Set HEROES_PYTHON to the project's Python interpreter with the locked PyTorch dependencies." >&2
    exit 2
fi
exec "$TRAINER_PYTHON" "$SCRIPT_DIR/tools/train_1000_games.py" "$@"
