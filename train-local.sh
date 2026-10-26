#!/bin/sh
# User-owned local training: no LLM API, recurring task, or cloud agent.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TRAINER_PYTHON=${HEROES_PYTHON:-"$HOME/Projects/Heroes/.venv/bin/python"}
if [ ! -x "$TRAINER_PYTHON" ]; then
    echo "Set HEROES_PYTHON to the project's Python interpreter with the locked PyTorch dependencies." >&2
    exit 2
fi
cd "$SCRIPT_DIR"
exec "$TRAINER_PYTHON" tools/run_local_training.py "$@"
