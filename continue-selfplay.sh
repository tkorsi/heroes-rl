#!/bin/sh
# Check by default; an explicit --hours value launches the remaining self-play games.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TRAINER_PYTHON=${HEROES_PYTHON:-"$HOME/Projects/Heroes-artifacts/python/next-launch-numpy235-v1/venv/bin/python"}
if [ ! -x "$TRAINER_PYTHON" ]; then
    echo "Set HEROES_PYTHON to the project's locked Python interpreter." >&2
    exit 2
fi
MODE=continuation-check
if [ "${1:-}" = "--check" ]; then
    shift
elif [ "${1:-}" = "--hours" ]; then
    MODE=continue
elif [ "$#" -ne 0 ]; then
    echo "Usage: $0 [--check | --hours HOURS] [--lineage PATH] [--acceptance PATH]" >&2
    exit 2
fi
if [ "$MODE" = continue ] && command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -i "$TRAINER_PYTHON" "$SCRIPT_DIR/tools/run_selfplay_training.py" "$MODE" "$@"
fi
exec "$TRAINER_PYTHON" "$SCRIPT_DIR/tools/run_selfplay_training.py" "$MODE" "$@"
