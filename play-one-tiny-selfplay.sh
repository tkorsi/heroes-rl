#!/bin/sh
# Check by default; --run explicitly plays one frozen game with the tiny model.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TRAINER_PYTHON=${HEROES_PYTHON:-"$HOME/Projects/Heroes-artifacts/python/next-launch-numpy235-v1/venv/bin/python"}
if [ ! -x "$TRAINER_PYTHON" ]; then
    echo "Set HEROES_PYTHON to the project's locked Python interpreter." >&2
    exit 2
fi
case "${1:---check}" in
    --check|--run) ;;
    *) echo "Usage: $0 [--check | --run] [--seconds SECONDS] [--output PATH]" >&2; exit 2 ;;
esac
if [ "${1:-}" = --run ] && command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -i "$TRAINER_PYTHON" "$SCRIPT_DIR/tools/run_tiny_selfplay_one_game.py" "$@"
fi
exec "$TRAINER_PYTHON" "$SCRIPT_DIR/tools/run_tiny_selfplay_one_game.py" "$@"
