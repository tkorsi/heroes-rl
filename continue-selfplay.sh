#!/bin/sh
# Check by default; select the ten-game experiment explicitly with --batch-ten.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TRAINER_PYTHON=${HEROES_PYTHON:-"$HOME/Projects/Heroes-artifacts/python/next-launch-numpy235-v1/venv/bin/python"}
if [ ! -x "$TRAINER_PYTHON" ]; then
    echo "Set HEROES_PYTHON to the project's locked Python interpreter." >&2
    exit 2
fi
MODE=continuation-check
if [ "${1:-}" = "--cycle" ]; then
    shift
    MODE=cycle-check
    case "${1:---check}" in
        --check) if [ "$#" -gt 0 ]; then shift; fi ;;
        --run) MODE=cycle-run; shift ;;
        --resume) MODE=cycle-continue; shift ;;
        *) echo "Usage: $0 --cycle [--check | --run | --resume --run-dir PATH] [--cycles N] [--hours HOURS]" >&2; exit 2 ;;
    esac
elif [ "${1:-}" = "--batch-ten" ]; then
    shift
    MODE=batch10-check
    case "${1:---check}" in
        --check) if [ "$#" -gt 0 ]; then shift; fi ;;
        --run) MODE=batch10-run; shift ;;
        --resume) MODE=batch10-continue; shift ;;
        *) echo "Usage: $0 --batch-ten [--check | --run | --resume --run-dir PATH] [--batches N] [--hours HOURS]" >&2; exit 2 ;;
    esac
elif [ "${1:-}" = "--check" ]; then
    shift
elif [ "${1:-}" = "--hours" ]; then
    MODE=continue
elif [ "$#" -ne 0 ]; then
    echo "Usage: $0 [--check | --hours HOURS] [--lineage PATH] [--acceptance PATH]" >&2
    exit 2
fi
case "$MODE" in
    continue|batch10-run|batch10-continue|cycle-run|cycle-continue)
        if command -v caffeinate >/dev/null 2>&1; then
            exec caffeinate -i "$TRAINER_PYTHON" "$SCRIPT_DIR/tools/run_selfplay_training.py" "$MODE" "$@"
        fi ;;
esac
exec "$TRAINER_PYTHON" "$SCRIPT_DIR/tools/run_selfplay_training.py" "$MODE" "$@"
