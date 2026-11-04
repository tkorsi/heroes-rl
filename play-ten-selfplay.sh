#!/bin/sh
# Check by default; explicit --run launches ten frozen games concurrently.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RUNNER_PYTHON=${HEROES_PYTHON:-"$HOME/Projects/Heroes-artifacts/python/next-launch-numpy235-v1/venv/bin/python"}
if [ ! -x "$RUNNER_PYTHON" ]; then
    echo "Set HEROES_PYTHON to the project's locked Python interpreter." >&2
    exit 2
fi
case "${1:---check}" in
    --check|--run) ;;
    *) echo "Usage: $0 [--check | --run] [--output PATH]" >&2; exit 2 ;;
esac
if [ "${1:-}" = --run ] && command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -i "$RUNNER_PYTHON" "$SCRIPT_DIR/tools/run_ten_selfplay.py" "$@"
fi
exec "$RUNNER_PYTHON" "$SCRIPT_DIR/tools/run_ten_selfplay.py" "$@"
