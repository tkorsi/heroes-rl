#!/bin/sh
# One manual 100-game continuation from the latest retained checkpoint.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case "${1:---run}" in
    --check)
        exec "$SCRIPT_DIR/continue-selfplay.sh" --cycle --check --cycles 1 --hours 10 ;;
    --run)
        exec "$SCRIPT_DIR/continue-selfplay.sh" --cycle --run --cycles 1 --hours 10 ;;
    *) echo "Usage: $0 [--run | --check]" >&2; exit 2 ;;
esac
