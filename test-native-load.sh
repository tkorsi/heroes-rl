#!/bin/sh
# Bounded diagnostic on disposable state; never starts official training.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TEST_PYTHON=${HEROES_PYTHON:-"$HOME/Projects/Heroes/.venv/bin/python"}
if [ ! -x "$TEST_PYTHON" ]; then
    echo "Set HEROES_PYTHON to the project's locked Python interpreter." >&2
    exit 2
fi
cd "$SCRIPT_DIR"
exec "$TEST_PYTHON" tools/test_native_load.py "$@"
