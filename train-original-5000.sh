#!/bin/sh
# The next manually launched experiment: final current easy state -> original map.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PARENT_RUN="$HOME/Projects/Heroes-artifacts/local-script-training/run-20261026T160706Z-0be8eb77"
exec "$SCRIPT_DIR/train-local.sh" --profile original --games 5000 --hours 48 \
    --eval-every 100 --parent-final "$PARENT_RUN" "$@"
