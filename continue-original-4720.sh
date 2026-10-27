#!/bin/sh
# Continue the official original5000 target from699 or its verified descendants.
# No gameplay starts until the user invokes this wrapper.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ORIGINAL_RUN="$HOME/Projects/Heroes-artifacts/local-script-training/original/run-20261026T181133Z-062cd75a"
exec "$SCRIPT_DIR/train-local.sh" --profile original --fast-path \
    --serial-verification --prefix-helpers 0 --replay-threads 2 \
    --continue-original "$ORIGINAL_RUN" --batch-wall-seconds 10800 \
    --hours 336 --eval-every 100 "$@"
