#!/bin/sh
# Continue the original5000 target from its latest verified official descendant.
# The disclosed interaction guard changes support, not terminal rewards or PPO.
# No gameplay starts until the user invokes this wrapper.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ORIGINAL_RUN="$HOME/Projects/Heroes-artifacts/local-script-training/original/run-20261026T181133Z-062cd75a"
exec "$SCRIPT_DIR/train-local.sh" --profile original --fast-path \
    --serial-verification --prefix-helpers 0 --replay-threads 2 \
    --continue-original "$ORIGINAL_RUN" --batch-wall-seconds 10800 \
    --interaction-guard visible-interaction-budget-v1 \
    --hours 336 --eval-every 100 "$@"
