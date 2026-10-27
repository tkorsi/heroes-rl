#!/bin/sh
# One-time original-map launch. Existing official runs must use their continuation.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ORIGINAL_ROOT="$HOME/Projects/Heroes-artifacts/local-script-training/original"
if [ -e "$ORIGINAL_ROOT/latest-run.json" ] || [ -e "$ORIGINAL_ROOT/latest-checkpoint.json" ] || [ -e "$ORIGINAL_ROOT/sequence-ledger.json" ]; then
    echo "An official original-map run already exists; refusing to reset its state to629." >&2
    echo "Use ./continue-original-4720.sh to check and continue its verified remaining target." >&2
    exit 2
fi
PARENT_RUN="$HOME/Projects/Heroes-artifacts/local-script-training/run-20261026T160706Z-0be8eb77"
exec "$SCRIPT_DIR/train-local.sh" --profile original --games 5000 --hours 48 \
    --eval-every 100 --parent-final "$PARENT_RUN" "$@"
