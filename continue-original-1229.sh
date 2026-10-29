#!/bin/sh
# Manual continuation with an explicit audited recovery; no model/API calls.
# Defaults to a one-hour trial. A later clean descendant is never reset to1229.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RECEIPT="$HOME/Projects/Heroes-artifacts/run-investigations/2026-10-29-1229-sleep/recovery-receipt.json"
exec "$SCRIPT_DIR/continue-original-4720.sh" \
    --recovery-receipt "$RECEIPT" --hours 1 --collection-workers 4 "$@"
