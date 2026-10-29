#!/bin/sh
# Manual native continuation after verified cloud closure and state import.
# This wrapper never changes ownership locks or selects a fixed checkpoint.
set -eu

usage() {
    cat <<'USAGE'
Usage: ./continue-original-native.sh --native-learner /absolute/manifest.json \
    --native-sha256 SHA256 [--check] [--hours HOURS] [--cpu-cap CPUS] \
    [--recovery-receipt /path/to/receipt.json]

Defaults to one hour and all available CPUs. --check performs offline checks only.
The existing runner verifies the exact native build and accepted source, restores
the latest verified original-map state, and computes the remaining 5,000-game target.
The four-game PPO batch, opponent, terminal rewards and visible interaction budget
remain fixed. No checkpoint number or remaining-game count is hard-coded here.

Cloud ownership must first be closed and the latest complete state and consumed-seed
ledger imported through a separately verified handoff. This script does not release
the Mac ownership fence; --check also refuses while that fence remains in place.
USAGE
}

fail() { echo "Native continuation: $*" >&2; exit 2; }
COMMAND=start
HOURS=1
NATIVE_MANIFEST=
NATIVE_SHA256=
CPU_CAP=
RECOVERY_RECEIPT=
SEEN=' '
while [ "$#" -gt 0 ]; do
    OPTION=$1
    case "$OPTION" in
        --help|-h) usage; exit 0 ;;
        --check|--hours|--cpu-cap|--native-learner|--native-sha256|--recovery-receipt) ;;
        *) fail "Unsupported or conflicting option: $OPTION (see --help)." ;;
    esac
    case "$SEEN" in *" $OPTION "*) fail "Repeated option: $OPTION." ;; esac
    SEEN="$SEEN$OPTION "
    shift
    if [ "$OPTION" = --check ]; then COMMAND=check; continue; fi
    [ "$#" -gt 0 ] || fail "Missing value for $OPTION."
    case "$1" in ''|--*) fail "Missing value for $OPTION." ;; esac
    case "$OPTION" in
        --hours) HOURS=$1 ;;
        --cpu-cap) CPU_CAP=$1 ;;
        --native-learner) NATIVE_MANIFEST=$1 ;;
        --native-sha256) NATIVE_SHA256=$1 ;;
        --recovery-receipt) RECOVERY_RECEIPT=$1 ;;
    esac
    shift
done
case "$NATIVE_MANIFEST" in /*) ;; *) fail "An absolute --native-learner manifest is required." ;; esac
[ "${#NATIVE_SHA256}" -eq 64 ] || fail "--native-sha256 must contain 64 lowercase hexadecimal characters."
case "$NATIVE_SHA256" in *[!0-9a-f]*) fail "--native-sha256 must contain 64 lowercase hexadecimal characters." ;; esac

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CAMPAIGN_ROOT="$HOME/Projects/Heroes-artifacts/local-script-training"
# Reading writability does not acquire, modify, replace or release either lock.
# The runner still acquires its existing campaign locks and verifies all evidence.
for LOCK in "$CAMPAIGN_ROOT/owner.lock" "$CAMPAIGN_ROOT/original/owner.lock"; do
    if [ ! -f "$LOCK" ] || [ -L "$LOCK" ] || [ ! -w "$LOCK" ]; then
        fail "Official Mac ownership is unavailable or fenced: $LOCK. Verified cloud closure and latest-state import are required before manual continuation or a readiness check."
    fi
done

set -- "$COMMAND" --profile original --fast-path \
    --continue-original "$CAMPAIGN_ROOT/original/run-20261026T181133Z-062cd75a" \
    --native-learner "$NATIVE_MANIFEST" --native-sha256 "$NATIVE_SHA256" \
    --interaction-guard visible-interaction-budget-v1 \
    --batch-wall-seconds 10800 --eval-every 100 --hours "$HOURS"
if [ -n "$CPU_CAP" ]; then set -- "$@" --cpu-cap "$CPU_CAP"; fi
if [ -n "$RECOVERY_RECEIPT" ]; then set -- "$@" --recovery-receipt "$RECOVERY_RECEIPT"; fi
exec "$SCRIPT_DIR/train-local.sh" "$@"
