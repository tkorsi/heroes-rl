#!/bin/sh
# Manual configurable continuation; retain the filename for existing shortcuts.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODE=--run
GAMES=500
HOURS=
METAL=1
while [ "$#" -gt 0 ]; do
    case "$1" in
        --check|--run) MODE=$1; shift ;;
        --metal) METAL=1; shift ;;
        --cpu) METAL=; shift ;;
        --games|--hours)
            [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 2; }
            if [ "$1" = --games ]; then GAMES=$2; else HOURS=$2; fi
            shift 2 ;;
        --help|-h)
            echo "Usage: $0 [10..1000 | --games N] [--check | --run] [--hours H] [--metal | --cpu]"
            echo "Default: 500 games, Metal PPO batches up to 100, one final diagnostic, no time limit."
            echo "--cpu selects CPU PPO; gameplay and diagnostics always remain on CPU."
            exit 0 ;;
        *[!0-9]*|'') echo "Unknown argument: $1" >&2; exit 2 ;;
        *) GAMES=$1; shift ;;
    esac
done
set -- --cycle "$MODE" --games "$GAMES"
if [ -n "$HOURS" ]; then set -- "$@" --hours "$HOURS"; fi
if [ -n "$METAL" ]; then set -- "$@" --metal; fi
exec "$SCRIPT_DIR/continue-selfplay.sh" "$@"
