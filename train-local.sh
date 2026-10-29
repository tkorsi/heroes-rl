#!/bin/sh
# User-owned local training: no LLM API, recurring task, or cloud agent.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TRAINER_PYTHON=${HEROES_PYTHON:-}
if [ -z "$TRAINER_PYTHON" ]; then
    TRAINING_PROFILE=easier8x8
    EXPECT_TRAINING_PROFILE=false
    for TRAINING_ARGUMENT do
        if [ "$EXPECT_TRAINING_PROFILE" = true ]; then
            TRAINING_PROFILE=$TRAINING_ARGUMENT
            EXPECT_TRAINING_PROFILE=false
        else
            case "$TRAINING_ARGUMENT" in
                --profile) EXPECT_TRAINING_PROFILE=true ;;
                --profile=*) TRAINING_PROFILE=${TRAINING_ARGUMENT#--profile=} ;;
                --) break ;;
            esac
        fi
    done
    # The original-map preparation has its own receipt-bound environment.
    if [ "$TRAINING_PROFILE" = original ]; then
        TRAINER_PYTHON="$HOME/Projects/Heroes-artifacts/python/next-launch-numpy235-v1/venv/bin/python"
    else
        TRAINER_PYTHON="$HOME/Projects/Heroes/.venv/bin/python"
    fi
fi
if [ ! -x "$TRAINER_PYTHON" ]; then
    echo "Set HEROES_PYTHON to the project's Python interpreter with the locked PyTorch dependencies." >&2
    exit 2
fi
cd "$SCRIPT_DIR"
exec "$TRAINER_PYTHON" tools/run_local_training.py "$@"
