#!/bin/sh
# Finish this campaign using its existing state. Pass that directory from another checkout.
set -eu
state=$(cd "${1:-$(dirname "$0")/.experiment/continue-1797-100/state}" && pwd -P)
cd "$(dirname "$0")"
parent_games=10372
target_games=$((parent_games + 100))
saved_games=$(/usr/bin/plutil -extract games raw -expect integer -o - "$state/latest.json")

if [ "$saved_games" -lt "$parent_games" ] || [ "$saved_games" -gt "$target_games" ]; then
    echo "Unexpected saved game count in $state/latest.json: $saved_games" >&2
    exit 1
fi
remaining=$((target_games - saved_games))
if [ "$remaining" -eq 0 ]; then
    echo "All 100 games are already saved."
    exit 0
fi

export HEROES_ARTIFACTS="${HEROES_ARTIFACTS:-$HOME/Projects/Heroes-artifacts}"
export HEROES_PYTHON="$HEROES_ARTIFACTS/python/mlx-v1/venv/bin/python"
export HEROES_VCMI="$HEROES_ARTIFACTS/runtime/local-v8-tile-changes-v1/bin/vcmiclient"
exec ./train.command "$remaining" --state "$state" --workers 5 --games-per-update 5 \
    --backend mlx --windows-per-minibatch 16 --windows-per-game 8 --learning-rate 8.5e-4
