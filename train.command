#!/bin/sh
# Choose the number of self-play games to collect before PPO.
# Every other argument goes to "python -m heroes train"; see --help.
set -eu
cd "$(dirname "$0")"
PYTHON=${HEROES_PYTHON:-"$HOME/Projects/Heroes-artifacts/python/mlx-v1/venv/bin/python"}
if [ ! -x "$PYTHON" ]; then
    echo "Set HEROES_PYTHON to the Python that has torch and mlx installed." >&2
    exit 2
fi
# "./train.command check" shows what a run would start from and plays nothing.
if [ "${1:-}" = check ]; then
    exec "$PYTHON" -m heroes "$@"
fi
if [ "$#" -eq 0 ]; then
    printf 'Сколько партий сыграть перед одним обновлением PPO? (10, 20, 40, 100, 1000): '
    IFS= read -r games || exit 1
    set -- "$games"
fi
# Keep the Mac awake until training ends; the display may still sleep. The
# trainer stays the foreground process, so Ctrl-C returns only once it has
# stopped its games.
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -is -w $$ &
fi
exec "$PYTHON" -m heroes train "$@"
