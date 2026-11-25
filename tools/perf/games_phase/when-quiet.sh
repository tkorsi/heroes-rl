#!/bin/sh
# Wait until no game of anyone else runs (no vcmiclient outside claude-g10 and no "heroes train"), then exec the rest.
while ps -A -o command | grep "[v]cmiclient" | grep -v claude-g10 > /dev/null || pgrep -f "venv/bin/python -m heroes trai[n]" > /dev/null; do sleep 10; done
echo "== $(date) the machine is quiet"
exec "$@"
