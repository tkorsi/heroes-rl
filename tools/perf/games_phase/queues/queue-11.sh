#!/bin/sh
# A third matched seed list (2026112503), not used during the work: main at 14 actors, then the final tree
# (src-wip41 + e2, orjson, 24 actors), each on a fresh copy of checkpoint 2036.
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10
say() { echo "== $(date) $*"; }
say "priority $(ps -o pri= -p $$)"
sh $G/scripts/when-quiet.sh true
sh $G/scripts/full-run2.sh base14-s3 $G/src-2c1672b 2026112503 > $G/logs/base14-s3.log 2>&1
sh $G/scripts/when-quiet.sh true
export PYTHONPATH=$G/pylib
RUNTIME=claude-g10-e2 sh $G/scripts/full-run2.sh fin24-s3 $G/src-wip41 2026112503 --workers 24 > $G/logs/fin24-s3.log 2>&1
say "queue 11 done"
