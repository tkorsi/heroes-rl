#!/bin/sh
# The final tree (src-wip41 + e2, orjson, 24 actors) on both seed lists, after queue 9; then its exactness run on the
# 20 games of v1 (results/verify-v6.jsonl against verify-v1-a.jsonl).
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10
CAND=$G/src-wip41; ENGINE=claude-g10-e2
export PYTHONPATH=$G/pylib
say() { echo "== $(date) $*"; }
say "priority $(ps -o pri= -p $$); waiting for queue 9"
until grep -q "queue 9 done" $G/logs/queue-9.log; do sleep 15; done
sh $G/scripts/when-quiet.sh true
RUNTIME=$ENGINE sh $G/scripts/full-run2.sh fin24-s1 $CAND 2026112501 --workers 24 > $G/logs/fin24-s1.log 2>&1
sh $G/scripts/when-quiet.sh true
RUNTIME=$ENGINE sh $G/scripts/full-run2.sh fin24-s2 $CAND 2026112502 --workers 24 > $G/logs/fin24-s2.log 2>&1
sh $G/scripts/verify-one2.sh v6 20 2026112501 16 $CAND $ENGINE
say "queue 10 done"
