#!/bin/sh
# The second matched seed set: main at 14 actors, then the candidate of the first set (src-wip15, engine e2, 20 actors).
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10
say() { echo "== $(date) $*"; }
say "priority $(ps -o pri= -p $$); waiting for the candidate batch of the first seed set to end"
until grep -q "queue done" $G/logs/queue-1.log; do sleep 20; done
sleep 20
sh $G/scripts/when-quiet.sh true
sh $G/scripts/full-run.sh base14-s2 $G/src-2c1672b 2026112502 > $G/logs/base14-s2.log 2>&1
sh $G/scripts/when-quiet.sh true
RUNTIME=claude-g10-e2 sh $G/scripts/full-run.sh cand20-s2 $G/src-wip15 2026112502 --workers 20 > $G/logs/cand20-s2.log 2>&1
say "queue 2 done"
