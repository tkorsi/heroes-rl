#!/bin/sh
# After queue 7: main and src-wip41 (+ e2, orjson) on all 100 games of seed list 1 at once (8 actors each), the batch
# in game order before PPO and every trajectory digested (results/verify-v5-a.jsonl, -b.jsonl). Not a timing run.
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10
export PYTHONPATH=$G/pylib
say() { echo "== $(date) $*"; }
say "priority $(ps -o pri= -p $$); waiting for queue 7"
until grep -q "queue 7 done" $G/logs/queue-7.log; do sleep 15; done
sh $G/scripts/when-quiet.sh true
sh $G/scripts/verify-pair.sh v5 100 2026112501 8 $G/src-2c1672b - $G/src-wip41 claude-g10-e2
say "queue 8 done"
