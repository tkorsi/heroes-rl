#!/bin/sh
# After the baseline of seed list 2: replay timing of wip31, the candidate (wip31 + e2, 24 actors) on both seed
# lists, then wip31 on the 20 games main was digested on (results/verify-v1-a.jsonl).
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10
CAND=$G/src-wip31; ENGINE=claude-g10-e2
say() { echo "== $(date) $*"; }
say "priority $(ps -o pri= -p $$); waiting for queue 4"
until grep -q "queue 4 done" $G/logs/queue-4.log; do sleep 15; done
sh $G/scripts/when-quiet.sh true
sh $G/scripts/ab.sh $G/src-wip16 $CAND
sh $G/scripts/ab.sh $G/src-wip16 $CAND $G/replays/s3-d120.jsonl.gz
sh $G/scripts/when-quiet.sh true
RUNTIME=$ENGINE sh $G/scripts/full-run.sh cand24-s1 $CAND 2026112501 --workers 24 > $G/logs/cand24-s1.log 2>&1
sh $G/scripts/when-quiet.sh true
RUNTIME=$ENGINE sh $G/scripts/full-run.sh cand24-s2 $CAND 2026112502 --workers 24 > $G/logs/cand24-s2.log 2>&1
sh $G/scripts/verify-one.sh v3 20 2026112501 16 $CAND $ENGINE
say "queue 5 done"
