#!/bin/sh
# The second matched seed set: main at 14 actors (the baseline of seed list 2026112502).
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10
say() { echo "== $(date) $*"; }
say "priority $(ps -o pri= -p $$)"
sh $G/scripts/when-quiet.sh true
sh $G/scripts/full-run.sh base14-s2 $G/src-2c1672b 2026112502 > $G/logs/base14-s2.log 2>&1
say "queue 4 done"
