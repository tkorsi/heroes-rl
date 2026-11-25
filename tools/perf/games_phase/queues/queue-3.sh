#!/bin/sh
# After the main-vs-main check: replay timing of the candidate of 01:45 (wip16) against the newest (wip22),
# then throughput windows of the newest at 20, 24 and 28 actors.
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10; P=$A/python/mlx-v1/venv/bin/python
CAND=$G/src-wip22; ENGINE=claude-g10-e2
say() { echo "== $(date) $*"; }
say "priority $(ps -o pri= -p $$)"
until grep -q "verify v2 done" $G/logs/verify-2.log; do sleep 10; done
sh $G/scripts/when-quiet.sh true
sh $G/scripts/ab.sh $G/src-wip16 $CAND
sh $G/scripts/ab.sh $G/src-wip16 $CAND $G/replays/s3-d120.jsonl.gz
for W in 20 24 28; do
  (cd $CAND && HEROES_VCMI=$A/runtime/$ENGINE/bin/vcmiclient $P $G/scripts/window_bench.py $G/states/window --workers $W --seconds 150 --warmup 40 --seed 31 --label wip22-$W --out $G/results/windows.jsonl)
  tail -1 $G/results/windows.jsonl
done
say "queue 3 done"
