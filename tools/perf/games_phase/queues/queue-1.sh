#!/bin/sh
# After the baseline batch: throughput windows (training configuration) for main at 14 actors and the candidate at
# 14, 16 and 18 actors; then the candidate's full batch on the baseline's seed list with the fastest count.
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10; P=$A/python/mlx-v1/venv/bin/python
CAND=$G/src-wip15; ENGINE=claude-g10-e2
say() { echo "== $(date) $*"; }
say "priority $(ps -o pri= -p $$); waiting for the baseline batch to end"
until grep -q "== .* exit" $G/logs/base14-s1.log; do sleep 20; done
sleep 30
mkdir -p $G/states/window; cp -r $G/base/. $G/states/window/
sh $G/scripts/when-quiet.sh true
say "windows"
(cd $G/src-2c1672b && $P $G/scripts/window_bench.py $G/states/window --workers 14 --seconds 150 --warmup 40 --seed 31 --label main-14 --out $G/results/windows.jsonl)
for W in 14 16 18 20; do
  (cd $CAND && HEROES_VCMI=$A/runtime/$ENGINE/bin/vcmiclient $P $G/scripts/window_bench.py $G/states/window --workers $W --seconds 150 --warmup 40 --seed 31 --label cand-$W --out $G/results/windows.jsonl)
done
BEST=$($P -c "
import json
rows = [json.loads(l) for l in open('$G/results/windows.jsonl')][-4:]
print(max(rows, key=lambda r: r['decisions_per_second'])['workers'])")
say "fastest count: $BEST"
sh $G/scripts/when-quiet.sh true
RUNTIME=$ENGINE sh $G/scripts/full-run.sh cand$BEST-s1 $CAND 2026112501 --workers $BEST
say "queue done"
