#!/bin/sh
# After queue 6: throughput windows of src-wip39 (+ e2, orjson) at 20, 24 and 28 actors, beside its 16 of queue 6.
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10; P=$A/python/mlx-v1/venv/bin/python
CAND=$G/src-wip39; ENGINE=claude-g10-e2
export PYTHONPATH=$G/pylib
say() { echo "== $(date) $*"; }
say "priority $(ps -o pri= -p $$); waiting for queue 6"
until grep -q "queue 6 done" $G/logs/queue-6.log; do sleep 15; done
sh $G/scripts/when-quiet.sh true
for W in 20 24 28; do
  (cd $CAND && HEROES_VCMI=$A/runtime/$ENGINE/bin/vcmiclient $P $G/scripts/window_bench.py $G/states/window --workers $W --seconds 150 --warmup 40 --seed 31 --label scale39-$W --out $G/results/windows.jsonl)
done
say "queue 7 done"
