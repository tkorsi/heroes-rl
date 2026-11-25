#!/bin/sh
# After queue 5: (1) how fast one actor plays when 1, 4, 8, 12, 16 play at once; (2) the interim candidate (src-wip39,
# engine e2, orjson from claude-g10/pylib, 24 actors) on both seed lists; (3) its exactness run on the 20 games of v1.
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10; P=$A/python/mlx-v1/venv/bin/python
CAND=$G/src-wip39; ENGINE=claude-g10-e2
export PYTHONPATH=$G/pylib
say() { echo "== $(date) $*"; }
say "priority $(ps -o pri= -p $$); waiting for queue 5"
until grep -q "queue 5 done" $G/logs/queue-5.log; do sleep 15; done
sh $G/scripts/when-quiet.sh true
for W in 1 4 8 12 16; do
  (cd $CAND && HEROES_VCMI=$A/runtime/$ENGINE/bin/vcmiclient $P $G/scripts/window_bench.py $G/states/window --workers $W --seconds 150 --warmup 40 --seed 31 --label scale39-$W --out $G/results/windows.jsonl)
  tail -1 $G/results/windows.jsonl
done
sh $G/scripts/when-quiet.sh true
RUNTIME=$ENGINE sh $G/scripts/full-run2.sh int24-s1 $CAND 2026112501 --workers 24 > $G/logs/int24-s1.log 2>&1
sh $G/scripts/when-quiet.sh true
RUNTIME=$ENGINE sh $G/scripts/full-run2.sh int24-s2 $CAND 2026112502 --workers 24 > $G/logs/int24-s2.log 2>&1
sh $G/scripts/verify-one2.sh v4 20 2026112501 16 $CAND $ENGINE
say "queue 6 done"
