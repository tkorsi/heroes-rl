#!/bin/sh
# After queue 8, on a free machine: main against src-wip41 in actor replays (one thread, no cyclic collector) and in
# live games played alone (seed 1, 40 days; seed 3, 120 days): processor time per decision of actor and engine.
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10; P=$A/python/mlx-v1/venv/bin/python
say() { echo "== $(date) $*"; }
say "priority $(ps -o pri= -p $$); waiting for queue 8"
until grep -q "queue 8 done" $G/logs/queue-8.log; do sleep 15; done
sh $G/scripts/when-quiet.sh true
sh $G/scripts/ab.sh $G/src-2c1672b $G/src-wip41
sh $G/scripts/ab.sh $G/src-2c1672b $G/src-wip41 $G/replays/s3-d120.jsonl.gz
for SEED in 1 3; do
  DAYS=40; [ $SEED = 3 ] && DAYS=120
  for SIDE in main new main new; do
    if [ $SIDE = main ]; then T=$G/src-2c1672b; unset HEROES_VCMI; unset PYTHONPATH
    else T=$G/src-wip41; export HEROES_VCMI=$A/runtime/claude-g10-e2/bin/vcmiclient; export PYTHONPATH=$G/pylib; fi
    (cd $T && $P $G/scripts/profile_one.py $G/prof $SEED --max-days $DAYS --tree $T 2>/dev/null | tail -1) | tee -a $G/results/live-alone.jsonl
  done
done
unset HEROES_VCMI PYTHONPATH
say "queue 9 done"
