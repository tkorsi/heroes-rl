#!/bin/sh
# Same seeded games, PPO input in game order, digests per trajectory:  verify-pair.sh LABEL GAMES SEED WORKERS TREE_A RUNTIME_A TREE_B RUNTIME_B
# (RUNTIME "-" = the default engine). Both runs at once, each with its own copy of checkpoint 2036. Writes results/verify-LABEL-{a,b}.jsonl.
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10; P=$A/python/mlx-v1/venv/bin/python
LABEL=$1; GAMES=$2; SEED=$3; W=$4
one() {  # side tree runtime
  S=$G/states/verify-$LABEL-$1; rm -rf $S; mkdir -p $S/checkpoints
  cp $G/base/checkpoints/update-002036.pt $S/checkpoints/; cp $G/base/latest.json $S/
  rm -f $G/results/verify-$LABEL-$1.jsonl
  if [ "$3" = "-" ]; then unset HEROES_VCMI; else export HEROES_VCMI=$A/runtime/$3/bin/vcmiclient; fi
  cd $2 && $P $G/scripts/verify_train.py $2 $G/results/verify-$LABEL-$1.jsonl -- train $GAMES --state $S --seed $SEED --workers $W > $G/logs/verify-$LABEL-$1.log 2>&1
}
echo "== $(date) verify $LABEL: $5 ($6) against $7 ($8), $GAMES games, seed $SEED, $W workers each"
one a $5 $6 & one b $7 $8 & wait
echo "== $(date) verify $LABEL done"
