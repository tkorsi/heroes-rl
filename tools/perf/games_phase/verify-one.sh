#!/bin/sh
# One tree on seeded games with PPO input in game order and trajectory digests:  verify-one.sh LABEL GAMES SEED WORKERS TREE RUNTIME
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10; P=$A/python/mlx-v1/venv/bin/python
LABEL=$1; GAMES=$2; SEED=$3; W=$4; TREE=$5; RT=$6
S=$G/states/verify-$LABEL; rm -rf $S; mkdir -p $S/checkpoints
cp $G/base/checkpoints/update-002036.pt $S/checkpoints/; cp $G/base/latest.json $S/
rm -f $G/results/verify-$LABEL.jsonl
if [ "$RT" = "-" ]; then unset HEROES_VCMI; else export HEROES_VCMI=$A/runtime/$RT/bin/vcmiclient; fi
echo "== $(date) verify $LABEL: $TREE ($RT), $GAMES games, seed $SEED, $W workers"
cd $TREE && $P $G/scripts/verify_train.py $TREE $G/results/verify-$LABEL.jsonl -- train $GAMES --state $S --seed $SEED --workers $W > $G/logs/verify-$LABEL.log 2>&1
echo "== $(date) verify $LABEL done"
