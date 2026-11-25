#!/bin/sh
# One normal training batch, timed: ./train.command 100 from a fresh copy of checkpoint 2036.
#   full-run.sh LABEL TREE SEED [more train arguments]
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10; LABEL=$1; TREE=$2; SEED=$3; shift 3
S=$G/states/$LABEL
rm -rf $S; mkdir -p $S/checkpoints; cp $G/base/checkpoints/update-002036.pt $S/checkpoints/; cp $G/base/latest.json $S/
[ -n "$RUNTIME" ] && export HEROES_VCMI=$A/runtime/$RUNTIME/bin/vcmiclient
echo "== $(date) priority $(ps -o pri= -p $$) label $LABEL tree $TREE seed $SEED engine ${HEROES_VCMI:-default} PYTHONPATH ${PYTHONPATH:-none} arguments: $*; power: $(pmset -g batt | head -1)"
echo "busiest:"; ps -Ao pcpu,command | sort -nr | head -4 | cut -c1-120
sh $G/scripts/sampler.sh $G/logs/$LABEL.cpu & W=$!
cd $TREE && /usr/bin/time -l ./train.command 100 --state $S --seed $SEED "$@"
STATUS=$?
kill $W
echo "== $(date) exit $STATUS"
