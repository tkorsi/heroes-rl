#!/bin/sh
# Replay recorded games with two trees, alternating A B A B; prints answers, exactness, digests, CPU per decision.
#   ab.sh TREE_A TREE_B [REPLAY]
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10; P=$A/python/mlx-v1/venv/bin/python
R=${3:-$G/replays/s1-d40.jsonl.gz}
echo "== $(date) priority $(ps -o pri= -p $$) replay $R"
for T in $1 $2 $1 $2; do
  (cd $T && $P $G/scripts/actor_replay.py replay $G/prof $R --tree $T 2>&1 | tail -n 1 | \
   $P -c "import json,sys; t=sys.stdin.read()
try:
    r=json.loads(t); print('$T'.split('/')[-1][:16].ljust(16), r['answers'], 'exact' if r['mismatch'] is None else r['mismatch'], r['answers_digest'], r['trajectory_digest'], r['us_per_decision'], 'us/dec')
except Exception: print('$T', 'FAILED', t[-600:])")
done
echo "== $(date) done"
