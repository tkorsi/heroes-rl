#!/bin/sh
# One live 40-day training game, seed S, with tree+engine A then B, twice, alternating: actor and engine CPU per decision.
#   live-ab.sh TREE_A RUNTIME_A TREE_B RUNTIME_B [SEED] [DAYS]
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10; P=$A/python/mlx-v1/venv/bin/python
S=${5:-1}; D=${6:-40}
echo "== $(date) priority $(ps -o pri= -p $$) seed $S days $D"
for pair in "$1 $2" "$3 $4" "$1 $2" "$3 $4"; do
  set -- $pair
  (cd $1 && HEROES_VCMI=$A/runtime/$2/bin/vcmiclient $P $G/scripts/profile_one.py $G/prof $S --max-days $D --tree $1 2>&1 | tail -n 1 | \
   $P -c "import json,sys; t=sys.stdin.read()
try:
    r=json.loads(t); print('$1'.split('/')[-1][:12].ljust(12), '$2'[:14].ljust(14), r['choices'], r['digest'], 'wall', r['wall_seconds'], 'actor', r['actor_us_per_decision'], 'engine', r['engine_us_per_decision'], 'us/dec')
except Exception: print('FAILED', t[-500:])")
done
echo "== $(date) done"
