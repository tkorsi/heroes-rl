#!/bin/sh
# The same seeded games on two engines: one digest per game of what the learner reads; days and choices too.
#   engine-ab.sh RUNTIME_A RUNTIME_B SEED GAMES MAXDAYS TREE
A=$HOME/Projects/Heroes-artifacts; G=$A/claude-g10; P=$A/python/mlx-v1/venv/bin/python
for R in $1 $2; do
  echo "== $(date) engine $R seed $3"
  (cd $6 && $P tools/perf/bench_play.py $G/prof --workers 8 --games $4 --max-days $5 --seed $3 --vcmi $A/runtime/$R/bin/vcmiclient \
      --label "$R" --out $G/results/engine-ab.jsonl > /dev/null)
done
$P - <<PY
import json
rows = [json.loads(l) for l in open('$G/results/engine-ab.jsonl') if l.strip()]
a, b = rows[-2], rows[-1]
same = sum(x == y for x, y in zip(a['digests'], b['digests']))
print(a['label'], a['wall_seconds'], a['decisions'], a['decisions_per_second'], '|', b['label'], b['wall_seconds'], b['decisions'], b['decisions_per_second'])
print('digests equal', same, 'of', len(a['digests']), '; days equal', a['days'] == b['days'], '; choices equal', a['choices'] == b['choices'], '; failures', len(a['failures']), len(b['failures']))
PY
