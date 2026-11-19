# Measuring training speed

These scripts start the real engine or time the GPU update. By rule 1 of
[AGENTS.md](../../AGENTS.md) the user launches them. They write only to the
state folder they are given and never train or save a checkpoint. Use a folder
of its own, never the one a training run is using.

Before timing anything, check that the Mac is not power-limited: the charge
must not be falling while on AC.

```sh
pmset -g batt
pmset -g ac        # the charger's wattage
```

Set up once. The folder needs a checkpoint and its `latest.json`; copying the
newest ones from a training state folder is enough.

```sh
P=~/Projects/Heroes-artifacts/python/mlx-v1/venv/bin/python
S=~/Projects/Heroes-artifacts/perf-opt
mkdir -p $S/checkpoints
cp ~/Projects/Heroes-artifacts/training/latest.json $S/
cp ~/Projects/Heroes-artifacts/training/checkpoints/$(sed -n 's/.*"checkpoints\/\(.*\)".*/\1/p' $S/latest.json) $S/checkpoints/
```

## Where a game's time goes

Plays one game in this process, stopped after day 60, and prints one JSON line:
decisions per second, and the seconds spent waiting for the engine, decoding
its messages, encoding frames, in the policy's forward pass, sampling, and
compiling the recipe.

```sh
$P tools/perf/profile_game.py $S --max-days 60 --out $S/results/profile-game.jsonl
$P tools/perf/profile_game.py $S --max-days 60 --profile $S/results/game.prof   # per function; slows Python
```

## A fixed batch of real games

Plays games with the folder's checkpoint and keeps each finished game, compiled
recipe included, in `$S/batch/`. Three actors leave room for other work.

```sh
$P tools/perf/collect_games.py $S 12 --workers 3
```

## The PPO update on that batch

Each value of `--refresh` (`state_refresh_steps`) runs from the same weights,
games and shuffle seed. `--steps` stops a run early; the line then carries the
time per step and a projection for the whole update. Do not time while another
update is using the GPU.

```sh
C=$S/checkpoints/$(sed -n 's/.*"checkpoints\/\(.*\)".*/\1/p' $S/latest.json)
$P tools/perf/bench_update.py --batch $S/batch --checkpoint $C --refresh 1,1024 --repeat 2 --steps 300 --out $S/results/update.jsonl
$P tools/perf/bench_update.py --batch $S/batch --checkpoint $C --refresh 1024 --out $S/results/update.jsonl   # a whole update
$P tools/perf/bench_update.py --batch $S/batch --checkpoint $C --refresh 1024 --steps 300 --profile $S/results/update.prof
```

Scripted games need neither VCMI nor a checkpoint and show how the time grows
with the length of a game:

```sh
$P tools/perf/bench_update.py --synthetic 20000,5000,5000 --cache $S/synthetic --refresh 1,1024 --steps 200
```

## What reusing states costs in accuracy

Runs the exact update and, beside it, compares the states, log-probabilities,
values and gradient each `--refresh` value would have given for the same
minibatch and weights. Reported for the whole run and by fifths, because
staleness grows as the weights move.

```sh
$P tools/perf/state_fidelity.py --batch $S/batch --checkpoint $C --refresh 64,256,1024,1000000000 --every 20 --out $S/results/fidelity.jsonl
```
