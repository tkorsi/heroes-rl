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
compiling the recipe. "Waiting for the engine" includes the JSON parsing of each
message, which runs in the reader thread.

The line also carries a `digest` of everything the learner reads of the game.
A game is determined by its seed, so a change to the play path that is meant to
be exact must leave the digest of a seeded game unchanged; run the same
`--seed` before and after, alternating, and compare digest and time.

```sh
$P tools/perf/profile_game.py $S --max-days 60 --out $S/results/profile-game.jsonl
$P tools/perf/profile_game.py $S --max-days 60 --profile $S/results/game.prof   # per function; slows Python
```

## How fast several actors play

Two measurements of the play phase alone, with the actor pool of training.

```sh
# the same seeded games on ten actors, to their end at day 60: one digest per game, time per decision
$P tools/perf/bench_play.py $S --workers 10 --games 10 --max-days 60 --out $S/results/play.jsonl
# every actor busy for 130 s after a 20 s warm-up: decisions per second of all games together
$P tools/perf/bench_play.py $S --workers 10 --seconds 130 --warmup 20 --out $S/results/play.jsonl
```

The first is how a change to the play path is shown to be exact on more than
one game: game `n` of a seed is the same game in every run, whatever the number
of actors, so every digest must stay as it was. If a game gives no trajectory
the run compares nothing, however fast it was: the report is still written,
with the reason of each failure, and the exit status is 1. A game stopped at
the day limit has a trajectory and counts. Every call plays in a folder of its
own inside the state folder, so two calls on the same folder and seed do not
remove each other's games; they still share the Mac, so only their digests
compare, not their times. The second is the rate a
training run collects at while no actor is idle. It depends on everything else
the Mac is doing, so compare versions alternating, and compare actor counts in
one sitting. `--vcmi` selects another engine build; `HEROES_VCMI` does the same.

Never edit the source tree a measurement is running from: an actor imports the
recipe compiler when its first game ends, and a file changed meanwhile fails
that game. Measure from a copy (`git archive HEAD | tar -x -C SOMEWHERE`).

`--both-seats` builds and compiles the opponent seat's trajectory as well, as
training does with that setting. The report then also carries `seat_digests`,
one digest per colour for every game, with `null` where a seat gave no
trajectory; `digests` stays the learner seat's, so it still compares with a
run without the flag. `--guard NAME` plays with another
interaction guard, for example
`--guard "$(python -c 'from heroes.game import guard; print(guard.version(stack_moves=4, windows=6))')"`.
A guard with other budgets samples from another support, so these are other
games: compare their counts (the report lists decisions, days and the learner
seats' actions by kind), not their digests.

One difference between two runs of the same code is not a change of play. When
a game ends naturally, the engine's network thread ends it while the AI thread
of the player who moved last may still be reporting that move or asking for
the next decision. The terminal then comes before the move's result, after it,
or after the next decision, which it withdraws. Training reads the same
samples in all three cases, since the last action owns the reward either way,
but a digest covers the seat's whole history and so has three values for that
seat of that game. On the 14 games of seed 3 to day 60 no seat showed this in
seven two-seat runs with the standard guard; with stack moves 4 and windows 6
one seat of one game did, in ten runs with the digests 7, 2 and 1 times. It
looks like this: one seat of one naturally finished game differs, while the
statuses, the days, the sample count and the learner seats' actions by kind
are the same. A difference in a game stopped at the day limit, in a sample
count, or in more than that one seat is something else.

## A fixed batch of real games

Plays games with the folder's checkpoint and keeps each finished game, compiled
recipe included, in `$S/batch/`. Three actors leave room for other work.

```sh
$P tools/perf/collect_games.py $S 12 --workers 3
```

## The PPO update on that batch

Every run starts from the same weights, games and shuffle seed. `--steps` stops
a run early; the line then carries the time per step and a projection for the
whole update. A second update on the same GPU makes every time longer, so
compare runs made under the same conditions, alternating.

```sh
C=$S/checkpoints/$(sed -n 's/.*"checkpoints\/\(.*\)".*/\1/p' $S/latest.json)
# the default update, 300 steps
$P tools/perf/bench_update.py --batch $S/batch --checkpoint $C --steps 300 --out $S/results/update.jsonl
# whole updates with 8 and 16 windows of a game sharing one replay
$P tools/perf/bench_update.py --batch $S/batch --checkpoint $C --windows-per-minibatch 16,32 --windows-per-game 8,16 --out $S/results/update.jsonl
# a batch of production size: every game used 8 times
$P tools/perf/bench_update.py --batch $S/batch --checkpoint $C --copies 8 --steps 100 --out $S/results/update.jsonl
# the same with a 16 GiB GPU plan cache instead of the 4 GiB training uses (this run only)
$P tools/perf/bench_update.py --batch $S/batch --checkpoint $C --copies 8 --steps 100 --plan-cache-gib 16 --out $S/results/update.jsonl
# where the Python side of the update spends its time
$P tools/perf/bench_update.py --batch $S/batch --checkpoint $C --copies 8 --steps 70 --profile $S/results/update.prof
```

Scripted games need neither VCMI nor a checkpoint and show how the time grows
with the length of a game:

```sh
$P tools/perf/bench_update.py --synthetic 20000,5000,5000 --cache $S/synthetic --steps 200
```

## What whole updates learn from the same games

Timing says nothing about whether a faster setting is as good an update. This
script makes one whole update per setting and shuffle seed, each from the same
checkpoint with its Adam state on the same batch, and then judges every set of
weights without taking a step, on the batch and on games no update saw: how far
the policy moved (`kl`, `clip_fraction`), the first-order estimate of how much
better it became (`gain`), the value error, each with an interval against a
reference update, and the lengths of the weight changes and the cosines between
them. Two seeds of one setting show how far apart chance alone puts two updates;
read every difference against that.

```sh
# games no update sees: a second folder with the same checkpoint, another seed
H=~/Projects/Heroes-artifacts/perf-opt-holdout
mkdir -p $H/checkpoints && cp $S/latest.json $H/ && cp $C $H/checkpoints/
$P tools/perf/collect_games.py $H 12 --workers 3 --seed 2
# the default twice, and 16 windows per step with and without its learning rate
$P tools/perf/update_quality.py --batch $S/batch --holdout $H/batch --checkpoint $C --work $S/quality \
    --settings 2:1,16:8,16:8:8.5e-4 --seeds 0,1 --out $S/results/quality.jsonl
```

Weights and verdicts stay in `--work`, so an interrupted comparison continues
and a later call adds settings without repeating the earlier ones. A default
update of 12 games takes about 17 minutes on a free GPU.
