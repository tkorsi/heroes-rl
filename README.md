# Heroes RL

A policy learns to play the Heroes III adventure map by playing against itself
in the real [VCMI](https://github.com/vcmi/vcmi) engine. It sees what a player
sees, as structured data, and chooses typed actions: move a hero, open a town,
recruit, build, answer a dialog, end the turn. Battles are fought by VCMI's
BattleAI. Screenshots are never an input.

Training is recurrent PPO. The whole loop is the `heroes/` package, about 8,000
lines, of which 3,200 are the typed game contract.

**More:** [how it works](docs/architecture.md) ·
[research journal](docs/research-log.md) ·
[current restrictions](docs/temporary-restrictions.md)

## Train

You run training yourself, in Terminal, on the Mac that has the prepared VCMI
runtime and your Heroes III data. Reading or editing this project never starts it.

```sh
cd ~/Projects/Heroes
./train.command 1000          # play 1,000 more games, one PPO update per 100
./train.command check         # show what a run would start from; plays nothing
```

`train.command` uses the training Python, keeps the Mac awake and passes every
other argument on:

| Option | Meaning | Default |
| --- | --- | --- |
| `--games-per-update N` | games played with the same weights before one update | 100 |
| `--workers N` | games in play at once, one process each | 10 |
| `--keep-game-files` | keep each game's engine folder and logs, in `kept-games/` | deleted |
| `--backend torch` | slow reference update without MLX | `mlx` |
| `--windows-per-minibatch N` | loss windows of 16 decisions per optimizer step | 2 |
| `--windows-per-game N` | windows of a minibatch that come from the same game and share one replay of it; see [Making it faster](#making-it-faster) | 1 |
| `--learning-rate X` | Adam step size; 8.5e-4 goes with 16 windows per step, see the same section | 3e-4 |
| `--state DIR` | where checkpoints and the log go | `~/Projects/Heroes-artifacts/training` |
| `--fresh` | start from a random policy when the state folder is empty | off |

Every default lives in [`heroes/config.py`](heroes/config.py). Paths follow
`HEROES_ARTIFACTS`, `HEROES_VCMI` and `HEROES_PYTHON` when those are set.

Press Ctrl-C to stop. Games of the unfinished batch are discarded and the next
run continues from the last saved update.

## What a run does

```text
newest checkpoint
   │  ten actor processes, each one game at a time, both seats the same weights
   ▼
100 games  ──►  one PPO update on the learner seat of each  ──►  next checkpoint
   ▲                                                                  │
   └──────────────────────────────────────────────────────────────────┘
```

1. **Restore.** The newest checkpoint in the state folder. An empty state
   folder imports the last checkpoint of the previous pipeline once (weights,
   Adam, shuffle generator and counters) and saves it as its first file.
2. **Play.** Each actor starts a headless `vcmiclient`, and the policy answers
   every decision of both seats. One seat is the learner, alternating by game.
   A game counts when the engine finished it with a verified win and loss, or
   when it was stopped at day 365. A crashed or inconsistent game is logged,
   never trained on, and replaced by another one.
3. **Update.** After 100 usable games, one PPO epoch over all of them. The
   update recomputes the recurrent state from the start of each game with the
   current weights; nothing stored during play is reused except the action
   taken, its probability and the value estimate. This replay is most of an
   update's time; see [Making it faster](#making-it-faster).
4. **Save.** One file per update, then back to step 2.

The state folder holds everything:

| Path | Content |
| --- | --- |
| `checkpoints/update-NNNNNN.pt` | model, Adam state, shuffle generator and counters after that update |
| `latest.json` | which checkpoint is newest; switched last, so a crash never leaves a half state |
| `log.jsonl` | one line when a game is handed out, one when it ends and one per update: outcome, days, samples, losses, timings |
| `games/` | engine folders of games in play, deleted as each game ends and emptied when a run starts |
| `lock` | held by the running trainer, so two runs cannot share one state folder |

## Where the code is

| Path | What it holds |
| --- | --- |
| [`heroes/train.py`](heroes/train.py) | the loop above and the pool of actor processes |
| [`heroes/actor.py`](heroes/actor.py) | play one game, build the learner's trajectory |
| [`heroes/engine/`](heroes/engine/) | launch VCMI, speak its socket protocol, classify how a game ended |
| [`heroes/game/`](heroes/game/) | typed observations, offers and actions; the interaction guard |
| [`heroes/model/policy.py`](heroes/model/policy.py) | the network and its action distribution |
| [`heroes/learner/`](heroes/learner/) | PPO targets and windows, the reference update, the MLX update |
| [`heroes/checkpoint.py`](heroes/checkpoint.py) | save, load, and the one-time import of earlier checkpoints |
| [`engine/`](engine/) | the C++ side: VCMI patches and the ExplorerRL plugin that asks the policy |
| [`tests/`](tests/) | tests against a scripted stand-in engine; no VCMI needed |

## Status

The last checkpoint of the previous pipeline is update **1,785** (9,922 games,
13.7 million decisions). This pipeline continues from it.

The refactored pipeline was first verified without the engine: three captured
games replay through the moved policy bit for bit, the MLX update receives
byte-identical inputs for them, the previous pipeline's checkpoints import with
identical weights and optimizer state, and the loop runs end to end against a
scripted engine.

On 2026-11-18 it ran on the Mac for the first time: two runs of ten games and
one PPO update each, into a scratch folder. All 20 games ended naturally, both
updates were saved, and the second run resumed from the first. A game stopped
at a limit, a failed game and an interrupted run have not been seen with the
real engine yet, and nothing has run in the real state folder.

To try a change without touching the real state, use a scratch folder:

```sh
./train.command 10 --games-per-update 10 --state ~/Projects/Heroes-artifacts/training-trial
```

The MLX update checks its first minibatch against the probabilities recorded
during play and stops if they disagree, so a mismatch between play and update
cannot train silently.

## Making it faster

`log.jsonl` records, for every game, its wall time and recipe compile time, and
for every update how long collecting, planning and the GPU steps took.
[`tools/perf/`](tools/perf/README.md) has scripts that measure the parts on real
games.

The two runs of 2026-11-18, ten games per update:

| | Learner decisions | Collecting | PPO update |
| --- | --- | --- | --- |
| Run 1 | 105,875 | 9 min 49 s | 10 min 52 s |
| Run 2 | 172,718 | 16 min 44 s | 28 min 12 s |

The first update of 100 games, on 2026-11-19, took 4 h 39 min for 1,372,976
learner decisions that took 1 h 50 min to collect, on a Mac that was short of
power for most of it. What these numbers and the measurements of 2026-11-19 in
the [journal](docs/research-log.md) say:

- **The charger.** Ten games need more power than a 65 W charger supplies.
  Once the battery is empty the Mac slows to what the charger gives: games fell
  from about 100 to about 34 decisions per second each, for the rest of the
  night. `pmset -g batt` must not show the charge falling while on AC.
- **The update is most of a cycle, and nearly all of it is replay.** Every
  optimizer step recomputes the recurrent state of the two games its windows
  come from, from their first frame with the current weights: about 600
  replayed decisions for each one that is trained on. This part of a game's
  cost grows with the square of its length. Run 2 above held one game with
  63,390 learner decisions and needed 9.8 ms per decision, run 1 needed 6.2 ms.
- **The replay cannot be skipped.** Keeping the states before a game's windows
  from one optimizer step to the next was built twice and removed twice. Kept
  for many steps, they changed the gradient beyond use, and replaying only the
  last 8,000 or 32,000 frames before a window did not help: the policy's memory
  spans the whole game. Kept for only 8 small steps, each gradient stayed close
  to the exact one, and a whole update still moved the policy 2.5 to 5 times
  further than the exact update, because a state from before earlier steps
  hides how far the policy has already moved on that game.
- **The replay can be shared.** `--windows-per-minibatch 16 --windows-per-game 8
  --learning-rate 8.5e-4` takes eight windows of a game per step, so one replay
  serves all eight, still with the current weights: every step is an exact PPO
  step on 256 decisions instead of 32. On 12 real games with the GPU otherwise
  free a whole update took 248 s against about 990 s by default, and on a batch
  of production size (those games eight times, 1.08 million decisions) the
  rates of the first 120 steps project to 38 minutes against 2.9 hours. The
  learning rate belongs to the setting: 8 times fewer steps at 3e-4 change the
  weights less than half as far as the default update, and at 8.5e-4 as far.
  Judged on 12 games no update saw, the update then moved the policy no further
  than the default and fitted the value at least as well. Which setting wins
  more games per game played is a question for long training runs, so the
  defaults are unchanged.
- **Ten games on ten actors take as long as the longest game.** With 100 games
  per update the actors stay busy and this evens out.
- A game plays at about 105 decisions per second of both seats together when
  the Mac has the power it needs, and its recipe compiles at about 1.6 ms per
  learner decision.
- `--workers`. Ten games at once was chosen for the old pipeline.
- `--games-per-update`. Fewer games per update means more frequent updates on
  less data.

About half of a game's wall time is spent waiting for the engine. It sends
every observation twice, as a view and again inside the decision that follows,
and writes about 130 KB of replay log per decision that nothing reads, several
gigabytes per long game. Its end-of-game shutdown is built on the replay
recorder ([patch 0015](engine/patches/0015-contextual-terminal-barrier.patch)),
so neither can be switched off from Python; both need an engine change.

A seat answers about 100 decisions per game day, most of them stack moves
within the guard's daily budget, and the quarter of games that reach day 200
hold 60% of all decisions. Shorter games would cut play and update time alike.
That is a question of the guard and the action space, not of code speed.

## Develop

```sh
python -m pip install pytest
python -m pytest              # about 30 seconds, no engine, no saved games
```

The tests play scripted games through the real sockets, environment, actor,
PPO reference update and checkpoint code. They cannot tell you whether VCMI
behaves; only a real run can.

To rebuild the engine see [the architecture guide](docs/architecture.md#engine).

## Rules of the project

- Read [AGENTS.md](AGENTS.md) before changing or publishing anything. The user
  launches all training.
- Record meaningful steps in the [research journal](docs/research-log.md).
- Weights, captures, runtime builds and game data stay in private storage.

VCMI is GPL-2.0-or-later; see [third-party notices](THIRD_PARTY_NOTICES.md).
Original Heroes III assets are user-supplied and are not redistributed here.
