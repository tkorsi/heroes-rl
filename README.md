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
   taken, its probability and the value estimate.
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

The last checkpoint of the previous pipeline is update **1,784** (9,822 games,
12.9 million decisions). This pipeline continues from it.

The refactored pipeline was verified without the engine: three captured games
replay through the moved policy bit for bit, the MLX update receives
byte-identical inputs for them, checkpoint 1,784 imports with identical weights
and optimizer state, and the loop runs end to end against a scripted engine.
It has **not yet played a real game on the Mac**. Start with a small run into a
scratch folder before pointing it at the real state:

```sh
./train.command 10 --games-per-update 10 --state ~/Projects/Heroes-artifacts/training-trial
```

The MLX update checks its first minibatch against the probabilities recorded
during play and stops if they disagree, so a mismatch between play and update
cannot train silently.

## Making it faster

`log.jsonl` records, for every game, its wall time and recipe compile time, and
for every update how long collecting, planning and the GPU steps took. Things
to try first, one at a time:

- `--workers`. Ten games at once was chosen for the old pipeline.
- `--games-per-update`. Fewer games per update means more frequent updates on
  less data.

One known waste needs an engine change first. The engine writes several
gigabytes of replay logs per long game that nothing reads, but its end-of-game
shutdown is built on the replay recorder
([patch 0015](engine/patches/0015-contextual-terminal-barrier.patch)), so the
logs cannot be switched off from Python.

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
