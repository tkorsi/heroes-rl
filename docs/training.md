# Training Heroes RL

The default trainer uses CPU actors and an Apple silicon GPU for recurrent PPO. The reference PPO implementation uses PyTorch. Both backends play through the same patched VCMI engine.

## Set up a new checkout

Use CPython 3.13 on Apple silicon. From the repository root, install the pinned learner dependencies in a virtual environment and choose a directory outside the checkout for game data, runtimes and training state.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r dependencies/learner-mlx.lock
export HEROES_PYTHON="$PWD/.venv/bin/python"
export HEROES_ARTIFACTS="$HOME/.local/share/heroes-rl"
mkdir -p "$HEROES_ARTIFACTS"
```

Supply a compatible Heroes III installation so that its original `Data` folder is available at `$HEROES_ARTIFACTS/game-data/Data`. Follow the [VCMI build instructions](architecture.md#engine) through the CMake build. For the artifact directory chosen above, run the freeze step there:

```sh
python3 tools/freeze_runtime.py "$HEROES_ARTIFACTS/runtime/local-v8-equipment-controls-v1"
```

To use a different prepared runtime, set `HEROES_VCMI` to its patched `vcmiclient` executable. For the Torch reference backend, install [the base learner lock](../dependencies/learner.lock) and pass `--backend torch` to both the check and train commands.

## Resume training

The default map is 36×36 with four neutral gold mines. The default state starts from the [published checkpoint](../checkpoints/update-002025.pt) on the first run. The trainer verifies the saved weight digest when loading. Subsequent runs resume the most recent checkpoint in that state. The default directories are `training-codex-2025-fast` for the Codex checkout and `training-2025-fast` for other checkouts, under `HEROES_ARTIFACTS`. The fast trainer starts its own history from 2025, in a separate directory from any runs started with the earlier default.

```sh
./train.command check
./train.command 100
```

For a separate run, create a state directory with its own copy of checkpoint 2025:

```sh
state=$(mktemp -d "$HEROES_ARTIFACTS/state-2025.XXXXXX")
echo "$state"
mkdir "$state/checkpoints"
cp checkpoints/update-002025.pt "$state/checkpoints/"
printf '%s\n' '{"checkpoint":"checkpoints/update-002025.pt"}' > "$state/latest.json"
./train.command check --state "$state"
./train.command 100 --state "$state"
```

To train a randomly initialized network in a separate empty state instead:

```sh
./train.command 20 --fresh --state "$(mktemp -d "$HEROES_ARTIFACTS/fresh.XXXXXX")"
```

## Choose a batch

`./train.command` prompts for a positive game count. With an initialized default state, `./train.command 20` collects 20 usable games and performs one PPO update. Pass `--state DIR` to select a particular run. Concurrent runs need distinct state directories. Ctrl-C stops the run; the next launch resumes from the last saved update. Memory use for a single batch of 1,000 full-length games is unmeasured.

| Option | Effect |
| --- | --- |
| `--games-per-update 5` | Update after each five usable games. |
| `--workers N` | Cap concurrent games. The default is 14, limited to the batch size. |
| `--keep-game-files` | Retain engine logs and replay ledgers for inspection. Full ledgers can grow by roughly 130 KB per decision. |
| `--both-seats` | Also learn from the opponent in decisive games. Both seats already contribute on 365-day draws. |
| `--stack-moves N`, `--windows N`, `--decisions N` | Change action budgets. Record these when comparing policies. |

The default MLX update makes one pass over the batch with ten trajectories per optimizer step and an Adam learning rate of 0.001. Use `--windows-per-minibatch` and `--windows-per-game` for the window-based update. Both seats use the same frozen weights. By default the learner color alternates between games. Runs record engine and policy seeds, outcomes, samples, optimizer steps, model-weight hashes and phase timings in `log.jsonl`. Use `--starts FILE` for a weighted mix of starting positions or layouts and `--seed N` to reproduce seed selection. The launcher lists further options with `./train.command 20 --help`.

## Read a training result

A verified win scores +1 and a loss scores −1. When neither side wins within 365 days, both seats score −1 and the game counts once as a cutoff. Crashes and inconsistent endings are replaced. Opening and immediately closing the same unchanged window twice in succession costs the trained seat 0.0001 on the second opening, capped at 0.2 per game. Opening a town to recruit or build is free.

Measure collection time and PPO time separately. Game length and the number of decisions can vary by checkpoint and seed. [Performance tools](../tools/perf/README.md) · [Research journal](research-log.md)

For development, run the existing tests in the learner environment. They exercise the socket protocol, actor, learner and checkpoint code against a scripted engine.

```sh
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest
```
