# Heroes RL

Learn Heroes III adventure-map strategy with recurrent PyTorch PPO and the real
VCMI engine. The policy receives structured, player-visible state and chooses
typed actions; it does not consume screenshots. Combat is automatic.

This README is the authoritative project overview and agreement. Updated
**25 October 2026**. Earlier stages are preserved in
[project history](docs/project-history.md); their status and authorization
paragraphs are historical, not new instructions to run experiments.

## Current state

The reviewed environment, learner, replay, cloud tooling and performance work
are consolidated on **`main`**. Start new work from `main`; intermediate
experiment branches and their PRs describe the history.

The [consolidation check](validation/project-consolidation-2026-10-25.json)
ran 1182 tests: 1156 passed and 26 required another platform or optional fixture.
A fresh nine-target macOS native build and the Swift viewer build also passed.
This check launched no gameplay.

| Area | Implemented | Current limit |
| --- | --- | --- |
| Environment | Real headless VCMI, fog-filtered structured history, explicit map/hero/town windows | Unsupported actions/dialog grammars fail explicitly |
| Actions | Coordinate targets with native routing, construction, recruitment, Tavern/hero hire, army operations and typical adventure dialogs | No tactical battle policy or path-cost oracle |
| Learning | Recurrent PyTorch actor/critic, hierarchical action probabilities, PPO, model/Adam/RNG/progress restoration | Winning strategy has not been demonstrated |
| Evidence | Native capture, independent request/state replay, policy reconstruction and strict sample admission | A seed alone does not reproduce all built-in AI randomness |
| Performance | Exact inference/cache optimizations and bounded benchmark tools | Workload timings are not universal whole-game/PPO throughput |
| Native reuse | Opt-in C++ EngineSession with load/reset/observe/step/close | Experimental; not enabled in Teacher or accepted as a general training runtime |

The fixed opponent is **Nullkiller on Easy**, with BattleAI resolving battles.
The ordinary map is the authored **hero-management-v1** scenario; see its
[configuration](configs/hero-management-v1.json).

Latest evidence is timestamped, not a live activity guarantee:

- [Teacher status](reports/cloud/teacher-status.json) and
  [continuation evidence](validation/cloud-continuation-2026-10-25.json).
- [Optimizer status](reports/cloud/optimizer-status.json) and
  [native session evidence](validation/native-engine-session-2026-10-25.json).
- [Contextual pipeline performance](validation/performance-2026-10-25.json).

The completed cloud session continued update 21 to **update 31**: 40 new
training games, ten PPO updates and eight evaluation games. Its 48 verified
games produced **zero wins, 47 natural losses and one day-limit cutoff**. The
checkpoint preserves 124 cumulative rollouts, 6,776 decisions and 253 Adam
steps; model, optimizer, RNG, progress and pool restored exactly from three
copies. The cloud Teacher stopped by its fixed deadline with no native
processes remaining. See the [final analysis](validation/cloud-final-analysis-2026-10-25.json).
Changed weights or occasional recruitment do not establish improved strategy.

## Experiment agreement

- Use existing VCMI rules, never a substitute simulator. macOS and Linux host
  the same rules; paths, transports and process supervision belong outside the
  policy/game-rule contract.
- Only information visible to the player's side enters its history. An enemy
  disappearing does not reveal its destination or fate. Window content becomes
  available only after explicitly opening that window.
- The policy chooses a map target `(x, y)`; the engine builds the route. Do not
  supply path cost or scripted movement/economy strategies.
- No demonstrations or handcrafted exploration rewards. Natural victory is +1,
  defeat -1. Technical failures and cutoffs are separate; verified cutoffs may
  use the declared value bootstrap, never a fabricated win/loss.
- Preserve source/runtime/map/checkpoint identities, every attempt, termination
  reasons and both-seat evaluation. Report failed and unstarted cases too.
- Keep opponent and interface fixed within a comparison. Self-play is a later
  direction. Interpretability is a future direction, not a gate for this work.
- Gameplay training needs its explicit experiment scope and budget. Unit tests
  may exercise synthetic PPO; they do not start a gameplay campaign.

### Current local continuation

The user authorized a new **local Mac session lasting at most two hours**, then
shutdown and analysis. New episodes use **365 game days** and **5120 total
choices** (4960 ordinary plus 160 reserve), whichever limit is reached first.
Use complete four-game PPO batches and both-seat fixed evaluation. The training
clock starts when the prepared campaign launches; allow time within the bound
for safe shutdown and checkpoint preservation.

Continue the preserved **cloud update 31**, including Adam, RNG, progress and
pool; never resume the stale local update 21. Preparation must first validate an
explicit, exactly bound Linux-to-Mac and source transition, then fresh native
capture/replay in both seats. The current ordinary cold-process backend remains
the training runtime. Claude's in-process engine work is separate and
experimental. Preparation is not training; dated local evidence must confirm
that the campaign actually launched.

The earlier cloud session is complete. Its compute cutoff was **2026-10-25
17:50 UTC** and final stop **17:53 UTC (19:53 Kyiv)**. Its dated tools remain
expired and its original 30-day configurations/results remain immutable. The
new local authorization does not restart cloud work or rewrite historical
metadata. Weights and full optimizer state remain in private storage.

Local365 preparation is recorded in
[prelaunch evidence](validation/local365-prelaunch-2026-10-25.json). The exact
cloud update-31 model, Adam, RNG, progress and pool restore on macOS. The three
reviewed performance modules are accepted only at their exact historical and
consolidated hashes, with a trained-policy parity proof; the generic portable
guard still rejects arbitrary policy changes. A fresh frozen two-seat gate
passed native replay and cleanup in 8.432 seconds (two losses, zero updates).

The new local configuration explicitly assigns `max_days=365`. A single durable
`local365/campaign-session.json` and exclusive controller lock cap the campaign
at two hours from actual launch, with five minutes reserved for cleanup and
publication. The entire controller has an independent deadline guardian; each
native game and optimizer batch retains its existing guards. The local config,
controller, parity checker and new authorization are bound into the session.
The expired cloud defaults and historical plans remain unchanged. Preparation
has not launched the PPO campaign; coordinator review/merge is still required.

## Source layout

| Path | Responsibility |
| --- | --- |
| `engine/ExplorerRL/` | Player-visible adapter and native actions |
| `engine/ExperimentReplay/` | Native replay and request accounting |
| `engine/NativeSession/` | Experimental reusable native host |
| `engine/patches/` | Canonical changes to pinned upstream VCMI |
| `experiments/headless/` | Environment, histories, capture/replay and bounded processes |
| `experiments/learner/` | Models, inference, PPO, admission, checkpoints and evaluation |
| `experiments/recording/` | Renderer footage and montages |
| `apps/replay-inspector/` | Offline macOS Swift inspector for a historical recorded match |
| `configs/`, `dependencies/` | Experiment configurations and dependency locks |
| `tests/`, `validation/` | Regression tests and compact experiment evidence |
| `tools/`, `reports/cloud/` | Operational utilities and dated progress snapshots |

This is one source repository. Generated `vcmi/` is ignored, not a root
submodule. All authored code and engine patches belong in the paths above.
Never keep a fix only in the generated dependency checkout.

## Setup and checks

Use a dedicated checkout. Source requires Python 3.11+; validated checkpoint
runtimes are more specific: macOS CPython 3.13/Torch 2.9.1 and Linux x86_64
CPython 3.12/Torch 2.9.1+cpu. A successful import does not authorize loading a
historical checkpoint. Dependency files retain exact pins.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r dependencies/learner.in
.venv/bin/python -m unittest discover -s tests -v
```

Default unit tests need no copyrighted game data. Native tests need a prepared
and rebuilt dependency; target-specific skips are reported. Linux setup and its
hashed locks are in [cloud setup](docs/cloud-environment.md). Do not install a
macOS lock on Linux.

```sh
python3 tools/prepare_vcmi.py
python3 tools/prepare_vcmi.py --check
```

Preparation pins VCMI 1.7.5 at
`6901839ae9760dfcd1d5ff2c4aa4ef6968fee752`, applies canonical patches/overlays and
refuses to overwrite unexpected changes. Preserve local edits before changing
the recipe. Install upstream VCMI build dependencies before configuring.

For an existing macOS Homebrew build environment:

```sh
cmake -S vcmi -B vcmi/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/homebrew -DENABLE_GOLDMASTER=ON \
  -DENABLE_STRICT_COMPILATION=OFF -DENABLE_EXPLORER_RL=ON \
  -DENABLE_LAUNCHER=OFF -DENABLE_EDITOR=OFF -DENABLE_DISCORD=OFF \
  -DENABLE_MMAI=OFF -DENABLE_NULLKILLER2_AI=OFF -DENABLE_TRANSLATIONS=OFF
cmake --build vcmi/build --parallel 2 \
  --target vcmiclient vcmiserver ExplorerRL Nullkiller BattleAI vcmireplay
python3 tools/prepare_vcmi.py --check
```

Use optional native sessions only through their
[documented workflow](docs/native-engine-session.md). Ordinary training keeps
the cold client and independently verified replay.

## Experiment entry points

These commands only show help; they launch no games or training:

```sh
.venv/bin/python -m experiments.learner.builtin_first_run --help
.venv/bin/python -m experiments.learner.builtin_training --help
.venv/bin/python -m experiments.learner.evaluate_builtin_opponent --help
.venv/bin/python -m experiments.headless.builtin_replay --help
.venv/bin/python tools/summarize_builtin_training.py --help
```

`builtin_first_run` is the bounded initial PPO pilot. `builtin_training` handles
verified continuation and frozen evaluation. Preparation writes an immutable
plan; execution is separate. One-off migration configurations bind historical
inputs and cannot be relabelled as new experiments. Do not admit a partial or
failed batch as an unconditional successful update.

There are two requested replay presentations: a schematic state-based view and
original VCMI graphics. The schematic video and Swift inspector show a retained
historical match with its viewed hero and army; they are not generic viewers for
the latest Teacher checkpoint. Exact recorded-match reconstruction through the
original renderer remains pending. Ordinary live renderer recording already
exists. See [viewer scope and commands](docs/replay-viewers.md).

Label which presentation is shown. Replay requires its input bindings and an
independent verifier; footage of a built-in bot is not learned-policy footage.
Earlier recorder commands remain in [project history](docs/project-history.md).

## Artifacts and development

Keep original game data, raw captures, videos, builds and large checkpoints out
of this public repository. Use ignored local artifacts and the authorized
private training-state repository. Publish compact evidence here. A backup is
complete only after verifying exact restoration of model, optimizer, RNG,
progress and opponent state.

Never delete/archive the only worktree holding a checkpoint or runtime. Preserve
and verify artifacts first. Branch cleanup must not interrupt an active pinned
Teacher, Optimizer or other checkout.

Branch from `main`, read `AGENTS.md`, edit canonical sources and run relevant
tests plus the native recipe check. Preserve provenance. Do not silently broaden
the interface or weaken admission to make a failed cohort pass.

- [Historical contracts and experiment chronology](docs/project-history.md)
- [Linux setup and historical frozen diagnostics](docs/cloud-environment.md)
- [Ordinary native shutdown correctness](docs/native-headless-shutdown.md)
- [Experimental EngineSession](docs/native-engine-session.md)
- [Replay worker and benchmark interpretation](docs/native-lifecycle-optimizer.md)

VCMI and its dependencies retain their upstream licenses; VCMI is
GPL-2.0-or-later. Original Heroes III assets are separately copyrighted,
user-supplied, and not downloaded or redistributed by this repository.
