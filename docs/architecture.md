# Architecture and research guide

The [README](../README.md) defines the current workflow and accepted settings.
This guide maps that workflow to its implementation. Historical contracts remain
in [project history](project-history.md); experiment evidence belongs in the
append-only [research journal](research-log.md).

## Follow one cycle

```text
run-100-games.command → continue-selfplay.sh → tools/run_selfplay_training.py
  collect + prepare a complete batch → PPO → save and restore-check
  repeat remaining batches → one independently verified diagnostic
```

| Stage | Code and responsibility |
| --- | --- |
| Manual entry | [run-100-games.command](../run-100-games.command) accepts game count/backend; [continue-selfplay.sh](../continue-selfplay.sh) selects the locked Python and macOS sleep assertion |
| Coordinate | [run_selfplay_training.py](../tools/run_selfplay_training.py) owns source checks, seed reservations, ten rolling actors, batch boundaries, publication and retirement |
| Observe and act | [contextual.py](../experiments/learner/contextual.py) defines typed state/actions; [contextual_rollout.py](../experiments/learner/contextual_rollout.py) collects player histories; [interaction_guard.py](../experiments/learner/interaction_guard.py) applies versioned action restrictions |
| Store and admit | [contextual_storage.py](../experiments/learner/contextual_storage.py) transfers bound captures; [selfplay_training.py](../experiments/learner/selfplay_training.py) checks receipts, prepares histories and restores complete checkpoint pairs |
| Learn | [ppo.py](../experiments/learner/ppo.py) owns GAE, shuffled windows, clipping and update order; [builtin_gameplay.py](../experiments/learner/builtin_gameplay.py) supplies transactional rollback |
| Inspect progress | [game_progress.py](../experiments/learner/game_progress.py) renders game cells and PPO counters; display files cannot authorize training |
| Verify a diagnostic | [run_selfplay_one_game.py](../tools/run_selfplay_one_game.py), [headless replay](../experiments/headless/) and retained [validation probes](../validation/probes/) reconstruct both policies and replay native actions |

Preparation overlaps capture; all declared training captures and owned-worker
cleanup must complete before an update. Capture wins are
engine-reported, whereas the final diagnostic has independent reconstruction and
replay. Saved updates and diagnostic success are distinct outcomes. Fresh seeds
remain consumed even when an attempt fails or never starts.

## Policy and update backends

[contextual_model.py](../experiments/learner/contextual_model.py) combines a typed
semantic tree, map CNN and owned-state features with recurrent history memory.
The adapter exposes player-visible information; it does not substitute an
omniscient state or image-based policy. [native_recipes.py](../experiments/learner/native_recipes.py)
compiles checked histories into the representation shared by update backends.

| Backend | Implementation | Boundary |
| --- | --- | --- |
| Metal, current default | [metal_backend.py](../experiments/learner/metal_backend.py), `metal_gru.py`, `metal_tree.py`, `metal_map_delta.py`, `metal_map_kernels.py` and [Metal shaders](../native/learner/metal/) | Adopted full-D source; FP32 MPS and custom GRU/tree/sparse-map kernels; feature size 32, hidden size 64 |
| CPU, explicit fallback | [native_backend.py](../experiments/learner/native_backend.py), [kernels.cpp](../native/learner/kernels.cpp), `text_replay.h`, `replay_tape.h` | Source-, binary- and runtime-bound native implementation; also used by diagnostic verification |

Metal's prefix cache stores immutable packed constants, bounded to 4 GiB on the
device. It never caches encoded recurrent features across weight updates. Each
prefix is recomputed with current weights. Sparse map execution avoids the large
neighborhood materializations that caused the initial full-cohort OOM. Custom
tree/backward operations and the full-D GRU arithmetic belong to this backend;
PPO settings and batch admission stay shared. Model and Adam tensors return to
CPU, including on failure, and the original optimizer configuration is restored.

The user accepted CPU/Metal numerical differences. Full D's measured speedup
and failed strict policy-comparison gate are both retained in the journal.
Those measurements predate integration into main; the new launcher default has
not received a fresh integrated runtime preflight or test-suite run under the
current no-launch instruction.
Later deterministic pool/object-reduction candidates are private investigations,
not part of this adopted implementation. These benchmarks did not publish an
official training checkpoint.

## Engine boundary

| Path | Responsibility |
| --- | --- |
| [engine/ExplorerRL](../engine/ExplorerRL/) | Visible-state adapter, typed adventure actions and policy transport |
| [engine/ExperimentReplay](../engine/ExperimentReplay/) | Native action replay, terminal/request accounting and verifier host |
| [engine/patches](../engine/patches/) | Canonical changes to pinned upstream VCMI |
| [engine/NativeSession](../engine/NativeSession/) | Historical opt-in reusable host, distinct from ordinary cold-process collection |
| [experiments/headless](../experiments/headless/) | Process ownership, environment protocol, capture and replay |
| [starting_equipment_runtime.py](../tools/starting_equipment_runtime.py) | Runtime/rule acceptance and v4 map selection |

Generated `vcmi/` is ignored, not a submodule or an independent source of fixes.
[dependencies/vcmi.json](../dependencies/vcmi.json) pins VCMI 1.7.5 at
`6901839ae9760dfcd1d5ff2c4aa4ef6968fee752` and declares all engine overlays.
Edit canonical root sources and patches; preserve exact agreement with generated
VCMI. The v4 scenario restores native calendar neutral spawning while keeping
starting taverns. [The restriction register](temporary-restrictions.md) records
the remaining simplifications and their removal boundaries.

## What the research preserves

The useful record includes the typed observation/action contract, full-history
recurrent PPO, native replay and checkpoint provenance, and measured costs of
collecting and updating on real game histories. Failed runs and negative
numerical controls are evidence alongside successful timings.

- [research-log.md](research-log.md): chronological hypotheses, criteria, source/runtime identities, measurements, failures and corrections.
- [project-history.md](project-history.md): earlier design decisions, contracts and preserved README chronology.
- [research/](research/): focused investigation write-ups; [validation/](../validation/) contains existing compact evidence and verifier code.
- [replay-viewers.md](replay-viewers.md): scope of renderer footage and the [Swift inspector](../apps/replay-inspector/); historical match viewers are not generic latest-checkpoint evaluations.
- [native-headless-shutdown.md](native-headless-shutdown.md), [native-engine-session.md](native-engine-session.md), [native-lifecycle-optimizer.md](native-lifecycle-optimizer.md): native lifecycle work and its measured limits.

Engine outcomes, verified outcomes, admitted games, published updates and playing
strength are different claims. One self-play diagnostic or a faster update does
not establish stronger play. Keep raw evidence privately and cite its exact
identity from the journal; do not publish captures, weights or diagnostic helpers.

## Compatibility and retained files

Some historical names still carry live dependencies. The current coordinator
imports helpers from `tools/benchmark_copied_training.py` and `validation/probes/`.
Source identity covers every learner module and native learner file; capability
acceptance also binds engine, headless, episode and verifier sources. Legacy
launchers and configurations participate in historical source-transition checks.
Deleting or moving them requires an explicit compatibility change, not merely
an absence of direct imports in the current launcher.

Existing tests, maps, configs and validation evidence therefore remain at their
bound paths. [Legacy local-training instructions](operations/local-script-training.md)
describe older lineages, not the current manual cycle. Likewise, dated cloud
reports and historical launch permissions are not active work orders.

The private artifacts directory contains execution dependencies as well as raw
logs: locked Python, game data, native runtime/build acceptance, checkpoint pairs,
seed ledgers and locks. The launcher retires only reviewed raw runs after copying
and exactly restoring retained state. Keep `docs/research-log.md` and its retention
markers in place; the coordinator reads them before cleanup.

## Setup reference

The prepared Mac launcher uses `HEROES_PYTHON`, or its locked interpreter
under `Heroes-artifacts/python/mlx-v1/`, installed from
`dependencies/learner-mlx.lock`. This adds MLX/MLX Metal 0.31.1 while preserving
the existing Torch 2.9.1 / NumPy 2.3.5 checkpoint and native-history dependencies.
The previous interpreter remains intact under `python/next-launch-numpy235-v1/`.
Accepted native manifests
also bind Python, Torch and compiled library identities. The validated macOS
runtime is CPython 3.13 / Torch 2.9.1; Linux CPU uses its separate CPython 3.12
lock. A successful import alone does not establish checkpoint compatibility.
See [dependency pins](../dependencies/) and [Linux setup](cloud-environment.md).

The following are **manual developer commands**, not actions performed by reading
or editing this guide. They do not authorize a training run:

```sh
python3 tools/prepare_vcmi.py
python3 tools/prepare_vcmi.py --check
```

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

Use the existing test suite and native checks appropriate to an authorized code
change. Tests requiring native assets report their requirements separately.
Read [AGENTS.md](../AGENTS.md) for launch and publication boundaries; validation
does not grant permission to publish new tests or auxiliary tooling.
