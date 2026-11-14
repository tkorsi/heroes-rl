# Heroes RL

Learn Heroes III adventure-map strategy with recurrent PyTorch PPO and the real
VCMI engine. The policy receives structured, player-visible state and chooses
typed actions; it does not consume screenshots. Combat is automatic.

Use `/Users/orsi/Projects/Heroes` as the local working checkout. The
[research log](docs/research-log.md) and [project history](docs/project-history.md)
record previous experiments and their limits.

## Baseline for new experiments

The current launcher accepts 10–1,000 training games (default 500), with ten
rolling actors and PPO updates in complete batches of at most 100 games.
One validated self-play game follows the final saved update.
The lineage began at checkpoint 1,753; continuation restores its latest complete
published checkpoint. The user launches project scripts and training outside
Codex by default, including offline PPO benchmarks and diagnostic experiments.
The agent prepares scripts and commands, and inspects existing runs on request;
it does not launch workloads or monitor them continuously without an explicit
instruction. Earlier one-time launch permissions do not authorize new runs.
See Rule 1 in [AGENTS.md](AGENTS.md).
Start other experiments with an explicit hypothesis, configuration, seed allocation, output directory and
acceptance criteria. Choose explicitly between fresh weights and a verified
checkpoint. Keep diagnostic games separate from training admission.

The baseline retains the native adapter and replay checks, checkpoint recovery,
rollout storage, progress display and shared performance fixes. Checkpoint 1,753
is a preserved self-play checkpoint available for deliberate reuse. Existing
continuation launchers and temporary rule variants belong to their documented
historical configurations. A new experiment must declare which rules it uses.

The previous checkout state is preserved on the local branch
`archive/experiments-through-2026-11-04` at `edf0b12`, including the historical
README and retired ten-game launcher. The last trial ran ten native games
concurrently and fully verified five within the time budget. The twenty-minute
acceptance criterion remained unmet. Checkpoints and raw captures are stored
separately in the private artifacts directory.

## Search for a first verified win

Use `./play-one-selfplay.sh --check --until-first-win` to inspect the configuration.
To launch manually, use `./play-one-selfplay.sh --run --until-first-win --seconds 7200 --max-games 100`.
The mode starts with up to ten workers. Use `--workers` to compare concurrency
under the same time budget. Waiting captures occupy worker slots, and one
additional verifier slot is reserved for a victory candidate. The disk reserve
defaults to 30 GiB and is configurable with `--disk-reserve-gib`.

The experiment uses the pinned original map and large frozen checkpoint 1,753,
with hero-relative coordinates, the existing cooldowns, artifact restrictions
and troop-slot rules. Only the gameplay horizons increase, to 5,840 days and
158,720 ordinary plus 5,120 reserved choices per player. Each game reserves fresh
seeds and retains verification evidence in private storage. A victory requires
exact policy reconstruction and native replay. Cutoffs are unscored, and
interrupted games are tracked separately. Training admission and PPO updates
are disabled.

To count wins across a fixed number of attempts, use
`./play-one-selfplay.sh --run --games 20 --seconds 7200`. This uses the same
frozen setup and continues after verified victories. The dashboard and final
report show Red and Blue wins, with separate counts for cutoffs, failures and
interruptions. The time budget can end the run before all attempts settle.

## Configurable training games followed by one validated game

The current cycle uses `hero-management-v4`: both starting taverns remain built,
and native calendar spawning of new neutral armies is restored. This allows
spawning when native special-month conditions apply, not every day or every month.
Initial neutrals, their growth, armies, resources and all other rules are unchanged.
The latest complete checkpoint is 1,765. The November 13 CPU run captured
200 games, but only its first 100-game update was published. The second update
aborted in native text/autograd memory cleanup at 5,226/10,347 optimizer steps;
no 1,766 checkpoint was published. The authorized November 13–14 Metal trial
recovered the 1,765 diagnostic (verified Blue win on day 87), then captured all
100 new games in 31m11s. Its update hit the retained 26.88 GiB MPS allocator
limit at 6,022/8,583 steps after 1h28m49s on the PPO progress clock. The update
rolled back; zero games were admitted, and no 1,766 or final diagnostic exists.
Post-failure full-state restoration of 1,765 passed. The Mac slept and resumed;
the attempt took 2h03m31s on the monotonic clock and 5h59m30s wall time, including
the recovery diagnostic. This is a failed full-cycle measurement, not evidence
of the targeted fivefold speedup. Keep this run's raw captures for the offline
investigation. New cycles have no default time limit.

A private memory-corrected Metal candidate subsequently completed the same
100-game update (273,913 samples, 8,583 optimizer steps) under the original 25%
MPS cap. Full trial D also aligns GRU expression order and disables FP
contraction in its GRU kernels. Its same-input CPU/Metal comparison measured
5h30m54s versus 46m34s
including preparation: **7.105x faster for the complete update**, or
11.473x for PPO alone. Both runs observed AC/Automatic power at their endpoints;
this is one pair, not a repeated benchmark or a full campaign timing. Gameplay
and final-game verification are excluded.

The full-update policies nevertheless failed the predeclared numerical
comparison bounds: 439/450 conditional argmax matches, with KL and value errors
also above their limits. These checks compare the resulting policies on recorded
states, not playing strength. The memory and timing results do not establish
preserved learning quality. Full candidates C and D failed those bounds despite
D's improved local numerical checks. Follow-up isolated changes removed two
sources of GPU forward variability in the sampled cases, but did not generally
improve CPU agreement; they have not received another complete-update test.
The candidates remain private and unadopted; the production backend and
published checkpoint remain unchanged. See the research log for exact scope,
failed attempts, power-mode corrections and private evidence references.

The earlier v3 variant disabled calendar spawning and was used through 1,763.
The v4 selector retains its pinned native acceptance, including the successful
native-default spawning control and replay, and checks that the map changes only
the spawning override and descriptive text. This reuses historical native evidence;
it does not claim a newly executed v4 mechanics capture. Historical maps remain.
Restrictions will continue to be removed one at a time.

The November 7 continuation completed all fifty games, one PPO update and its
fully validated diagnostic, advancing checkpoint 1,756 to 1,757 in 01:23:14.
The diagnostic reached the 365-day limit without a winner. The worker-registration
race and duplicate native-recipe storage were fixed; numerical parity checks
passed without changing gameplay, PPO settings or native resource bounds.
The subsequent 100-game launch
`run-20261107T155027Z-dfd8ff19` stopped at the 30 GiB disk reserve after 13 captured
terminals, with no admitted batch or PPO update. Its analysis is in the journal.
This completed cycle establishes execution correctness, not improved playing strength.

After artifact retirement, `run-20261107T161908Z-3d80d63a` completed 100/100
training captures and one PPO update to checkpoint 1,758 in 05:38:13 on November 7.
Engine-reported training outcomes were Blue 69, Red 14 and 17 day-limit cutoffs;
training replay/policy reconstruction was intentionally not performed. The fresh
1,758 diagnostic ended in a Blue victory and passed native replay and exact
two-policy reconstruction. The November 9 read-only audit restored the full
1,758 state and confirmed clean shutdown, no active workers and no newer
registered launch. The completed behavioral analysis found a day-18 capture of
Red's empty town after Red lost both heroes attacking Blue; neither side built
or recruited town troops. The preceding 100 games already used 1,757 weights
on v3 and had 83 engine-reported natural terminals, so the decisive diagnostic
does not establish improvement from the update. Findings are in the research
journal. Its reviewed raw artifacts were retired at the next launch after the
full 1,758 checkpoint pair was copied and restored in retained storage.

The authorized November 9 run `run-20261109T141819Z-14d160b4` captured all 100
games from 1,758: 66 engine-reported Blue wins, 16 Red wins and 18 day cutoffs,
with no capture failures. The user stopped it on November 10 at 02:11 Kyiv,
before the shared deadline. PPO had reached 6,826/9,073 minibatches but had not
published a checkpoint: zero games/samples were committed to training, 1,758
remained the latest complete state, and no diagnostic ran. Closed-lid sleep
consumed over three hours of the wall-clock budget; the existing `caffeinate -i`
does not prevent that sleep. Post-stop full-state restoration and owned process
cleanup passed. Its analyzed raw data were retired at the explicitly authorized
November 10 retry, which started from 1,758 with fresh seed offset 541.

That retry, `run-20261110T113903Z-94c3b975`, completed all 100 captures, one PPO
update and the validated diagnostic on November 10 in 07:52:07 (coordinator
time). Training outcomes were 60 engine-reported Blue wins, 19 Red wins and
21 day cutoffs, with zero capture failures. All 100 games and 341,569 samples
were admitted; 10,697 optimizer steps produced checkpoint 1,759. Exact full-state
restoration, both diagnostic policy reconstructions and native replay passed,
with owned cleanup verified and protected inputs unchanged. Blue won the
diagnostic on day 19: it captured Red's town on day 12 after defeating its
14-pikeman garrison; Red's surviving 25-pikeman hero did not recover a town
within seven days. The earlier capture and a modest reduction in window-action
share are descriptive observations, not proof of stronger play. Analysis is in
the journal. Its reviewed raw artifacts were retired at the November 11 launch
after preserving the checkpoint; that reservation started at offset 642.

The user-launched `run-20261111T115614Z-5c919e30` completed on November 11
in 04:28:18 coordinator time: 100 captures, 253,760 admitted samples, 7,956
optimizer steps and checkpoint 1,760. Training engine outcomes were Blue 65,
Red 16 and 19 day cutoffs, without capture failures. The fully verified
diagnostic reached the 365-day limit without a winner or town capture. Both
sides built and recruited; garrison swaps visibly added 162 pikemen to Red's
hero and 316 pikemen plus 91 halberdiers to Blue's hero. These forces did not
produce a town capture. Final Red had no heroes and 445 garrison pikemen;
Blue retained two heroes. Full restore/preflight and owned cleanup passed.
Analysis is in the journal and this run is retention-reviewed. Keep its raw
artifacts until the next authorized launch; the next fresh seed offset is 743.

Use the existing `run-100-games.command` filename (kept for shortcuts):

```sh
./run-100-games.command --check          # Preview default 500; no games or cleanup
./run-100-games.command                 # Manually launch 500 training games
./run-100-games.command 1000             # Manually launch 1,000
./run-100-games.command --games 250      # Any integer from 10 through 1,000
./run-100-games.command 500 --hours 30   # Opt into a 30-hour wall deadline
```

Double-clicking starts 500 games. The script restores the latest complete
checkpoint and retires the preceding reviewed raw run as described below.
Games are divided into balanced complete batches of 10–100: 500 uses five
updates of 100; 101 uses updates of 51 and 50. Each batch uses its starting
policy as the frozen opponent. Every update is saved and restore-tested before
collecting the next batch. The diagnostic runs only after the final update;
an outstanding diagnostic from an interrupted prior launch is recovered first.
An interruption preserves all completed updates, but not an incomplete batch.
The existing raw `--cycles` interface retains its historical per-cycle diagnostic.

Cycles have no default campaign, capture, policy-response or replay time limit.
Sleep pauses computation; when macOS wakes, the same processes continue from
their in-memory state. A genuine stall also waits until manual cancellation;
process exits, protocol errors and resource limits still fail the run. Optional
`--hours` explicitly selects the historical bounded mode (up to ten hours per
100 games), whose wall deadline includes sleep.
On macOS the launcher runs training under `caffeinate -is`: idle system sleep
is blocked, with an additional system-sleep assertion on AC power. Display
sleep is allowed and follows the existing macOS display timer. Assertions end
when the training command exits. Leave Terminal running; minimizing it or
locking the screen is fine. Closed-lid or explicitly requested sleep can still
pause training; it does not perform games while asleep. Keep power connected
to avoid battery exhaustion. Restart, power loss or killing the process loses
the incomplete batch; the latest published full checkpoint remains recoverable.
No global power-management setting is changed.
All gameplay still stops at 365 days with 158,720 ordinary plus 5,120 reserved
choices per player. Ten rolling actors, disk/RAM reserves and PPO/Adam remain.

The dashboard keeps the old half-card width and five paired groups across.
Above 100 requested games, each cell represents five consecutive games. A cell
header names its first game; the five symbols below it retain individual results:
`B` Blue win, `R` Red win, `C` cutoff, `E` error. Waiting is `.`, preparation is
`~`, and a playing game's `0`–`9` shows calendar progress in tenths of the day
limit, not a prediction of finishing or winning. Outcome letters can appear
before preparation finishes; the separate Ready count tracks PPO readiness.
Colors supplement these symbols. Prior batches remain visible during subsequent
updates, and the PPO/checkpoint/diagnostic status remains separate from the grid.

500 games use 100 cells / ten grid rows; 1,000 use 200 cells / twenty rows,
exactly twice the working height. At 80x28 and 80x48 respectively, each cell
shows its header and five markers. Smaller windows use one-line cells where
possible or state the minimum required height. Spare rows show exact active-game
days and Red/Blue choices. The terminal's alternate screen and cursor are
restored on exit; redirected output remains plain text. PPO has a completed/total
minibatch counter and an approximate ETA. Display snapshots remain disposable
and cannot authorize training or checkpoint publication.

Training games have no native replay, policy-probability reconstruction or
behavior analytics. Their Red/Blue win totals come from engine-reported outcomes
and are explicitly unverified. Every capture in the declared batch must complete
before its PPO update. Basic source, seed, action, boundary, terminal reward and
checkpoint integrity checks remain active. Checkpoint metadata explicitly records
`training_verification=not_performed`. The historical verified mode remains a
separate contract. Previously published ten-, forty- and fifty-game cycle checkpoints restore with
their original metadata; new launches declare each complete batch size explicitly.
The existing lineage/namespace names are historical. Old eleven-, forty-one- and fifty-one-slot reservations
remain consumed; new launches reserve N training seeds plus one final diagnostic.

Collection runs ten independent actor processes, each with one PyTorch/BLAS
thread. A rolling dispatcher immediately refills an available slot; it does not
wait for a whole wave. Main prepares completed games serially while actors keep
playing. All policy binding/model loading happens before the dispatcher starts;
preparation stays on main so its signal-based deadline remains active. Completed
receipts are sorted by schedule index before the unchanged PPO shuffle. Each
PPO update starts only after its complete batch and owned-worker cleanup succeed.
The inherited native execution profile continues to govern PPO; capture/preparation
worker counts are explicitly recorded in the cycle configuration. Each completed
game is compiled into a native PPO program, then its Python history is released.
The owned worker transfers immutable typed rollouts in a compact binary file,
bound to its completion token and file hashes. A restricted reader permits only
the project's record classes. Training omits duplicate bridge/history JSONL and
the independent post-game history audit. Native terminal evidence remains active.
Diagnostic games retain full JSONL, audit and replay. Temporary raw training
streams are removed after compilation. The update retains
at most one hundred learner trajectories at a time, the existing shuffled PPO windows and the original
PPO/Adam settings. An owned-process resident-memory guard leaves 48 GiB of the
Mac's physical RAM as headroom. It stops the attempt safely if the budget is
exceeded. A system-wide memory-pressure check also stops below 20% free.
The existing 30 GiB disk reserve applies. Long-game completion
within these limits still requires observation during the user's manual run.

After the final batch state is saved and restore-tested, one fresh diagnostic self-play
game runs with both policies frozen at the new checkpoint. That game retains its
capture, runs exact two-policy reconstruction and native replay, and never enters
PPO. A failed diagnostic stops the loop while preserving the saved update.
Continuation performs any outstanding diagnostic first, using fresh seeds.
One self-play game supports inspection, not a statistical claim of stronger play.

A separate lineage and seed namespace keep this experiment distinct from the
previous failed runs. Subsequent manual launches use
`./run-100-games.command --games N` (or its read-only `--check` preview).
All reserved slots remain consumed, including unstarted games. If continuation
needs an outstanding diagnostic, one additional block of N+1 slots is reserved
for it and its unused slots remain consumed. Reports separately record engine
wins, cutoffs, PPO steps, whether weight bytes changed, the saved checkpoint and
the diagnostic validation result. No training starts during `--check`.

## Historical verified batches of ten

These are historical commands, dependent on raw history that the single-run
retention policy retires. Use the current `--cycle` launcher for continuation.
Previously, `./continue-selfplay.sh --batch-ten --check --batches 1 --hours 10` provided a
read-only preflight. To launch one batch manually, use
`./continue-selfplay.sh --batch-ten --run --batches 1 --hours 10`. Set
`--batches 10` for ten successive batches (100 training games), subject to the
same overall ten-hour deadline. No held-out evaluation games are added.

Each batch collects ten games concurrently with balanced learner seats and a
frozen copy of the current policy as opponent. Only the scheduled learner's
trajectory from each game enters PPO. All ten games must pass policy
reconstruction and native replay before one update and a complete checkpoint
roundtrip. A partial or failed batch never trains. Captured game cutoffs and
verification failures are separate, potentially overlapping counts; Red and Blue
win totals include verified wins only. Unverified captured victories remain
explicitly labelled. Reports include checkpoint and weight changes per batch.

This separate lineage preserves the full original 1,753 checkpoint and uses
fresh, durably consumed seeds. Subsequent launches require
`./continue-selfplay.sh --batch-ten --resume --run-dir PRIVATE_RUN_PATH --batches 1 --hours 10`;
the path must identify its latest run. Continuation restores the latest complete
published checkpoint, never an in-flight update. The launcher rejects concurrent
game/training activity and retains a 30 GiB disk reserve. Do not change the
source checkout while a run uses it.

The map, large network, PPO/Adam settings, terminal rewards and current temporary
rules are unchanged. The experiment retains the first-win mode's extended
horizons: 5,840 days and 158,720 ordinary plus 5,120 reserved choices per player.
Ten games per update replaces the historical four-game batch contract. More
decisive self-play games alone do not establish stronger play; evaluation against
frozen 1,753 is recorded as a future option and is currently disabled. Meaningful
research steps and corrections belong in [the research journal](docs/research-log.md).

## Experimental Metal PPO backend

The `codex/metal-ppo` branch adds an explicit, offline numerical candidate for
Apple Silicon, using FP32 MPS tensors and custom Metal GRU/tree kernels. It keeps
PPO settings, full current-weight recurrent prefixes, native recipe validation,
and the CPU checkpoint/Adam format. It currently supports feature size 32 and
hidden size 64 and requires the tested PyTorch 2.9.1 MPS shader API. The experimental launcher selects it only with explicit `--metal`; gameplay,
policy inference and diagnostic verification remain on CPU. The ordinary
launcher default remains CPU.

The internal `_update(..., use_metal=True)` entry point requires an explicit
native binding and histories prepared through `metal_backend.checked_backend`.
It restores model and Adam tensors to CPU, including on failure, and restores the
optimizer's previous `foreach` setting. Its process uses a 25% MPS recommended
memory limit. Immutable prefix-plan device storage is cached up to 4 GiB;
eviction keeps packed CPU constants and reuploads them without rebuilding
metadata or reusing encoded features across optimizer steps. Raw captures, weights, benchmark workers and numerical probes stay
in private storage. The research journal records measurements and acceptance
limits; kernel-only speedups do not establish the requested fivefold complete
update speedup or equivalent playing strength.

The November 13 follow-up measured 6.41–6.49x complete-update speedup on the
long offline history (including preparation and return to CPU), while the short
history remained below fivefold. The main dispatch fix replaces quadratic-like
MPS row writes with exact direct Metal writes within manual-autograd/no-grad
boundaries. Exact canonical JSON reuse reduces preparation without removing
validation. Numerical and held-out evidence and negative experiments are recorded
in the research journal. Campaign-scale memory and playing-strength acceptance
remain separate from these offline measurements. The authorized first full-cycle
measurement uses the existing launcher in this experimental checkout:

```sh
./run-100-games.command 100 --metal --check
./run-100-games.command 100 --metal
```

It recovers an outstanding diagnostic before the new 100-game batch. Reports
identify the selected learner backend and record collection/preparation, full
update, checkpoint and diagnostic phases; wall time includes any system sleep.
Preparation overlaps gameplay, so its per-game durations must not be added
again to the combined collection/preparation duration.

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

Keep artifacts for one manual cycle launch at a time: record, analyze, then remove
the previous launch before starting the next one. Record meaningful findings and
failures in `docs/research-log.md`, including ``Retention-reviewed run: `RUN_ID`.``
for the latest analyzed run. Cleanup refuses an unreviewed or live run. A failed
or interrupted run also needs analysis before its artifacts are retired.

The launcher first audits the complete available publication chain, copies and
exactly restores the initial checkpoint and latest published policy/resume pair,
then atomically saves a compact continuation baseline under
`Heroes-artifacts/selfplay-ten-then-check/retained/`. This keeps model weights,
Adam, RNGs, progress, sampler and opponent pool. It also preserves consumed seed
reservations, including unstarted attempts, and any pending diagnostic obligation.
Only then are old run directories deleted. Interrupted cleanup can be retried.
Fresh-run admission, replay, native/source/map checks and the 30 GiB launch reserve
remain enforced. `--check` is read-only; `./continue-selfplay.sh --cycle --cleanup`
performs the same retirement without launching games.

Raw history is deliberately no longer re-verifiable after retirement; journal
conclusions and the audited checkpoint baseline replace that storage dependency.
Engine/runtime acceptance, Python, game data, locks and model backups are retained
as execution dependencies. Historical experiment launchers that require deleted
raw history are retired; they are not an alternative continuation route.

Keep original game data, raw captures, videos, builds and large checkpoints out
of this public repository. Use ignored local artifacts and the authorized
private training-state repository. Publish compact evidence here. A backup is
complete only after verifying exact restoration of model, optimizer, RNG,
progress and opponent state.

Never delete/archive the only worktree holding a checkpoint or runtime. Preserve
and verify its continuation state first; the single-run rule authorizes retiring
analyzed raw captures. Branch cleanup must not interrupt an active pinned
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
