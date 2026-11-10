# Heroes RL

Learn Heroes III adventure-map strategy with recurrent PyTorch PPO and the real
VCMI engine. The policy receives structured, player-visible state and chooses
typed actions; it does not consume screenshots. Combat is automatic.

Use `/Users/orsi/Projects/Heroes` as the local working checkout. The
[research log](docs/research-log.md) and [project history](docs/project-history.md)
record previous experiments and their limits.

## Baseline for new experiments

The current prepared experiment repeats one hundred capture-only training games with ten
rolling actor processes, then one PPO update and one validated self-play game.
The lineage began at checkpoint 1,753; continuation restores its latest complete
published checkpoint. Launches remain manual by default; on November 7 the user
authorized the agent to fix worker registration and execute one complete
fifty-game continuation with progress reports every thirty minutes.
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

## One hundred training games followed by one validated game

The current cycle uses both starting taverns already built and
calendar spawning of new neutral armies disabled (`hero-management-v3`). Initial
neutrals and their growth remain native. These are temporary restrictions to be
removed gradually. Native controls, a short frozen-1,757 prefix and the
read-only continuation preflight passed. The first complete 100-game cycle on
this variant advanced checkpoint 1,757 to 1,758; the November 10 retry advanced
it to 1,759. The next continuation restores 1,759 with these rules.

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
the journal; this run remains available until the next authorized launch. The
next fresh reservation starts at offset 642.

Use `./continue-selfplay.sh --cycle --check --cycles 1 --hours 10` for the
read-only preflight, or use `./run-100-games.command --check`. The prepared
`./run-100-games.command` starts one manual continuation from the latest complete
checkpoint. Before launch it retires the previous analyzed run as described below;
the script remains reusable and does not contain a historical run path.
One cycle has one hundred training games, one PPO update and one diagnostic game. Both diagnostic seats
use the newly saved weights. All stages share the ten-hour deadline. Both
training and diagnostic
games stop at 365 game days; the existing per-player choice limits remain
158,720 ordinary plus 5,120 reserved choices.

In an interactive terminal, each former game card is split into two independent
games: five groups across and ten rows, showing all 100 numbered slots. Each
half keeps its own activity and outcome marker. PPO and diagnostic validation
have separate status lines. The panel uses the alternate screen and restores the
terminal/cursor on exit. Elapsed time uses `hh:mm:ss`. At 80x24, dense cards show
all 100 games, with exact days and action counts for the ten active games below.
At 80x28, two-line cards add each game's day; tall windows show per-game Red/Blue
counts. Compact result markers are `R`/`B` for wins, `D` for the day limit and `C`
for other cutoffs; `>` means playing, `~` preparing, `OK` ready and `!` failed.
Widen smaller terminals to 80 columns for the complete grid.
Redirected output remains plain text. During PPO the
panel shows completed/total optimizer minibatches, a progress bar, elapsed time
and an approximate ETA from the current update. ETA stays unknown until at least
three minibatches and three seconds have elapsed. Optimizer completion and
checkpoint saving/restore verification are separate phases. Disposable progress
snapshots are throttled to once per second apart from phase transitions; their
failures cannot block or change learning.

Training games have no native replay, policy-probability reconstruction or
behavior analytics. Their Red/Blue win totals come from engine-reported outcomes
and are explicitly unverified. All one hundred fresh completed captures must exist
before one PPO update. Basic source, seed, action, boundary, terminal reward and
checkpoint integrity checks remain active. Checkpoint metadata explicitly records
`training_verification=not_performed`. The historical verified mode remains a
separate contract. Previously published ten-, forty- and fifty-game cycle checkpoints restore with
their original metadata; new cycles require one hundred games before one update.
The existing lineage/namespace names are historical. Old eleven-, forty-one- and fifty-one-slot reservations
remain consumed; new cycles reserve one hundred training seeds plus one diagnostic.

Collection runs ten independent actor processes, each with one PyTorch/BLAS
thread. A rolling dispatcher immediately refills an available slot; it does not
wait for a whole wave. Main prepares completed games serially while actors keep
playing. All policy binding/model loading happens before the dispatcher starts;
preparation stays on main so its signal-based deadline remains active. Completed
receipts are sorted by schedule index before the unchanged PPO shuffle. A single
PPO update starts only after all one hundred games and owned-worker cleanup succeed.
The inherited native execution profile continues to govern PPO; capture/preparation
worker counts are explicitly recorded in the cycle configuration. Each completed
game is compiled into a native PPO program, then its Python history is released.
The owned worker transfers immutable typed rollouts in a compact binary file,
bound to its completion token and file hashes. A restricted reader permits only
the project's record classes. Training omits duplicate bridge/history JSONL and
the independent post-game history audit. Native terminal evidence remains active.
Diagnostic games retain full JSONL, audit and replay. Temporary raw training
streams are removed after compilation. The update retains
all one hundred learner trajectories, the existing shuffled PPO windows and the original
PPO/Adam settings. An owned-process resident-memory guard leaves 48 GiB of the
Mac's physical RAM as headroom. It stops the attempt safely if the budget is
exceeded. A system-wide memory-pressure check also stops below 20% free.
The existing 30 GiB disk reserve applies. Long-game completion
within these limits still requires observation during the user's manual run.

After the full state is saved and restore-tested, one fresh diagnostic self-play
game runs with both policies frozen at the new checkpoint. That game retains its
capture, runs exact two-policy reconstruction and native replay, and never enters
PPO. A failed diagnostic stops the loop while preserving the saved update.
Continuation performs any outstanding diagnostic first, using fresh seeds.
One self-play game supports inspection, not a statistical claim of stronger play.

A separate lineage and seed namespace keep this experiment distinct from the
previous failed runs. Subsequent manual launches use
`./continue-selfplay.sh --cycle --resume --run-dir PRIVATE_RUN_PATH --cycles 1 --hours 10`.
All reserved slots remain consumed, including unstarted games. If continuation
needs an outstanding diagnostic, one additional 101-slot block is reserved
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
