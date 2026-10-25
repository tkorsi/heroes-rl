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
Completed campaigns used the authored **hero-management-v1** scenario. The
expanded interface uses **hero-management-v2**, which removes only the native
level-one cap; see its [configuration](configs/hero-management-v2.json). Historical
map bytes and reports remain unchanged.

Latest evidence is timestamped, not a live activity guarantee:

- [Completed local365 Teacher analysis](validation/local365-final-analysis-2026-10-25.json).
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

### Completed local365 continuation

The authorized local session ran from **18:27:13 to 20:07:24 UTC on 25 October
2026** (20:27:13–22:07:24 Kyiv), stopping after 100.2 minutes on a genuine
unsupported-interface failure, before its two-hour hard deadline. It used the
ordinary cold runtime, Nullkiller Easy/BattleAI, complete four-game PPO batches,
**365 days and 5120 total choices**, and the existing terminal-only rewards.
The active source stayed pinned at `aebe2fb`; no cloud or experimental worker
was started. See [launch evidence](validation/local365-launch-2026-10-25.json)
and the [final analysis](validation/local365-final-analysis-2026-10-25.json).

The exact cloud update-31 state advanced to **update 92**: **61 new PPO updates,
244 admitted training games, 14,286 transitions and 530 Adam steps**. Cumulative
state is 368 rollouts, 21,062 decisions and 783 Adam steps. Every completed batch
was independently restored and privately published. Final canonical, backup
and private copies restore the exact model, Adam, RNG, progress and pool.

All **244 training games**, **24 scheduled evaluation games** and the **two
initial frozen gate games** were verified natural losses: zero wins and zero
cutoffs or timeouts. A terminal-history audit identified 268 standard losses
and two defeats under the seven-days-without-a-town rule. No town capture was
observed. The fixed Red check improved from a
day-2 loss at update31 to days8–9 at updates70–90, with recruitment; Blue
remained a day-2 loss except for a transient day-8 result at update50. These
repeated seeds demonstrate limited behavior changes, not general strategic
improvement. The last held-out checkpoint is update90, two updates before the
final preserved checkpoint; no games ran after the correctness stop.

Three admitted training games exceeded the old 30-day horizon, ending naturally
on days65,66 and107. Longer trajectories were expensive: the largest batch took
1,283 seconds, compared with a 35.7-second median. The new horizon supplied
additional terminal feedback but did not produce a verified learner victory.

Batch93 stopped in its first game on day62 after 1,337 choices with
`unsupported_interaction: informational notice`, raised by the structured
informational-notice handler. The original failure log omitted the rejected
template. A subsequent
[offline diagnostic replay](validation/local365-day62-notice-investigation-2026-10-25.json)
reproduced all 1,112 recorded server inputs and 5,605 outgoing packets without
state or packet-hash divergence: Thant won a defensive battle, then raised four
Skeletons. The resulting necromancy notice (`core.genrltxt.145`, automatic
display, creature/count component) was rejected by the existing normalizer.
This adapter coverage defect is now fixed in the v7 native/Python notice path;
see the [repair and coverage audit](validation/necromancy-support-2026-10-25.json).
A preceding visible battle-win event is not a verified game win.
The failed attempt is unscored and supplies **zero samples**; three remaining
games were not started. Its raw capture is privately preserved. No retry,
repair campaign or restart was performed. Native and controller cleanup passed,
and no owned native/training processes remained. The Mac window was released.

The post-campaign repair accepts the exact singular/plural necromancy notice,
including the displayed creature and count, without adding a policy action or
reward. Its recorded server prefix still matches all 1,112 inputs and 5,605
packets; the repaired native payload also passes the Python v7 decoder.
This diagnostic does not complete or admit the failed game. Legacy interfaces,
model tensor shapes, checkpoint92 and the frozen campaign runtime are unchanged.
The repaired ExplorerRL and replay targets build, and 61 focused tests pass.
No new game or training campaign was started.

The source audit identified these gaps in the completed v7 campaign. Their v8
implementation and acceptance work is described below; the necromancy repair
alone did not supply them:

| Priority | Missing interface | Reachability |
| --- | --- | --- |
| First | Eagle Eye spell-learning notice; Castle Stables bonus notice | Existing Tavern recruits and town construction can trigger them |
| Next | Army overflow window when neutrals join a full army | New-month monster spawns can reach this branch in long games |
| Next | Marketplace exchange; spellbook/war-machine purchases; artifact inventory/equipment | Town and hero controls missing from the current action set |
| Before lifting the level cap | Hero level-up skill choices | Current scenario explicitly caps heroes at level 1 |
| Later map expansion | Boats, external dwellings, universities, quests and further object windows | Coverage is limited to explicitly supported object families |

Creature upgrades, split/merge/swap, garrison transfers, own-hero army exchange
and town Tavern hiring are already implemented. Tactical battles remain
automatic. This audit is a source inventory, not proof of every live branch.

The final parent is
`/Users/orsi/Projects/Heroes-artifacts/runs/local365-continuation-092/plan.json`,
with a second copy under `Heroes-artifacts/checkpoints/local365-continuation-092/`.
Private checkpoint commit `58b7826` and final raw-evidence commit `d610424` are on
`codex/local365-teacher-results` in the private training-state repository.
The consumed campaign lock/session remain in place; this authorization is
complete and does not permit a restart.

### Expanded interface and next local campaign

The user requested the complete listed expansion **before** another training
campaign: marketplace exchange; spellbook and war-machine purchases; artifact
inventory, equipment, own-hero transfer and combination assembly/disassembly;
adventure spells; and native hero level-up choices. Eagle Eye/Stables notices
and full-army neutral joining are part of the same acceptance work. This is
`adventure-capabilities-v1`, history/actions v8, protocol 10. The ordinary v7
interface and historical observations retain their original serialized forms.

The [final native acceptance](validation/capabilities-native-2026-10-25.json)
passed all **28 cases** (14 mechanics in both seats) against one frozen
implementation/runtime. Every case passed independent history and native replay;
3,036 transactions were checked with none skipped. Matched Dimension Door cases
showed identical visible observations and 224 target candidates with/without an
unseen blocker, followed by genuine success/rejection. The retained draft
failures are separate and supplied no training samples. These are controlled
mechanics checks, not evidence of learned strategy or ordinary-game victories.

Available controls appear in the currently open hero/town/exchange window.
Inventory, prices, spell choices and inspection details appear only after the
corresponding window or target is explicitly opened. Visions uses the engine's
actual visible-object popup information, including native Disguise handling.
View Earth/Air expose only their delivered map payload. Fogged Dimension Door
coordinates are public-range attempts; hidden destination occupancy must never
filter the action list. Flight/Water Walk use native movement layers over the
player's knowledge, without exposing route costs. Battles remain automatic.

New capability actions use the existing typed candidate encoder; model v6 has
the same named parameter tensors as the v7 history model. Update92 requires an
explicit migration preserving tensors, Adam slots, RNG and progress. Additional
actions can change probabilities even when all tensors remain equal. The new
both-seat baseline therefore belongs to a separate v8 cohort.

The authorized next campaign is **one fresh local session of at most two hours**,
with 365 days, 5120 total choices, Nullkiller Easy/BattleAI, complete four-game
PPO batches, terminal-only rewards and both-seat held-out evaluation. Launch
remains gated on all requested native feature checks, exact capture/replay,
checkpoint migration and a separate deadline/watchdog. Preparation is not
training; see [the migration preparation](docs/update92-v8-prelaunch.md). The old
session is consumed and must never be restarted.

The [PPO prefix benchmark](validation/ppo-prefix-performance-2026-10-25.json)
reuses deterministic encodings within one current-weight minibatch and keeps
every recurrent transition. On the archived long-trajectory benchmark it
reduced prefix reconstruction from 88.81 to 7.29 seconds and the measured
minibatches from 95.27 to 14.27 seconds, with bit-exact gradients, weights and
Adam state after two steps. This is a numerical workload measurement, not a
whole-game throughput claim. Claude's separate embedded-engine work remains
experimental until its own real-game equivalence and timing checks pass.

The [prelaunch evidence](validation/local365-prelaunch-2026-10-25.json),
[local configuration](configs/builtin-local365-v1.json) and
[exact training migration](configs/builtin-local365-migration-training-v1.json)
retain the reviewed source/runtime transition, trained-policy parity proof and
fresh two-seat replay gate. The earlier cloud session remains stopped; its
expired deadlines, original 30-day plans and historical metadata are unchanged.

### Gated v8 preparation

A new local two-hour authorization is gated on all requested v8 capabilities,
full native validation and coordinator release. The [prelaunch checklist](docs/update92-v8-prelaunch.md)
binds the exact update92 parent, explicit checkpoint/Adam migration requirements,
fresh session and watchdog, both-seat baseline and burned sequence93. The
[parent audit](validation/update92-v8-resume-audit-2026-10-25.json) verifies all
three preserved copies. Built-in collection/replay now selects explicit v7/v8
profiles; ordinary resume still refuses a schema transition. Explicit conversion and a
new single-use controller now require a privately restored migrated baseline,
all native feature gates and coordinator release. The target ordinary map is
hero-management-v2, with native hero leveling enabled. Temporary conversion of
the actual update92 pair passed and was removed; no permanent migrated
checkpoint, new session, gameplay probe or training update has been created.

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
