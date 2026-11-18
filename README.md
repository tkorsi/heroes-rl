# Heroes RL

Learn Heroes III adventure-map strategy with recurrent PPO in the real VCMI
engine. The policy sees structured, player-visible information and chooses typed
actions. Combat is handled by BattleAI; screenshots are not policy inputs.

**Start here:** [architecture and key code](docs/architecture.md) ·
[research journal](docs/research-log.md) ·
[current restrictions](docs/temporary-restrictions.md)

## Run a training cycle

Use the main checkout at `/Users/orsi/Projects/Heroes`. The user runs these
commands in Terminal; editing or reviewing this project never starts training.
The prepared Mac runtime and private game assets must already be available.
The MLX source completed a fresh 100-game continuation from checkpoint 1,765 to
1,766 and its exact two-policy reconstruction/native-replay diagnostic. The
targeted existing test run had 94 passes and three errors also reproduced on the
prior source; it is not a fully green suite. The commands below are for the user
to run unless assistant execution is explicitly authorized.

```sh
cd /Users/orsi/Projects/Heroes
./run-100-games.command --check       # Preview; no games or artifact cleanup
./run-100-games.command 200           # 200 games using CPU policy and MLX PPO
./run-100-games.command              # Default: 500 games
./run-100-games.command 200 --cpu     # Explicit CPU fallback
./run-100-games.command 200 --metal   # Previous PyTorch/Metal PPO, CPU policy
```

The filename remains compatible with existing shortcuts. Any integer from
**10 to 1,000** is supported, including `--games 250`. Double-clicking launches
the default 500. There is no default time limit; `--hours 30` adds an explicit
wall-clock deadline. No automatic launch or restart is scheduled. See
[the launch policy](AGENTS.md).

| Current contract | Behavior |
| --- | --- |
| Published state | Latest complete checkpoint **1,779**; restore model, Adam, RNG and progress together |
| Collection | Ten rolling actors; balanced learner seats against a frozen copy of the batch-start policy |
| Updates | Complete batches of at most 100 games; save and restore-check each update |
| Backend | CPU policy and MLX PPO; compatible checkpoint and sampling boundaries |
| Final game | One fresh diagnostic after the last update; recover any outstanding diagnostic first |
| Scenario | `hero-management-v4`, starting taverns built, native calendar neutral spawning restored |
| Game horizon | 365 days; 158,720 ordinary and 5,120 reserved choices per player |

Training outcomes are engine-reported and do not receive post-game replay or
policy reconstruction. Source, seed, action, boundary and reward checks remain.
The final diagnostic receives exact two-policy reconstruction and native replay
and never enters PPO. A partial batch cannot publish an update; completed
checkpoints survive a later failure. One diagnostic is not evidence of stronger play.

The dashboard shows individual outcomes (`B`, `R`, `C`, `E`) and separate PPO
progress. Above 100 games, each cell holds five games; 1,000 games use twice the
grid height of 500. On macOS, `caffeinate -is` permits display sleep while keeping
idle computation awake. Closing the lid or requesting sleep may still pause the
process; it continues after wake. Keep Terminal open and power connected.

Current cycles use CPU policy inference and MLX PPO, with restored manual
troop-slot management and upgrades under an explicit support version. New
cycles exclude swaps that return to an army layout already observed since the
last meaningful visible progress; reopening windows does not reset this memory.
All other restrictions and PPO settings remain. Historical captures retain their original
policy support. The first restored-stack attempt failed before PPO; its repaired
retry completed and published checkpoint **1,769**. See the research journal for
the saved-failure reproduction and subsequent campaign evidence.

The completed 1,000-game continuation from 1,769 used ten batches of 100,
ten PPO updates and one final diagnostic, publishing **1,779**. Cycle-aware
swap support and all other restrictions remain. Only exact layout cycles are
excluded; novel but unhelpful rearrangements can still occur. No further
training launch is scheduled or performed by the assistant.

## MLX results and remaining performance work

The 1,000-game continuation published checkpoints 1,770–**1,779**: all games
admitted, ten PPO updates, 4,892,383 samples, 14h21m02s total. Final diagnostic
reconstruction/replay passed. Final process cleanup initially timed out. A fresh identity-aware audit of
2,003 receipts / 1,991 groups subsequently certified closure, preserving the
original failure and the new audit separately. Report status is completed;
saved 1,779 policy/resume hashes and seed consumption are unchanged. Cleanup
now batches process-table inspection and distinguishes reused PGIDs from live
original sessions; ambiguous/live originals still block continuation.
Both final diagnostic seats use 1,779 weights; earlier references below to the
Blue diagnostic seat as the previous checkpoint are incorrect. See the explicit
research-log correction; those diagnostic outcomes do not compare checkpoints.

The first restored-stack cycle after the repeat-count fix completed from 1,768
to **1,769**: all 100 captures admitted, zero failures, 466,381 samples and 14,598
optimizer steps. Collection/preparation took 36m58s, complete PPO 29m52s and the
full cycle 69m30s. The final diagnostic passed exact two-policy reconstruction
and native replay; Blue (frozen 1,768) beat Red (new 1,769). Stack operations
were exercised without the previous failure; high rearrangement counts require
behavioral analysis, and one diagnostic does not establish strength changes.

The user-run **CPU policy / MLX PPO** cycle continued 1,767 to **1,768** with
100 new games, 228,930 samples and 7,179 optimizer steps. Collection/preparation
took **23m10s**, complete PPO **600.1246 s**, and the whole cycle **34m26s**.
Compared with the preceding MLX-policy cycle (35m41s / 12m09s / 51m30s), total
time fell 33.15%, saving 17m04s; samples fell 4.16%. Different trajectories and
conditions limit causal attribution, but this supports retaining CPU inference.
The strict complete-PPO <=600 s target is still missed by 0.1246 s; the inner
adapter took 596.9127 s. All captures completed, checkpoint restore passed,
and the final diagnostic passed exact two-policy reconstruction/native replay.
Red (new 1,768) beat Blue (frozen 1,767); one game does not prove stronger play.
`--mlx` now selects MLX PPO with CPU gameplay and diagnostics. The initial
hybrid attempt failed before gameplay because of a stale capture-plan gate;
that failure and its correction are retained in the research journal.

The subsequent user-run optimized cycle continued 1,766 to **1,767**: 100 new
games, 238,862 samples and 7,487 optimizer steps. Complete PPO took **729.14 s
(12m09s)**, collection/preparation 35m41s, and the entire cycle 51m30s. PPO was
35.9% shorter than the preceding run, with approximately 1.44x throughput after
normalizing by samples/steps; different recurrent histories and conditions still
limit attribution. The 600-second goal remains unmet. Checkpoint restore and
exact two-policy reconstruction/native replay passed; diagnostic Blue (frozen
1,766) beat Red (new 1,767). One game does not establish a strength change.

The MLX production migration completed a fresh 100-game continuation from
checkpoint 1,765: 259,090 PPO samples and 8,119 optimizer steps. The complete
production PPO phase took **1,137.24 seconds (18m57s)**, including admission,
planning and state import. This misses the **600-second target**; optimization
continues. Collection and overlapping preparation took 42m51s; the complete
cycle took 63m02s. All observed power samples showed AC power. Follow-up changes
overlap current-weight prefixes with CPU planning, remove unused plan/gate
buffers, and validate shared admission inputs once before and once after the
individual captures. A saved two-game probe improved from 33.66 to 31.74–31.85
seconds across two candidate repeats; this does not establish the full-100
target. The **next fresh cycle is user-launched**.
The canonical network/PPO code is
`experiments/learner/mlx_model.py` and `mlx_backend.py`; live neural scoring is
`mlx_policy.py`. Torch remains at checkpoint, native-history and sampling
boundaries. Existing CPU and PyTorch/Metal modes remain explicit fallbacks.

Saved-input integration checks completed 1,089 samples / 35 steps and verified
transaction rollback. Live MLX inference passed unchanged exact reconstruction
for 1,089 actions/values and 1,861 conditionals. That serial inference probe was
about three times slower than CPU; gameplay acceleration is not assumed.

The first user-run comparison on `codex/ppo-framework-comparison` completed:
one shared ten-game cohort, followed by current PyTorch/Metal, Python/MLX and
Swift/MLX updates. Full update times were 70.86, 204.18 and 251.99 seconds for
10,885 samples and 343 steps (one epoch). The implementations have different
kernel optimizations and validation overhead; this does not rank framework or
language performance in isolation. See the results and comparison contract in `docs/framework-comparison.md` on the preserved local experimental branch.
It does not replace the campaign backend or publish an official checkpoint.
The authorized optimized comparison on two saved games is now complete: all nine
trials passed, with 1,089 samples and 35 steps per trial. Median ready-bundle full
updates were 9.159 s (PyTorch/Metal), 1.681 s (Python/MLX) and 1.526 s (Swift/MLX).
MLX consumes plans/targets prepared during a separate 2.012 s export, whereas
Metal rebuilds them inside its timer. Charging that export entirely to MLX gives
conservative accounting speedups of 2.48x and 2.59x, not a measured production
pipeline result. Numerical agreement was close; playing strength and larger-batch
scaling remain untested. The comparison code is in local commit `8da9caa`.
The user authorized consolidating the production CPU-policy / MLX-PPO path into `main`; comparison runners and Swift prototypes remain on the preserved experimental branch.

The previous Metal backend uses the **full-D implementation** measured on the same saved
100-game cohort: 273,913 samples and 8,583 optimizer steps. On the same Mac,
CPU took **5h30m54s** and Metal **46m34s**, including preparation: **7.105× faster for the complete
update**. This single paired measurement excludes gameplay and the final
diagnostic. It is not a promise about total campaign time.

The user accepted CPU/Metal FP32 differences when choosing this backend.
Full D failed the earlier strict numerical comparison: 439/450 conditional
argmax matches, with KL and value errors above their bounds. Adoption changes
the accepted numerical contract; it does not establish equal learning quality
or playing strength. Later deterministic-pooling prototypes remain separate
from the adopted full-D source. Exact results and failed attempts remain in
[the journal](docs/research-log.md).

Metal keeps the original PPO settings, full current-weight recurrent prefixes,
CPU checkpoint format, 25% MPS memory cap and 4 GiB prefix-constant cache.
The process retains the 30 GiB disk reserve. At the user's direction the cycle
no longer has an owned-process RAM budget or system memory-pressure stop
(removed 2026-11-18); MLX keeps its own allocator cap during PPO. The fresh
MLX cycle published checkpoint 1,766; offline benchmarks do not publish
training checkpoints.

## Find the implementation and evidence

| Read or change | Start here |
| --- | --- |
| Collection, scheduling, retention and progress | [Cycle coordinator](tools/run_selfplay_training.py) |
| Policy representation and recurrent memory | [Contextual model](experiments/learner/contextual_model.py) |
| MLX network, PPO and live inference | [MLX model](experiments/learner/mlx_model.py), [MLX update](experiments/learner/mlx_backend.py), [MLX policy](experiments/learner/mlx_policy.py) |
| PPO and CPU/Metal execution | [PPO](experiments/learner/ppo.py), [Metal backend](experiments/learner/metal_backend.py) |
| Admission and checkpoint restoration | [Self-play training](experiments/learner/selfplay_training.py) |
| Native game actions and replay | [ExplorerRL](engine/ExplorerRL/), [ExperimentReplay](engine/ExperimentReplay/) |
| Hypotheses, failures, measurements and limitations | [Research journal](docs/research-log.md), [historical notes](docs/project-history.md) |

The [architecture guide](docs/architecture.md) explains the execution path,
setup and compatibility boundaries. Historical launchers, maps and validation
sources are retained where imports, replay or source hashes still depend on
them; they are not alternative entry points for the current campaign.

## Preserve the research state

Record, analyze, then retire the preceding reviewed raw run at the next manual
launch. Cleanup preserves and restore-checks the complete checkpoint pair,
consumed seed reservations and pending diagnostic obligation. An unreviewed or
live run cannot be retired. Keep the current investigation's saved cohort while
it remains required evidence; the journal records retention decisions.

Weights, raw captures, runtime builds and original game data stay in private
storage. Keep the append-only journal and its `Retention-reviewed run` markers
at their existing path. Read [AGENTS.md](AGENTS.md) before changing source or
publishing work; no experiments or validation tooling are published by default.

VCMI is GPL-2.0-or-later; see [third-party notices](THIRD_PARTY_NOTICES.md).
Original Heroes III assets are user-supplied and are not redistributed here.
