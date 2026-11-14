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
The consolidated source passed current-checkpoint/runtime inspection and one
replay-verified diagnostic from checkpoint 1,765. A complete Metal training cycle
and the test suite have not been rerun after consolidation. The commands below
are for the user to run; the single-game check does not establish full-cycle readiness.

```sh
cd /Users/orsi/Projects/Heroes
./run-100-games.command --check       # Preview; no games or artifact cleanup
./run-100-games.command 200           # 200 games using Metal PPO
./run-100-games.command              # Default: 500 games
./run-100-games.command 200 --cpu     # Explicit CPU fallback
```

The filename remains compatible with existing shortcuts. Any integer from
**10 to 1,000** is supported, including `--games 250`. Double-clicking launches
the default 500. There is no default time limit; `--hours 30` adds an explicit
wall-clock deadline. No automatic launch or restart is scheduled. See
[the launch policy](AGENTS.md).

| Current contract | Behavior |
| --- | --- |
| Published state | Latest complete checkpoint **1,765**; restore model, Adam, RNG and progress together |
| Collection | Ten rolling actors; balanced learner seats against a frozen copy of the batch-start policy |
| Updates | Complete batches of at most 100 games; save and restore-check each update |
| Backend | Metal PPO by default; gameplay, inference and diagnostic verification stay on CPU |
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

## Metal result and its limits

The default uses the **full-D implementation** measured on the same saved
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
The process retains the 48 GiB RAM reserve, 20% system-free-memory check and
30 GiB disk reserve. The latest published checkpoint remains 1,765; offline
benchmarks did not publish a training checkpoint.

## Find the implementation and evidence

| Read or change | Start here |
| --- | --- |
| Collection, scheduling, retention and progress | [Cycle coordinator](tools/run_selfplay_training.py) |
| Policy representation and recurrent memory | [Contextual model](experiments/learner/contextual_model.py) |
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
