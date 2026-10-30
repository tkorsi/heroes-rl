# Removal of training performance telemetry, 30 October 2026

## Prospective protocol

The user requests removal of the CPU instrumentation subsystem. This change
removes native active-worker statistics, phase clocks, counters and Python phase
profiling. It retains parallel execution, numeric ordering, bounded queues,
deadlines, ownership, replay/admission checks, checkpoint preservation and the
manual launcher's total elapsed-time output. The cost of the removed profiling
has not been isolated; no speedup is claimed.

The completed manual session continued checkpoint 1585 to 1728 in 143 complete
updates, admitting 572 losses. Its 24 evaluation games were excluded from PPO.
The original target has 604 games remaining and sequence 1369 is next. The
canonical and backup full state restore exactly. These files, the original
campaign ledger and all historical raw evidence remain unchanged.

Validation uses a clean native build in a new private directory, exact numerical
and failure-cleanup controls, a complete copied update, fresh native controls,
and frozen copied checkpoint1728 policy games. Every diagnostic has bounded
deadlines, separate outputs and verified cleanup. No diagnostic experience is
admitted to official training. A new acceptance descriptor must pass the existing
source, engine, replay and checkpoint gates before selecting the new build.

The user explicitly chose explanation only for self-play. Nullkiller Easy,
the map, policy architecture, PPO configuration, four-game batches, terminal-only
rewards and revealed-movement support remain unchanged. Self-play is a separate
possible experiment; the current v8 continuation cannot switch opponent names
without a new validated provenance and admission contract.

## Results

The clean native learner build is bound by manifest SHA256
`53970605c4560bbe14b655442df373d3577efef3de0c5f824b95cfa22a74b573`
and actual library SHA256
`3d7099e10b8ad3ba73e0a8f30ef0642d714689e3cd45e88b029c79bf5f6db19b`.
The engine is unchanged, and `tools/prepare_vcmi.py --check` passed. The native
learner implementation is frozen at `221d319`; the policy diagnostic source
`4b83e6c` adds only the native-control report.

- All 43 focused native tests passed, including exact model, gradient, Adam and
  RNG checks at different worker budgets and failure cleanup. The removed
  statistics API is absent. Scheduling state and bounded queues remain.
- One archived 363-sample batch was recomputed in isolated copies under both
  implementations: 13 optimizer steps, 39 model tensors, 35 gradient tensors and
  117 Adam tensors were bitwise equal, as were RNG and diagnostic bookkeeping.
  Progress, sampler and pool deliberately stay at the copied parent state;
  this is not a new official update. Candidate PPO phase timings are absent.
- All 16 garrison and 28 legacy native controls passed replay, history and
  mechanics checks. Both artifact controls passed exact inventory restoration.
- The accepted policy cohort ran 16 fresh frozen-1728 games, eight per verification
  arm and four per seat per arm. All were natural losses. All 2,030 choices
  reconstructed with zero difference; all 142 movement targets were revealed.
  The full parent state restored exactly and remained unchanged. All receipts
  were revoked; admitted samples and official optimizer updates are zero.
- The strict runtime acceptance gate passed. Its descriptor SHA256 is
  `09a7f0f484b097bf27bb4712cfc515cd099ca7688db388a64f9127665d15b831`.
  The new release is `telemetry-free-1728-v1`. It separately requires the
  frozen-1728 policy proof and the unchanged checkpoint-1392 terminal engine proof.

The first full offline suite ran 2,061 tests: three launcher assertions still
expected the old manifest, with 33 platform/optional skips. The shared expected
manifest path/hash were updated, and all 23 launcher tests passed. Final offline
suite and real main-checkout preflight results follow below when complete.

## Failed attempts and limits

The first frozen-policy attempt completed eight excluded sequential-arm games
(971 reconstructed choices), then correctly stopped before the pipeline arm
because a concurrently written validation report made the checkout unclean.
The retained error is `ValueError: Selected checkout must be clean and committed`.
Cleanup passed. Its final parent-unchanged/sampler assertions were not reached;
all 15 protected canonical inputs were independently rehashed unchanged. Its
reserved starts 1730000040–1730000047 were consumed and not reused. After committing
the report, the accepted second attempt used fresh starts 1730000048–1730000055,
paired only between its explicitly declared diagnostic arms.

Earlier focused test runs also exposed stale error-text and manifest expectations;
these were corrected without weakening checks and the affected modules passed.
All original failed logs remain private. There was no official failed batch,
seed reservation, sample admission or restart during this work.

No instrumentation overhead percentage or training speedup is established. The
single copied PPO pair ran under concurrent validation load (6.577 versus 6.702
seconds). The two policy arms did not have identical native semantic digests,
so their timing ratio is not an identical-workload speedup. Production CPU
counters and training phase profiling are removed. Coarse attempt timestamps,
deadlines, failure evidence and explicit diagnostic outer timing remain;
standalone bounded evaluation tools retain their diagnostic phase timing.

## Observed travel and self-play discussion

The existing 572 training captures show own-hero position changes in every game,
a median of 14 distinct hero/position pairs (including starting positions), and
76.84 newly observed tiles per game on average after the initial observation.
There were three verified town captures: two captures of an enemy starting town
and one recapture of an initially owned town. All three towns were subsequently
lost and all three games ended in loss. The median game ended on day 2. These
are observations from existing evidence, not additional rewards or a learning claim.

The policy is the actor that maps visible history to action probabilities. The
value head estimates expected return; PPO and Adam train both. Self-play would
replace Nullkiller with the learner's own policy or frozen earlier versions,
with separate observations and recurrent memory for each player. Similar-strength
opponents could allow longer games, but two weak policies can also stay passive.
Terminal-only rewards do not automatically teach exploration or town capture.

A future experiment could use frozen opponent snapshots and evaluate against
held-out past opponents and Nullkiller. A current-policy mirror match tends toward
50% decisive wins by symmetry and is not by itself evidence of improvement.
Although low-level two-policy routing exists, the accepted current v8 training
controller binds built-in-opponent provenance and admission; changing an opponent
name would not establish a valid v8 self-play campaign. The user requested
explanation only, so no such change was implemented.

## Manual readiness

Final preflight pending. The launcher remains `./continue-1585.sh --hours 10`;
it follows the latest verified checkpoint and computes the remaining target.
No official training was launched.
