# Heroes RL

Learn Heroes III adventure-map strategy with recurrent PyTorch PPO and the real
VCMI engine. The policy receives structured, player-visible state and chooses
typed actions; it does not consume screenshots. Combat is automatic.

This README is the authoritative project overview and agreement. Updated
**30 October 2026**. Earlier stages are preserved in
[project history](docs/project-history.md); their status and authorization
paragraphs are historical, not new instructions to run experiments.

Use `/Users/orsi/Projects/Heroes` as the working checkout and run the manual
launchers there. Keep existing worktrees referenced by active processes or
historical validation evidence intact.

## Current state

The user stopped the small-network experiment. Its initialization, completed
capture and seed history remain preserved for reproducibility; it is no longer
the selected next experiment. Use the trained **checkpoint 1,753** model
(64 hidden / 32 features, 74,344 parameters) for the next manual one-game test:
`./play-one-selfplay.sh --check`, then explicit `./play-one-selfplay.sh --run`.
This retains the temporary troop-slot restriction and passive own-army information,
with one frozen self-play game and zero PPO updates. Progress includes the observed
in-game day beside both action counters. See the
[manual preparation report](docs/research/large-one-game-day-progress-2026-10-30.md).

The retained user-requested temporary restriction is:
troop-slot rearrangement, splitting, merging, individual transfers and stack
upgrades are disabled. Recruitment and whole-hero entry/exit from the town
garrison remain available. The opt-in `own-army-details-v1` observation provides
owned hero/town composition and current numeric combat statistics without clicks.
See the [implementation and validation report](docs/research/recruit-only-passive-armies-2026-10-30.md).
The [temporary restriction register](docs/temporary-restrictions.md) explicitly
requires restoring army management in a later version; automatic combat is also
recorded there. The clean rebuilt runtime passed two sets of six bounded native
probes, including exact policy reconstruction and fabricated-stat rejection.
No full game or training campaign was launched by this change.

The [self-play storage and progress update](docs/research/rollout-storage-2026-10-30.md)
stores policy results separately from the existing hash-bound player histories,
and removes a duplicate transport-history reconstruction. All policy and native
replay checks remain mandatory. On one copied long game the rollout artifact
shrank from 2.48 GB to 30.3 MB, and write plus strict reload fell from 73.7 to
45.0 seconds; this is not a whole-game speedup. Manual self-play launchers show
Red/Blue action-budget bars and separate storage/verification phases. Existing
checkpoints and learning settings remain unchanged; no game or training campaign
was started by this update.

A separate [tiny-policy candidate](docs/research/tiny-policy-2026-10-30.md) reduces
the recurrent hidden size from 64 to 8 and feature width from 32 to 4: 15,460
parameters instead of 74,344. It has fresh random weights, zero games/updates,
and an independently saved initial policy/RNG/opponent pool. This is a width
reduction of the existing encoder/CNN/GRU architecture. Offline native parity
and synthetic PPO/checkpoint controls passed. A separate
[historical manual one-game diagnostic](docs/research/tiny-one-game-2026-10-30.md) used:
`./play-one-tiny-selfplay.sh --check`, then explicit `--run`. It uses the untrained
tiny weights, the same doubled choice limits, progress display and full replay
checks, with no PPO update. The first user-launched tiny game subsequently passed
replay/reconstruction and reached the day limit with no winner or weight update.
The new restricted/passive-information experiment has separate native runtime
acceptance; a tiny training campaign remains unaccepted. The existing
1753 launcher keeps its original checkpoint.

The separate [100-game self-play experiment](docs/research/selfplay100-2026-10-30.md)
completed at **checkpoint 1,753**, with 100 admitted games and 25 complete PPO
updates from 1,728. Outcomes were three learner wins, eight losses and 89
choice-budget cutoffs. The learner captured three towns; stack swaps accounted
for 69.6% of submitted choices. Every naturally decided game was won by Blue,
including opponent wins. These observations do not establish improved strength.
Four games from the earlier user-interrupted batch remain excluded.

The user has requested a separate [one-game evaluation with doubled choice
limits](docs/research/selfplay-one-game-2x-2026-10-30.md). The prepared
`play-one-selfplay-2x.sh` checks by default; explicit `--run` plays one frozen
self-play game from the latest verified completed checkpoint, currently 1,753.
Both players receive 9,920 normal choices and 320 reserved dialog choices, with a
10,240-choice hard ceiling. This evaluation performs no PPO update and changes no
training checkpoint, sampler, seed reservation or completed campaign target.

Training completion releases the campaign lock before any optional behavior
report. Detailed reporting is a separate, manually invoked command with a
deadline and per-game progress; it cannot admit games or update weights. The
[reporting repair](docs/research/reporting-lock-fix-2026-10-30.md) records the
safe shutdown of the old report process and the preserved checkpoint.

The old `continue-selfplay.sh` retains its completed 100-game target and must not
be used to start this evaluation. The separate original Nullkiller campaign and
its launcher remain unchanged at checkpoint 1,728, with 604 games remaining.

The latest user-launched one-hour session advanced **checkpoint 1,585 to 1,728**
in **143 complete PPO updates / 572 admitted games**, all natural losses. Its
24 evaluation games were excluded from PPO. It stopped normally at the compute
reserve, with exact final model, Adam, RNG, progress, sampler and pool restoration
and verified owned shutdown. The original 5,000-game target now has **4,396
admitted games and 604 remaining**; sequence **1,369** is next.

The [telemetry removal](docs/research/telemetry-removal-2026-10-30.md) removes
native CPU worker counters and training phase profiling. Parallel workers,
pipelined verification, deadlines, replay/admission checks and the launcher's
total elapsed-time output remain. A fresh C++ learner build passed exact numerical
controls and a complete copied PPO update, followed by 44 native controls and
16 frozen-checkpoint policy games with 2,030 exactly reconstructed choices.
All 142 movement targets in those diagnostics were revealed. The overhead of
the removed instrumentation was not isolated; no speedup is claimed. Final
main-checkout preflight passed with exact checkpoint restoration and 604 games
remaining; the offline suite passed 2,028 tests (33 skipped). No official training
was started during this change.

The launcher remains `./continue-1585.sh --hours 10`. Its historical filename
selects the latest verified descendants, now checkpoint 1,728, and recalculates
the remaining original target on every manual invocation. It uses the warning-free
NumPy environment and accepted collection pipeline. The map, architecture, PPO,
four-game batches, Nullkiller Easy opponent and terminal-only +1/-1/0 rewards
remain unchanged. The earlier 772-game goal is complete.

The [revealed-movement intervention](docs/research/revealed-movement-2026-10-30.md)
limits `move_to` targets to tiles present in the hero's current delivered
observation on the same map level. Its version remains
`visible-interaction-budget-revealed-v1`, preserving visible interaction budgets
without a path-feasibility or strategic mask. The latest official run selected
4,311/4,311 revealed movement targets across training and evaluation. Existing
captures show travel and three training town captures, but all games were lost;
this does not establish strategic improvement. The telemetry-removal change
included explanation only for self-play; the separate experiment authorized
afterward is described above.

The accepted [CPU and NumPy changes](docs/research/cpu-next-launch-2026-10-29.md)
remain integrated. The separate environment adds NumPy 2.3.5 with unchanged
Python/Torch libraries. Sealed-game replay verification can overlap remaining
collection when the CPU budget permits; all four games must still pass before
PPO. The separate C++ graph-compaction candidate was slower at the production
thread budget and remains excluded. Repeatable full-training throughput
improvement, full CPU saturation and better playing strength are not established.

Historically, at the preceding failure boundary, the user-started 1,000-game
run had admitted **228 games in 57 complete updates**, then stopped on an unsupported
terminal opponent request in replay verification. The failed batch admitted no
samples or update; full model/Adam/RNG/progress/sampler/pool state is preserved.
The original target then had **3,052 admitted games and 1,948 remaining**; sequence
**1,033** was next and **772 games remained** in the interrupted 1,000-game goal.

The [checkpoint 1392 repair report](docs/research/checkpoint1392-repair-2026-10-29.md)
records both fixes, a clean rebuilt runtime, 44 fresh native controls and eight
frozen-policy games with 1,136 exact reconstructed choices. The prepared manual
launcher is `./continue-772.sh --hours 10`; it recalculates the remaining goal
across manual windows and prints total runtime. The first main-checkout check
exposed a historical cloud-machine guard; its narrow correction and the final
real main-checkout check passed before that user launch. The check restored checkpoint 1,392 exactly,
confirmed the 772-game goal and next sequence 1,033, and launched no gameplay.
The checkpoint, failed-batch evidence and final diagnostic package passed private
remote readback. No official training was launched during repair. Earlier
milestones below remain historical.

The user has authorized a **visible-history interaction budget** to prevent
repeated window, army/artifact and no-effect movement loops. This is an explicit
policy-support intervention, not another exact acceleration. Terminal-only
rewards and the PPO/model architecture stay unchanged. The
[prospective protocol and results](docs/research/interaction-guard-2026-10-27.md)
record eight individually verified games with frozen checkpoint 717: 2–117 choices,
all natural losses. Postcollection harness bookkeeping failed and was separately
corrected and audited; its original failure is retained. Full restoration and owned
shutdown are verified. The guarded run from checkpoint 717 to 770 stopped on an unsupported owned-mine
query; its [garrison repair](docs/research/garrison-adapter-2026-10-28.md) was
accepted before the next user launch. The earlier failed manual run advanced **770 to
916** in **146 complete updates / 584 admitted natural losses**, then stopped in
sequence 555 on a terminal native-request replay rejection. That entire failed
batch admitted no samples or update, including its two completed games.

Checkpoint **916** is preserved and independently restored from its private
remote copy, including Adam, RNG, sampler, progress and pool state. The original 5,000-game
target then had **1,148 admitted games and 3,852 remaining**, with unused sequence
**556** next. The [terminal replay repair](docs/research/terminal-replay-repair-2026-10-28.md)
aligns Python and native accounting for a cancelled Nullkiller artifact-transfer
request and an existing marketplace case. All packet, terminal-barrier, history
and admission checks remain enforced. Its clean rebuilt runtime passed 594 native
accounting checks, 16 garrison controls, 28 legacy controls and eight frozen-916
policy games (591 choices, all natural losses); diagnostics supplied no PPO data.
The manual continuation keeps the original map, architecture, PPO, opponent,
terminal-only rewards and existing visible interaction budget. No official
continuation was launched during repair.
Subsequent user-launched one-hour sessions stopped normally at their compute
reserve: **916 to 992**, then **992 to 1,095**. The latter completed session
admitted 412 games in 103 updates, all losses, with exact final restoration and
owned cleanup.

The next user-launched one-hour session advanced **1,095 to 1,229** with four
collectors: **536 admitted losses in 134 complete PPO updates**. It stopped
uncleanly after Mac lid-close sleep crossed the fixed wall deadline; the guardian
killed the worker on wake. This was a deadline interruption during periodic Blue
evaluation, with no evidence of a new gameplay or adapter rejection. The
[investigation](docs/research/local-sleep-interruption-2026-10-29.md) independently
restored checkpoint **1,229** and its durable backup, including Adam, all RNG,
progress, sampler and pool state, and found no remaining matching processes.
At that boundary, the original target had **2,400 admitted games and 2,600 remaining**;
sequence **869** is next. The interrupted four-game evaluation cohort is excluded.
A separate [audited manual recovery](docs/research/checkpoint1229-recovery-2026-10-29.md)
now passes the real main-checkout one- and ten-hour prelaunch checks. It preserves
the failed shutdown, complete checkpoint state and every consumed reservation.
`./continue-original-1229.sh --hours 1` starts a fresh manual window with four
collectors; use `--hours 10` for ten hours. Keep the Mac awake with its lid open.
The private checkpoint copy was independently downloaded and restored exactly.
The authorized cloud lifetime probe was blocked before launch by missing
managed-environment prerequisites; unattended cloud execution remains unverified.
A subsequent explicitly authorized ten-hour cloud launch also stopped at that
setup gate before downloads or training; see the
[attempt record](validation/cloud-launch-attempt-2026-10-29.json).
No training was restarted. Improvement in winning on the original map has not
been demonstrated.

The reviewed environment, learner, replay, cloud tooling and performance work
are consolidated on **`main`**. Start new work from `main`; intermediate
experiment branches and their PRs describe the history.

The authorized **26 October easier-start curriculum campaign is complete**.
Local training advanced update 146 to 254; cloud then reached **update 412**,
adding **632 cloud training games and 158 PPO updates** (563 natural wins,
69 losses, no horizon cutoffs). The final fixed Linux check improved from
**2/8 to 8/8 wins** on the same eight starts. This is narrow easier8x8 evidence:
the later transfer study below tests varied starts and the original board.
A recruit-then-attack strategy has not been demonstrated.
The expired window authorizes no further training. See the
[overnight results and limitations](docs/research/overnight-results-2026-10-26.md).
Operational curriculum/overnight source remains on its recorded immutable
experiment commits; this report synchronization does not merge that code or
Claude's separate experimental engine work.

The subsequent [checkpoint 412 acceleration diagnostic](docs/research/checkpoint412-acceleration-2026-10-26.md)
passed 72 same-host Mac games with identical paired histories and no learning.
Embedded collection was 3.64 times faster; complete probe rows with replay and
policy checks were 2.09 times faster (1.69 times including wrapper startup).
The subsequent [frozen transfer study](docs/research/transfer-eval-2026-10-26.md)
passed a narrow eight-game Linux gate and completed **168 valid transfer games**:
40 natural wins and 128 losses, with no horizon cutoffs. Checkpoint 412 won
**24/24 varied easy starts**, compared with 14/24 for update 254 and 2/24 for
update 146; all three won **0/32 original-scenario games**.
Four authored easy layouts placed a hero anchor at x=0, producing an off-board
native visitable position at x=-1. The same fixture failed in embedded and cold
runtimes: two startup failures remain unscored and 22 affected entries remain
unstarted. This does not establish an acceleration-specific regression. The
frozen schedule and map bytes were retained; no failed attempt was replayed.

All valid transfer games passed replay and strict policy reconstruction. The
first 36 used embedded execution and the remaining 132 used accepted original
cold execution; the first 36 additionally reconstructed under original cold
source. The study admitted no training data or optimizer updates, restored all
three full checkpoint states, and verified owned-process shutdown. These
results support spatial transfer on the tested easy subset, while original-map
strategy remains unsuccessful. Long-run memory stability and full PPO throughput
remain unproven. PR 29/30 executable code is not merged by this report update.

The user now requests a **local script launched manually**, with an explicit
configuration and run budget, instead of a persistent cloud LLM session. The
script preparation and report synchronization have not started a new gameplay
or PPO campaign. Closed cloud sessions and their deadlines remain closed.
The completed [cloud-to-local handoff](docs/operations/teacher-cloud-to-local-handoff-20261026.md)
at `7287573700a30aa67634f05e2acb786567c0e8fd` binds full checkpoint 412,
its update-411 dependency, both maps and the original continuation seed ledger.
Published closure receipts record no live owned workers/watchdogs, held locks
or queued restart. A new manual run allocates a separate recorded seed namespace;
it does not reopen the closed cloud session.

The first manually launched local easy-map series has now finished at
**update 629** (868 new training games and 217 PPO updates), with exact final
restoration and owned shutdown recorded at 17:58:39 UTC on 26 October.
Its [final checkpoint and full raw archive](validation/local-script-final629-sync-2026-10-26.json)
are privately published; the downloaded checkpoint restored exactly.
The prepared original-map 5,000-game script continues that finalized state;
its launch remains manual. The user renewed Claude's optimization assignment
through **27 October 2026, 13:00 Europe/Kiev (11:00 UTC)**, prioritizing the
complete verified local PPO cycle on the original map. See the
[overnight performance assignment](docs/operations/claude-original5000-overnight-20261026.txt).
This authorizes isolated performance benchmarks and implementation, not automatic
launch of the 5,000-game experiment or promotion of experimental learner changes.

The [consolidation check](validation/project-consolidation-2026-10-25.json)
ran 1182 tests: 1156 passed and 26 required another platform or optional fixture.
A fresh nine-target macOS native build and the Swift viewer build also passed.
This check launched no gameplay.

The later [v8 integration check](validation/v8-integrated-validation-2026-10-25.json)
ran 1253 tests: 1227 passed and the same 26 platform/optional cases were skipped.
All 34 targeted checks after the final launch-controller and release-gate fixes
also passed. Native gameplay acceptance is recorded separately below.

| Area | Implemented | Current limit |
| --- | --- | --- |
| Environment | Real headless VCMI, fog-filtered structured history, explicit map/hero/town windows | Unsupported actions/dialog grammars fail explicitly |
| Actions | Native coordinate routing, construction/recruitment, Tavern, armies, dialogs, marketplace, artifacts, adventure spells and level-up choices | Supported object families only; no tactical battle policy or path-cost oracle |
| Learning | Recurrent PyTorch actor/critic, hierarchical action probabilities, PPO, model/Adam/RNG/progress restoration | Varied easy starts improved to 24/24; update 412 lost all 32 tested original-board games |
| Evidence | Native capture, independent request/state replay, policy reconstruction and strict sample admission | A seed alone does not reproduce all built-in AI randomness |
| Performance | Exact inference/cache optimizations and bounded benchmark tools | Workload timings are not universal whole-game/PPO throughput |
| Native reuse | Opt-in C++ EngineSession with load/reset/observe/step/close | Used in bounded diagnostics; not accepted as a general training runtime |

The fixed opponent is **Nullkiller on Easy**, with BattleAI resolving battles.
Earlier campaigns used the authored **hero-management-v1** scenario. The
completed v8 continuation uses **hero-management-v2**, which removes only the native
level-one cap; see its [configuration](configs/hero-management-v2.json). Historical
map bytes and reports remain unchanged.

The map's authored hero inventories are empty, but native lobby defaults also
request a **random starting bonus** for each player (artifact, gold or resources).
The [recorded inventory audit](docs/research/artifact-inventory-audit-2026-10-28.md)
confirmed that the failed game gave Orrin a Collar of Conjuring and Edric a
Necklace of Swiftness. Orrin looted Edric's necklace before attempting the
cancelled equipment replacement. No starting-bonus rule was changed.

Latest evidence is timestamped, not a live activity guarantee:

- [Completed frozen transfer comparison](docs/research/transfer-eval-2026-10-26.md),
  [source-bound results](validation/transfer-eval-2026-10-26.json) and
  [final cloud transfer status](reports/cloud/teacher-transfer-status.json).
- [Completed overnight curriculum results](docs/research/overnight-results-2026-10-26.md),
  [cloud analysis](validation/overnight-cloud-final-2026-10-26.json) and
  [local curriculum analysis](validation/curriculum-v3-final-2026-10-26.json).
- [Research log for the seminar](docs/research-log.md), with a source-bound
  [experiment index](validation/research-experiment-index.jsonl).
- [Completed v8 Teacher analysis](validation/local-v8-final-analysis-2026-10-26.json)
  and [local Teacher status](reports/local/teacher-status.json).
- [Completed local365 Teacher analysis](validation/local365-final-analysis-2026-10-25.json).
- [Teacher status](reports/cloud/teacher-status.json) and
  [continuation evidence](validation/cloud-continuation-2026-10-25.json).
- [Optimizer status](reports/cloud/optimizer-status.json) and
  [native session evidence](validation/native-engine-session-2026-10-25.json).
- [Contextual pipeline performance](validation/performance-2026-10-25.json).

The first released v8 campaign stopped during native startup before any policy
choice or optimizer step: the built-in-opponent native guard still required v7.
Its consumed session and failed baseline are retained. Explicit repair attempt2
support uses a separate session and output namespace, but inherits the original
absolute and monotonic deadlines; it grants no fresh two-hour budget. A new
reviewed release, repaired native acceptance and exact privately preserved
migration are required before repair gameplay. See
[repair preparation](docs/update92-v8-prelaunch.md#original-window-startup-repair).

The [mixed-opponent repair check](validation/builtin-v8-repair1-2026-10-25.json)
then completed two real v8/Nullkiller games with the fixed update92 policy: both
were natural losses, on days 9 and 2. All 211 sampled decisions reconstructed
exactly and 2,693 native transactions replayed with none unchecked. These
diagnostics supplied no PPO samples. The repaired runtime also passed the full
28-case capability suite; the original failed startup remains unscored.

The subsequent repair2 campaign completed two verified baseline games (both
natural losses, 274 decisions), then stopped on a native heap-corruption crash
after 39 choices on day2 of its first training game. No PPO update or training
sample was admitted; three remaining games were not started. Update92 and all
failed evidence are preserved, and owned-process cleanup passed. The source
trace identifies an invalid same-backpack artifact insertion after source
removal. The user requested the fix and another launch within the original
window. The reviewed repair now passes all 28 real capability checks, including
both-seat backpack reorders and transfers; 3,120 native transactions replayed
without unchecked operations. The initial stale-verifier packaging failure is
retained separately. See the [new runtime evidence](validation/runtime-artifact-fix-2026-10-26.json). See the
[dated failure analysis](validation/v8-artifact-crash-2026-10-26.json).

The [completed repair3 continuation](validation/local-v8-final-analysis-2026-10-26.json)
ran from **22:20:31 to 23:43:28 UTC on 25 October** (00:20:31–01:43:28 Kyiv on
26 October), using pinned source `fe62a80` and the frozen artifact-fix runtime.
It stopped cleanly before the original **23:44:09 UTC compute cutoff and
23:49:09 UTC hard stop**. No owned processes remain, and no restart is authorized.
It admitted **216 training games and 54 PPO updates**, advancing update 92 to
**146**, plus 24 evaluation games. All 240 games passed their replay and
reconstruction checks and ended in natural losses; repair3 had no cutoffs or
technical failures. The final model, Adam state, RNG, progress and opponent pool
are privately preserved; the cumulative state is 584 training rollouts, 40,641
decisions and 1,475 Adam steps.

This run did not demonstrate strategic improvement. Recruitment occurred in
67 of 216 training games, but all 10 initiated battles targeted neutral monsters;
there were no enemy-town target moves or town captures. About 77% of choices
were window navigation/selection or army rearrangement. The last fixed evaluation
was **update 145**, not final update 146: Red changed from a day-7 loss with one
recruitment operation and three completed movements to a day-2 loss with neither;
Blue retained its day-2, 17-choice loss. This small fixed comparison does not
establish a general regression, but it provides no evidence of improvement.
No further evaluation or training was started after shutdown. Initial launch
evidence remains in the [dated launch report](validation/local-v8-repair3-launch-2026-10-26.json).

The completed cloud session continued update 21 to **update 31**: 40 new
training games, ten PPO updates and eight evaluation games. Its 48 verified
games produced **zero wins, 47 natural losses and one day-limit cutoff**. The
checkpoint preserves 124 cumulative rollouts, 6,776 decisions and 253 Adam
steps; model, optimizer, RNG, progress and pool restored exactly from three
copies. The cloud Teacher stopped by its fixed deadline with no native
processes remaining. See the [final analysis](validation/cloud-final-analysis-2026-10-25.json).
Changed weights or occasional recruitment do not establish improved strategy.

The separate [scripted recruitment/rush diagnostic](validation/recruit-rush-2026-10-26.json)
establishes that a short winning path exists on the current map. Red won all
three corrected trials on day 1 after buying 14 pikemen and transferring them to
its starting hero (39 total). Attack began on typed command 8; nine commands were
submitted including the subsequent level-up response, which the natural terminal
superseded. Blue lost both trials after its first attack; Nullkiller had already
used its first turn to recruit and transfer the same 14 pikemen. A preliminary
Red trial that accidentally left the purchased troops in town also won and is
retained separately. All six games passed native replay. These authored probes
are excluded from training and are neither learned-policy wins nor proof of a
globally shortest sequence. The Teacher's policy and campaign settings were not
changed by this diagnostic.

## Manual local execution

Training is launched by the user with a local script, without continuous cloud
LLM supervision. The easier8x8 parent closed at checkpoint 629. The user-launched
original two-Castle 20x20 experiment then admitted **280 of its 5,000 games**
(70 updates), ending at **checkpoint 699**. Its next batch 336 exceeded the inner
update allowance and rolled back completely; no partial update was admitted.
The official next unused sequence is 337. Preserve 699 and the failed attempt.

The prepared `./continue-original-4720.sh` continues that verified lineage and
recalculates the remaining target after each manually stopped/restarted session.
It does not select 629 or a disposable performance checkpoint. The Mac validation
measured one identical long original-map batch at 90.57 minutes for the accepted
implementation, 43.34 minutes for exact serial acceleration and 35.53 minutes
with two replay threads (2.55x versus accepted). Complete training state matched;
the missing accepted Blue update was checked separately on existing captures.
The original two-update full-cycle protocol remains incomplete because the
accepted runner preserved its conservative budget reserve. These results do not
predict the duration of all remaining games. The wrapper uses the measured exact
two-thread path with a three-hour batch allowance, unchanged
PPO/native collection limits and a bounded 14-day manual session cap. The generic
runner retains its 48-hour maximum; `--hours 48` selects a shorter continuation. The user starts
training; preparing or checking the wrapper starts no games. See
[continuation details](docs/operations/original699-continuation.md) and the
[Mac comparison](docs/research/mac699-acceleration-2026-10-27.md).

The policy samples its own current distribution and PPO updates every complete
four-game batch: at most 1,250 updates and 2,500 training games per seat. Rewards
remain terminal-only; every verified training game counts, not only wins.
Separate paired evaluation occurs before training, every 100 updates and at the
end. The declared session cap, manual stop, or technical failure may end the run short
of 5,000; reports distinguish target completion from an incomplete stop. The
source map is unchanged hero-management-v2, including its original small set of
mines, gold and neutrals, native Tavern support and normal hero leveling.

`./train-local.sh check --profile original` checks the prepared asset bundle
without gameplay. The actual final-parent restore and native baseline gate happen
at manual launch. The old `./train-local.sh --hours 2` remains the easier8x8
preset. See the [launch and result instructions](docs/operations/local-script-training.md)
and [prospective original-map experiment](configs/local-original-5000-20261026.json).
Preparing this next experiment does not start games or alter the active release.

The [four-process integration](docs/research/parallel-continuation-2026-10-28.md)
adapts the isolated eight-process collector to the existing **four-game** batch.
`--collection-workers 4` collects its four independently seeded games concurrently;
the generic runner defaults to one collector. Current original-map launches use
the current accepted machine descriptor; historical descriptors stay immutable. Workers activate the immutable
plan's visible interaction budget and execution choices. Every game must pass
native replay and policy reconstruction after verified worker cleanup; an
incomplete batch cannot update PPO. Batch size, optimizer cap, schedule ordering,
map, opponent and terminal rewards remain unchanged.

Claude's latest Python performance work is already incorporated through merges
`8b81931` and `3e2a268`, including the later text-leaf replay fixes. The older
experimental embedded native backend remains unaccepted on the current Mac
runtime; its stale branch is not restored over the garrison and terminal repairs.

The historical [eight-process experiment](docs/research/parallel-eight-collection-2026-10-27.md)
remains a separate diagnostic contract. Its checkpoint-699 benchmark performed no
PPO update: collection took 53.69 versus 161.20 seconds, and collection plus
verification took 189.10 versus 293.08 seconds. Two paired histories differed,
so these observed timings do not establish identical-workload acceleration or
current four-process performance. Eight-game batches are not enabled by the
manual continuation.

## Experiment agreement

The [Mac native learner](docs/research/native-mac-utilization-2026-10-29.md) now
uses private parallel frame graphs, exact retained text-backward schedules and
ordered gradient reduction. On two archived PPO workloads, the final verified
build reduces update time by **35–36%** versus the prior accepted native build;
final model, gradients, Adam, all RNG and complete continuation state match exactly.
The forward phase reached **16 concurrent workers**. Mean whole-update CPU use
is about **3.3–3.6 cores**, below the prior build because synchronization overhead
was removed; this is not continuous full-system saturation. The four-game batch,
model, PPO settings, policy support, terminal rewards and opponent stay fixed.
The full offline suite passed 1,753 tests with 47 documented skips. Measurements
cover PPO, not complete campaign throughput or policy quality. Python admission,
independent behavior checks, serialization, Adam and supervision remain authoritative.

`continue-original-native.sh` is an opt-in manual launcher with exact build and
source acceptance gates. The subsequent cloud run stopped on a private Git
publication read-back timeout after computing checkpoint **1,335**. Its
[recovery audit](docs/research/cloud1335-recovery-2026-10-29.md) verifies all 106
cloud checkpoint pairs and the complete failed-batch raw inventory. Checkpoint
1,335 restored exactly without another optimizer update. At that historical
boundary the original target had **2,824 admitted games and 2,176 remaining**,
with sequence **975** next. The original failed cloud report remains unchanged.
The subsequent Mac repair preserved checkpoint **1,392**. The user resumed from
that anchor, and the current run's verified checkpoint continues to advance.

For the interrupted 1,000-game goal, use `./continue-772.sh --check --hours 10`,
then launch `./continue-772.sh --hours 10` manually. It verifies the checkpoint
1,392 anchor and recalculates the remaining admitted games across manual windows;
it never resets a later checkpoint or replays consumed batch-1,032 seeds.

For a manual session targeting **1,000 additional admitted training games**, use
`./train-1000.sh --check` for offline preflight, then `./train-1000.sh` to launch
after verified cloud closure and latest-state import. The script uses the accepted
native build, a budget covering all available CPUs and the unchanged four-game
PPO batch. Serial dependencies mean this is not continuous full CPU occupancy. It stops
after 250 complete updates, or earlier on failure, a manual stop or its default
ten-hour safety cap (`--hours` overrides that cap). Evaluation games do not count
toward 1,000. It rejects a request larger than the verified remaining original
5,000-game target. It prints total wall-clock running time on success, failure or
a handled interruption, including startup, evaluation and cleanup. A clean early
stop is not proof of completing 1,000 games; check `target_games_met` and admitted
games in the runner summary. See the
[launcher validation](docs/research/manual-1000-game-launcher-2026-10-29.md).

For a short, isolated CPU and memory check, run `./test-native-load.sh`. It uses
the accepted build and disposable copies of archived batch 846, compares native
and Python PPO within a 120-second comparison deadline, and prints CPU capacity
use, peak parallel frame/replay workers, process lifetime peak RSS and exact-state
checks. It starts no gameplay or official training and leaves the cloud ownership
fence intact. It measures the
PPO workload, not full-campaign or GPU utilization.

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

The [final native acceptance](validation/capabilities-repair1-2026-10-25.json)
passed all **28 cases** (14 mechanics in both seats) against one frozen
implementation/runtime. Every case passed independent history and native replay;
3,036 transactions were checked with none skipped. The independently frozen
[macOS runtime](validation/runtime-freeze-v8-repair1-2026-10-25.json) removes donor-build
loader references and copies its resources; the complete suite was repeated
against those exact signed bytes. Matched Dimension Door cases
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

The historical v8 authorization was **one fresh local session of at most two hours**,
with 365 days, 5120 total choices, Nullkiller Easy/BattleAI, complete four-game
PPO batches, terminal-only rewards and both-seat held-out evaluation. Launch
was gated on all requested native feature checks, exact capture/replay,
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

### Reviewed v8 launch tooling

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
the actual update92 pair passed and was removed. Permanent migration must preserve
the original tensors and optimizer state in independent local and private copies
before a fresh session starts. Preparation evidence alone does not claim gameplay
or a new training update; the dated launch/campaign report records actual activity.

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
