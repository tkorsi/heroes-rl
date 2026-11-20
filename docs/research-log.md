# Heroes RL research log

Created on 26 October 2026 for a future scientific seminar. This is an evidence
index and laboratory notebook, not a replacement for the authoritative
[README](../README.md) or a claim that the agent has learned to win.

The historical entries below were reconstructed **retrospectively** from saved
reports. They were not preregistered hypotheses. Exact run timestamps, source
identities, settings and evidence bindings remain in the linked reports. Report
filenames sometimes use the Kyiv date while their timestamps use UTC.

Machine-readable stage records are in
[research-experiment-index.jsonl](../validation/research-experiment-index.jsonl).
Each record binds its sources by SHA256 and identifies its model lineage.

## Research question and fixed principles

Can a recurrent neural policy learn useful Heroes III adventure-map decisions
through interaction with the real VCMI rules and terminal rewards?

- The learner receives structured, player-visible state, including explicit
  window contents only after opening them. It receives no screenshots or hidden
  enemy state. Battles use the native automatic BattleAI.
- Natural victory gives +1, natural defeat gives -1, and ordinary transitions
  give 0. Technical failures are unscored. A verified horizon cutoff is a
  separate outcome, with only the bootstrap treatment declared for that run.
- Demonstrations, scripted learner strategies and reward shaping are excluded.
  Authored mechanics probes and winning-path diagnostics are separate evidence
  and are never silently admitted as training experience.
- A successful PPO update, changed weights, longer survival or occasional
  recruitment does not by itself establish improved winning ability.

The observation/action interface, scenario, opponent, horizons and runtime have
changed over the project's history. They are experimental conditions, not fixed
constants across every row below. Comparisons must respect those changes.

## Retrospective experiment chronology

| Stage | Verified observation | Interpretation and boundary |
| --- | --- | --- |
| [First restricted-interface PPO experiment](../validation/first-gameplay-2026-10-25.json), with [later evaluation](../validation/after-evaluation-comparison-2026-10-25.json) | Ten games: four before evaluations, two training games and four after evaluations. One PPO update from 294 transitions; ten Adam steps. All games reached the 256-choice horizon. | Initial execution covered six games; the last four evaluations were separately authorized. No natural wins/losses and no demonstrated improvement. This historical model lineage is separate from the current one. |
| [Longer frozen-neural evaluation](../validation/long-dialog-evaluation-2026-10-25.json) | Eight games scheduled with 4,096 choices and 200 days; two verified cutoffs, one failed replay and five unstarted games. No PPO. | An eight-game schedule was not eight completed games. The failed prefix stayed unscored. |
| [Repaired neural/built-in comparison](../validation/local-revalidation-summary-2026-10-25.json) | [Frozen neural](../validation/local-neural-revalidation-v2-2026-10-25.json): four verified cutoffs. [Nullkiller Easy](../validation/local-builtin-revalidation-v2-2026-10-25.json): four verified natural learner losses. Each schedule stopped on its fifth attempt; three later games were unstarted. | Nullkiller made natural terminal feedback reachable. No PPO; reused seeds and repaired verification did not retrospectively admit failed rows. |
| [First natural-outcome PPO pilot](../validation/builtin-ppo-pilot-2026-10-25.json) | Eight verified natural losses: two before evaluations, four training games and two after evaluations. One PPO update, 106 transitions and five Adam steps in 62.161 seconds. | Established a working bounded training path. Its trained checkpoint was subsequently lost during managed-worktree archival; it is not an ancestor of update 146. The incident and recovery are recorded in [project history](project-history.md). |
| [Recovery failure](../validation/recovery-2026-10-25-001.json) and [fresh continuation](../validation/builtin-continuation-2026-10-25.json) | The first recovery failed with no PPO samples. Fresh recovery and subsequent accepted batches reached update 21: 84 training games, 3,944 decisions and 149 Adam steps. Of those admitted games, 81 were natural losses and three verified cutoffs. | The current model lineage begins again from the preserved random v7 baseline. Evaluation games and verified games from rejected batches are excluded from these training counters. |
| [Cloud continuation, updates 21–31](../validation/cloud-final-analysis-2026-10-25.json) | Forty new training games, ten PPO updates, 2,832 decisions and 104 Adam steps; eight evaluations. The 48 verified games contained 47 natural losses and one day-limit cutoff. | The action limit was 5,120; the day limit remained 30. Linux gate/debugging attempts remain separate unscored evidence. |
| [Local365 continuation, updates 31–92](../validation/local365-final-analysis-2026-10-25.json) | 244 new training games, 61 PPO updates, 14,286 decisions and 530 Adam steps; 24 evaluations and two frozen gate games. All 270 completed games were natural losses, with no cutoff/timeout. | Increasing the horizon to 365 days allowed long natural endings, but did not demonstrate winning. The last fixed evaluation was update 90, not final update 92. |
| [Day62 unsupported notice](../validation/local365-day62-notice-investigation-2026-10-25.json) and [necromancy support](../validation/necromancy-support-2026-10-25.json) | A later game failed after 1,337 choices; three remaining games were unstarted. No samples from the failed batch were admitted. Additional native capabilities were implemented and validated. | Interface coverage is an engineering result. It does not establish that the policy learned to use the capability strategically. |
| [v8 startup/artifact failures](../validation/v8-artifact-crash-2026-10-26.json) | Attempt 1 failed before any policy choice. Attempt 2 completed two baseline losses, then crashed after 39 choices in its first training game; three games were unstarted. Neither attempt admitted PPO samples. | Expanded action support required a new baseline even though model/Adam/RNG state was preserved. The [artifact repair and native acceptance](../validation/runtime-artifact-fix-2026-10-26.json) preceded the next fresh session. |
| [v8 repair3, updates 92–146](../validation/local-v8-final-analysis-2026-10-26.json) | 216 new training games, 54 PPO updates, 19,579 decisions and 692 Adam steps; 24 evaluations. All 240 games were verified natural losses. The campaign stopped cleanly at 23:43:28 UTC on 25 October, before its deadlines. | No demonstrated strategic improvement. Recruitment occurred in 67 training games; all ten initiated training battles targeted neutrals. Final update 146 was not separately evaluated; the latest fixed evaluation was 145. |
| [Authored recruitment/rush diagnostic](../validation/recruit-rush-2026-10-26.json) | Three corrected Red trials won on day 1 with 39 pikemen: attack command 8, nine submitted commands overall. Both Blue trials lost. A preliminary Red control also won with the starting 25, leaving its purchased troops in town. All six replays passed. | Proves a short winning path exists, not that the network discovered it. It does not prove recruitment necessary or nine commands globally minimal. The last modal response was superseded by natural terminal. **Zero training samples.** |

The early implementation history, including movement, coordinate routing,
explicit town/hero windows, recruitment, army operations and later capabilities,
is preserved in [project history](project-history.md). Its old uses of “current”
refer to historical milestones, not the present project state.

The rush diagnostic ran concurrently with repair3 training. The table groups
related milestones; it does not imply that every row started after the preceding
row finished.

## Model lineage and accounting

These counters describe only the current recovered lineage. Earlier pilot
updates, diagnostic games, evaluations and rejected batches are not added to it.
Run sequence numbers and model update numbers are distinct identifiers.

| Accepted stage | New PPO updates | New training games | New decisions | New Adam steps |
| --- | ---: | ---: | ---: | ---: |
| Fresh recovery through update 21 | 21 | 84 | 3,944 | 149 |
| Cloud, 21→31 | 10 | 40 | 2,832 | 104 |
| Local365, 31→92 | 61 | 244 | 14,286 | 530 |
| v8 repair3, 92→146 | 54 | 216 | 19,579 | 692 |
| **Cumulative update 146** | **146** | **584** | **40,641** | **1,475** |

The admitted training games in this lineage comprise **580 natural losses and
four verified horizon cutoffs, with zero natural wins**. This is a derived
accounting total over accepted batches, not a new experiment. The final v8 run
alone had no cutoffs. Its final model, optimizer, RNG, progress and opponent pool
are preserved privately with exact restoration evidence.

An environment episode, a player rollout, a typed interface choice, a game day,
a PPO update and an Adam optimizer step are different quantities. Reports must
name which one they count. In these single-learner/built-in-opponent batches,
one admitted training game contributes one learner rollout.

## Latest fixed evaluation and its limits

| Metric | Red at update 92 | Red at update 145 | Blue at update 92 | Blue at update 145 |
| --- | ---: | ---: | ---: | ---: |
| Natural outcome | Loss | Loss | Loss | Loss |
| Last public game day | 7 | 2 | 2 | 2 |
| Typed choices | 257 | 62 | 17 | 17 |
| Completed recruitment operations | 1 | 0 | 0 | 0 |
| Changed movement endpoints | 3 | 0 | 0 | 0 |
| Initiated battles | 0 | 0 | 0 | 0 |

Source: [final v8 analysis](../validation/local-v8-final-analysis-2026-10-26.json).
Update 146 has no separate held-out evaluation. Repeated fixed seeds are useful
regression checks, but are not independent samples for estimating a general win
rate. Native opponent randomness is not fully specified by the engine seed.
This small comparison supports “no demonstrated improvement,” not a universal
claim that every aspect of the policy became worse.

Across the 216 training games, roughly 77% of choices concerned window
navigation/selection or army rearrangement. Recruitment volume and longer
individual games are useful observations, but they did not establish a learned
recruit-then-attack strategy. Accepted recruitment quantities and visible army
redistributions must not be mistaken for an independently reconstructed army
inventory; repeated transfers can count the same units multiple times.

## Newly authorized curriculum experiment

On 26 October 2026 the user authorized easier initial game states, a diagnostic
gate showing both natural wins and losses under the network's own sampled
actions, followed by a fresh two-hour local PPO campaign. This paragraph records
the authorization after task dispatch; it is **not** a claim that the gate or
training has completed, and it is **not** a preregistration of unspecified settings.

The starting checkpoint is 146. Real rules, terminal-only rewards and the ban on
demonstrations remain in force. The earlier authored rush traces stay excluded.
Changing initial conditions is an explicit curriculum intervention; success on
that distribution must not be presented as success from the original map start.
Doing nothing must not reliably win automatically in the selected curriculum.

The Teacher published its first preparation study at commit `3f40a374`:

- [Curriculum protocol and research record](https://github.com/tkorsi/heroes-rl/blob/3f40a374a3ed558071b01f153957d74d7f94dcc0/docs/research/curriculum-v1.md).
- [Append-only experiment events](https://github.com/tkorsi/heroes-rl/blob/3f40a374a3ed558071b01f153957d74d7f94dcc0/validation/curriculum-v1-research-log.jsonl).
- [Selection results](https://github.com/tkorsi/heroes-rl/blob/3f40a374a3ed558071b01f153957d74d7f94dcc0/validation/curriculum-v1-selection-2026-10-26.json).

The published selection report records 32 verified diagnostic games, 2,490
sampled choices, zero wins and 32 losses, split equally between seats. The 20×20
candidate accounted for eight games; the open 8×8 candidate for 24. No move
targeted the opponent town entrance. Native replay and policy reconstruction
passed, but the mixed-outcome gate did not. No PPO session started and update 146
remained unchanged. Do-nothing controls and held-out comparison were not run
because selection had already failed. The two pre-game loader errors and the
negative selection result are retained. This is a report-bound preparation
result, not evidence that winning is impossible.

Further candidate selection must be logged as a new experiment. Exact gates,
seeds, settings and hypotheses must be timestamped **before their corresponding
experiments**. Preparation is separate from the fresh two-hour campaign clock,
whose persisted deadline starts at actual launch and includes checks,
evaluation and preservation.

## Overnight handoff authorization, 26 October 2026

The user subsequently requested that the current local Teacher finish its
campaign, then continue the final verified state in cloud until **12:00 Kyiv
(10:00 UTC) on 26 October 2026**. The cloud compute cutoff is 09:55 UTC, leaving
five minutes for preservation and shutdown. This authorizes a sequential
continuation; it does not permit overlapping local/cloud training or extending
the cloud deadline if the local handoff arrives late.

The [recorded authorization and handoff contract](../validation/overnight-handoff-authorization-2026-10-26.json)
and [prepared English cloud task](operations/teacher-overnight-20261026.txt)
specify the required final checkpoint dependencies, both seat-specific maps,
restoration evidence and explicit ownership transfer through private Git.
At this entry's creation, cloud dispatch was blocked by access/Cloudflare
checks and the user was handling browser verification. **No cloud task creation
or training launch was confirmed.** This paragraph is a dated authorization
record, not a live activity indicator or an experiment result.

## Recording protocol for the seminar

For every new stage, preserve:

1. **Before execution:** research question, hypothesis, changed variables,
   unchanged controls, source/runtime/map/checkpoint identities, actual seed and
   seat schedules, opponent, observation/action interface, reward, PPO settings,
   budget, diagnostic gate and held-out success/failure criteria.
2. **During execution:** UTC timestamps, actual launch/deadlines, all run IDs,
   attempted/completed/admitted counts, failures and unstarted attempts,
   parameter changes with reasons, and paths/hashes binding the raw evidence.
3. **After execution:** natural outcomes versus cutoffs/failures, PPO and Adam
   updates, fixed evaluation, behavior measures, checkpoint restoration and
   shutdown receipts, interpretation, alternative explanations and limitations.

Keep previous entries intact. Append corrections with a reference to the entry
being corrected; do not rewrite an earlier hypothesis after seeing the result.
Label retrospective reconstructions explicitly. Do not invent missing timing,
resource use, uncertainty estimates or telemetry. A narrative claim must point
to a source report or be identified as a hypothesis.

Keep simulation, inference, experience reconstruction, PPO, replay verification,
startup/reset and preservation timings separate when measured. A short scripted
match's speed is not the training pipeline's throughput. Code/tests and compact
English evidence belong in public Git; large checkpoints, raw captures and
private preservation material stay in the private repository. Proprietary game
assets are not published.

The seminar narrative should retain negative results and infrastructure
failures. At present, the defensible conclusion is that the environment and
bounded training/evidence pipeline work, a short Red winning path exists, and
the completed network training has **not yet demonstrated learned winning play**.


## Completed curriculum and overnight continuation, 26 October 2026

This entry is retrospective. The prospective protocols, every failed preparation
study, the source-integrity incident, and subsequent technical corrections remain
in the immutable Teacher journals linked from the
[completed results](research/overnight-results-2026-10-26.md). Earlier authorization
and readiness paragraphs above retain what was known when they were written.

Local training added 432 games and 108 PPO updates (64 wins, 368 losses), moving
146 to 254. Cloud then added 632 games and 158 PPO updates (563 wins, 69 losses),
reaching update 412: 1,648 cumulative training rollouts, 95,836 decisions and
3,545 Adam steps. There were no horizon cutoffs in these two campaigns. Their
combined new training is 1,064 games, 266 updates, 55,195 decisions and 2,070 Adam
steps. Diagnostic/control/held-out and failed/partial experience stay excluded.

Fixed Linux evaluation improved from 1/4 wins per seat at update 254 to 4/4 per
seat at final update 412. The full nonmonotonic trajectory is retained: an earlier
8/8 had regressed to 2/8 at update 311 before recovering. The 272 cloud evaluation
games reuse the same eight starts; they are not 272 independent test cases.
Improvement on this easier open8x8 cohort does not establish original-map
generalization or a particular learned strategy. Cloud behavior counts combine
training and evaluation histories and show zero recruitment/army transfers; no
buy-army-then-attack claim is supported.

The cloud campaign stopped at 09:48:38 UTC, its controller at 09:49:22, final
held-out gameplay at 09:52:25, and preservation at 09:53:26, before the unchanged
09:55/10:00 cutoffs. Final checkpoint 412 and failed-attempt evidence are privately
preserved with exact model/Adam/RNG/progress/pool restoration. Publication faults
are retained separately from gameplay: sequence 118's operational record remains
failed while its complete valid update was subsequently acknowledged. The Blue
evaluation 286 publication fault was reconciled without replay or PPO admission.

The [final root audit](../validation/overnight-root-final-audit-2026-10-26.json)
binds imported report hashes, private receipts, independent counter/sequence
checks and the separate post-deadline owned-process verification. No games were
launched for the root audit and the expired authorization permits no restart.

## Frozen checkpoint 412 acceleration and transfer preparation

The [prospective protocol and completed result](research/checkpoint412-acceleration-2026-10-26.md)
bind the isolated source revisions, repeated starts and unchanged checkpoint.
All 72 diagnostic games passed with identical paired histories; no training
samples or updates were admitted. Complete probe rows improved 2.09 times, or
1.69 times including wrapper startup, on the same Mac. The startup interpreter
failure and Linux/Mac numerical reconstruction difference are retained.

The [English Teacher assignment](operations/teacher-transfer-eval-20261026.txt)
is a new bounded frozen comparison of checkpoints 146, 254 and 412 across
varied easier positions and the original scenario. Its 85-minute compute and
90-minute hard limits begin at the first new game. Neither this report nor
that assignment authorizes PPO. Preparation/delivery is not completed evaluation.

At 12:58 UTC the existing cloud Teacher visibly received the English assignment
and acknowledged the new frozen scope. The direct message tool returned403;
the already authenticated browser route succeeded. This is preparation, not
completed Linux acceptance or evaluation. See the
[dispatch receipt](../validation/teacher-transfer-dispatch-2026-10-26.json).

## Additional acceleration assignment

The user requested another twofold training speedup over Claude's current
optimized implementation. The [English task](operations/claude-acceleration-phase2-20261026.txt)
targets complete verified four-game PPO batches, including restoration and
durability, with component timings and fixed resource budgets. Isolated
disposable-state optimizer benchmarks are allowed; official checkpoint412 and
the active Teacher transfer study remain unchanged. Any numerical alternatives
are separate experiments, not silently accepted replacements. This records an
engineering objective, not a measured speedup or a new learning campaign.

At 13:25 UTC (minute precision), Claude visibly acknowledged the additional
twofold complete-cycle target in the existing Linux task and began reading
the assignment and profiling prerequisites. The [dispatch receipt](../validation/claude-acceleration-phase2-dispatch-2026-10-26.json)
records delivery and acknowledgment. No additional speedup has yet been measured.

## Completed frozen transfer and local script decision

The [finished transfer report](research/transfer-eval-2026-10-26.md) and its
prospective manifest, append-only journal and final status were reviewed and
imported byte-for-byte from Teacher commit
`82e022e6145814ef2b96c9b21ec463f94e26ffc7`. No cloud executable controller or
worker was imported by this report synchronization. The 192-entry transfer
schedule yielded 168 verified games (40 wins, 128 losses, no cutoffs), two
unscored startup failures and 22 unstarted invalid-fixture entries; the eight
Linux gate games are separate. Independent accounting matched the 178 unique
launch intents, cell totals and preserved schedule.

The failure diagnosis corrects the preliminary embedded-backend suspicion:
an authored hero anchor at x=0 gives native visitable x=-1, and both embedded
and accepted cold runtimes failed on that invalid fixture. Four affected
layouts were excluded by the same structural rule for all models and seats;
map bytes, seeds, failed attempts and original hypotheses were retained.
Checkpoint 412 won 24/24 valid varied easy starts, versus 14/24 for 254 and 2/24
for 146. All three lost all 32 original-scenario games. This supports transfer
between the tested easy positions, not success on the original board or a
demonstrated recruit-then-attack strategy.

The final report binds raw captures to private commit
`d00675305595e62078f8c9bab85b22d67b990504` and independent audit/stop evidence to
`24ae5b2ecb9ef035484887eb1ac9df47682ce347`. It records unchanged checkpoint
bytes, exact full-state restoration, zero PPO/Adam/sample admission and no live
owned workers. Root reviewed these published receipts; no gameplay or fresh
cloud runtime audit was performed for this import.

The user requested a manually launched local training script as the next
operating mode, without a persistent cloud LLM session. This records a tooling
decision and preparation work, not a new campaign launch or a reset of any
closed cloud deadline. The manual script must bind its own explicit input
configuration, checkpoint, budget and output evidence before a future run.

### Completed Teacher handoff review

The subsequent [cloud-to-local handoff](operations/teacher-cloud-to-local-handoff-20261026.md)
and [machine receipt](../validation/teacher-cloud-to-local-handoff-20261026.json)
were reviewed and imported exactly from
`7287573700a30aa67634f05e2acb786567c0e8fd`. The status and append-only Teacher
journal were refreshed from that same commit; the finished study report and
prospective manifest remain unchanged from `82e022e`. Private handoff
`eb37149e06e751b6215fd4aad370badca94275cd` binds checkpoint 412 and its
update-411 initial dependency, original checkpoint metadata, both easier maps,
and the original map. No raw private assets or cloud executable files were
imported in this report synchronization.

The published receipt records no live owned controllers, workers, native
processes or watchdogs, unheld ownership locks, and no queued automatic restart
or recurring cloud model supervision. Root reviewed these receipts without
starting new games or a cloud process audit. Exact restoration is distinct from
cross-platform numerical reconstruction: the previously recorded strict
Linux-to-Mac reconstruction mismatch remains, with no relaxed tolerance.

The original continuation ledger identifies sequence 266, Red, engine
911065--911068 and policy 921065--921068. The proposed manual local experiment
uses separately recorded 20,000,000/30,000,000 seed bases with an allocation
ledger; this is an explicit new experiment namespace, not a claim that the
original cloud schedule was continued unchanged. The handoff itself performs
zero games, PPO updates or Adam steps and authorizes no new campaign.

### Manual local runner: prospective design and offline validation

The new [manual runner](operations/local-script-training.md) restores full
checkpoint 412 and reuses the existing collector, replay verification and PPO.
The first preset retains the accepted easier8x8 stage3 maps; the proposed harder
owned-town/active-enemy-hero bridge is not silently substituted. Training seeds
use the new namespace recorded above. Each manual invocation freezes source and
records an explicit wall/monotonic budget before current-runtime baselines.
Both four-game baseline seats must pass before training. Only complete verified
four-game training batches advance a durable independently restored checkpoint.
Evaluation, partial and failed attempts never enter PPO.

[Offline validation](../validation/local-script-training-2026-10-26.json) passed
77 tests, including synthetic controller admission/preservation failures,
interruption and aggregate deadline cleanup. Exact model/Adam/RNG/progress/pool
restoration of 412 passed on the prepared Mac. The cold runtime's canonical
engine recipe and accepted historical identities match; first-launch native
acceptance remains required. No native game or real campaign update was run
during preparation. The script invokes no language-model service and stops on
technical failure. Raw runs and two verified checkpoint copies stay local
outside managed worktrees; remote archival is a separate analysis step.

### Frozen local release check

Executable commit `cf2021cd3b2225ced440b1459ad9220581a028c0` was copied to a
durable release. Canonical VCMI preparation and `prepare_vcmi.py --check` passed.
Both seat-specific baseline plans passed actual source/runtime/map/checkpoint
identity checks without execution. Full checkpoint 412 restored exactly:
1,648 rollouts, 95,836 decisions and 3,545 cumulative Adam steps. The checks
created no campaign, consumed no training seed, and invoked no native game.
A further 26 existing first-run compatibility tests passed, for 103 targeted
passes in total. The [frozen check](../validation/local-script-frozen-check-2026-10-26.json)
binds these receipts and the source commit. The obsolete paused handback
automation was deleted; future training starts only on manual invocation.

## Original-map 5,000-game manual experiment: preregistration

The user requested training, rather than a frozen-policy probability estimate:
5,000 stochastic games on the original two-Castle map, launched manually after
the current easy run, continuing its **final** complete state. The
[prospective configuration](../configs/local-original-5000-20261026.json) fixes
the original hero-management-v2 map, native opponent/rules, terminal-only rewards
and unchanged PPO. A positive outcome may supply useful advantage estimates;
neither obtaining a win nor learning a reliable strategy is assumed. The
previous offline likelihood of one scripted prefix is not used as an estimate
of whole-game win probability, a demonstration or training data.

The new script targets 1,250 complete four-game updates, alternates seats and
uses a separate40M/50M seed namespace. It excludes baseline/evaluation/failed
experience from the5,000 training count. Paired evaluation every100updates plus
baseline/final yields112evaluationgames for full completion. A48-hour safety
cap includes evaluation, verification and preservation; an early budget or
technical stop must be reported as a shortfall, never as completing5,000.

A shared lock rejects overlap with the existing easy run. The wrapper binds
its parent to the closed run's report and shutdown evidence, not a moving
latest pointer or rollback to412. The narrow adapter transition is pinned to
known old/current source identities; model, PPO and optimizer mathematics are
unchanged. [Offline checks](../validation/local-original5000-preparation-2026-10-26.json)
passed126tests, restored an immutable intermediate state only for compatibility,
and verified rejection while the existing owner lock is held. No original-map
gameplay or training was launched during preparation; the active frozen release
and its state were not modified. Final-parent selection remains deferred until
the user's later manual invocation after clean shutdown.

The original-map adapter and controller were then frozen at
`dd819e6f89929b1f2c4db3a39c4e225229da6f01`. Canonical dependency preparation and
exact checking passed. A read-only immutable intermediate snapshot567 restored
exactly (2,268rollouts,121,485decisions,4,578Adamsteps), and both original-map
baseline plans passed full identity checks without execution. This snapshot is
**not** the selected final parent. The [frozen receipt](../validation/local-original5000-frozen-check-2026-10-26.json)
binds the offline evidence; no training sequence or native game was consumed.

## 26 October 2026 — overnight original-map throughput assignment

The user renewed Claude's existing performance task until 27 October 13:00
Europe/Kiev (11:00 UTC). Heavy benchmarks stop at 10:50 UTC for preservation.
The prospective target is substantially less wall time for the manual original-map
5,000-game experiment, including replay, policy audit, PPO and durable restoration.
The phase-2 additional 2x target remains a target, not a result. The local runner
does not publish every batch to GitHub; local and network costs must stay separate.

The easy local series has closed at update629; the original-map parent remains
its exact final full state. Isolated disposable benchmarks do not enter that
lineage. The user retains manual launch of the 5,000-game campaign. The previous
opted-in reordered-gradient experiment remains separate from the exact default.
See [the prospective assignment](operations/claude-original5000-overnight-20261026.txt)
and [authorization](../validation/claude-original5000-authorization-2026-10-26.json).

The renewed assignment was delivered once to the existing Claude task at
20:13 Kyiv. Claude acknowledged the 11:00 UTC deadline and began work on
the assignment/watchdog. Installation is not yet independently verified.
[Dispatch receipt](../validation/claude-original5000-dispatch-2026-10-26.json)
records this distinction and the bounded root follow-ups.

## 26 October 2026 — completed manual easy run and private629 handoff

The user-launched local run closed at 17:58:39 UTC, advancing412 to629
with868 admitted training games and217 PPO updates. All868 training games
and352 separate evaluation games won on the easier8x8 distribution. All305
batches passed;217 unique training sequences266–482 were consumed. These
results do not establish original-map performance. The last paired fixed check
was627 (4/4 each seat); no final629 pair was run.

The final model/Adam/RNG/progress/pool restored exactly in local copies and
from an independent private Git download. [The synchronization receipt](../validation/local-script-final629-sync-2026-10-26.json)
binds the remote checkpoint commit and input dependency. A single535MB full
archive push failed with HTTP408; the complete local archive remains intact.
Checkpoint publication is confirmed; full-archive publication is tracked
separately and is not yet claimed. This transport failure occurred after clean
training shutdown and invalidates no admitted experience. No games or PPO were
started by this synchronization.

The full archive was subsequently published successfully without force pushes
or omitted evidence. Eight bounded metadata/chunk stages retained remote
acknowledgements; the original complete package commit is
`87e814e6d2de8a0690fed7e158d2e50f87b59267`, with publication receipt
`78afd6f27b589a71414f143f6f866d6ded0cca49` on the private branch
`codex/local-script-final629`. All 13 archive-part Git blobs match the locally
verified SHA256-bound archive of 38,337 regular files. Game-data symlinks were
recorded without copying their targets. The full archive was not redownloaded;
the smaller final checkpoint was downloaded and restored independently.
The original HTTP408 publication failure remains in the receipt and Git history.

Claude acknowledged the immutable629 handoff at 20:32 Kyiv and will validate
Linux relocation on disposable copies. This is not evidence of completed Linux
acceptance or a measured speedup for629. The deadline and manual launch remain
unchanged.

## 27 October 2026 — original-map timeout and exact699 preservation

The user manually launched the original-map campaign from629. It admitted280 games
and70 complete PPO updates, ending at699:0 wins,275 losses and5 collection-window
cutoffs. Batch336 collected four verified games but its16058-sample PPO update
exceeded the inherited3510-second effective inner deadline after466 attempted Adam
steps. Transactional rollback admitted no samples or update from336; no after
checkpoint exists for it. Owned shutdown and full699 restoration passed. The next
unused training sequence is337 and4720 games remain.

[The preservation receipt](../validation/local-script-final699-sync-2026-10-27.json)
binds a separately published private699 package, remote blob readback and exact
model/Adam/RNG/progress/pool restoration. Full raw evidence remains in the original
artifact directory; this audit does not claim a new remote full-archive copy.
The user authorized disposable Mac performance checks and timeout/continuation
preparation. The official continuation remains a manual user launch.

## 27 October 2026 — exact Mac699 acceleration and manual continuation

Disposable full699 copies on the user's M4 Max produced a matched long Red batch
in90.57 minutes with accepted code,43.34 minutes with exact serial acceleration,
and35.53 minutes with two replay threads (2.55x versus accepted). Semantic
training histories and full model/Adam/RNG/progress/sampler/pool state matched.
The captured Blue supplement also matched accepted and selected exact execution,
including gradients, without new games or official admission.

The planned two-update full-cycle gate remains incomplete: accepted A preserved
its conservative reserve before its second update and ran its ordinary final
pair. Retain all52 diagnostic native games and the original failed completeness
verdict. This is one ordered workload, not a whole-campaign duration forecast.
The [research record](research/mac699-acceleration-2026-10-27.md) and
[compact evidence](../validation/mac699-acceleration-results-2026-10-27.json)
preserve this deviation, the late baseline-only view difference and exactness.

The manual wrapper continues official699 with4720 games remaining and fresh
sequence337, selected exact two-thread execution, a three-hour batch allowance
and a bounded14-day fresh-session cap. It recalculates the remainder after later
manual closures; no automatic retry,629 reset or benchmark promotion is allowed.
The official training campaign has not been launched by this preparation.

The final readiness check was run from the user's actual main checkout at
`32b3adf`, after a narrowly scoped source-discovery correction for a pre-existing
ignored generated virtualenv. The earlier main refusal is retained; learner math
and native runtime were unchanged. [Main readiness](../validation/original699-main-readiness-2026-10-27.json)
confirms exact699 restoration,4,720 remaining games and fresh sequence337, with
zero launched official games or updates. Both ownership locks are free.

## 27 October 2026 — prospective visible interaction budget

The user requested fixing repeated actions without reward shaping. The
[prospective protocol](research/interaction-guard-2026-10-27.md) discloses a
versioned policy-support intervention, its management limits and mandatory-choice
escapes. It requires frozen-policy native replay/reconstruction and unchanged
full checkpoint state before enabling a manual continuation. This is not an
exact-performance claim; no official launch is part of preparation.

## 27 October 2026 — guarded diagnostics and manual checkpoint717 continuation

The previous user-launched run closed cleanly at checkpoint717 (72 new admitted
games,18 updates,62 losses and10 collection cutoffs). Its full state is privately
published and independently restored. The original target has4,648 games left,
with fresh training sequence355; no automatic continuation was launched.

Eight fresh frozen717 diagnostics with the visible interaction budget individually
passed native replay and strict policy reconstruction:352 choices in total,
2–117 per game, all natural losses. No zero-MP/current-tile move was selected;
optional navigation stayed within two uses per semantic target/progress epoch.
No diagnostic data entered PPO. This is a policy-support intervention with
terminal-only reward, not demonstrated improvement in winning or a paired speedup.

The original diagnostic report remains failed: after all games passed, a strict
OrderedDict/dict comparison and a duplicate shutdown-timestamp keyword broke
final bookkeeping. The missing in-memory assertion chain is not claimed to have
passed. Three regression tests cover the corrected harness; separate read-only
audits verify persisted full state, every saved game proof and actual shutdown.
The earlier zero-game preparation path error is also retained. Full raw evidence
and these corrections are privately preserved. See the [complete results and
limitations](research/interaction-guard-2026-10-27.md) and
[compact report](../validation/interaction-guard-results-2026-10-27.json).

The final offline check passed from the actual main checkout at `5b683ea`: complete717 restoration,4,648 remaining games and fresh sequence355. The guard version is bound explicitly. [Main readiness](../validation/interaction-guard-main-readiness-2026-10-27.json) started no games or updates.


## 28 October 2026 — map-garrison adapter repair and checkpoint770 preservation

The manual guarded original-map run stopped after53 complete batches and212
natural losses, advancing717 to770. Sixteen evaluation losses remain separate.
The valid `move_to` in failed sequence408 revisited an owned sawmill, whose
native garrison query was unsupported by the adapter. No failed experience
entered PPO; the next unused sequence is409 and4,436 games remain.
[Full-state preservation and remote restoration](../validation/local-script-final770-sync-2026-10-28.json)
passed. The user will launch any continuation manually.

The [prospective repair protocol and retained attempts](research/garrison-adapter-2026-10-28.md)
extend query support to map garrisons, preserve explicit policy choices and
terminal-only rewards, and test native replay and permission boundaries.
Related validation exposed a mixed C++ object build and a native locked-garrison
deposit-direction defect. Rejected candidates and evidence remain retained;
only a fully rebuilt runtime with complete acceptance may become the manual default.

The rebuilt adapter is now accepted:16 new native controls,28 legacy native
regressions and8 actual frozen770 policy games passed identity/replay/cleanup.
The policy games were8 natural losses and580 decisions, with0 PPO samples or
optimizer steps. A separate126-decision offline inference audit on scripted
fixture histories also passed. All rejected candidates and diagnostics are
privately preserved with independent remote hash read-back. The manual runtime
gate requires these proofs. [Final repair results](../validation/garrison-adapter-results-2026-10-28.json)
record770,4,436 remaining games and unused sequence409; no launch occurred.

The actual main checkout at25ce435 now passes the default manual wrapper's
offline check with the accepted runtime:770 restored,4,436 games left, sequence409,
10-hour cap. [Main readiness](../validation/garrison-main-readiness-2026-10-28.json)
records configuration activation and independently verified private receipts.
No official training was launched.


## 28 October 2026 — checkpoint916 and terminal-request replay repair

The latest manual original-map run admitted 584 natural losses in 146 complete
updates, advancing 770 to 916. Sixteen evaluation losses were separate. Sequence
555's third game ended naturally but replay accounting rejected a cancelled
Nullkiller artifact-transfer request; its two preceding completed games were
also excluded, and no batch update began. The next unused sequence is 556;
1,148 of 5,000 original-map games are admitted and 3,852 remain.

The [repair and retained unsuccessful attempts](research/terminal-replay-repair-2026-10-28.md)
identify a Python/native terminal-tail allowlist gap and a related marketplace
drift. Supplemental replay of the preserved failure checked all 90 native
inputs, 532 packets and 777 transactions, without changing the failed cohort.
The clean rebuilt runtime passed 594 accounting checks, 16 garrison controls,
28 legacy controls and eight frozen-916 policy games (591 choices, eight losses),
with exact reconstruction and no training samples or updates. Full checkpoint
state remained unchanged. The stable offline suite ran 1,628 tests: 1,595 passed, 33 skipped.
These results establish repair coverage, not improved game strategy. Fresh
policy games did not reproduce the rare artifact-transfer cancellation; the
original captured-command replay and native fixtures cover it explicitly.

The runtime release is `terminal-replay-v1`. Original map, architecture, PPO,
Nullkiller Easy/BattleAI, terminal-only +1/-1/0 rewards and the existing visible
interaction budget remain unchanged. [Preservation](../validation/local-script-final916-sync-2026-10-28.json)
and [manual readiness](../validation/terminal-replay-main-readiness-2026-10-28.json)
record the complete state and exact ten-hour check. No official training or
cloud work was launched by this repair.


Checkpoint916 and all failed-run evidence are now privately published and
independently restored. All 19,536 archived run files and 1,707 diagnostic files
passed remote read-back; Adam, all RNG state, sampler, progress and pool matched
exactly. Final private receipt commit is `0082b8317f386e443af55d84d48534ddc47e405b`.
The intermittent publication/read-back transport failures are recorded separately
from the native replay rejection and supplied no training data.

## 28 October 2026 — transparent artifact-origin audit

The user questioned whether the cancelled terminal request addressed a real
artifact on the simple map. The [inventory audit](research/artifact-inventory-audit-2026-10-28.md)
confirms empty authored inventories but native random starting bonuses: Orrin
received Collar of Conjuring and Edric received Necklace of Swiftness. Exact
recorded-state reconstruction follows necklace instance 242 from Edric's neck
through battle loot into Orrin's backpack. It existed at request 91; replacing
the collar was slot-compatible but the equipment request remained unexecuted.

A separately compiled diagnostic records presence, identity, source/destination
contents and execution status without changing the accepted runtime or exposing
opponent inventory to the policy. Missing-holder and empty-slot controls report
absence. All 90 inputs, 532 packets and 777 transactions replayed exactly.
The initial incompatible CLI-option attempt and intermediate successful capture
are retained. No training samples, optimizer updates or new games were produced.
Production cancellation accounting remains a provenance check, not a statement
of full gameplay legality. Historical checkpoint 916 was not modified. The
follow-up manual check refused an occupied training lock: a separate user run
was already active on frozen release `36e0b90` and reported checkpoint 953 at
16:06 UTC. Its workers and locks were left alone. The old 3,852-game remaining
count belongs to the earlier handoff and is not a current active-run count.


## 28 October 2026 — reviewed concurrent collection integration

[The integration report](research/parallel-continuation-2026-10-28.md) records
merging Codex's parallel collector while preserving four-game PPO batches and
the 640-step cap. Child interpreters now activate the plan's execution/visible
interaction budget, load the exact parent metadata, and independently reject
recorded support that differs from the plan. Failed batches revoke receipts and
preserve retention/cleanup errors. Claude's latest Python improvements were
already merged; its old embedded native experiment remains unaccepted here.

An initial preflight failed before games because the old native acceptance bound
older collector source. The strict gate was retained: 44 fresh native controls
and eight frozen-1095 evaluations (530 choices) refreshed acceptance on the same
binaries. Historical acceptance and machine files remain unchanged. A separate
16-game diagnostic then passed 1,073 choices, four isolated PPO updates, exact
checkpoint round trips and full original-state restoration. All 24 policy games
were losses; none entered official training. Different fresh-seed cohorts mean
these timings establish no identical-workload speedup. Four collectors remain
opt-in; parallel evaluation has offline coverage only.

The latest user-run session closed normally at checkpoint 1,095 with budget
reserve, 103 updates / 412 admitted losses and complete cleanup. The original target
has 1,864 admitted games, 3,136 remaining, and next sequence 735. The merge does not
launch a campaign or consume official seeds. The report retains failed
preflight/development attempts and the full validation limits.

Main checkout check mode also passed for one- and ten-hour continuations with
four collectors: exact 1095 restore, 3,136 remaining games, next sequence 735. Main
engine preparation matched the unchanged recipe. Historical machine bytes and
official checkpoints remain preserved; no official training was launched.

## 29 October 2026 — manual session interrupted by lid-close sleep

The next user-launched one-hour session on frozen `f5a7f25` admitted 536 natural
losses in 134 complete PPO updates, advancing 1,095 to 1,229. The Mac entered
clamshell sleep at 20:10:29 UTC on 28 October and woke at 20:26:15 UTC, after the
20:21:06 hard wall deadline. The guardian sent SIGKILL on wake; historical
shutdown/deadline certification failed. This was a deadline interruption during
periodic Blue evaluation, with no evidence of a new engine or adapter rejection.

The [investigation](research/local-sleep-interruption-2026-10-29.md) corrects the
stale outer report: all four evaluation games reached terminal states, three
passed replay/reconstruction, and the fourth verification was interrupted.
The entire pending cohort remains excluded and supplied no PPO data. The
session's accepted evaluation total was 20 losses; final evaluation is absent.

All 134 admitted batch reports and preservation copies passed audit. Checkpoint
1,229 and its backup restored identical model, Adam, all RNG, progress, sampler
and pool state; a later process audit found no survivors. The original target
has 2,400 admitted games, 2,600 remaining, and unused sequence 869 next. These
later observations do not rewrite historical cleanup failure. The real manual
check correctly refuses continuation pending a separate recovery path.

No engine/runtime code, experiment settings, official artifacts or seed ledger
were changed. No gameplay or PPO was launched by the investigation. Private
receipts retain sleep/process evidence and complete state hashes; remote
publication of this latest run was not verified. An initial check-output
redirection failed before script execution; the subsequent actual check returned
the expected refusal. No new regression or speedup claim is made.

## 29 October 2026 — audited manual recovery; cloud lifetime unmeasured

The separate [checkpoint1229 recovery](research/checkpoint1229-recovery-2026-10-29.md)
now permits a new manual window after revalidating all 134 admitted updates,
exact canonical/backup state, actual runtime bytes, failed evidence and current
process closure. The original missed deadline and failed shutdown remain true.
No official games, updates, reservations or restart occurred. Main one-hour and
ten-hour checks and a frozen-release ten-hour check passed with 2,600 games
remaining and unused sequence869. Canonical engine preparation still matches.

Review caught an inherited-recovery seed-ledger gap before publication: an
unadmitted reservation could be removed while completed ancestry still passed.
Exact ledger append checks now retain original rows and failed/preparing attempts.
The first broad test run overlapped source edits and failed its source-pin checks
(three failures, seven subtest errors); that log remains preserved. The stable
rerun passed 1,663 of 1,710 tests, with 47 platform/optional skips. All 110 final
focused tests passed. No integrity or replay gate was relaxed.

A 45-file private checkpoint/evidence snapshot was independently downloaded and
restored to the exact complete state; a separate private recovery receipt also
passed remote readback. This is not a full raw-lineage or Linux-ready handoff.
One authorized cloud chat and one inspection follow-up stopped before the
process probe: required environment status and network-policy inputs were
unavailable. Cloud lifetime is unmeasured, not observed to fail. The standalone
trainer has no model calls; no recurring model supervision or cloud training was
started. No new strategic improvement or throughput claim is made.

## 29 October 2026 — authorized ten-hour cloud launch blocked during setup

The user explicitly authorized a new cloud chat and a ten-hour official script.
The new chat `Setup blocked report` (conversation
`6ae2a223-538c-83eb-9bcc-b55cb9f7df99`) stopped before any repository download or
launch. Its executor again lacked the required managed status tool and
network-policy snapshot; credential readiness remained unverified. An attempt
to inspect cloud setup through Computer Use was denied by the app-access policy.
No settings were changed and the ten-hour window never started. This is a setup
block, not a gameplay failure or evidence about detached process lifetime.

The current Mac ten-hour check still passed at checkpoint 1229, with 2,600 games
remaining and sequence 869 next. Both owner locks were free, no matching old
processes remained, and 21 checked files stayed byte-exact. Source inspection
confirmed that those locks are local flock files, not a cross-host lease. A
cloud start still needs accepted Linux binaries, complete lineage relocation
and enforceable ownership. No official experience or optimizer update occurred.
See the [attempt record](../validation/cloud-launch-attempt-2026-10-29.json).

## 29 October 2026 — native Mac learner accepted; cloud ownership retained

The user's requested [C++ learner rewrite](research/native-mac-training-2026-10-29.md)
now executes semantic/spatial encoders, recurrent windows, selected-action
likelihoods, PPO losses and ordered parallel text backward in a freshly compiled
ATen extension. Python retains admission, the existing Adam object, checkpoint
serialization and process supervision. The four-game batch, original map,
architecture, PPO settings, opponent, terminal rewards and disclosed visible
interaction budget remain unchanged. Bounded fresh-process replay verification
also passed four actual archived native captures with no unchecked requests.

The first numerical candidate matched state exactly but was 0.528–0.546x as fast
as the optimized Python reference. It was rejected for performance: it omitted
the existing text-leaf reuse and concurrent backward replay. All attempts and
failed preparation tests remain private. Review also exposed unsupported aliases,
worker startup/error cleanup gaps, module-order checks and a lower-level worker
acceptance bypass; these were corrected without weakening integrity gates.

On final source `7101636`, both execution orders of two recorded four-game PPO
updates matched all 13 comparisons, including gradients, model, Adam, all RNG and
complete continuation state. The 210-sample update improved 1.699–1.700x and the
584-sample update 1.805–1.808x. Fifteen simultaneous text replays were observed;
mean whole-update use was 4.24–4.39 cores on the 16-core Mac. These results do not
establish full campaign throughput, learning improvement or continuous use of
all cores. The earlier small fixture and every failed performance pair remain
separately identified in the [acceptance record](../validation/native-mac-learner-2026-10-29.json).

The full offline suite passed 1,719 of 1,766 tests with 47 platform/optional skips.
After the last module-order/introspection guard tightening, 139 of 140 focused
checks passed, with one skip; these included the compiled controls, source gates,
replay and continuation. Final recorded comparisons used that last source.
Canonical engine preparation matched exactly. Independent review found no
remaining issue within the supported canonical model contract.

The manual native launcher and accepted build are prepared, but current Mac
launch readiness is false: the real ten-hour check correctly refuses the existing
read-only ownership fence. The separately authorized cloud run's frozen source
and ownership were not changed. A later verified cloud closure and latest-state
import must precede local continuation; local checkpoint 1229 is not a substitute
for the cloud's newer state. This work launched no official games or updates,
reserved no seeds, changed no checkpoint and removed no fence.

## 29 October 2026 — repeatable native load-check launcher

The requested `./test-native-load.sh` now runs the accepted native/Python
comparison on disposable copies of archived batch 846, with a 120-second bound.
It reports actual PPO CPU use against host capacity and process lifetime peak
RSS, while preserving all 13 exact-state and owned-cleanup checks. Review closed
an output-symlink placement gap and rejected incomplete cleanup evidence before
the real test. All 12 focused launcher/benchmark tests passed.

The [actual main-checkout invocation](../validation/native-load-launcher-2026-10-29.json)
on source `ea354ae` passed: native PPO took 15.77 seconds versus Python's 28.42
(1.80x), averaging 4.59 of 16 CPU cores (28.7% of available capacity), with 15
concurrent text replays and 0.77 GiB process lifetime peak RSS. This is one paired
order and measures neither GPU use nor full campaign utilization. It confirms
parallel work, not continuous use of every core. All official pointer/lock
snapshots stayed unchanged; the cloud fence remains. No official training,
gameplay, seed reservation or official update occurred.

## 29 October 2026: Mac utilization bottlenecks, exact numerical repair

The [utilization investigation](research/native-mac-utilization-2026-10-29.md)
profiles the actual 28.7% load result. Replay workers often waited for the producer;
allocation/disposal and PyTorch callback-state scheduling added substantial overhead.
An initial reducer-only candidate preserved state but did not establish a native
speedup. Strict text-backward execution schedules and private parallel frame graphs
then reduced two archived update workloads by 35–36% versus the accepted native build.
Peak frame concurrency reached 16, while average whole-update CPU use remained
3.3–3.6 cores; increased occupancy is not claimed as the result. All paired state,
model, gradient, Adam, RNG and sampler checks were exact. Build/import failures and
a corrected test-cardinality assertion remain disclosed in the report.

The final runtime passed the 1,800-test offline suite (1,753 pass, 47 documented
skips), reviewed source pins and canonical generated-engine agreement. Raw state
and captures remain private. No new game seeds, official samples or updates were
admitted; the Mac/cloud ownership fence and checkpoint pointers are preserved.
The [compact evidence](../validation/native-mac-utilization-2026-10-29.json)
distinguishes measured PPO speed from full-campaign speed and policy quality.

The synchronized main-checkout load test then passed: 10.39 seconds native,
29.10 seconds Python, 16 peak frame workers, 22.4% average CPU capacity, all 13
exact checks and cleanup. The real ten-hour continuation check correctly refused
at the unchanged ownership fence before the runner. The checkpoint/run pointers
and locks matched their pre-task hashes, modes and identities.

## 29 October 2026 — timed manual 1,000-game launcher

The requested `./train-1000.sh` targets 1,000 additional admitted games (250
complete PPO updates), preserving original campaign accounting and excluding
evaluation games. It uses the accepted native runtime and available CPUs, keeps
the learning settings fixed, and prints total monotonic elapsed time after
controller cleanup. Its default ten-hour safety cap may stop the session early;
the runner's admitted count and target flag remain authoritative.

The [launcher report](research/manual-1000-game-launcher-2026-10-29.md) records
143 passing checks and one Linux-only skip across 144 focused tests, including
isolated signal/cleanup controls and a mocked 250-update stop. Two macOS path
expectation errors and the intermediate stale-source-pin rejection are disclosed.
Reviewed source pins, generated-engine agreement and loading the actual unchanged
native library passed. The acceptance gate now includes the entire manual
launcher chain. No new performance or policy-quality claim follows from these tests.

The real worktree preflight correctly refused the existing cloud ownership fence
and printed its elapsed time. All six official pointer/lock snapshots remained
unchanged. No seeds, games or official updates were created; verified cloud
closure and latest-state import are still required before local continuation.

The synchronized main checkout independently passed native-library loading and
canonical engine agreement. Its exact `./train-1000.sh --check` command refused
the unchanged fence with status 2 and printed total running time of 0.29 seconds.
The final compact receipt binds implementation commit `4f14064` and preserves
the separate historical performance acceptance evidence.

## 29 October 2026 — cloud checkpoint 1335 publication recovery

The [publication recovery investigation](research/cloud1335-recovery-2026-10-29.md)
identified a 120-second `git show` timeout while the old blobless-clone verifier
read back batch 974's private raw archive. The push and complete PPO update had
already succeeded. The campaign correctly retained checkpoint 1334 as admitted
and checkpoint 1335 as pending; this was a publication failure, with no evidence
of an engine, policy-action, adapter or replay failure. The 5.66 MB part does not
explain the underlying network stall, which remains unknown.

Explicit bounded blob fetches now precede strictly local, streamed verification.
Exact commit/manifest/source bindings, all raw hashes, full policy/resume restore
and expected-head publication remain mandatory. The failed transaction passed
fresh remote readback in 9.58 seconds without another commit or optimizer update.
Private return-to-Mac certificates subsequently passed publication/readback in
8.75 seconds. All 106 checkpoint pairs and the pending archive's 134 regular
files passed their separate evidence audit.

The recovered checkpoint is update 1335, with 5,340 cumulative rollouts, 680,677
decisions and 23,014 Adam steps. Original-campaign accounting is 2,824 admitted
games and 2,176 remaining; sequence 974 remains consumed and 975 is next. The
model, Adam, all RNG, sampler, progress and pool are preserved. The original map,
learning settings, terminal-only rewards and visible interaction-budget contract
remain unchanged, as do the engine and accepted native learner runtime.

The cloud app reported a deleted, nonresumable environment. A separately bound
terminal-training certificate uses the immutable closure, consumed claim and
frozen source control flow; it does not claim a fresh outer-supervisor process
observation or remote lock-mode change. The Mac import requires the exact pinned
receipt, complete ancestry/seed audit and real continuation verification before
unfencing. Partial application remains fenced with a durable journal.

The import passed after correcting historical PID reuse with strict process-birth
and group-leader proofs; no unrelated processes were signalled. The installed
state passed continuation verification before both original lock modes were
restored. The first real main preflight then exposed an acceptance record that
predated the merged performance changes. Fresh validation passed 44 native
controls and 8 frozen-1335 policy games, with 1,504 exact reconstructed choices,
all natural losses, full state unchanged and no PPO input. A separate accepted
runtime descriptor and new immutable machine file preserve the original machine;
only acceptance metadata changes. Both the historical lineage and current
acceptance remain independently mandatory.

The full suite passed 1,861/1,908 tests with 47 platform/optional skips before the
final refresh seams; final implementation passed 289/290 focused tests with one
skip. The actual native library passed 13 exact disposable PPO comparisons and
cleanup. Both canonical engine checks passed. Main's exact
`./train-1000.sh --check` now passes in 6.81 seconds: checkpoint 1335, 1,000 new games,
2,176 original games remaining, next sequence 975 and a ten-hour cap. It performs
no gameplay or PPO. All failed preparation, source-gate, PID, ownership-lock,
acceptance and test-selection attempts are retained in the detailed report.
No official training, seed reservation, duplicate update or automatic restart
occurred. Raw captures, state and operational receipts remain private.

## 29 October 2026 — checkpoint 1392 replay and rejected-batch repair

The user-started 1,000-game session admitted 228 natural losses in 57 complete
updates, advancing 1335 to 1392. Batch 1032 then failed replay verification on an
unapplied terminal Nullkiller `GarrisonHeroSwap` request. A second reporting
contract error obscured that cause after whole-batch receipt revocation. All four
failed-batch games remain excluded and sequence 1032 remains consumed. The
[repair report](research/checkpoint1392-repair-2026-10-29.md) records exact model,
Adam, RNG, progress, sampler and pool restoration, with 772 games left in this
manual goal and 1,948 left in the original target.

Both Python and canonical C++ accounting now cover the reviewed town-hero swap
without relaxing packet, identity, terminal or replay checks. Rejected batches
retain individual verification evidence and their original cause; revoked
histories receive rejected status and are not scored. The supervisor remains
strict. A fresh nine-target runtime build passed actual library-load checks,
715 native accounting controls, terminal cancellation controls and a supplemental
replay of the copied failure. That supplemental record remains ineligible for
PPO and does not change the failed cohort's result.

Fresh native acceptance passed 16 garrison controls, 28 legacy controls and eight
frozen-1392 games with 1,136 exactly reconstructed choices, all natural losses.
The fresh policy cohort had no type-189 terminal tail (five type-182 tails);
type 189 is covered by the exact failed capture and native accounting controls.
Every diagnostic had bounded deadlines and verified cleanup; none entered PPO.
The full suite passed 1,864/1,936 tests with 72 platform/optional skips. A separate
41-test native learner suite and 178 focused tests passed; these overlapping
counts are not additive. Small policy acceptance cohorts do not establish
learning improvement, long-run stability or full CPU saturation.

The first real `./continue-772.sh --check --hours 10` invocation exited 2 after
2.14 seconds because the historical cloud-machine guard refused the new runtime.
A narrowly registered immutable runtime transition passed 44 tests and independent
review. The corrected final combined selection passed 182 tests, and the actual
main-checkout retry at `9eb67debd056ea8426eb70331b153177cb46251d` passed in 10.42
seconds: exact checkpoint 1392, 772 remaining goal games, next sequence 1033,
ten-hour cap, four collectors and four verifiers within a 16-CPU budget. It
launched no gameplay or PPO. Failed test-selection imports and a macOS fixture
path-normalization mismatch are retained and disclosed.
An inherited acceptance annotation was separately corrected with a supersedes
binding, preserving the original descriptor and all native/policy evidence.

Private commit `a489c0e4ae133616bf69569db740a57c0e4c63cb` passed fresh remote
readback and full checkpoint restoration. It preserves 130 failed-batch regular
files, undereferenced Data links and 1,378 run metadata files. Successful official
raw histories and intermediate checkpoint pairs remain durable locally and are
explicitly outside that package. Final private commit
`375b61c2f22f99c65552aabd654e3ca53e82948f` separately passed fresh readback in
21.54 seconds for 1,694 diagnostic/proof files and 52 undereferenced symlinks.
It contains both acceptance descriptors and the final source/main-readiness
proofs, without runtime binaries or game data.
The final closure audit found no owned processes across 841 deadline records,
verified both locks available and left all five official/original-machine
bindings unchanged, with no signals. Official state and consumed seed
reservations remain unchanged. The new launcher recalculates the same 772-game
goal across manual windows, never resets the latest
checkpoint, and prints total running time. This repair launched no official
training or restart.

## 2026-10-29 — CPU pipeline and isolated NumPy environment for the next manual launch

The user requested preparation without interrupting the active continuation.
That run stayed on frozen `10d3d4f`, its original environment and accepted native
binary. A read-only audit at 22:00 UTC restored checkpoint 1,459 exactly, with
504 games then remaining in the 772-game goal. Moving official state continued
advancing independently; no diagnostic modified its seed ledger, admitted
experience, checkpoint or ownership locks.

The prepared environment adds NumPy 2.3.5 while preserving all ten existing
package versions, Python/Torch ABI and native library hashes. Tensor/NumPy bridge
and numerical smoke checks emitted no warnings. A copied complete batch-1033
update reproduced model, Adam, RNG, pool and numerical PPO results exactly.
Diagnostic progress and sampler bookkeeping intentionally stayed at the parent;
the original admitted counters were independently audited, not admitted again.

Replay verification can overlap remaining collection only after a completed
collector exits, its capture is sealed and cleanup verifies. Resource plan v2
budgets four collectors, four verifiers and a coordinator at nine or more CPUs;
smaller budgets retain sequential phases. All four games still must pass before
PPO, results retain schedule order, and failure revokes earlier receipts.

Two successful fresh cohorts passed 32 games and 7,674 reconstructed choices,
all natural losses, with full parent state unchanged and all receipts revoked.
The first matched eight pairs and took 33.613 versus 31.834 seconds. The second
matched seven pairs and took 33.103 versus 34.765 seconds; one pair differed in
native opponent observations despite identical learner actions. That second
comparison has no identical-workload speedup estimate. Both were baseline-first
under live-run contention. Reliable net throughput gain and full CPU saturation
are not established.

The additional C++ graph-compaction candidate remained bitwise exact but was
11.27% slower and used 5.40% more CPU time at the production 16-thread budget
across both execution orders. It was rejected and the accepted C++/PPO bytes
restored. Hook-contract gaps, missing-type fallback, the redundant-niceness
launch failure, cancelled unrun comparisons and the reporting-only error remain
in private evidence. The fresh-pipeline setup refusal consumed its private seed
reservation and was not relabeled as a successful run.

The unchanged runtime passed 16 garrison and 28 legacy native controls. The full
suite ran 1,990 tests (1,932 passed, 58 skips); final combined selection ran 217
(213 passed, four skips), with bounded cleanup. Subsequent source-closure tests
reject mutation of every newly inventoried acceptance helper. A generic-launcher
interpreter compatibility issue found in final review was corrected without
rewriting old machine descriptors. The compact [validation result](../validation/cpu-next-launch-2026-10-29.json)
and [full report](research/cpu-next-launch-2026-10-29.md) record the final main check,
source bindings, additional wrapper tests and retained limitations. Native source
preparation checks pass. Raw captures, copied tensors and build evidence stay
private. No official training or automatic restart was launched by this preparation.

The final wrapper-corrected selection passed 223/227 tests with four skips.
The real main-checkout continuation check at `7cd022f` passed the environment,
machine and native gates, then refused the active official owner lock in 7.79
seconds with exit 2 and no NumPy warning. This is the required duplicate-worker
refusal, not a completed continuation preflight; the user repeats it after the
current run ends. The final 22:06 UTC audit restored checkpoint 1,468 exactly
(468 goal games and 1,644 original-target games remaining then), confirmed the
active frozen source and unchanged protected inputs, and found no owned
diagnostic processes. No official process was signalled.

## 2026-10-30 — Completed checkpoint 1585 and prospective revealed-movement validation

The user-launched checkpoint-1392 continuation finished its 772-game goal in
193 complete PPO updates at checkpoint 1585. It admitted 771 natural losses and
one verified collection-window cutoff; 32 evaluation games were excluded. All
193 admission rows and sequences 1033–1225 were audited. Canonical and backup
model/Adam/RNG/progress/sampler/pool restoration passed, as did final shutdown;
no owned training process remains. The original target has 3824 admitted games
and 1176 remaining, with sequence 1226 next. This local integrity audit does not
by itself certify private remote publication of checkpoint 1585.

Read-only movement analysis found 41416 of 47712 attempts targeted coordinates
absent from the current delivered known tiles; all returned no_route_known
without changing the endpoint. The explicit visible-interaction-budget-revealed-v1
intervention limits movement to tiles in the hero's current observation on the
same level. It retains the existing budgets and uses no route-feasibility or
strategic mask. This changes policy support; it is not an exact acceleration.
The map, model, PPO, opponent, four-game batch and terminal-only rewards stay fixed.

The private prospective protocol binds checkpoint 1585 and 15 protected inputs,
with SHA-256 bd9d7547225f9c1e34922943701dfae385a70664973d9a3f0bb99c7cf8fc23fd.
It calls for 16 fresh native diagnostic games, eight per verification arm and
four per seat in each arm, under a 900-second deadline. Both arms use the new
intervention. Acceptance requires exact replay/reconstruction, zero sampled
unrevealed targets, full parent preservation, revoked receipts and owned cleanup.
No diagnostics enter PPO or official seed accounting. Native validation, full
suite, runtime acceptance and final main-checkout preflight are pending.

The separate continue-1585.sh launcher defaults to ten hours, uses the accepted
native manifest and isolated NumPy environment, prints elapsed time and
recalculates the remaining original goal from verified descendants. Its focused
mock-only continuation/launcher selection passed 34 tests. A pure goal audit
against actual accepted-main ancestry confirmed 1176 remaining and sequence1226;
this is not final runtime readiness. Earlier report-writer and preflight probe
failures are retained and disclosed in the
[prospective report](research/revealed-movement-2026-10-30.md).

Remote inspection confirms accepted NumPy/pipeline and Claude performance work
already on main at c289408; the old prefix branch has an equivalent patch in
main. The rejected slower C++ compaction remains excluded. No repeatable
full-training speedup, full CPU saturation or policy-quality gain is claimed.
No official training or automatic restart was launched by this preparation.

### 30 October 2026 — revealed movement accepted for manual continuation

The final source 14cb529 passed 2024 of 2057 offline tests (33 skipped), 44 fresh
native controls and 16 policy games from copied checkpoint 1585. Every one of 128
movement targets was revealed, with 1838 choices reconstructed exactly. Both
verification arms covered both seats; all 16 games lost naturally. No quality
or speedup claim follows. All diagnostic receipts were revoked; zero official
samples/updates were admitted.

Two artifact-fixture failures exposed extra native inventory and legitimate
disassembly-to-backpack behavior; explicit diagnostic transfers and equipment
restoration now retain exact ownership/slots and all original regression gates.
The first policy cohort was excluded after an old-only capture whitelist
rejected the new version; the registered-version gate and real private-pipe
capture tests now pass. All failures, consumed schedules and immutable captures
are retained in the private evidence and disclosed in the
[final report](research/revealed-movement-2026-10-30.md).

The real main-checkout check passed at 1585 with 1176 remaining and sequence 1226
next, a 16-CPU resource budget, four collectors/four verifiers, pipelining and
the warning-free NumPy environment. Full model/Adam/RNG/progress/sampler/pool
state and all 15 protected inputs remain unchanged. The accepted runtime pin
is 877c1919964b5f9037b7f348015dcfa58e55c28f06ca58cceefb49c3000a35a6.
The user alone launches `./continue-1585.sh --hours 10`; it recalculates the same
remaining goal and prints elapsed time. No official training was started.

## 2026-10-30: performance telemetry removal protocol

The user requested removal of CPU counters and phase profiling, retaining parallel execution and all learning/integrity settings. Self-play is explanation only. Full checkpoint 1728 and its completed-run evidence are protected. The [prospective protocol](research/telemetry-removal-2026-10-30.md) requires a fresh native build, exact numerical parity, excluded native/policy diagnostics and a real manual preflight. No training launch is authorized.

## 2026-10-30: telemetry removal accepted for manual continuation

The telemetry-free C++ learner and Python training path passed 2,028 offline tests
(33 platform/optional skips), 43 focused native tests, a bitwise-equal copied PPO
update, 44 native controls and 16 frozen-1728 policy games. All 2,030 choices
reconstructed exactly and all 142 movement targets were revealed. The first
policy attempt stopped at the clean-checkout gate after eight excluded games;
its consumed starts were retained and the accepted attempt used fresh starts.
Three stale launcher assertions were corrected; failed logs remain private.
No instrumentation-overhead percentage or speedup is established.

The actual main-checkout ten-hour check passed: checkpoint 1728 restores exactly,
604 original-target games remain and sequence 1369 is next. Both native binary
and source inventory were verified. All 15 protected canonical inputs remain
unchanged, and no matching owned training or diagnostic processes remain.
No training or self-play was launched. The [report](research/telemetry-removal-2026-10-30.md)
and [compact evidence](../validation/telemetry-removal-2026-10-30.json) distinguish
production telemetry removal, retained integrity evidence, observed map travel,
and the explanation-only self-play discussion.

## 2026-10-30: authorized 100-game self-play protocol

The user now requests 100 actual self-play training games and an examination of
travel, aimless repetition and time-limit endings. The [prospective protocol](research/selfplay100-2026-10-30.md)
starts from full checkpoint 1728 in a separate lineage, retains four-game PPO
batches and all other learning settings, and uses frozen batch-start opponents.
Validation must precede training. Technical failures remain unscored and stop
the campaign; verified collection/day cutoffs retain their ordinary bootstrap
contract. Post-run travel metrics are never policy inputs or rewards.

## 2026-10-30: self-play stopped for a ten-game report

The accepted separate experiment ran from 1728 to 1730 with two complete updates
and eight admitted games. User stop at ten raced with four-way collection: twelve
games verified, but the interrupted third update never published and all its four
games remain excluded. First-ten outcomes are one learner win, one loss and eight
action-budget cutoffs; these are not wall timeouts. Learner movement included
1,149 revealed targets, 18 no-route responses, one town capture and substantial
repetition (88.1% position revisits; 69.6% stack swaps). No playing-strength claim
is supported. Complete 1730 state and evidence are preserved and restored exactly.

The manual continuation follows the published lineage and reserves 92 fresh games
after the entire consumed 100-seed window. Default check performs no gameplay or
reservation. All original Nullkiller state stays unchanged. The
[report](research/selfplay100-2026-10-30.md) and
[compact results](../validation/selfplay100-stopped-2026-10-30.json) separate all
denominators and disclose diagnostic failures and the stop race. No restart was
performed after the user's stop.

## 2026-10-30: one manual self-play evaluation with doubled choices

The separate 100-game target completed at checkpoint 1753: three learner wins,
eight losses and 89 collection cutoffs, with three learner town captures. All
natural victories belonged to Blue. The user now requests one manually launched
game with twice the choice budget. The [protocol](research/selfplay-one-game-2x-2026-10-30.md)
uses frozen 1753 policies, 9920 normal plus 320 reserved choices per player, and a
10240 hard ceiling. It preserves the day horizon, map, support and rewards, and
performs no PPO update. Preparation must not launch the game.

## 2026-10-30: optional reporting separated from training ownership

The completed self-play supervisor retained both campaign locks while serially
analyzing 184 histories (33.6 GB) without a reporting deadline or progress. The
user authorized safe unblocking and a fix. After exact process identity,
checkpoint 1753 restoration and verified shutdown checks, only the report PID
received SIGINT. All 370 old worker/guardian PIDs were absent; the supervisor and
its caffeinate child exited, the locks became available, and all 128 protected
files stayed unchanged. The original optional full report remains incomplete.

Training now finishes before optional reporting. Explicit reports use separate
private outputs, progress and an independent deadline without a training lease.
The focused suite passed 105 tests, canonical engine agreement passed, and a
real three-second reporting timeout verified cleanup while the campaign lock
was held by a diagnostic checker. The initial synthetic path-alias failure and
interrupt-receipt review correction are disclosed in the [report](research/reporting-lock-fix-2026-10-30.md).
The actual main-folder one-game check passed exact full-state restore at 1753,
with no gameplay, seed reservation or update. The [compact evidence](../validation/reporting-lock-fix-2026-10-30.json)
records the narrow reporting-controller pin; all engine/learner/replay sources
and learning settings remain unchanged. No new training or evaluation was run.

## 2026-10-30: self-play replay knowledge views and game audit

The user requested Red/Blue knowledge maps, removal of collected/defeated objects,
and mine-owner marks. The Swift viewer now uses each player's recorded visibility
for drawing and object selection, opaque unknown tiles, and owner-color mine
bars. A shared projection intersects visible IDs with current native objects;
a Swift-executed regression covers stale resource/hero IDs and all mine owners.
The release build and actual app opening passed. Initial build errors from a
stale selection variable and SwiftUI multi-file entry-point mode were corrected.
No engine, policy, learning configuration or training artifacts changed.

The doubled-choice frozen1753 game ended on day104 without a winner. Blue defeated
a neutral on day1 and lost its hero to Red on day5. Red hired three additional
heroes on days62/80/85 (four total). Blue's town appeared in Red's delivered
observation on day87; it was not captured. Red revealed356/400 tiles, made9920
policy choices, and spent6443 of them swapping stacks. It never selected a
move_to target at the Blue town's15,7 coordinate. Both gold piles and both neutral
mines remained untouched. The defeated neutral and Blue hero disappear from the
native frames. Player-local handles were resolved by object type/owner/position,
rather than equated with native IDs. These facts show failure to finish the
opponent in this game; they do not establish what the policy intended.
See [compact evidence](../validation/replay-player-knowledge-2026-10-30.json).

## 2026-10-30: independently initialized tiny-policy candidate

The user requested a smaller network for a separate experiment, keeping the
available actions. A width-only8-hidden/4-feature candidate has15460 parameters
versus74344 in the existing64/32 model. CNN/GRU/tree architecture remains. Fresh
random initial policy/RNG/pool was saved privately and restored exactly; no
checkpoint 1753 weights or Adam state were transferred. All126 protected original
files remain unchanged. Sixteen unique offline controls and an actual saved-tiny
C++/Python prefix parity check passed. A limited128-frame warm-prefix comparison
was1.18x faster; full gameplay/training speed and learning are unmeasured. The
[report](research/tiny-policy-2026-10-30.md) discloses test-selector/manifest and
metadata-gate diagnostic corrections, and [validation](../validation/tiny-policy-2026-10-30.json)
keeps raw initialization/checkpoints private. No gameplay or training started.
The tiny candidate is not accepted or selected by an existing training launcher.

## 2026-10-30: lossless self-play storage and within-game progress

The user requested faster storage/checking and action progress inside each game.
A versioned envelope now references the existing manifest-bound player histories
instead of embedding a second copy. Full behavior support, probabilities, values,
prefixes and samples remain explicit; the strict typed and native checks remain.
Training now consumes the native verifier's freshly reconstructed history proof
instead of invoking that same reconstruction twice. Display snapshots are separate
from evidence, throttled, and cannot grant admission or affect policy RNG.

On a fresh copy of the completed 1753 doubled-choice game, exact decoded equality
and independent seeded reconstruction passed for 89,446 frames, 11,763 actions, 11,765
values and 30,487 conditionals. The actual accepted native verifier checked 102,220
transactions with none unchecked. All 158 protected originals remained unchanged
and owned cleanup passed. The artifact shrank from 2.48 GB to 30.3 MB; replacing only
that duplicate reduces the retained episode by 37.53%. Write plus strict reload
fell from 73.73 s to 44.97 s (1.64x) in one measurement on one archived game, with
unflushed caches. This is not evidence of full-game/training throughput.

The full suite ran 2,177 tests (2,103 passed, 74 skipped); final focused integration
ran 153 (152 passed, one skipped). Canonical engine agreement and the real main
checkout's manual `--check` passed with exact full-state restore at 1753. No game,
training update, sample admission or seed reservation occurred. The tiny policy
is still untrained and unselected. The [report](research/rollout-storage-2026-10-30.md)
and [validation](../validation/rollout-storage-2026-10-30.json) disclose stale unit
fixtures, progress-error controls and the benchmark minimum-deadline correction;
the full copied-game benchmark passed on its first attempt.

## 2026-10-30: manual single-game launcher for the untrained tiny model

The user corrected the request to require their own launch with the small network.
A separate `play-one-tiny-selfplay.sh` defaults to offline check mode; explicit
`--run` selects independent frozen copies of the preserved random 8/4 initialization
(15,460 parameters), with no PPO/Adam update or admission. Existing doubled budgets,
revealed movement support, progress display, compact storage and full native replay
checks remain. A separate consumed-seed ledger prevents retries from reusing seeds.
The protected 1753 state is a comparison baseline, never the tiny model's parent.

Fifty-eight focused offline tests passed, including 16 new controls. The actual
main-checkout tiny and legacy shell checks both passed with empty stderr and exact
state restoration. Canonical engine agreement passed. No game, native worker,
training update, seed reservation or tiny output namespace was started. Initial
legacy/custom-worker mock signature failures were corrected without weakening
production checks. Native tiny gameplay and speed remain unmeasured. See the
[report](research/tiny-one-game-2026-10-30.md) and
[validation](../validation/tiny-one-game-2026-10-30.json).

## 2026-10-30: temporary recruit-only army experiment (preregistered)

The user requested a second explicit simplification alongside automatic combat:
remove troop-slot rearrangement, splitting, merging and individual transfers.
Stack opening and upgrades are temporarily disabled. The user explicitly retained
whole-hero garrison entry/exit so purchased town troops can reach the hero, and
requested current own-army composition/combat statistics without clicks.

The source audit corrected a false assumption: the previous interface exposed
only troop IDs/counts after opening a hero/town/exchange; stack clicks supplied
upgrade offers, not combat statistics. The new versioned own-army extension is
owner-only and binds its native ready/header, manifest, finalization, history,
policy input and replay. Historical wire/features retain their prior shape.

The permanent [temporary restriction register](temporary-restrictions.md) includes
an explicit removal checklist. This support change must be lifted in a separately
versioned experiment, not silently erased from historical reports. Rewards remain
terminal-only and model tensors/PPO are unchanged. Native acceptance is pending: a
clean build, both-seat bounded mechanics prefixes, exact replay, a fabricated-stat
rejection control and a frozen tiny technical prefix are required before release.
All diagnostic experience remains excluded from training.

## 2026-10-30: recruit-only/passive-army native acceptance and manual readiness

The clean native build compiled 782 objects and produced a new frozen runtime,
with actual dynamic-library loading checked. Two fresh-seed diagnostic attempts
passed six technical prefixes and 24 controls each. The final attempt reconstructed
all 62 Red and 64 Blue choices exactly. Recruitment, actual town garrisons,
whole-hero garrison entry/exit and artifact-modified current stats passed for both
seats. Legacy optional-field absence passed. Both native fabricated-attack
controls were rejected at regenerated-record comparison. No full scored policy
game or training campaign was launched; no sample entered PPO and no weight changed.

The first full offline run (2,239 tests) had 10 failures and 13 errors, primarily
stale historical source pins and a reference encoder's absent-field handling.
A focused attempt loaded pins before their writer completed and also found a
tuple/set test typo; its stable replacement passed. The second full run (2,252
tests) had one failure: another legacy expectation still admitted all globally
registered support versions. That expectation now names the two historical
versions explicitly. Final affected-module/launcher/native validation ran 125
tests with 124 passes and one skip; the full suite was not repeated after this
last test-only correction and receipt pin. All attempts remain recorded privately.

The compatibility review fixed a real isolation gap: the historical trainer no
longer inherits new support versions from a shared registry and rejects passive
army declarations in its plan/session/metadata/lineage. The new tiny experiment
is explicit; historical checkpoints, semantics and reports are not relabeled.
Both actual main-checkout shell checks passed with empty stderr. All 136 protected
files are unchanged; exact tiny initialization/pool and baseline 1753 full state
restore, and no native process remains. The next manual tiny seed is unused;
check mode reserved none. The first user-run tiny game remains historical evidence
under the old mode (day-limit cutoff, 3,613 Red / 6,907 Blue choices, no update).

The [restriction register](temporary-restrictions.md) retains the required removal
checklist. Numeric adventure stats and three traits do not constitute a complete
special-ability catalog. Short controls establish integrity, not stronger play or
faster full games. See the [implementation report](research/recruit-only-passive-armies-2026-10-30.md),
[native acceptance](../validation/passive-armies-native-2026-10-30.json), and
[manual readiness](../validation/recruit-only-readiness-2026-10-30.json).

## 2026-10-30: stop tiny experiments, prepare trained game with day progress

The user stopped the tiny-model experiment and requested one more manual game,
then asked for the current day in the progress bar. The next launcher is
`play-one-selfplay.sh`: trained checkpoint 1753, hidden 64 / features 32, 74,344
parameters. It retains the requested troop-slot restriction, passive own-army
information, original map, automatic combat, doubled choice budgets and day 365
horizon. It runs one frozen self-play diagnostic without PPO updates. Tiny
artifacts and the shared large-model seed ledger remain preserved.

Day progress reads only calendar metadata already delivered to the collector.
It adds no engine query or history scan and keeps the existing one-write-per-second
throttle. Exact synthetic collector results and RNG match with the observer
absent, present or failing. Existing snapshots remain readable; unknown days are
shown honestly and a delivered day 366 cutoff is not clamped to365.

The actual 1753 policy consumed both seats' hash-verified recorded passive-army
observations. Python and the accepted C++ backend matched bit for bit for action
log probability, value, recurrent state, conditional probabilities and native
prefixes with 1/4 workers. Weights/RNG were unchanged and no gradient/update was
created. Two offline harness failures (stdin syntax; frozen parameter flags) and
a disposable launcher fixture path failure were corrected without relaxing any
production guard. Final focused validation ran 124 tests, with 123 passes and one
skip; shell syntax and canonical engine agreement passed.

The prior native acceptance receipt remains immutable. An exact three-file
reviewed display transition binds the UI, its callback and source-pin metadata;
all other native-accepted implementation bytes remain exact. This is explicitly
inherited native evidence, not a claim that a new game was run. The full suite and
native gameplay were not repeated. Real manual readiness and preservation are
recorded in the [preparation report](research/large-one-game-day-progress-2026-10-30.md)
and its compact validation receipt. No game, training, restart or official seed
reservation is launched by this preparation.

The actual main-checkout new and historical shell checks both passed with empty
stderr. The new launcher selects the large 1753 model, unchanged weights, retained
recruit-only/passive-army contract and unused engine seed 1,960,000,001. All 136
protected files are unchanged; exact full-state restoration passed. No seed or
game was started. See [compact readiness](../validation/large-one-game-day-progress-2026-10-30.json).

## 2026-10-30: readable colored terminal settings and progress

The user requested better readability and colors for the selected rules/options.
Terminal output now groups Setup, Game rules and Limits; allowed actions,
temporary restrictions, learning-off status and Red/Blue progress have distinct
colors plus explicit text labels. Current day, choices and dialog reserves remain
visible. Saving, verification, a natural outcome and an unscored cutoff are shown
separately. Non-terminal output remains compatible; explicit `--json` emits clean
JSON stdout with timing on stderr. `NO_COLOR`, `TERM=dumb` and narrow terminals
have plain readable fallbacks. No dependency or profiling subsystem was added.

An active user game was discovered before editing. Implementation and validation
used an isolated sibling checkout while main stayed clean at its original commit.
The focused suite ran 132 tests (131 passed, one skipped), followed by 13 successful
launcher checks after final spacing and movement-label edits. Visual review used
a synthetic preview of the actual ANSI renderers, not a game. Independent review
found no blocker. Collection logic, progress schema/throttle, policy inputs,
learning settings, models, seed handling and verification gates remain unchanged.
Source pins account only for the reviewed display change and matching pin metadata.

Main synchronization must wait for the active run's verified shutdown and hold
the existing campaign lease. No game, training, replay, automatic retry or seed
reservation is launched by this task. See the
[UI report](research/terminal-ux-2026-10-30.md) for scope and validation boundaries.

The existing user run shut down before synchronization; its processes were
absent and cleanup/protected-state/plan checks passed. Main was fast-forwarded
under the existing campaign lease. Actual main `--check --json`, canonical engine
agreement and two real PTY checks passed; all 137 protected files remain exact.
No seed was reserved and no gameplay was started. See
[compact validation](../validation/terminal-ux-2026-10-30.json).

## 2026-10-31: explicit empty-equipment start and completed-game explanation

The user requested artifact-free starts and explicitly included spellbooks and
war machines. The final opt-in `no-artifacts-v1` disables every artifact for every hero throughout
the game, including ordinary items, spellbooks, war machines and acquisitions
from hires, purchases, loot and rewards. This follows two explicit scope
clarifications; narrower candidates were not accepted. Native initialization RNG,
map, starting armies and gold/resource bonuses remain preserved. This changes
starting siege and magic capabilities and is recorded in the temporary restriction
register. The trained-1753 frozen one-game launcher requires new runtime acceptance;
no full game, training run or official seed reservation is started by preparation.

The latest completed game's initial artifact panels contain only catapults, not
treasure. Later spellbook/tent items arrived with hired heroes. Blue lost to Red's
town garrison on day 18; Red's original hero survived until day 197. Four new-month
neutral camps really spawned. Red repeatedly attacked them without collecting the
large recruited town army and lost 12 heroes. Red hired Edric after Blue lost him,
which can make the replay's allegiance changes confusing. See the
[source-bounded audit and validation protocol](research/no-starting-equipment-2026-10-31.md).

The 4 November follow-up established that Blue never built a Tavern, while Red
built one on day 165 and first hired on day 187. Red targeted the left edge in
86.8% of movement commands; ten of twelve fatal camp encounters followed adjacent
guarded-tile moves. Neither side learned during the frozen game. Exact enemy
garrison/camp quantities were absent from policy observations.

Final `no-artifacts-v1` acceptance passed 220 native controls, nine paired
comparisons, 438 native parser controls and 97 exactly reconstructed frozen-policy
choices. Validation also fixed ordinary artifact pickup notice recognition and
an uninitialized empty-Tavern packet role. Failed diagnostic attempts and their
consumed seeds remain private. No tests, fixtures, probes or validation tooling
are included in the publication diff; the pinned acceptance receipt stays local.
No full game or official training was started.

## 2026-11-04: one-action town cooldown

In the archived checkpoint-1753 game, 139 of 466 town closes were followed by
opening the same town on the same day. In 126 of those 139 cases the player had
chosen nothing inside the town. The next manual frozen game tests a separate
policy support version that suppresses all town openings after a confirmed town
close until another action completes. It preserves town purchases, ending the
turn, movement, and mandatory dialogue choices. Alternating between hero and town
windows can still satisfy the rule. The weights and coordinate encoding remain
the same; hero-relative coordinates are a separate experiment. No performance
or outcome comparison is available until the user launches the game.

Offline checks passed 60 existing tests. At both seats' archived reopen decisions,
the new support blocked the submitted reopen before sampling, and independent
history reconstruction produced the same support. The old support still allowed
those choices. Synthetic controls verified that the cooldown covers multiple
towns, survives a rejected action and clears after an accepted end turn.

## 2026-11-04: hero-relative destinations and exact command cooldown

The completed town-cooldown game reached the 365-day limit with neither town
captured. It verified all 12,723 choices with zero updates. There were zero
immediate reopens after 344 town closes. Opening and closing empty artifact
windows accounted for 4,480 choices. Red still chose the leftmost column for
1,452 of its 1,696 movement commands.

The worker ran for 21 minutes 45 seconds. The manifest and rollout timestamps
divide this into about 8 minutes 2 seconds through finalized gameplay, 3 minutes
39 seconds to assemble and save the rollout, and 10 minutes 4 seconds to reload
the evidence, reconstruct the frozen policies, run the native replay, and close
out the report. These are phase estimates from file and process timestamps, not
per-function profiling. The original player histories totaled 1.59 GB, and the
independent replay wrote a 1.60 GB contextual tape. The 30 MB rollout is only the
envelope that references those histories.

The optional `--hero-relative` launcher mode keeps checkpoint 1753 and changes
the destination scorer's two coordinate inputs to normalized offsets from the
selected hero. Native map indices and candidate ordering remain intact. The
mode also adds a cooldown for the same command from that player's last choice.
A different hero, destination or quantity remains available. Required answers,
window exits and a last-resort end turn preserve progress. Both changes are
enabled in this game, so it cannot isolate their effects. The launcher without
the flag retains the previous experiment. Offline controls checked both seats
with checkpoint 1753, command filtering, quantity probabilities, and agreement
with the native recipe. Gameplay remains a manual launch.

## 2026-11-04: ten concurrent frozen games, first deadline attempt

The user requested ten simultaneous games with ten visible terminal rows and a
20-minute end-to-end limit, retaining the current checkpoint-1753 hero-relative
experiment. The runner was committed before testing (894c7cc, followed by
8c0ea73). No tests, probes or raw evidence were published.

The first private trial observed all ten native leaders alive concurrently.
All ten reached their original game boundary, but only eight passed full
verification before the work deadline. External elapsed time was 1,159.30
seconds; this is a failed acceptance, not a ten-game success. The two longest
captures contained 12,429 and 10,833 choices. Both stopped during verification
because of the declared wall limit; the replay report for the latter wrapped
its timeout as an unsuccessful replay. There was no demonstrated replay
mismatch. All recorded worker/native groups were independently confirmed closed,
and all protected checkpoint inputs were unchanged. No PPO updates occurred.
The initial aggregate incorrectly combined game failures with cleanup status;
the next runner revision separates these outcomes.

The existing focused suite passed 58 of 59 tests. Its source-inventory fixture
failure was reproduced against pre-change commit 054d99d with identical test
bytes. The real runtime preflight and canonical engine preparation check passed.
A bounded 45-second profile of a preserved long rollout intentionally expired:
it showed substantial repeated map-record serialization and strict history
loading. A two-second native stack sample was also taken late in the failed
trial; this attempt is not a clean standalone speed benchmark.

The next candidate streams history hashes, avoids redundant prefix scans and
copies, specializes serialization of simple immutable map records, and retains
just-validated canonical frame text through paired validation. Every exact
round-trip, hash, chronology, policy and native replay check remains. Analytics
now counts submitted terminal moves even when no ordinary result follows.
Terminal disconnects cannot suppress private result reporting. These changes
are committed before their validation and fresh-seed batch; acceptance remains
pending. Historical seeds and failed captures remain consumed and private.

The 489840a candidate passed 117 existing core tests. Differential local checks
covered 976 wire records across 34 record types and seven schemas, 84 transition
comparisons, 21 paired-history checks and 84 tampered-hash rejections. Initial
local harness retries corrected an import path, a legacy-map fixture/profile
combination and a missing synthetic cutoff declaration; no production changes
were needed for those harness issues. Closed-terminal and failed-worker controls
passed. Independent reconstruction of all 10,833 choices from a preserved long
capture was exact (87.09 seconds strict load, 142.24 seconds policy checks).
Native replay intentionally rejected that historical capture's old adapter hash;
this was not counted as full verification, and the source-binding gate remained.

The second fresh-seed ten-game trial again verified eight games before its work
deadline, with external elapsed time 1,158.69 seconds. All ten finished gameplay,
but native replay for the 11,906- and 11,425-choice captures timed out. Cleanup
and protected-state checks passed. These two failures are preserved privately;
they are not admitted games or successful throughput results.

The next revision overlaps independent native replay with strict reload and
policy reconstruction using the existing replay CLI and guarded subprocess
ownership. Every proof and before/after input hash is still required. Parent
cleanup covers the replay process group as well as the game process group.
The ordinary one-game verification path remains sequential by default. This
runner revision is committed before its next validation and fresh-seed trial.
The design follows the profiling and independent-work overlap principles in
[Sample Factory's profiling documentation](https://www.samplefactory.dev/07-advanced-topics/profiling/)
and [EnvPool](https://proceedings.nips.cc/paper_files/paper/2022/hash/8caaf08e49ddbad6694fae067442ee21-Abstract-Datasets_and_Benchmarks.html);
no external throughput figures are claimed for Heroes and no asynchronous
learning or policy-lag settings are introduced.

The 46a4af5 parallel-verification candidate passed 41 existing process/progress
checks, two sequential-verification controls, nine local guarded-process failure
controls and a copied 210-choice native positive control (3.94 seconds). The third
fresh-seed trial nevertheless failed acceptance: seven of ten games verified in
1,159.03 external seconds. All ten native games overlapped and finished gameplay;
the 11,208-, 11,585- and 11,724-choice captures exhausted verification time.
Protected state and owned cleanup passed, with zero PPO updates. This batch is
preserved as a failed attempt, not a successful ten-game performance result.

The next candidate removes complete-history copying from the encoder and the
adapter's incremental wire cache. Encoder snapshots retain fixed prefix bounds
and branch safely after failed speculative encodings. Finalization can deliver
validated immutable frames directly inside the process, retaining both structural
and exact canonical prefix comparisons. The independent persisted audit, strict
reload, full policy reconstruction and native replay remain required. No day or
choice limit, model, action support, reward or learning setting changes. The
Markdown analytics now defers final batch status/time to the authoritative JSON
report, avoiding a preliminary success label near the deadline. These production
changes are committed before validation; performance acceptance remains pending.

The 237afe4 candidate passed 228 existing tests and the actual launcher preflight
and canonical engine preparation check. The broader adapter test invocation
initially failed because its legacy imports require the tests directory on the
module path; unittest discovery supplied that path and all 31 adapter tests
passed. Local validation also passed 31 typed-history controls, 239,904 bounded
slice cases, 952 index cases, eight post-append rollback controls, 469 baseline
encoder/snapshot comparisons, 36 exact model-state frames and nine exact sampled
choices/RNG comparisons. No test or diagnostic source was published.

The typed-history harness first needed the headless module import directory.
A deliberate object.__setattr__ mutation of a copied frozen frame carrying an
old private canonical cache can defeat in-process cached-text comparisons; this
violates the immutable-object contract and was not treated as a supported input
path. Normal frozen construction rejects that malformed frame, and fresh forged
records, persisted JSON tampering and typed/JSON prefix changes were rejected.
The adapter accessor returns only already decoded and chronology-validated frozen
frames. Independent disk reload and replay remain mandatory.

The fourth fresh-seed batch ended with failed acceptance: 5/10 verified, external
wall time 1,159.26 seconds (internal 1,158.84 seconds including analytics). All ten
native games were independently observed running concurrently, with ten persistent
terminal rows. The terminal recorded the last collecting state at 666.8 seconds
and all gameplay complete at 667.3 seconds. The captures contained 87,446 choices.
Eight reached the day boundary and two reached a player's ordinary-choice limit;
these capture declarations do not certify the five unverified games.

Games 2, 3, 5, 6 and 10 timed out during verification after collection/storage had
completed. None has a completed policy-reconstruction or native-replay proof.
The five successful games were 1, 4, 7, 8 and 9; all were unscored day-limit
cutoffs with zero wins and zero confirmed town captures. Across their 33,022
choices, 23,754 (71.93%) opened, closed or navigated windows. Their 4,905 movement
attempts all targeted revealed tiles, 4,781 changed the observed endpoint and
76 returned no_route_known. These analytics cover only the five verified games.

All 30 owned worker/native/replay process groups were independently confirmed
empty after exit; no recorded leader, guardian or writer PID remained. All 805
protected checkpoint bindings rehashed unchanged, with zero admitted samples and
zero PPO updates. Failed captures and consumed seed reservations remain private.
The main conclusion is limited: ten-game concurrency and terminal progress work,
but full saving, verification and analytics for all ten still exceed the target.
The different seeds and substantially different choice totals across attempts
do not establish a controlled speedup ratio. The remaining measured delay is
post-game history processing and verification; checks were not weakened.

During the fourth run the user explicitly requested finishing this batch, making
conclusions and stopping. No fifth trial, training campaign or further performance
change was started. The candidate remains committed and checkable, but it is not
accepted as satisfying the requested 20-minute ten-game result.

## 2026-11-04 — Outcome reporting correction and ten-game training preparation

The user requested separate Red wins, Blue wins, game cutoffs and verification
failures, followed by training in ten-game batches to investigate improvement.
They also requested a mandatory research journal and explicitly retained manual
control of script launches. `AGENTS.md` now requires recording meaningful
investigation, design, implementation, validation, failure and result steps,
including corrections and the limits of the evidence. No new training or native
game is authorized for automatic execution during this preparation.

The preceding explanation of apparently sudden victories was incomplete. An
audit of the individual private reports found that the earlier ten-game trials
already contained three fully verified wins with analytics enabled: trial 1,
game 1 was a Blue victory on day 14 (316 choices); trial 2, game 2 was a Red
victory on day 9 (210 choices); trial 3, game 10 was a Blue victory on day 16
(383 choices). Their policy reconstructions had zero differences and their
native replays passed. The corresponding overall batches failed the requirement
to finish verification of all ten games within their time budgets. Across four
trials, 28 of 40 games were verified: three victories and 25 cutoffs. The other
12 captures did not complete verification and are not accepted outcomes.

The two later first-win searches stopped on Blue victories on days 21 and 5,
respectively, then interrupted nine other games each. They used the same frozen
checkpoint 1,753, original map, native runtime and policy-support version as the
prior hero-relative trials. Their 16-fold horizons cannot explain these early
victories: the horizons only control cutoff boundaries, not early action masks
or network inputs. Analytics was downstream of gameplay and did not influence
decisions. First-win searches do not estimate a win rate because the unfinished
games are censored. Evidence remains private under `selfplay-ten/trial-20261104-*`
and `selfplay-one-game-2x/first-win-1473804f9d17` and
`selfplay-one-game-2x/first-win-8997e617ba83` in the durable artifacts root.

At preparation start, the manually launched frozen run
`selfplay-one-game-2x/win-count-1acc104c29f3` was active in the main checkout at
`06346c3164cf6820cec181e3126b20014d21ca40`. To preserve its source identity and
processes, changes are being prepared in a separate checkout on
`codex/selfplay-batch-learning`. The active run, its seeds and its artifacts are
not reused as training inputs. Main-checkout synchronization must wait until
its owned processes have stopped and its final evidence is sealed.

The new experiment is explicitly separate from the historical four-game update
contract. The proposed hypothesis is that ten-game PPO batches reduce unscored
cutoffs and improve performance against frozen checkpoint 1,753. Decisive
self-play frequency is a completion measure, not by itself a strength measure.
The proposed protocol starts from the complete verified 1,753 model, Adam, RNG,
progress, sampler and opponent-pool state; keeps the large architecture, PPO and
Adam hyperparameters, original map, automatic combat, terminal-only rewards,
hero-relative coordinates, existing cooldowns, revealed movement support,
own-army observations, equipment prohibition and troop-slot restrictions; and
retains the explicitly requested 5,840-day and 158,720 ordinary plus 5,120
reserved choices per-player limits. Ten physical games, alternating learner
seats, replace four per update; only the scheduled learner trajectory is trained
from each game. Both policies are frozen during collection.

Training requires ten fresh, fully verified receipts before one transactional
PPO update. A failed or incomplete cohort cannot train, but its verified game
outcomes must remain visible. The optimizer-step safety ceiling must be derived
from the declared batch/choice/window configuration (51,200 at the maximum),
rather than retaining the obsolete 640-step ceiling; this is a safety bound,
not a change to PPO epochs or minibatch settings. Checkpoint publication follows
an exact model/Adam/RNG/progress/sampler/pool roundtrip. A separate diagnostic
evaluation against frozen 1,753 uses fresh disjoint seeds, balanced seats and no
training admission, before training and after saved updates. Its failures must
not erase an already verified training checkpoint. Ten-game results are noisy
pilot evidence, not sufficient evidence for a paper's claim of improvement.

Implementation and validation are in progress. No completed training result,
speedup or learning improvement is claimed by this entry. An initial patch to
the journal rule used the wrong context and made no change; the corrected patch
added the rule successfully. Existing tests and additional local checks will
remain outside the published change unless separately authorized.

The user then selected only ten training games per batch and deferred the
additional frozen-1,753 evaluation games. The final implementation scope has no
extra baseline or evaluation games. Retain the fixed-opponent, balanced-seat,
held-out evaluation protocol above as a future option; do not activate it
automatically. Batch outcomes can measure decisive-game frequency and cutoff
frequency, while an improvement in strategic strength remains unproven without
a suitable independent comparison.

A second correction concerns an earlier conversational explanation of learning
without wins: this project's PPO configuration has `entropy_coefficient=0.0`.
There is no entropy-loss exploration term. Value estimation, advantages and
cutoff bootstrapping can still produce updates without a positive terminal
reward, but those updates do not supply an example of a successful victory.

Initial local validation passed 21 existing admission/checkpoint tests and six
new inline synthetic controls, including a real PPO update on temporary synthetic
test trajectories, second-admission rejection, incomplete/duplicate/reordered/
forged receipt rejection, diagnostic exclusion, new-plan drift rejection, bound
arithmetic and exact new-metadata checkpoint restoration. These controls did not
admit any captured game or modify an official checkpoint. Reporting validation
passed eight local mocked controls and four existing launcher/UI tests, covering
mixed wins/failures, unverified claims, cutoff/verification overlap, post-horizon
terminals, interruption and cleanup failure. No test or probe files were added
to the published source. An independent read-only review found no admission
defect in these two learner files; controller integration is still pending.

The unchanged canonical engine recipe was checked read-only in the main checkout:
`python3 tools/prepare_vcmi.py --check` passed. No runtime rebuild is needed for
the current Python reporting/admission changes. Runtime artifact identity and
historical native acceptance still require the actual launcher preflight.

Independent controller review found two defects before acceptance: the supervisor
mixed interruption errors with owned-process cleanup failures, incorrectly
blocking continuation after a clean stop; and an interruption after atomic
checkpoint publication but before report persistence could leave stale admission
counts. These must be corrected by separating operational status from cleanup
evidence and reconciling reports only from the committed publication chain, never
from orphan checkpoint candidates.

A broader existing offline selection ran 60 tests (self-play behavior, collection,
one-game launcher, recruit-only launcher and progress display), with six failure
events and five errors. Running the identical selection read-only at the unchanged
main commit `06346c3` reproduced the same failures. These older fixtures omit
equipment-policy/source-transition fields or mock superseded runtime interfaces;
this is not an all-green test-suite claim. Test files and production integrity
checks were left intact. Targeted new-contract controls remain necessary in
addition to the working existing tests.

The controller implementation now has explicit check/run/continue commands in
the existing launcher, a separate seed namespace and checkpoint lineage, ten
simultaneous collectors, and no optional behavior analytics or held-out games.
All ten verified learner trajectories precede one unchanged PPO operation.
Live completed-batch lines and the final table show Red/Blue wins, captured
cutoffs, verification failures and checkpoint advancement. Captured victories
without completed verification remain separate unverified claims.

The recovery findings were corrected and independently reviewed. Finalization
reconciles only publications reachable from the atomic latest pointer; orphan
publications and rollback/missing-pointer cases block continuation for explicit
recovery. Operational interruptions remain distinct from actual cleanup failure,
and final reports use the supervisor's authoritative cleanup/protection evidence.
The existing 30 GiB operational disk reserve is retained, checked before launch
and every two seconds by the supervisor. No automatic restart is added.

All 37 existing controller tests passed again on the final controller. Additional
temporary controls passed for fresh consumed seeds, missing ledgers, mixed game
and verification outcomes, publication/report crash recovery, orphan exclusion,
idempotent finalization, clean interruption, cleanup failure and low-disk stopping.
Independent reviewers found no remaining defect in the reviewed admission,
checkpoint and finalization paths. Exact source inventory validation passed for
215 files, including three negative source-tamper controls. The native binaries,
engine sources, encoding and PPO kernel are unchanged; the exact reviewed Python
transition retains prior native acceptance and does not claim a repeated native
capture. An initial full source-identity call correctly refused the still-dirty
checkout; full manual preflight will be run from the committed candidate.

The final scope is manual training only, with no baseline/evaluation games and
no automatic publication of raw evidence. Checkpoint 1,753 and the active frozen
run remain untouched. Preparing this candidate did not launch any native game
or official optimizer update. Tests, fixtures and diagnostic scripts are not part
of the committed change.

The real `./continue-selfplay.sh --batch-ten --check --batches 1 --hours 10`
preflight passed at candidate `2fd20c3` in 8.99 seconds. It restored checkpoint
1,753 with exact model/Adam/RNG/progress/sampler/pool state, verified the existing
runtime and map bindings, and reported zero gameplay and optimizer updates.
The new training artifacts directory was still absent afterward: no seeds were
reserved. Review of its output found unused historical seed lists and an old
wall-time field inherited into the displayed configuration. They were removed
from the ten-game configuration, which now declares its actual budget and
read-only next-seed preview. This correction changes no sampling or PPO behavior.

At the final integration check the main checkout remained clean at `06346c3`,
and its manually launched fixed-count frozen run was still active. The candidate
therefore remains isolated on `codex/selfplay-batch-learning`; merging it into
main while that run owns the source would invalidate the run. After the current
run finishes, manual integration is `git merge --ff-only codex/selfplay-batch-learning`
from `/Users/orsi/Projects/Heroes`, followed by the read-only preflight above and
the corresponding `--run` command. Do not merge while the run is active.

The corrected code at `4c87a3f` passed the same real preflight in 5.55 seconds:
checkpoint 1,753, exact complete-state restoration, ten physical training games,
zero held-out games, one expected update, a 36,000-second budget, and fresh engine
seeds starting at 1,970,000,000 with 20 policy seeds starting at 1,972,000,000.
The preview explicitly reports `reserved=false`; the training artifacts directory
remains absent. The private transcript is under
`Heroes-artifacts/diagnostics/batch-ten-preparation-20261104/4c87a3f0224c84fd20e70a754955181963de0492-preflight.log`.
A read-only check of an existing verified native manifest confirmed that its
army-information/equipment fields match the new admission checks. This is reuse
of existing native evidence, not a newly executed training batch. Full native
execution of the new ten-game training mode remains for the user's manual launch.


## 2026-11-05 — Authorized capture cleanup and checkpoint 1,753 continuation

The user explicitly authorized deleting bulky recordings and failed-run data
while preserving everything required to resume checkpoint 1,753. The stopped
first ten-game training attempt had zero admitted games and zero updates. Its
consumed reservation remains intact for all 100 planned games, including the
ten attempted games and 90 unstarted games. The
latest run is `selfplay-batch10-training/run-20261104T220221Z-eaf201f5`.

Under the existing exclusive activity lease, cleanup removed 492 raw JSONL
capture files from that stopped training attempt, the interrupted fixed-count
`selfplay-one-game-2x/win-count-1acc104c29f3` experiment and the failed
`selfplay-ten` trials. Removed file types were native contextual replay, bridge,
paired history and ordinary replay streams. Compact manifests, reports, seed
ledgers, plans, checkpoint ancestry, process-deadline evidence, model/resume
pairs, runtimes and Python dependencies were retained. Historical captures in
these groups can no longer be replayed from their deleted raw streams. Earlier
verification results remain historical results, not newly reproducible claims.

The 805 protected input bindings were checked unchanged after deletion. An
independent copy of the selected policy/resume pair and its descriptor was
preserved under private `diagnostics/cleanup-20261105/checkpoint1753`. Full
post-cleanup restoration passed with progress updates=1753, rollouts=7012,
decisions=1268962 and the original weight digest. Model, Adam, global/minibatch
RNG, progress, sampler and pool restoration checks remain enforced.

Deleted logical file size was 341,995,816,230 bytes. Observed filesystem free
space increased from 37,863,718,912 to 379,875,463,168 bytes, recovering
342,011,744,256 bytes (318.52 GiB). Filesystem allocation and unrelated background
activity can make this differ slightly from summed logical sizes. The exact
inventory, before/after counters and restore result are private under
`Heroes-artifacts/diagnostics/cleanup-20261105/`. The manual CLI preflight
transcript is retained there as `resume-preflight.log`. The next launch uses
`--resume --run-dir` for the existing stopped run, retaining its consumed seeds.
Preparation performs no native game or official optimizer update.

The actual CLI resume preflight passed after cleanup and journal commit. Its
fresh-seed preview starts at engine seed 1,970,000,100 and policy seed
1,972,000,200. A local post-check assertion initially expected only ten consumed
games and failed. Inspecting the immutable launch and durable ledger confirmed
that the user had reserved ten batches (100 games) before that attempt. All 100
remain consumed under the existing reserve-including-unstarted rule. The
corrected check against the actual ledger passed. No ledger was altered.


## 2026-11-05 — Ten capture-only training games and one validated self-play game

The user rejected full validation and analytics on every training game and
requested a repeating ten-training/one-validation cycle. They explicitly retain
manual launches. They selected both diagnostic seats using updated weights,
instead of a frozen-1753 opponent. The new experiment begins at the full preserved
checkpoint 1,753 and uses separate consumed seed namespaces. The hypothesis is
that removing repeated native replay and old-probability reconstruction reduces
training overhead while a retained diagnostic game permits qualitative review.
A single self-play game cannot establish strategic improvement.

The new contract explicitly discloses unverified training experience. Training
outcomes come from the native engine. Ten fresh completed learner captures are
required for one PPO update. No diagnostic, historical, partial, forged or reused
experience is admitted. Basic collection, source, seed, terminal reward and
checkpoint integrity constraints remain enforced. Native replay and frozen
policy reconstruction run on the one post-update diagnostic only. Existing
verified-training admission remains separate. PPO/Adam settings, architecture,
map, opponent policy support, temporary restrictions and gameplay horizons remain
as declared previously.

The previous memory warning exposed unbounded concurrent full-history retention.
The stopped run's report records KeyboardInterrupt, zero updates and clean owned
process cleanup, rather than a captured Python MemoryError or measured memory
peak. Current work therefore makes no claim about that exact peak. The new cycle
collects one game at a time, compiles its learner history into native programs,
retains compact loss records for all ten games and releases full Python history
before the next capture. Temporary raw training streams are then removed. An
owned-process memory guard leaves 48 GiB of physical RAM as headroom, alongside
the existing 30 GiB disk reserve. This scheduling decision prioritizes memory
safety and is independent of the PPO batch size of ten.

The saved model/Adam/RNG/progress/sampler/pool pair precedes the diagnostic. A
failed diagnostic preserves that saved update and stops the loop. Continuation
must finish an outstanding diagnostic with fresh seeds before more training.
All reserved training and diagnostic slots remain consumed after interruption.
Acceptance requires offline protocol/admission/recovery controls, exact native
PPO parity on disposable synthetic trajectories, complete checkpoint restore
and the actual manual CLI preflight. No native game or official update is
launched by the assistant during preparation.

Initial validation passed 58 existing self-play admission, checkpoint and
controller tests. A local actual-native control used ten synthetic trajectories
and five Adam steps: model, Adam and minibatch RNG matched the original kernel
exactly, while the new path did not call old-probability reconstruction. Seven
live-admission controls passed for incomplete, reordered, duplicate, forged,
diagnostic and repeated admission cases plus the complete scheduled batch.
Explicit metadata drift and disjoint eleven-slot seed reservation controls also
passed. No synthetic trajectory entered an official checkpoint. An initial
collector mock omitted its hard-deadline fields and failed before exercising the
new behavior; that fixture needs correction. A first tool orchestration block
had a JavaScript syntax error and made no filesystem changes before retry.


Final local protocol controls passed two ten-plus-one cycles with synthetic
optimizer changes and real temporary checkpoint save/restore. The simulated
checkpoints numbered 1,754/1,755 belong only to disposable test fixtures. The
actual preserved training checkpoint remains 1,753. Injecting a diagnostic
failure preserved the published fixture update. Its actual publication chain
restored, marked the outstanding diagnostic, and a continuation completed that
diagnostic before a fresh training batch and its own diagnostic. Both diagnostic
policies used identical updated weight digests. Full report counts remained
separate for training and diagnostic phases.

The corrected collector fixture passed sequential capture/compile interleaving,
zero replay calls during training, explicit unverified outcomes and removal of
temporary streams while retaining compact manifests. An initial coordinator
fixture created its output before reserving seeds and was correctly rejected by
the missing-ledger safety check; reserving first fixed the fixture. Seven live
admission controls and the strict metadata controls remained successful.

An actual prepared-native PPO update on synthetic trajectories was interrupted
after partial optimizer work. The transaction restored model, Adam, minibatch
RNG, global Torch RNG and Python RNG exactly. Another 34 existing native/PPO/UI
tests passed using the installed native backend, for 92 existing tests overall.
Memory controls passed owned-descendant accounting, unrelated-app exclusion,
the 80 GiB resident threshold on a 128 GiB model and the additional system-wide
20% free-memory floor, which also covers pressure from compression and other
applications. The real macOS pressure query format was checked read-only.

The canonical VCMI recipe check passed. Engine, model encoding, native learner
sources and binary ABI are unchanged. The installed native binary itself was
loaded for numerical parity and rollback controls. Exact source-inventory and
reviewed-transition checks cover all 215 implementation files, with six negative
tamper controls. No test, fixture or diagnostic script is staged for publication.
Real long-game peak memory, end-to-end cycle completion time and gameplay
behavior remain unmeasured until the user's manual run. They are not inferred
from the small synthetic fixtures.


The actual main-checkout `--cycle --check --cycles 1 --hours 10` preflight passed
at candidate `cf92730` in 2.66 seconds. It restored the complete 1,753 state,
reported ten training games and one diagnostic, and previewed fresh unreserved
seeds beginning at engine 1,980,000,000 and policy 1,982,000,000. The private log
is `Heroes-artifacts/diagnostics/ten-then-check-20261105/preflight-cf92730.log`.
The new lineage directory remains absent and no seeds were consumed.

Four real-parent offline plan controls then passed: training and diagnostic
plans for two cycles, actual frozen checkpoint loading, and prelaunch capture
binding. They confirmed ten/one schedule lengths, correct phase labels and
balanced learner seats even after eleven-slot offsets. An initially hand-built
control launch omitted policy_support and failed before plan validation. Adding
the field provided by the real prepare function corrected that fixture. These
controls used disposable snapshots, started no native process, changed no parent
weights and rehashed all protected inputs unchanged.

The cycle terminal output is intentionally limited to engine-reported Red/Blue
wins, cutoffs, PPO updates, weight changes, checkpoint and diagnostic result.
Full source and recovery evidence remains in the private JSON report. The check
command retains its explicit JSON configuration display.


## 2026-11-05 — Cycle handoff failed on an unregistered progress phase

The manually launched cycle `run-20261105T111051Z-4911fcaa` stopped after
1,092.05 seconds. Game 1 completed collection as an unscored game-day-limit
cutoff, with 45,454 Red choices and 24,247 Blue choices. Immediately afterward,
the coordinator called set_phase("preparing PPO"), which raised
ValueError("Invalid progress phase") before preparation or PPO admission.
The new collector also emitted "captured", another phase omitted from the
progress schema and label map. This was an implementation defect in the new
cycle UI integration. The earlier collector mock replaced set_phase, so it did
not exercise the actual validator. The earlier green checks did not cover this
normal runtime handoff and should not be read as evidence that it had worked.

The repair registers both phases and their display labels. It retains unknown
phase rejection and captured-state finalization. A local control now uses real
GameProgress/set_phase/snapshot reads, preserves the observed choice/day fields,
checks plain and terminal rendering, and verifies that finalization retains the
captured state. An AST control checks that every literal set_phase emitted by
the controller belongs to the phase schema and every phase has a label. All 19
existing progress tests passed. Source pins bind the exact display-only delta.

The actual run admitted zero games and performed zero PPO updates. Checkpoint
1,753 and protected inputs remain preserved; shutdown verified owned-process
cleanup. The failed capture and all eleven reserved slots remain consumed and
are retained. The next manual launch must use --cycle --resume with this run
path, which allocates fresh seeds. No game or official update is launched during
repair. Real progress controls stay local and outside the published diff.

The actual main-checkout resume preflight passed at `71f4b6a`, restoring the full
1,753 state and retaining all eleven consumed slots. The next fresh engine seed
is 1,980,000,011. Its private transcript is
`diagnostics/ten-then-check-20261105/progress-fix-preflight-71f4b6a.log` under the
artifacts root. No game or optimizer update occurred during this check.


## 2026-11-05 — Main-loop and trajectory-processing acceleration preparation

The user requested maximum practical game/training throughput and realistic
measurements, retaining detailed review for one diagnostic game. Manual launch
ownership remains with the user. Baseline is main commit 036d1be, preserved in a
detached diagnostic worktree. Checkpoint 1,753 restored before changes. No native
game or official PPO update is authorized for automatic launch during this work.

The preregistered comparison uses identical prefixes from the latest actual
captured game, frozen 1,753 weights, per-player policy seeds and action/support
traces. Offline actor timing excludes engine/transport and therefore cannot be
reported as end-to-end throughput. Candidate acceptance requires identical
choices, values, probabilities, supports and RNG progression, unchanged training
admission/rewards, existing regressions and the actual manual preflight. Bounded
microbenchmarks must report their workload and limits. The original game used
69,701 sampled choices, approximately 798 seconds through its last bridge event,
817.27 seconds through manifest finalization and 1,092.05 coordinator seconds.
It produced about 20 GiB of raw JSONL data and a 200 MB rollout envelope.

Initial code inspection found full-history tuple copies on each environment
request, ignored typed-frame handoff in the self-play collector, redundant final
frame decoding, and an independent full history audit still active during the
explicit capture-only training mode. The candidate will reuse the existing
checked typed-frame handoff used by the builtin collector, preserve mutable-wire
tamper fallback, remove avoidable whole-history copies, and omit the independent
file audit only under the declared capture-only training contract. Validated
diagnostic games retain their full validation path. Raw captures and all timing
artifacts remain private. No benchmark/test tooling is staged for publication.

### Implemented candidate and offline measurements

The main loop now avoids whole-history tuple copies per request, reuses typed
frames only after exact canonical-wire/type agreement, and retains the original
strict decode fallback for changed records. The observation memo's tile-key
fast path produces the identical token sequence as the generic traversal.
Capture-only training now omits duplicate bridge/frozen/live-history JSONL and
its independent post-game history audit; native recording and terminal barriers
remain enabled. A protocol-5 binary handoff replaces JSON re-decoding only for
the fresh owned training child. The coordinator binds its completion token,
process identity, seal, manifest and payload hashes before restricted loading.
It still checks actual policy/runtime/support identity, eligibility and complete
boundaries, then compiles the existing native PPO program. A file alone cannot
be admitted or retrospectively become a verified episode. Diagnostics keep the
original full JSONL/audit/reconstruction/replay path. The engine, installed native
learner, model architecture, PPO/Adam, rewards, action support and game budgets
are unchanged. Source transitions bind the exact reviewed Python delta.

Private evidence is under `diagnostics/loop-speed-20261105` in the artifacts
root. Benchmarks use checkpoint 1,753 and actual recorded inputs from the failed
capture above; none runs a native game or an optimizer update:

- `baseline-final.json` and `candidate-final.json`: identical 2,000 choices
  (1,000/player), three repetitions. Median offline actor time 16.3978 ->
  14.2762 seconds (1.149x throughput; 12.94% less time). Individual times overlap
  (baseline 15.58-17.54, candidate 13.62-16.10 seconds), so host noise matters.
  All six action/log-probability/value traces share SHA-256
  `cefdb3f85c5f242458dff4b0d02e9ff6a6092d11019cd48bcf2093bd6da2cad5`,
  with identical final RNG hashes for both players. Engine/transport/PPO excluded.
- `storage.json`: serialize/fsync/hash and reload 5,000 actual typed frames,
  three repetitions. Median combined JSON time 1.12719 seconds, binary 0.05350
  seconds (21.07x). 44,411,511 -> 4,286,750 bytes (10.36x smaller). Exact frame
  equality checked. This isolates frame storage, not the full episode handoff.
- `request-baseline.json` and `request-candidate.json`: 100 request-tail calls
  at a synthetic 300,000-frame history, three repetitions. Median 0.11987 ->
  0.00466 seconds (25.73x). This measures removal of history-size-dependent
  copying; repeated frames are a copy-cost fixture, not a valid played game.

No end-to-end candidate gameplay, full ten-game update timing, peak-memory
claim or one-minute-game result is established. The previous real game's
approximately 13.3-minute gameplay plus finalization cannot be scaled by the
storage benchmark's 21x. A next manual cycle is needed to measure the combined
result. Native contextual recording remains a known source of disk/engine cost;
this change does not claim to remove all recording.

### Validation and failed diagnostic attempts

The locked NumPy-enabled interpreter passed 246 existing tests covering
headless/contextual adapters, storage, collectors, self-play admission/controller,
checkpoint/progress and inference exactness; all nine native training tests
passed against the installed native learner manifest. `python3
tools/prepare_vcmi.py --check` passed exact canonical/generated agreement; no
C++ source changed and no rebuild was needed. Local controls passed: actual
owner-token binary transfer through real collection checks and installed native
program compilation on disposable synthetic data; payload tamper and executable
pickle-global rejection; general diagnostic-reader rejection of the owned
format; default ledger guards; lean finalization; and preservation of the
independent diagnostic audit. Exact generic/optimized memo keys matched 300
actual frames and deliberately type-mutated edge cases. These controls do not
constitute native gameplay validation or training admission.

Failed preparation attempts are retained as development evidence: an initial
trace serializer used a nonexistent `to_dict` on conditional records; a first
tile-key implementation accidentally extended the traversal stack rather than
the output and failed three existing exactness tests (fixed, with identical
baseline keys and final traces afterward); an initial lean constructor hit the
base replay/ledger guard (fixed with the explicit capture-only exception while
preserving default guards). Early local controls omitted fixture seed/dialog
metadata or attempted to construct an intentionally invalid tile directly;
they stopped before the tested handoff/finalization and were corrected. Some
commands lacked required project/test/headless import paths and were rerun with
the correct paths. The buggy intermediate `candidate2.json` is not used for a
performance claim. No failed attempt touched official model/Adam/RNG/progress,
reused consumed seeds or entered PPO. Benchmark/probe scripts remain private
and outside the committed diff.

The real main-checkout continuation preflight passed at commit `88e2461`:
`./continue-selfplay.sh --cycle --check --run-dir
/Users/orsi/Projects/Heroes-artifacts/selfplay-ten-then-check/run-20261105T111051Z-4911fcaa
--cycles 1 --hours 10`. It restored checkpoint 1,753 with exact complete
model/Adam/RNG/progress/sampler/pool agreement, performed zero gameplay and zero
optimizer updates, and retained all eleven previously consumed slots. The next
fresh engine seed is 1,980,000,011 and the policy base is 1,982,000,022; this check
did not reserve them. Private transcript: `diagnostics/loop-speed-20261105/
preflight-88e2461.log`. No owned training/game workers remained. The temporary
owned baseline worktree was removed after benchmarks; its source commit and
private results remain preserved. Only production source and documentation were
committed; no benchmark/probe/test tooling was staged or pushed. Launch remains
manual: ten training games, one update and one fully validated diagnostic game.

## 2026-11-05 — Read-only check of the user's accelerated manual continuation

The user requested progress and real-speed confirmation without interrupting
training. Inspected lineage, process inventory, launch/report/progress manifests,
process deadline records and unchanged parent policy/resume file hashes. No game,
update, replay, stop or restart was issued. The latest launch is
`run-20261105T122341Z-c2124692`, source `94c3404`, engine seed 1,980,000,011,
policy seeds 1,982,000,022/23, preserved checkpoint 1,753. It actually used the
owned-binary capture-only configuration and existing no-artifacts runtime.

The run had already stopped itself: its first game reached the 5,840-day limit,
with 46,239 Red and 59,031 Blue sampled choices, no winner. Native PPO recipe
preparation then failed with `RuntimeError: Native history exceeds bounded
recipe size`. This is a native learner preparation-size rejection, not a game
engine crash, timeout, replay-verification failure or checkpoint publication
failure. The rejecting guard in `native/learner/kernels.cpp:70` requires fewer
than ten million recipe nodes and fewer than one million chronological frames.
This read-only check did not reconstruct the enormous recipe to distinguish
which count exceeded its bound. The small synthetic parity/handoff controls
from implementation did not establish full long-game recipe capacity.

Observed time through manifest finalization: previous 69,701 choices / 817.27
seconds = 85.29 choices/second; latest 105,270 / 1,168.39 = 90.10 choices/second,
a 5.64% higher observed normalized rate. The latest game took 19m28s rather than
13m37s because it sampled about 51% more choices. Different seeds and game
states prevent a controlled causal speedup claim. Full latest launch elapsed
23m12s including failed preparation; zero games admitted, zero optimizer updates,
checkpoint still 1,753. Both parent policy and resume file sizes/SHA-256 match
the launch bindings. Report/shutdown confirm protected inputs unchanged and
owned cleanup complete; no training/game workers remained during inspection.

Private compact evidence: `diagnostics/loop-speed-20261105/
manual-run-20261105T122341-audit.json`. Raw failed-attempt evidence is preserved.
The loop remains blocked by long-game native recipe capacity; no repair or
limit change is included in this status-check task. Do not report this launch
as a successful learning cycle or the offline 21x storage result as whole-game
acceleration.


## 2026-11-05 — Restore the manual cycle's 365-day limit

The user explicitly requested returning to 365 game days and a manual launch
command after the long-game recipe-capacity failure. New ten-then-check training
and diagnostic plans and checkpoint metadata now agree on max_days=365. The
historical verified ten-game contract retains its separate 5,840-day declaration;
checkpoint 1,753 is unchanged. Architecture, PPO/Adam, terminal rewards, existing
choice limits, opponent, supports, ten-game update cadence and one validated
diagnostic per cycle remain unchanged. This limits game length; it is not a
repair or proof of unbounded native recipe capacity. Acceptance: existing
self-play/checkpoint/controller regressions and the actual continuation preflight
must pass with 365 in its configuration, full 1,753 restore and fresh seeds.
No training launch is authorized here. Raw failed attempts and consumed slots
remain preserved. The new manual command must reference the latest failed run,
run-20261105T122341Z-c2124692, rather than its predecessor.

All 77 existing self-play/checkpoint/controller/progress tests passed. Local
controls passed exact source-transition binding, new-cycle metadata max_days=365,
all other gameplay fields unchanged, and rejection of 5,840 as new-cycle metadata.
There are no owned active training workers. No tests/probes were added to Git.

The actual main-checkout continuation preflight passed at `6c9575e`, confirming
configuration max_days=365, exact full checkpoint 1,753 restore, zero gameplay
and zero optimizer updates. All 22 consumed slots remain consumed; next engine
seed is 1,980,000,022, next policy base 1,982,000,044, with no reservation by the
check. Transcript: `diagnostics/loop-speed-20261105/preflight-365.log` under the
private artifacts root. Manual launch remains with the user.

## 2026-11-05 — Fixed ten-game terminal dashboard preparation

The user requested a stationary 5-column by 2-row panel showing all ten game
slots, active status, days/actions and clear results on the next manual launch.
The current 365-day cycle is active in the main checkout (sixth capture observed
at investigation), so implementation is isolated in Heroes-ui-next on branch
codex/batch-grid-ui from 93e15e9. No active-run source or artifact is edited.
This is a display-only change; gameplay, scheduling, learning and evidence gates
must remain unchanged. Acceptance: existing progress/controller tests, local
terminal render/resize/failure controls and source identity checks; no native
game or official update will be launched for UI validation. Keep training
outcomes separate from verification/admission, and keep all validation scripts
and captures private. Integrate into main only after its owned workers exit.

Implemented the existing supervisor's optional BatchProgressDisplay with all ten
slots visible, a cyan/text-marked active slot, Red/Blue action counts, observed
day, action-budget bars and separate result/preparation states. A failure after
a reported engine win retains the win and an error label. The report footer
separates engine totals, PPO and validated evaluation. Completed training cards
are frozen while the diagnostic reuses episode slot 1; a later batch cannot
reuse stale prior-batch snapshots. Worker scheduling, callbacks, sidecar format,
models, PPO, native runtime and all gameplay limits remain unchanged.

TTY output uses an alternate screen and redraws in place, leaving the last
column/row unused to avoid implicit scrolling. Terminal dimensions are refreshed
on each poll; narrow/short windows use a clipped compact list. NO_COLOR removes
color but preserves geometry/state. Redirected/dumb-terminal output retains the
existing plain-text path. Cursor and original screen restore on completion or
failure. UI parsing/render exceptions are best-effort and cannot become a
training gate. Existing two-second report polling supplies status; no extra
history scans, analytics, profiler or per-action writes were introduced.

All 77 existing progress/controller/checkpoint/training tests passed. Private
controls under diagnostics/batch-grid-20261105 passed 80/140-column grids,
60/30-column and short-window bounds, active/result/error labels, evaluation
isolation, screen/cursor restoration, NO_COLOR and redirected output. Actual
supervisor integration passed with mocked subprocess success/failure, preserving
exit status and cleanup handling. Exact display source transition and mutation
rejection passed. An initial local multi-size fixture reused sidecars after
finish correctly marked the active slot failed; recreating its active phase
between independent cases corrected the fixture. No native game or optimizer
update ran for validation; no tests/probes are included in the staged diff.

The new renderer also consumed a read-only snapshot of the real active run's
existing report/progress into an in-memory terminal: all ten captured day-limit
games appeared in their correct slots with their real action counts, and the
footer showed PPO UPDATING. The preview called only screen-close, never progress
finish/cleanup, and did not write any active-run file. Private plain/ANSI captures
are actual-read-only-preview.* in the same diagnostic directory. This validates
display compatibility only; it does not claim a completed optimizer update.

Clean committed source identity passed at 52d5c1f in the isolated worktree.
At final handoff the main run was still active in its PPO-updating phase, with
all ten games captured and zero published updates so far. Its shutdown record
was absent. Therefore main integration and a real continuation preflight are
deferred until owned workers exit; neither active source nor active-run files
were modified. After current shutdown, fast-forward main from
codex/batch-grid-ui to enable this UI for the next manual launch. The same
existing launcher will select it automatically for an interactive terminal.


## 2026-11-05 — Completed 365-day cycle analysis and elapsed-time format

The user requested hh:mm:ss running time and analysis of the completed cycle.
Read-only inspection confirmed latest run run-20261105T133736Z-09cc564e completed,
return code zero, protected inputs unchanged and owned cleanup verified; no
active training/game processes remained. Main was clean, so the previously
prepared dashboard branch was reviewed and fast-forwarded into main at 084c43b.
The dashboard is now in the user's Heroes checkout. The only additional behavior
change requested here is a shared hh:mm:ss duration formatter for the panel and
final console runtime; machine-readable elapsed_seconds fields remain numeric.
Acceptance: existing UI/controller regressions, duration boundaries and real
continuation preflight; no training or gameplay will be launched.

The completed cycle admitted ten fresh learner trajectories (25,473 samples),
performed one PPO update with 799 Adam steps, and published checkpoint 1,754.
Both policy/resume file hashes match their published bindings; full save/restore
was recorded true. Weights changed from cb40725f... to 2cdac931.... All ten
training games reached the 365-day cutoff, engine Red/Blue wins 0/0, with zero
capture/preparation/verification failures. Training games remain explicitly
unverified as agreed; their published cutoff samples carry zero rewards and
own-decision value bootstrap. Thus weights can change without a +1/-1 training
terminal, and this batch supplied no observed winning training example.

Total coordinator elapsed time was 1,772.899 seconds, displayed as 00:29:32
(whole elapsed seconds are truncated, not rounded). Sum of training manifest
wall times was 769.516 seconds; range 56.085-139.921 seconds/game. There were
53,195 sampled choices across both seats; only the alternating learner seats
entered PPO. Existing file modification timestamps estimate collection plus
per-game preparation as 933.770 seconds (00:15:33), last capture receipt through
checkpoint publication as 817.515 seconds (00:13:37), and evaluation plan through
validated receipt as 19.246 seconds (00:00:19). These phase estimates include
serialization/restore overhead and are not instrumented optimizer-kernel timing;
report ppo.timings_seconds is null. Small startup/finalization gaps account for
the remaining total. PPO/checkpoint work now consumes about 46% of the cycle;
diagnostic validation is about 1%. Do not blame replay for the remaining runtime
or attribute shorter games solely to acceleration: the day limit changed.

The frozen post-update diagnostic used 1,754 on both seats, 334 choices
(Red 141 / Blue 193), engine/manifest wall 5.770 seconds. Blue won naturally on
day 13; replay verified the complete terminal barrier and server/client streams,
and both policy reconstructions had maximum absolute difference zero. Blue's
own observation changes from one town at (15,7,0) to two including (4,7,0) on
day 6; Red loses its only town at the same boundary. Red's day-13 notice explicitly
reports seven days without a town, followed by natural loss. Blue won neutral
battles on days 2 and 3, recruited 14 creatures on day 3 and 28 on day 11, and
revealed tiles increased 92->393; Red's increased 92->312. Red still spent 96/141
choices on opening/closing hero/capability windows; Blue also retained many
management/navigation choices. This is exploration/capture behavior in one
validated game, not evidence of reduced aimless actions across the ten games,
whose detailed analytics were intentionally omitted. Different seeds, a single
updated-policy self-play game and no matched pre-update evaluation prevent a
causal learning/strength conclusion. The validated win was never admitted to PPO.

Private compact analysis: diagnostics/completed-cycle-20261105/analysis.json.
The 56 existing progress/controller tests passed; local duration boundary controls
passed minute/hour rollover and durations beyond 24/100 hours. Numeric evidence
fields are unchanged. No tests/probes or raw captures are staged for publication.

The real main-checkout continuation preflight passed at b070efb with the merged
dashboard and duration formatter. It restored complete checkpoint 1,754 with
exact model/Adam/RNG/progress/sampler/pool agreement, max_days=365, zero gameplay
and zero optimizer updates. All 33 consumed slots remain consumed; next fresh
engine seed is 1,980,000,033 and policy base 1,982,000,066, not reserved by this
check. The actual console ended `Total running time: 00:00:02`. Private transcript:
diagnostics/completed-cycle-20261105/preflight.log. No training was restarted.

## 2026-11-05 — Prepare forty-game 5 by 8 terminal view

The user requested forty training games first and a 5-column/8-row grid.
PPO cadence is explicitly pending clarification: preserve ten-game updates with
one diagnostic after all forty, or collect forty before one update/diagnostic.
Do not change training semantics before that answer. Current main is running
run-20261105T142055Z-4ffa5e98, so changes are isolated on codex/forty-game-grid in
Heroes-next-40 from c40ff89. No active source or active-run artifacts may change.
Independent UI work can support forty slots now. Acceptance: all forty visible
at a normal 80x24 terminal using compact cards, exact active counts separately,
full cards in tall windows, no scrolling, unchanged hh:mm:ss, and existing
regressions plus private layout controls. No training is launched automatically.


The user explicitly selected forty captures before one PPO update, followed by
one diagnostic game. The capture-only --cycle contract now requires forty fresh
captures per update; legacy --batch-ten remains ten. Existing complete ten-game
cycle checkpoints retain their exact metadata and restore support. Publication
ancestry audits now sum each bound plan's declared 10/40 receipts and matching
checkpoint rollout delta instead of multiplying all update counts by ten.
The existing seed namespace remains continuous: historical eleven-slot blocks
remain consumed and new blocks explicitly declare width 41. Publication,
recovery/report counts and sampler consumed-game offsets use actual admitted
games. No partial batch can train and no consumed capture may update twice.
The opponent stays frozen at batch start for all forty games; both diagnostic
seats use the saved updated policy. Architecture, PPO/Adam hyperparameters,
365-day cap, action support, rewards and memory/disk guards stay unchanged.
Collecting forty before one update is the user's explicit batch-size change.

Local controls passed forty-only fresh-receipt gating (39 rejected), order
checking, exactly one mocked update with forty prepared native inputs, repeated
update rejection, restoring original ten-game metadata and new forty-game
metadata, and rejecting unknown batch sizes. A private copy of the actual seed
ledger preserved every historical reservation and accepted a fresh 41-slot block
without official writes. Forty-slot 5x8 rendering passed at 80x24 and 140x65;
compact counters are rounded down to whole thousands, exact active counts stay
visible. All 77 existing progress/training/checkpoint/controller tests passed.
No native forty-game batch or official optimizer update has been run; forty-game
peak memory/runtime remains unmeasured and guarded. Validation/probe code stays
private under diagnostics/forty-grid-20261105, outside the published diff.

An isolated snapshot containing only the closed historical lineage through
checkpoint 1,754 passed the actual ten_inspect restore/audit under the new code.
All actual policy/Adam/RNG/progress/sampler/pool components restored. Private
prepare_ten and ten_batch_plan then produced strict forty-training plus one
same-policy diagnostic plans with 41 distinct engine and 82 distinct policy
seeds; incomplete 39-game plan rejection passed. Only the temporary lineage
copy reserved seeds or wrote copied policy files. Active official lineage was
never modified, and this is not a preflight of its still-running latest launch.
Script: diagnostics/forty-grid-20261105/closed-lineage-preflight.py (private).

A related UI audit found that two future forty-game cycles must display the
current forty, rather than leaving the second batch off-screen. This was fixed
and local controls now cover 1x40, 2x40 and historical 4x10 numbering at both
terminal sizes. No historical training contract is changed by the UI case.
Main integration remains deferred until the active run's shutdown is complete.
The handoff command must check the existing run first, fast-forward the prepared
branch only after that check passes, then manually launch the forty-game cycle.

All nine installed-native learner numerical regressions passed. The clean-source
closed-lineage preflight passed again at 4706c63 after the grid visibility fix;
its private transcript is closed-lineage-preflight.log in the diagnostic folder.
At final handoff main still had the newer ten-game run actively collecting,
with checkpoint 1,754 and no shutdown record. Main stayed clean and unchanged.
The prepared branch is not yet integrated into the active checkout. The manual
command first uses main's existing --check to require that run's completed owned
shutdown; only a successful check permits the fast-forward and next --resume.
The next launch restores whichever complete checkpoint that run publishes,
without resetting to 1,754, and allocates fresh forty-plus-one seeds. No claim of
a full forty-game execution, optimizer parity across batch sizes or peak-memory
measurement is made; only the user's chosen batch size changes the learning
collection/update cadence.

## 2026-11-05 — PPO progress and estimated remaining time

The user requested an ETA and progress bar during PPO. Main is still running
the prior ten-game cycle (checkpoint 1,755 already published, shutdown absent),
so this work extends the isolated codex/forty-game-grid branch in Heroes-next-40.
No active source or run artifact is changed. Acceptance: report completed/total
actual optimizer minibatches using the existing after-step hook, estimate only
from this update's observed pace after warm-up, distinguish optimizer completion
from checkpoint save/publication, preserve numerical results/Adam/RNG and rollback,
and make every display error fail open. No training campaign or native game is
authorized for validation. The existing forty-game cadence remains unchanged.


Implemented an optional observer on transactional_update's existing completed-step
hook; the numerical PPO kernel, native ABI and minibatch shuffle are unchanged.
The exact planned step count uses the same per-rollout windows/epoch arithmetic.
The existing forty-game runner supplies a PPOProgress sidecar callback; default
callers do not change. The sidecar is disposable, bounded, and atomically replaced
at most once per second except phase changes. ETA uses elapsed/completed pace
only after three completed minibatches and three seconds. It is approximate,
with no historical promise or training-data scan. The 40-slot panel reads the
sidecar independently of evidence. A 100% optimizer bar is explicitly separate
from pending/saving/restore-checking/published checkpoint stages; rollback and
failure have their own labels. Observer exceptions cannot trigger rollback or
replace the actual failure. No learning setting or admission gate changes.

All 84 existing progress/builtin-gameplay/self-play-controller/training tests
passed. Private controls exercised actual disposable synthetic PPO updates with
no observer, a recording observer, an observer throwing OSError, and the actual
file-writing PPOProgress observer. Model weights, gradients, Adam state,
Torch/Python/minibatch RNG and numerical PPO reports matched exactly, including
when the installed native backend was selected with 16 prefix threads. A genuine
two-step attempt then deadline failure restored the full pre-update state and
reported rolled_back. ETA warm-up, one-second write throttling, malformed or
oversized snapshots, disk failure, saving labels and forty-slot rendering passed.
Private scripts/logs are under diagnostics/ppo-progress-20261105. An initial
synthetic control inherited PPOConfig's default two epochs while declaring a
six-step cap; the existing cap rejected it before any optimizer step. The fixture
was corrected to the intended explicit one epoch. No official update/game ran.

During this work the prior manual run completed naturally and published 1,755;
its shutdown reports zero errors, protected inputs unchanged and owned cleanup
complete, with no owned training/game processes remaining. Main can now be
fast-forwarded safely under the existing activity lease, including the prepared
forty-game panel and PPO ETA, then checked against the real latest state.

Main was fast-forwarded from c40ff89 to b4147b2 while holding the existing
activity lease, after the latest shutdown and clean checkout were verified.
The real main-checkout continuation preflight then passed: complete checkpoint
1,755 restored exactly (model/Adam/RNG/progress/sampler/pool), forty training
games, one held-out diagnostic, one expected PPO update, max_days=365, and zero
gameplay/optimizer updates performed by the check. All 44 consumed seed slots
remain consumed; next fresh block is 41 engine seeds from 1,980,000,044 and 82
policy seeds from 1,982,000,088, not reserved by preflight. Transcript:
diagnostics/ppo-progress-20261105/main-preflight.log. The final console runtime
was 00:00:03. The panel and PPO ETA are now in ~/Projects/Heroes for the next
manual launch. No official training was started or interrupted.


## 2026-11-05 — Completed forty-game cycle through checkpoint 1,756

The user requested analysis after completion. Fixed run identity is
run-20261105T153955Z-e97254e5, launched from b584fcf with the no-artifacts runtime
and forty-game/365-day capture-only contract. Report and shutdown agree on
completed/returncode 0, protected inputs unchanged and owned cleanup complete;
no active game/training workers were observed. No game, replay or official
optimizer update was launched during this analysis.

The forty training games produced engine-reported Red wins 2, Blue wins 11 and
27 day cutoffs: 13/40 decisive (32.5%). They remain explicitly unverified as
agreed; zero capture/preparation failures occurred. All forty were sampled using
frozen checkpoint 1,755 before the single update, so their within-batch outcomes
cannot show improvement from that update. Alternating learner seats supplied
five winning, eight losing and 27 cutoff trajectories, 71,758 admitted decisions
from 132,743 total sampled choices. PPO performed 2,251 Adam minibatches, changed
weights, and published complete checkpoint 1,756. Recorded approximate KL is
0.00316454, clip fraction 0.0275091 and value loss 0.0845457; these numerical
metrics alone do not establish playing strength.

Hash-bound binary captures of all thirteen decisive training games were read
with the restricted record reader for analysis only. They never entered any
fresh-capture/admission registry or optimizer. Nine losing-player histories
explicitly include the seven-days-without-a-town terminal notice; the other four
lose their last town shortly before their terminal without that notice. This is
capture analysis, not retrospective native verification or proof of tactical
skill. A Blue/Red seat imbalance exists (11 vs 2) despite a shared frozen policy;
map/start-state/seed/behavior effects are not separated by this sample.

The separate fully validated 1,756-vs-1,756 diagnostic reached the 365-day limit
without a winner, with 2,096 Red and 9,521 Blue choices. Its two exact policy
reconstructions had maximum absolute difference zero, and full native replay
passed with no divergence. Both towns remained with their starting owners.
Red revealed 363/400 tiles, Blue 400/400. Red's single hero died to neutrals on
day 63; Red hired no replacement and kept recruiting into its town. Blue hired
22 heroes (22 successful hired results), lost 20 heroes/battles and won two
battles. Matching same-day visit evidence classifies its twenty battle losses
as thirteen neutral encounters, six attacks on Red's town and one attack on
Red's hero. It reached seven simultaneous heroes. These observations describe
the one diagnostic, not all forty training games.

Window interactions still dominate: Red used 872/2,096 choices (41.60%) and Blue
7,030/9,521 (73.84%) on open_hero/open_capability/close_capability/close alone.
Blue's most common four-command pattern was open_capability, close_capability,
open_capability, close_capability (885 overlapping occurrences). One-command
cooldown does not eliminate alternating multi-command loops. Productive actions
also occurred: Red recruited 32 times and constructed twice; Blue recruited 19
times and constructed three times. Red/Blue movement no_route_known counts were
5/149 and 66/1,761 respectively (71/1,910 combined, 3.72%). Reaching known tiles
and exposing the map therefore coexist with window loops and poor battle choice.
No strategic filter, reward, action limit or game rule was changed by analysis.

Coordinator elapsed time: 5,430.422 seconds, shown 01:30:30. File timestamps
estimate collection/preparation 00:40:54, PPO/checkpoint publication 00:38:08 and
the diagnostic including checks 00:11:23. Raw training manifest wall times sum
to 00:32:36; the 27 full day-cutoff games averaged about 65 seconds through
manifest finalization. The diagnostic itself took 258.251 seconds (00:04:18),
then 423.688 seconds (00:07:03) from manifest to validated receipt. The native
replay report measured 58.1335 seconds within that verification work. Phase
estimates include I/O/preparation and are not optimizer-kernel profiling.

Private compact evidence and bounded analysis script are under
 diagnostics/analysis-1756-20261105/ in the artifacts root. Player histories were
streamed; only decisive training captures were decoded, with a 180-second bound.
No raw captures, analysis scripts or new tests are staged for publication.
The actual continuation preflight passed, restoring full 1,756
model/Adam/RNG/progress/sampler/pool exactly; current policy/resume hashes also
match publication bindings. All 85 reserved slots remain consumed; next fresh
engine base is 1,980,000,085 and policy base 1,982,000,170 (41/82 seeds), not
reserved by the check. Transcript: preflight.log in that private folder.

The cycle successfully delivered real positive/negative terminal training
examples and a preserved update. A win rate increase from earlier 0/10 and 1/10
to 13/40 is descriptive across different seeds, policies and sample sizes; it is
not a controlled estimate of strength. The single post-update diagnostic did
not win. No frozen-opponent comparative evaluation or new campaign was started.

## 2026-11-05 — Fifty-game batch and rolling ten-process collection preparation

The user requested fifty games before one update, ten concurrent workers and a
script, with measured acceleration. Fifty slots require a 5x10 grid. Parent is
complete checkpoint 1,756 from run-20261105T153955Z-e97254e5; no owned training
process was active. Main stays unchanged while implementation/diagnostics use
codex/parallel-fifty in Heroes-parallel50 from 6ab6268. Mac reports 16 logical
CPUs and 128 GiB physical RAM. PPO/day/action/opponent/reward settings remain
unchanged except the explicitly requested fifty-game batch. No official training
will be launched; bounded frozen diagnostic games may measure collection speed.

Inspection found collection hard-coded sequential, with wave barriers in older
parallel mode. Per-game native preparation averaged about eleven seconds in the
last run, so merely raising wave size would leave slots idle. Plan: a lightweight
rolling dispatcher replenishes ten independent single-inference-thread processes
while main retains serial, deadline-bounded preparation; sort receipts by original
schedule index before the unchanged PPO shuffle/update. Maintain fresh owned
completion tokens, source/runtime hashes, guards, fail-closed cleanup and no
partial admission. Reports distinguish active captures from queued preparation.
Acceptance: rolling/cancellation/order controls, numerical regressions, actual
bounded serial/parallel frozen-policy comparisons (including semantic equality,
individual reconstruction/replay and cleanup), and real continuation preflight.
Never infer full-cycle 10x acceleration from collection-only measurements.


Implemented fifty-game admission/metadata while preserving 10/40-game checkpoint
restore and old 11/41-slot reservations; new cycles reserve 51 slots. The actual
capture configuration records ten processes, one math thread per actor and one
preparation worker; the inherited numerical native-execution profile is unchanged.
A lightweight dispatcher refills slots independently of main-thread, signal-bounded
preparation. Shared report writes are locked. Captured-but-not-prepared work is
separate from active games in the UI; deterministic receipt/attempt sorting
retains schedule order for PPO. Failure stops dispatch and joins it before owned
process cleanup; incomplete batches never update. Each actor receives its own
seed/model/engine directory and bounded math-pool environment. Resource, deadline,
completion-token and checkpoint gates remain enabled.

All 77 existing training/checkpoint/controller/progress tests passed. A real
collect_batch control with fifty lightweight disposable child processes observed
peak ten, refilled past a deliberately slow first job, overlapped preparation,
kept all model bindings before threading, returned sorted receipts and cleaned
up. Cancellation stopped further launch and joined dispatch. UI controls passed
all fifty slots and ten active jobs at 80x24/28 and 140x80. Local fifty-receipt
controls passed one-update/partial/order/reuse rejection, legacy metadata and
mixed seed-ledger checks. These controls launch no native game or PPO update.

Failed development controls: an initial unconditional receipt sort broke a
legacy mock receipt without a binding (fixed by sorting only the rolling live
path, retaining historical-wave behavior); a lightweight child fixture omitted
sealed artifact files and hit the real artifact binding check (fixture corrected,
cleanup ran); the copied forty-game configuration control still expected one
worker (updated to the requested ten). A symlink to the main generated VCMI was
rejected by prepare_vcmi's path-containment check; it was replaced by an owned
APFS clone of the dependency, and prepare_vcmi.py --check then passed exactly.
No engine source or binary changed. Validation scripts/results remain private
under diagnostics/parallel50-20261105, outside the staged diff.

The preregistered native benchmark uses twelve paired frozen-1,756 diagnostic
games per arm, one versus ten actors through the production rolling dispatcher,
512 ordinary plus 32 reserved choices/player, 365 days, bounded deadlines and
cleanup. Both arms use identical private engine/policy seeds, isolated profiles,
unchanged map/support/rewards, and zero training admission/updates. Parent native
program preparation is included and timed separately; every capture must later
pass two-policy reconstruction and native replay. Compare action/probability/value
and semantic pre-finalization history hashes; report any divergence and do not
claim matched-workload acceleration if it occurs. Short prefixes and diagnostic
full recording differ from full lean training episodes, so do not extrapolate
this result to end-to-end fifty-game/PPO throughput. Raw evidence stays private.

Native benchmark completed at source aee4fe6 using the accepted
local-v8-no-artifacts-v1-repair2 engine and existing remove-telemetry-v1-attempt1
native learner. Twelve paired bounded frozen-1,756 games per arm produced
12/12 identical decision/probability/value traces and 12/12 identical semantic
pre-finalization histories. All 24 captures passed exact two-policy reconstruction
and independent native replay. Observed peak actors: one and ten. Capture wall
time was 188.00s versus 28.77s (6.54x); capture plus serial native preparation
was 190.03s versus 65.79s (2.89x). Native preparation itself was 51.14s versus
51.20s and remains serial. Both arms overlap capture/preparation; verification
follows both timed arms. No optimizer update, official seed reservation or
training admission occurred, owned groups closed, and protected inputs matched.
Private paired seeds, plans, captures, timings and shutdown evidence are under
diagnostics/parallel50-20261105/native-benchmark-attempt4. This is a short
full-recording diagnostic comparison, not a measurement of fifty complete lean
training games or end-to-end PPO throughput.

Failed benchmark harness attempts remain private: the first compared two
different source-identity envelope shapes and rejected before native capture;
the second omitted policy_support and rejected before native capture. The third
captured one prefix but rejected native preparation because the private parent's
ATen thread count was not one; the next in-flight diagnostic was stopped and
cleanup/protected-input checks passed. The harness now configures the parent
through the existing configure_native function and uses bounded individual
stage limits rather than nesting the signal-based wall_limit context. No runtime
or learning setting was changed to accommodate these harness errors.

The final existing regression runs passed 77 training/checkpoint/controller/UI,
28 Python PPO numerical/acceleration and 16 actual compiled-native numerical
tests (121 total). Private dispatcher failure controls also rejected nonzero
actor exit, launch failure and deadline expiry, joined dispatch and closed owned
children without admission. Tests/probes and raw evidence remain outside Git.

A private continuation preflight restored the actual complete checkpoint 1,756
(model/Adam/RNG/progress/sampler/pool), copied the existing closed lineage/ledger,
reserved 51 slots only in that private copy and validated the actual fifty-game
training and one-game diagnostic plans. All fifty fresh source/model/seed
capabilities bound in 5.59s without changing parent Torch RNG. Next official
unconsumed offset remains 85 (engine 1980000085, policy 1982000170); official
pointers/reservations and protected inputs were unchanged. This checks planning,
not a fifty-game production completion or update. Canonical prepare_vcmi.py --check
passed again. Main-checkout synchronization and its read-only launcher preflight
remain the final handoff gates.

Main was fast-forwarded cleanly from 6ab6268 to 1e1edbe under the existing
activity lease, without unrelated changes. Its actual manual launcher passed
`./continue-selfplay.sh --cycle --check --run-dir PRIVATE_LATEST_RUN --cycles 1
--hours 10`: checkpoint 1,756, exact model/Adam/RNG/progress/sampler/pool,
fifty training plus one diagnostic, ten rolling actors, one planned PPO update,
and a non-reserving preview of 51 fresh slots at offset 85. Actual gameplay and
optimizer updates were zero; main prepare_vcmi.py --check passed. Evidence:
diagnostics/parallel50-20261105/main-continuation-check.log. Manual launch remains
the user's responsibility; full fifty-game memory/runtime completion is not yet observed.

## 2026-11-07 — Status reconstruction of the first fifty-game attempt

Question: where did the experiment stop after the fifty-game handoff? Acceptance
for this status review is agreement between the latest launch/report/shutdown,
checkpoint pointer and seed ledger; no game, replay or optimizer launch is part
of this review. The preceding handoff entry describes preparation, not the later
manual attempt discovered in private artifacts during this review.

Run `run-20261105T190107Z-9321e3b1` started on November 5 at 21:01 Kyiv time,
from source `989dcd03364a3f8b5db0a0e9412882d45e518b20` and parent checkpoint
1,756. Its hash-bound launch records the runtime/configuration identities:
fifty training games, ten rolling single-thread actors, one preparation worker,
365-day limit, capture-only training and one planned post-update diagnostic.
The reserved 51-slot block starts at offset 85, engine seed 1,980,000,085 and
policy seed 1,982,000,170. All slots, including unstarted ones, remain consumed.

The coordinator report records `stopped` after 119.482 seconds with
`ValueError: Register one live owned child and private completion token`.
The fifty attempt statuses are nine captured, ten interrupted, one failed and
thirty not started. Zero games/samples were admitted and zero optimizer updates
occurred; the latest published checkpoint remains 1,756. These attempt statuses
are distinct from engine outcomes and overlapping cutoff counters. No training
victory in this capture-only attempt is independently replay-verified.

Both shutdown reports record owned cleanup complete; campaign shutdown has
return code 1 and protected inputs unchanged. No matching active game/training
process was observed during the status review. Evidence is `launch.json`,
`report.json`, both `shutdown.json` files and `coordinator-deadline.jsonl` under
`/Users/orsi/Projects/Heroes-artifacts/selfplay-ten-then-check/` followed by the
run ID above, plus that namespace's `latest-checkpoint.json` and
`seed-reservations.json`. This review did not reproduce or establish the root
cause of the registration error. The next engineering step is to investigate
that failure before another manual fifty-game attempt; full-cycle throughput
and any learning improvement remain unestablished.

## 2026-11-07 — Registration-race repair and authorized fifty-game continuation

The user explicitly authorized investigation, repair, an agent-launched full
fifty-game cycle and progress reports every thirty minutes. This supersedes the
earlier user-only launch restriction for this attempt. Parent remains the latest
complete checkpoint 1,756. No learning, reward, gameplay, worker-count or
verification setting is to change. Work starts from `989dcd0` on
`codex/fix-capture-registration`; validation scripts and raw evidence remain
private under `diagnostics/capture-registration-20261107` in Heroes-artifacts.

Hypothesis: the actor can complete a short game while the launch-registration
binding recheck runs, before the subsequent live-process check. The failed
episode 20's worker ran from 19:02:47.950 to 19:02:53.754 UTC and retained a
natural-terminal manifest without worker stderr, consistent with this ordering.
The planned repair uses the existing subprocess stdin channel: the actor waits
for its private token, which the coordinator sends only after successful live
registration. Source rechecks, live-child requirements, completion tokens,
artifact hashes, deadlines and full-batch admission remain enabled.

Acceptance before launch: reproduce the old ordering with disposable real child
processes; prove that delayed registration cannot start gameplay, rejected or
abandoned registration cannot produce a completion, and fifty rolling actors
still produce ordered complete receipts with owned cleanup. Run relevant
existing regressions and canonical engine-recipe verification. Inspect staged
and complete target-branch diffs before the local source commit required by the
clean-source launch gate; publish no tests or auxiliary tooling.

Continuation acceptance: restore checkpoint 1,756 exactly, reserve fresh seeds
after the failed run's consumed 51 slots (next offset 136), collect all fifty
fresh games, perform exactly one PPO update, save and restore-check checkpoint
1,757, then complete exact two-policy reconstruction and native replay of one
fresh diagnostic. Retain the ten-hour deadline, 30 GiB disk reserve and existing
memory guards. Record outcomes, failed attempts, timings and actual validation
separately; do not infer learning improvement from one self-play cycle.

The ungated ordering reproduced the original error with a real successfully
exited child and a deliberately delayed binding recheck. With the production
stdin gate, the child remained alive without entering capture until registration
completed. Wrong-token, EOF, deadline and rejected-registration controls produced
no capture; owned groups closed. The actual rolling collector then completed
fifty disposable guarded children at peak ten, using real launch registration
and token/file-bound completion checks, returning fifty ordered receipts and
closing its process pipes/groups. These controls used synthetic artifacts and
did not launch VCMI or PPO; their scope is process ordering, not game validity.

The existing 87-test selection passed 85 and failed two controller fixtures
whose fake Popen objects lack stdin. An external local copy changed only that
fixture to provide BytesIO; all 37 controller tests then passed. The repository
tests remain unchanged and are not represented as an entirely green unmodified
suite. Logs and the adjusted fixture stay in the private diagnostics directory.
`python3 tools/prepare_vcmi.py --check` passed exact canonical agreement.
The production change is limited to worker release ordering and pipe closure;
the launch/completion admission code and numerical learner are unchanged.

The first read-only continuation preflight at `0f3a802` rejected the changed
controller through the existing exact runtime-source inventory, before any
reservation or gameplay. The repair also needs an explicit before/after
controller digest transition in the existing source validator. That transition
binds only the inspected registration/pipe change and retains rejection of any
other source drift; historical native captures are not claimed to have repeated.

The actual 212-file source inventory passed the added exact transition, and
twelve changed/missing controller, PPO and engine-source cases were rejected.
Seventeen of eighteen existing evaluator tests passed; the remaining historical
fixture omits the starting-equipment dependency. Running that same fixture
against the unchanged main-branch evaluator reproduced the identical failure.
The new transition accepts only the controller change and records
`native_capture_repeated=false`. A first local inventory control called the
clean-checkout entry point while this pin edit was uncommitted and correctly
failed; the corrected read-only control used the implementation inventory
directly. No seeds or gameplay were involved in these checks.

The clean `e4a593d` continuation preflight passed exact restoration of checkpoint
1,756 and previewed offset 136 without reserving. The agent launched
`./continue-selfplay.sh --cycle --resume --run-dir
/Users/orsi/Projects/Heroes-artifacts/selfplay-ten-then-check/run-20261105T190107Z-9321e3b1
--cycles 1 --hours 10 --output
/Users/orsi/Projects/Heroes-artifacts/selfplay-ten-then-check/run-20261107-registration-retry1`
at 13:35:01 UTC. It consumed 51 slots at offset 136 (engine base 1,980,000,136,
policy base 1,982,000,272), using the same engine/native learner and configuration.
All fifty actors registered and completed capture. Forty-nine preparations
succeeded; episode 31 failed native preparation with
`RuntimeError: Native history exceeds bounded recipe size`. The failed learner
seat had 15,075 choices; the other seat had 3,053. Total engine outcomes were
one Red win, fifteen Blue wins and thirty-four day cutoffs, all unverified.
No games were admitted and no PPO update occurred; checkpoint remains 1,756.
Shutdown return code was 1, with owned cleanup complete and protected inputs
unchanged. Total launcher time was 00:12:01. This establishes successful live
registration for the full batch, not successful training-cycle acceptance.

New hypothesis before another attempt: frame-local compilation duplicates exact
tree recipes across frames until the unchanged native ten-million-node bound is
reached. Investigate exact structural sharing after each frame's existing Python
equality/first-use handling. Preserve signed-zero choices, frame order, every
action, per-frame gradient graphs, native limits and the existing binary.
Acceptance requires numerical/gradient/update parity against the old compiler,
plus successful compilation of the actual rejected long capture without
admitting that historical capture to training. Only then repeat a fresh complete
fifty-game cycle from 1,756, with the failed 51-slot block left consumed.

Implemented cross-frame sharing in the existing recipe arena, preserving its
separate frame-local equality cache. Numeric keys retain original type/value
and signed zero so preprocessing collisions cannot merge distinct frame-local
nodes. The unchanged native binary still enforces its original size/depth bounds.
The actual rejected history contained 135,467 frames, 15,075 learner choices and
12,473,634 old frame-local nodes. It compiled successfully with 139,016 shared
nodes in 61.20 seconds, without dropping frames/actions or changing source
captures, model weights or RNG. Old/new native execution on an eight-decision
prefix of that game matched every output and gradient byte. Signed-zero,
int/float equality and rounded-number controls preserved within-frame graph
identity. No retained capture was registered for training or admitted to PPO.

All 43 distinct selected numerical/acceleration tests passed, including actual
native gradient/Adam/weight/RNG parity. The first invocation skipped seven native
tests because that module uses a different manifest environment variable; all
seven passed when explicitly selected with the correct variable. The exact
212-file inventory accepted only the recipe compiler delta; sixteen alteration
and omission cases were rejected. A first private inventory probe requested a
native-kernel path absent from this inventory and raised KeyError; its corrected
version used the recorded engine source, while the native build remains checked
by its separate unchanged manifest. Evidence: `compact-recipe-controls.json`,
`compact-recipe-controls.log`, `compact-recipes-tests.log`,
`compact-recipes-native-tests.log` and `recipe-source-transition.log` in the same
private diagnostics directory. Next fresh official offset is 187 (engine base
1,980,000,187, policy base 1,982,000,374); the next attempt must recollect all fifty.

## 2026-11-07 — Completed fifty-game continuation through checkpoint 1,757

The final source was `70373d3` on `codex/fix-capture-registration`. Its read-only
preflight passed exact restoration of checkpoint 1,756. The agent then invoked
the same cycle launcher with `--resume`, `--cycles 1 --hours 10`, latest run
`run-20261107-registration-retry1` and fresh output
`/Users/orsi/Projects/Heroes-artifacts/selfplay-ten-then-check/run-20261107-registration-retry2`.
Launch time was 13:51:43 UTC. The 51-slot reservation consumed offset 187 through
237, starting at engine 1,980,000,187 and policy 1,982,000,374. Source/runtime,
configuration and parent identities are bound in that run's `launch.json`.

All fifty fresh training captures prepared successfully, with zero failed or
interrupted games. Engine-reported outcomes were zero Red wins, seventeen Blue
wins and thirty-three day cutoffs; training verification was not performed as
specified by this experiment. All fifty learner trajectories were admitted,
containing 87,916 decisions. Exactly one PPO update performed 2,760 Adam
minibatches and changed weight digest from
`2ceb90214d71654c419f0d2c7ea84a5ace85d6a013ebd611c8ff37836bf0cfb5` to
`11aebbe1228f108b3027d54273f120e7d1e2d72becfae5bb82e78739356a19ee`.
Checkpoint 1,757 passed full save/restore verification and became the latest
complete publication. Reported approximate KL was 0.0022677753, clip fraction
0.0260589654 and value loss 0.0437530742. These metrics do not prove strength.

The fresh 1,757-versus-1,757 diagnostic used engine seed 1,980,000,237 and policy
seeds 1,982,000,474/475. It reached the 365-day cutoff without a winner, with
16,196 Red and 1,341 Blue sampled choices. Both exact policy reconstructions
reported maximum absolute difference zero and unchanged weights; independent
native replay passed. This diagnostic was excluded from PPO. It validates
recording/reconstruction/replay, not improvement against a fixed opponent.

The campaign report ended `completed`, with one optimizer update, fifty admitted
games, owned cleanup verified and protected inputs unchanged. Supervisor exit
code was zero with no cleanup errors. Coordinator elapsed time was 4,988.748
seconds; total launcher time was 01:23:14. The PPO progress sidecar recorded
3,558 seconds through save, not an isolated optimizer-kernel measurement.
The final read-only launcher preflight restored the full 1,757 model, Adam, RNG,
progress, sampler and opponent pool exactly, with zero games/updates and no seed
reservation. Next unconsumed offset is 238 (engine 1,980,000,238 and policy
1,982,000,476). No further cycle was launched.

Evidence: the run's `report.json`, `shutdown.json`,
`batch-train-000/publication.json`, `evaluation-000/execution/episode-000001/training-receipt.json`
and associated reconstruction/native replay reports; diagnostics retain
`cycle-retry2-terminal.log`, `preflight-retry2.log` and `post-run-preflight.log`.
All raw evidence, controls and adapted test fixtures remain private and outside
Git. The local branch contains the minimal production fixes, exact source pins
and English project documentation; no remote push or PR was performed.

## 2026-11-07 — Diagnostic 1,757 versus 1,756: analysis plan

Question: what behavior prevented a win in the completed 1,757 self-play
diagnostic, and which observed differences from the preceding 1,756 diagnostic
could support or contradict an improvement hypothesis? This is read-only
analysis of existing evidence, with no new game, replay, training, seed
reservation or policy/configuration change. The source at analysis start is
`3012b93` on `codex/fix-capture-registration`.

Acceptance criteria: stream both players' manifest-bound histories for both
games with identical metric definitions; distinguish sampled policy choices
from the cutoff continuation; verify input hashes against the completed run
receipts; reconstruct town ownership, hero roster changes, battle results and
preceding visits, successful development actions, exploration, movement results
and repeated window actions. Preserve compact evidence privately under
`Heroes-artifacts/diagnostics/analysis-1757-vs-1756-20261107/` and record the
observed behavior and limitations here. Exact reconstruction and native replay
results are read from the existing validated receipts, not rerun. The diagnostic
seeds differ and both seats change checkpoint, so one self-play game per
checkpoint cannot estimate a strength increase or isolate the PPO update.

## 2026-11-07 — Observed behavior and comparison of diagnostics 1,757 and 1,756

The bounded private streaming analysis completed for all four player histories.
Their byte counts and SHA-256 digests match the completed run receipts; submitted
learner-history actions exactly match the recorded policy sample counts. The
live cutoff continuation is excluded. The first scratch aggregation used a
generic battle-event name; this was corrected to paired `battle_start` and
`battle_complete` events and all four streams were reread before reporting.
Neither native replay nor gameplay was rerun. Both existing diagnostic receipts
confirm exact two-policy reconstruction (maximum absolute difference zero) and
successful native replay. Capture-manifest replay flags predate verification;
the later verification receipts own that result.

The compared games are `run-20261105T153955Z-e97254e5` (checkpoint 1,756; engine
seed 1,980,000,084; policy seeds 1,982,000,168/169) and
`run-20261107-registration-retry2` (checkpoint 1,757; engine seed 1,980,000,237;
policy seeds 1,982,000,474/475). All capture-manifest runtime/map/adapter identity
fields match except the project revision. Both use the same no-artifacts profile,
20-by-20 map and 365-day cutoff. The first retained own views have different
resources (1,756 Red/Blue wood and ore 5/12; 1,757 Red/Blue 15/5), so even initial
observed economic conditions are not matched. Both seats use their respective
checkpoint; this is not a fixed-opponent or paired-seed experiment.

In 1,757, Blue attacked Red's hero on day 6 and lost its only hero. It hired no
replacement during the remaining game. Its town-building decision on day 17
is especially informative: the hall showed the tavern as `allowed`, but the
policy built the archer dwelling instead, spending its last five wood and five
ore. The tavern subsequently appeared as `no_resources` on day 41. No further
building or hero-hiring action occurred, and wood/ore stayed zero. The production
Tavern adapter requires an owned town with a tavern before hiring. Thus the
observed path stranded Blue without a hero and without the wood needed to enable
hiring; it was not merely repeatedly declining an already available hire action.
Blue recruited into its town forty times. At the cutoff boundary the garrison
contained 710 pikemen and 459 archers, with zero heroes. It retained its original
town throughout and revealed 286/400 tiles, all by day 6.

Red won a neutral battle on day 2 and defended against Blue's hero on day 6. It
built a tavern on day 31, first hired on day 81, hired fourteen heroes overall,
and reached eight simultaneous heroes. Fourteen recruit actions and six
`swap_garrison` actions included three observed transfers of town troops into
hero armies: 140 pikemen on day 81, 154 on day 150 and 336 on day 322. The other
three swaps caused no net troop transfer. These troops were used in the field:
the first reinforced army won against neutrals on day 102; another reinforced
army won neutral fights on days 230 and 232. Red uncovered all 400 tiles by day
100. It won five neutral battles and its one defensive hero battle, but lost
fifteen battles/heroes: twelve neutral encounters and three attacks on Blue's
town, on days 123, 305 and 331. These classifications use same-day visits
immediately preceding battle starts, corroborated by the other player's town
defenses for the three sieges.

The attacking armies were poorly matched to the town defenders. On day 331,
Red attacked with sixteen pikemen and seven archers against Blue's recorded
town army of 640 pikemen and 369 archers. On day 327 its original hero attacked
neutrals with just three pikemen and died; on day 334 another hero did so with
one goblin. The last reinforced hero, carrying 352 pikemen and four archers,
died to neutrals on day 359. All fifteen heroes (initial plus fourteen hired)
were lost. Red finished with neither heroes nor garrison troops, but retained
its original town. Blue had no hero to take that empty town, and Red never took
Blue's defended town. Both original towns remained owned; the experiment ended
at the day limit, not a native terminal draw or a verification failure.

Window repetition remained prominent. Red devoted 12,372/16,196 sampled choices
(76.39%) to `open_hero`, `open_capability`, `close_capability` and `close` alone.
The four-action open/close-capability/open/close-capability pattern occurred
1,596 times, counted with overlap. Blue's post-loss behavior was mostly town
opening/closing, turn ending and intermittent recruitment. Across both seats,
the same four window-action kinds account for 12,783/17,537 choices (72.89%),
versus 7,902/11,617 (68.02%) in 1,756. These are descriptive action families, not
a claim that every window opening is useless. Movement `no_route_known` was
130/3,176 (4.09%), versus 71/1,910 (3.72%); movement exhaustion is counted
separately and not treated as a route error.

Identically defined aggregate observations (both seats, one game each):

| Metric | 1,756 | 1,757 |
| --- | ---: | ---: |
| Enemy towns captured | 0 | 0 |
| Neutral battles won | 3 | 5 |
| New heroes hired | 22 | 14 |
| Hero losses | 21 | 16 |
| Heroes remaining at cutoff | 3 | 0 |
| Observed nonzero town-to-hero troop transfers | 1 | 3 |
| Window-action fraction as defined above | 68.02% | 72.89% |
| Movement with no known route | 3.72% | 4.09% |

There are useful local behaviors in 1,757: fielding accumulated recruits and
winning more neutral fights in this particular game. They coexist with resource
planning failure, weak attack selection, repeated window sequences and eventual
loss of every field hero. The active/passive color roles largely reversed from
1,756 (whose Red lost its hero on day 63 and never replaced it). This evidence
does not establish stronger play or a reliable improvement from the update;
fewer total losses also cannot be called improved survival when one side stopped
fielding heroes on day 6. A strength comparison would need matched seeds and
swapped seats against a fixed opponent, with multiple games and uncertainty
reported. No such new experiment was launched.

Private evidence: `Heroes-artifacts/diagnostics/analysis-1757-vs-1756-20261107/`
contains `comparison.json`, the bounded `analyse.py`/`analysis.log`, compact
per-player action/event records and `blue-tavern-evidence.json`. Full histories,
armies and source identities remain in private storage. Only this English
research-journal entry is changed in the repository; no publication occurred.

## 2026-11-07 — Current restriction audit after diagnostic 1,757

The user requested the currently effective limits. Read-only inspection compared
README, the temporary restriction register and implementation against the last
completed cycle's launch/evaluation plan. The active support is
`visible-interaction-budget-hero-relative-cooldown-v1`, with
`own-army-details-v1` and `no-artifacts-v1`. Current interaction-guard,
coordinate-model and no-artifacts patch hashes match the executed plan. No rule,
weight, launch configuration or runtime was changed; no game was started.

The inherited restrictions still block manual stack splitting, merging, swapping,
transfer, stack inspection and stack upgrades; recruitment and whole-hero
`swap_garrison` remain available. All artifacts, including spellbooks and war
machines, remain disabled throughout the game. Movement targets must be present
in the delivered revealed tiles and the hero must have movement points. Repeated
no-progress targets, the current tile, immediate repeated commands and immediate
town reopenings are filtered as specified in the guard. The navigation budget
is two uses of the same semantic control until visible progress/day renewal;
free management is capped at 32 uses per involved subject per day. Mandatory
choices and exits are preserved. These filters do not eliminate alternating
window loops. Hero-relative coordinate encoding is a representation choice,
not an additional short movement-radius limit.

The cycle uses 365 game days, 158,720 ordinary and 5,120 reserved dialog choices
per player, a shared ten-hour wall deadline, ten capture actors, a 30 GiB disk
reserve, a 48 GiB RAM reserve and the existing 20% free-memory pressure stop.
Fifty training captures precede one PPO update and one separately validated
frozen diagnostic; training captures omit full replay/reconstruction as already
declared. Rewards remain terminal-only, with genuine cutoff bootstrapping rather
than treating the day limit as a natural draw/loss.

Interpretation clarification for the previous analysis: exact opposing army
counts in the siege comparisons were obtained offline by matching both players'
own histories. Those counts were not delivered to the attacking policy.
`own-army-details-v1` reveals its own armies only and is not a full creature
special-ability or enemy-strength interface. Combat remains automatic BattleAI.

## 2026-11-07 — Prebuilt taverns and no calendar monster spawns: plan

The user requested two additional temporary restrictions for the current
experiment: both starting towns have a tavern, and no new neutral armies appear
over time. They intend to remove restrictions gradually in later experiments.
Interpretation: grant the native Tavern building at map initialization without
resource deductions; suppress calendar-driven new wandering-monster objects,
while retaining initial neutrals, ordinary growth, special-week town growth and
other existing rules. No training campaign or full policy evaluation is requested.

Acceptance criteria: version the map/rules without overwriting historical maps
or runtimes; verify both initial taverns and native hiring availability; exercise
a month that would normally spawn neutral stacks and verify none appear under
the new rule, while a native-default control still spawns; preserve initial
objects/resources and native replay; wire the current cycle to the validated new
configuration and pass read-only continuation preflight from checkpoint 1,757.
Use existing build, freeze, launch and verification mechanisms. Local controls
and artifacts stay private, outside the published source diff. Record actual
validation scope and any failed attempts before calling the variant ready.

Implementation checkpoint: the new map adds only the two Tavern buildings, a
map-scoped boolean disabling monthly monster spawning and descriptive text.
The new engine setting defaults to native spawning, is stored in map settings,
and is retained by JSON map saving. Its enum was appended so existing setting
indices do not change. The first incremental build failed because the map loader
needed the IGameSettings definition; adding that include fixed the build.
Canonical dependency agreement and the exact map semantic-diff check passed.
All eighteen selected existing scenario/runtime-freeze tests passed. Native
acceptance and cycle selection remain pending at this implementation checkpoint.

## 2026-11-07 — Tavern/calendar native acceptance and cycle selection

Built and froze the canonical engine at implementation commit `81eda14` into
`Heroes-artifacts/runtime/local-v8-taverns-static-neutrals-v1`; the historical
runtime was not modified. The freeze retains the v2 map for existing historical
launch modes. The new cycle selector binds v3 separately, with its own native
acceptance. Map SHA-256 is
`2ddd04177e17f588922cc7733ec4fa049be10bb0f020f7387fd7285ea1604056`.
The frozen build contains patch 0032 with monthly spawning enabled by default;
v3 alone sets `allowMonthlyMonsters=false`.

Reused the private existing no-artifacts validator with fresh durably reserved
seeds. All 220 controls, nine initialization pairs, three native random-bonus
branches, later-hire/artifact-acquisition checks, wrong-setting replay rejection,
the frozen-1,753 two-policy prefix and process cleanup passed in 30.38 seconds.
Evidence: `Heroes-artifacts/starting-equipment-diagnostics/attempt-20261107-taverns-runtime/report.json`.
The complete fresh receipt is pinned as the new runtime's `native-acceptance.json`.

Private rule controls and scripts are retained under
`Heroes-artifacts/diagnostics/taverns-static-neutrals-20261107/`. An initial
script import failed before seed allocation or gameplay. Attempt 01 then spent
its declared diagnostic schedule beginning at engine seed 2,020,000,000, but
failed when the diagnostic tried `close` directly from a recruitment overlay.
The environment correctly rejected that invalid action before transmission.
No policy or engine correction was needed: the private control now uses the
existing explicit panel-back/unwind actions. Its failed capture was preserved.

Attempt 02 reserved the fresh engine seeds 2,020,000,032–34 and policy seeds
2,021,000,032–33. Each same-seed comparison was declared before launch, excluded
from training, and separate from the official cycle namespace. Four native
mechanics captures and their independent replays passed:

- v2 and v3 initial-state comparison: both players' initial resources, heroes,
  armies, town identities and visible objects matched exactly. The v3 map
  exposed native Taverns immediately. Each side moved its initial visitor and
  hired another hero on day one for the displayed native gold price.
- Two maps differing only in the calendar-spawn header ran through day 113.
  Both received the same native Gog month notification on day 85. The default
  control created one new Gog object at `(15,8,0)`; the restricted control had
  no spawn calls and no native `NewObject` packets in its complete trace.
  This conclusion uses engine logs, not absence behind player fog.
- Both town recruitment projections matched on days 1, 8 and 29. Special-week
  notifications remained enabled. Existing neutral growth and town special-week
  calculations are unchanged source paths; no separate quantitative neutral
  growth or Deity-of-Fire fixture was run.

The first post-capture report incorrectly read only `*.log`, omitting the real
`VCMI_Client_log.txt`, so its positive spawn assertion failed. The failed report
remains unchanged. `complete_acceptance.py` reread the retained native traces,
checked actual object-creation packets and matching month notices, and produced
`attempt-02/corrected-calendar-analysis.json` without rerunning mechanics games.
It then used the already reserved prefix seed to exercise checkpoint 1,757 on
v3: 64 Red and 53 Blue choices, exact reconstruction of both policies with zero
maximum difference, and independent native replay all passed. This was a bounded
unscored diagnostic, not a complete evaluation game or evidence of learning.
No optimizer update or training sample admission occurred.

Final native evidence is `attempt-02/acceptance.json`, copied intact to the new
runtime's `scenario-acceptance.json` (SHA-256
`0c328103d291c3576d9f76e0ae87fdf950f733528bb5ebe082c5fc191ba092b2`).
Only the fifty-game cycle selects v3, through the existing runtime validator;
source/map/runtime/acceptance bindings remain strict. Other modes retain v2 on
the new default-compatible binary. The first continuation `--check` stopped
at the existing clean-committed-checkout requirement before gameplay or seed
allocation; commit the integration before rerunning that check. No full cycle
was launched.

Integration unit selection: 59 existing tests ran, with 56 passing and three
errors. Both completed-capture fixture errors lack the `stdin` handshake added
by the earlier worker-registration fix; the old release-pin fixture omits
starting-equipment dependencies. All three errors reproduced unchanged in a
private `git archive 3012b93` baseline. Logs are `integration-tests.log` and
`baseline-tests.log` under the private diagnostic directory. No tests were
modified or excluded from the reported result, and no gate was weakened.

Final integration: the committed source at `2d322ef` passed the actual
`./continue-selfplay.sh --cycle --check --cycles 1 --hours 10` in three seconds.
It restored the complete checkpoint 1,757 state, selected 50 captures plus one
validated diagnostic, and left the next official seed offset at 238 with
`reserved=false`. A separate read-only inspection saved
`effective-cycle-configuration.json`, confirming the v3 map, new frozen runtime,
both acceptance receipts, no pending diagnostic, and unchanged latest-checkpoint
and seed-ledger bindings. The final canonical `prepare_vcmi.py --check` passed.
The variant is ready for the next manual cycle. No full training or evaluation
game was launched; the protected checkpoint remains 1,757.

## 2026-11-07 — One hundred games and split dashboard: plan

The user requested 100 training games per cycle and a script they will launch.
They clarified that each old dashboard card should contain two separate games,
so all 100 games fit in the footprint of the former 50-card grid. This is not a
Red/Blue split. Keep ten rolling actors, one PPO update after all 100 captures,
one fully verified diagnostic, the v3 Tavern/calendar rules, checkpoint 1,757,
the ten-hour deadline and existing disk/RAM safeguards. No game or training
launch is authorized in this preparation task.

Acceptance criteria: preserve historical checkpoint/seed formats while requiring
100 fresh captures for new updates; reserve 101 slots only on actual launch;
show every game exactly once in a five-group, ten-row layout with two games per
group; preserve active/result/error markers and PPO/diagnostic status. Validate
small and large terminal layouts locally, run relevant existing checks, and pass
the actual read-only continuation preflight without consuming seeds. Keep all
new validation scripts and previews in private artifacts. The requested launch
script may be tracked as the explicit user-facing entry point.

Implementation and local validation: the cycle contract now requires 100 games;
checkpoint restoration still accepts the historical 10/40/50 sizes, and seed
auditing retains the old 11/41/51 reservation widths alongside the new 101.
Actor count remains ten. The existing native runtime and both accepted scenario
receipts are retained through an exact four-file source transition: batch size,
metadata/seed compatibility, error/help text and dashboard rendering only. No
native engine, policy sampling or PPO math source changed.

All 44 selected existing progress, self-play checkpoint/training and runtime
tests passed. Private `validate_preparation.py` passed 24 rendering combinations,
full dashboard checks at 80x24, 80x28, 120x40 and 160x86, and a narrow-terminal
fallback. Each complete grid contains IDs 001–100 exactly once; status lines,
active-game details, terminal width bounds and cursor restoration were checked.
Metadata checks accept historical parents and the new size while rejecting
unknown sizes/actor settings. Incomplete batches cannot call the optimizer.
The exact source transition rejects unrelated native or model edits. A read-only
101-slot preview preserved the seed ledger. The requested script's run/check/error
argument paths were exercised against a disposable launcher, never a game.
Evidence and previews are private under
`Heroes-artifacts/diagnostics/hundred-game-ui-20261107/`.

`run-100-games.command` is a thin entry point to the existing continuation
launcher, bound to the completed checkpoint-1,757 run. Default/manual execution
requests one 100-game cycle plus its diagnostic within ten hours. `--check`
performs only preflight. A superseding lineage causes the old run path to be
rejected rather than silently selecting another parent. Full-cycle execution,
memory use and duration remain unobserved; no seeds or training were started.
Clean-checkout preflight follows the implementation commit.

Final readiness: committed implementation `f2c1f99` passed the actual
`./run-100-games.command --check` in three seconds. It restored checkpoint 1,757
and reported 100 training games, one diagnostic, one expected PPO update, ten
rolling actors and the 365-day limit. The next reservation is 101 physical-game
seeds with 202 policy seeds, starting at the unchanged offset 238;
`reserved=false`, `gameplay_performed=false` and `optimizer_updates=0`.
The v3 map and both native acceptance gates passed through the declared exact
source transition. `preflight.log` retains the complete output. The 80x24
preview was visually inspected as simulated progress, with all 100 games and
details for all ten active games visible. No native game or optimizer was run.

## 2026-11-07 — User-launched hundred-game cycle stopped at disk reserve

The user launched `run-20261107T155027Z-dfd8ff19` and requested a progress check.
Read-only inspection found the run already stopped, with no remaining coordinator,
capture workers or native clients. Source was `9e38b6c`, parent checkpoint 1,757,
the accepted v3 map and the hundred-game/ten-actor configuration. The launch began
at 17:50:27 Kyiv time; shutdown evidence was finalized at 17:51:55, approximately
88 seconds later. The supervisor reported `Ten-game disk reserve reached;
stopping owned workers` (the message retains its historical mode name).
Current filesystem free space was 29.98 GiB, below the unchanged 30 GiB reserve.
The stopped run directory occupied approximately 5.4 GiB.

Of 100 scheduled training games, 13 captures completed naturally, ten were
interrupted during shutdown and 77 never started. Engine-reported results were
11 Blue wins and two Red wins, with no completed day-limit cutoffs. These are
unverified training outcomes; policy reconstruction/native replay were not run
for those captures under the declared training contract. The capture report had
zero individual capture failures; the overall batch failed because the resource
safeguard stopped it. Zero games/samples entered PPO and zero updates occurred.
Checkpoint 1,757 remains the latest publication; both policy and resume files
were freshly hash-checked against the unchanged latest pointer.

The reserved 101-slot schedule beginning at offset 238 remains consumed,
including interrupted and unstarted slots. Shutdown reports verified owned
process cleanup and unchanged protected inputs. Evidence: the run's `report.json`,
`shutdown.json`, `launch.json` and retained capture manifests. No source/runtime
fix, file deletion, reserve reduction or restart was performed during this status
check. A future retry must follow the latest registered run with fresh seeds and
sufficient free disk space; the older run path in the one-use launch script is
now superseded.

## 2026-11-07 — Keep one run, then retire its artifacts: plan

The user explicitly replaced indefinite artifact preservation with a one-run
policy: capture a run, analyze it, retain important conclusions in this journal,
then delete old artifacts before the next run while keeping the neural network.
For exact training continuation, "network" includes the paired policy/resume
files, Adam state, RNGs, progress and opponent pool. The small initial lineage
checkpoint and consumed-seed ledger also remain; engine/Python/game assets are
execution dependencies, not game captures. Historical raw re-verification will
no longer be possible after the authorized deletion.

The current launcher recursively requires all previous game evidence. Replace
that dependency with an explicitly audited compact continuation baseline before
deleting anything. Acceptance: exact checkpoint-1,757 restoration before/after
rotation, unchanged native/map/PPO gates and seed consumption, no deletion of a
live/unreviewed run, bounded paths, interrupted-rotation retry, and a clean
read-only preflight after deletion. New captures remain available until their
analysis is recorded. No gameplay or optimizer launch is part of this task.

Retention-reviewed run: `run-20261107T155027Z-dfd8ff19`.
Its completed status analysis is the immediately preceding disk-reserve entry;
13 unverified terminal captures are not an admitted batch or evidence of learning.

### Implementation and local validation before retirement

Implemented cycle-only retirement in the existing controller and manual launcher.
The retained baseline holds the original 1,753 anchor and latest full 1,757 pair,
accepted inherited configuration, consumed-seed prefix and pending-diagnostic
state. Future launches recheck current native/source/map gates and audit only
publications after that baseline. Checkpoint copies are restored before atomic
baseline publication and deletion; the old metadata bindings allow interrupted
deletion to resume. The reusable 100-game command now selects the latest state.
Disk reserve enforcement remains before seed allocation/game launch; read-only
inspection and cleanup can run below the reserve to reclaim storage.

Private controls initially exposed two errors caused by macOS temporary-path
aliases: pruning compared unresolved paths with canonical checkpoint paths.
Canonical comparisons fixed the issue before any real artifact deletion. All
12 controls now pass: real 1,753/1,757 pair copy/restoration, exact seed ledger
preservation, live native/coordinator and owner-lock refusal, unreviewed-run and
symlink/path refusal, corrupt-copy abort, interrupted deletion retry, pending
diagnostic carry, seed rollback rejection, and a fixture-only next 100-game
launch descriptor at offset 339 (101 slots). No fixture game or update ran.
The 44 existing game-progress/checkpoint/self-play/passive-runtime tests passed.
Both native acceptance source transitions pass and a changed controller digest
is rejected. Private controls/logs are under `/tmp/heroes-retention-validation/`;
they are not committed. Real history audit, retirement and post-cleanup preflight
are still pending at this entry.

### Observed retirement and continuation readiness

Implementation commit `99e33be` passed the actual full legacy/cycle chain audit.
`./continue-selfplay.sh --cycle --cleanup` completed in 10 seconds and retired
all nine registered cycle run directories, including the analyzed disk-stop
attempt. The cycle directory now occupies approximately 3.7 MiB, containing
the two checkpoint pairs, compact audited baseline and original seed ledger.
Its free-space result was 106.35 GiB, versus approximately 30 GiB before cleanup.

A bounded, local-only maintenance pass then retired historical raw experiment
data in 274 selected paths under `Heroes-artifacts`: the old local-script,
self-play, ten-game, evaluation, investigation and validation captures. It held
the existing activity lease, refused live workers, checked every deletion tree
against current protected files, and preserved both existing owner-lock inodes.
Nested Git/source/model-backup repositories were excluded. Current runtime and
native acceptance, Python, native learner and game assets were preserved. This
pass took 38.48 seconds and increased free space by 727.92 GiB; final free space
was approximately 833 GiB. These free-space deltas include normal filesystem
activity and are not an exact sum of raw logical file sizes.

After both deletions the controller freshly restored the same policy/resume
pair, all protected inputs matched, and the seed ledger was byte-for-byte
unchanged (SHA-256 `e2e4214667edba00c9ff6faba819225842a9642ece9a971753ba7b8a0a1ac8fc`).
Retained checkpoint 1,757 has 7,122 rollouts, 1,475,242 decisions and 48,378
optimizer steps. Policy SHA-256 remains
`6b4c8adbdcefa9bbc4e3ead1d3d55180f281a34cf47b3f0d4bb0e06820ac1686`;
resume SHA-256 remains
`4e6d35e55a73e882c875e4f1ec22dc66f7671dbf02703e70a71550fd1c54bdcb`.
The existing 1,757 diagnostic is accepted, so no recovery diagnostic is pending.

The final manual `./run-100-games.command --check` passed after raw history was
gone: 100 training games, one PPO update and one diagnostic are planned, not
executed. The next reservation is still unallocated: offset 339, 101 slots,
engine base 1,980,000,339 and policy base 1,982,000,678. The v3 tavern/static-neutral
scenario and current native/source gates remain accepted. Native recipe check
also passed: pinned VCMI plus canonical patches/owned source match exactly.
No real game, PPO update, sample admission or seed reservation occurred in this
retention task. Important conclusions remain in this journal; deleted raw
history cannot be promised for future replay or behavioral re-analysis.

Durable continuation evidence is
`Heroes-artifacts/selfplay-ten-then-check/retained/state.json` and the copied
checkpoint pairs. Local maintenance/preflight details are in
`/tmp/heroes-retention-validation/{cleanup.log,legacy-cleanup-result.json,final-preflight.log}`;
those auxiliary files remain uncommitted and are not needed for continuation.

## 2026-11-09 — Status audit of the user's 100-game launch

Question: are the manually launched 100 games progressing, and did capture,
training, checkpoint publication and diagnostic validation complete separately?
Acceptance for this read-only investigation: reconcile the registered launch,
report, publication, shutdown and process inventory, then restore the latest
complete checkpoint without allocating seeds, launching games or deleting data.

The only registered post-retirement run is
`Heroes-artifacts/selfplay-ten-then-check/run-20261107T161908Z-3d80d63a`.
It started November 7 at 18:19:08 Kyiv time and its final report/shutdown were
written at 23:57:27; coordinator elapsed time is 20,293.32 seconds (05:38:13).
The source was `77ecfbaa4320a87b8891011ab7d41731862dcc53`, parent 1,757,
accepted runtime `local-v8-taverns-static-neutrals-v1`, and map
`hero-management-v3.vmap` (SHA-256
`2ddd04177e17f588922cc7733ec4fa049be10bb0f020f7387fd7285ea1604056`).
Settings remain 100 training games, ten rolling actors, 365 days, one PPO
update and one held-out diagnostic. Both starting taverns and disabled calendar
neutral spawning were active. Reservation offset 339 consumed 101 slots:
engine seeds 1,980,000,339–1,980,000,439 and policy seeds
1,982,000,678–1,982,000,879, including the diagnostic.

Observed: all 100 training captures completed and were admitted, with learner
seats balanced 50/50. Engine-reported outcomes were 69 Blue wins, 14 Red wins and
17 `game_day_limit` cutoffs; all other 83 terminations were natural. Capture
failures, interruptions and reported errors were zero. Training replay and
two-policy reconstruction were intentionally not performed; these outcome
counts must not be described as independently verified wins.

One PPO update admitted 296,184 learner samples and performed 9,281 optimizer
steps, advancing cumulative optimizer steps from 48,378 to 57,659. Published
checkpoint 1,758 has 7,222 rollouts and 1,771,426 decisions. Weight digest changed
from `11aebbe1228f108b3027d54273f120e7d1e2d72becfae5bb82e78739356a19ee`
to `475ba121c6da24c055c756e1b26c460c3ed5ccca264e48b434e7586417a70d08`.
The policy file SHA-256 is
`38933a5073f44a562aecacd339db8b1dc24108c4e89517e3efc1b01cc67cbe4c`;
resume SHA-256 is
`ea0ed7551b06f54bf090f1670d38f8fabd18829fde173803d49875ffbfbbea37`.
Reported approximate KL was 0.011269 and clip fraction 0.031092. These metrics
confirm the executed update, not improved playing strength.

The separate diagnostic used 1,758 in both seats, ended naturally in a Blue
victory (Red 147 / Blue 420 sampled decisions), and passed native replay plus
exact policy reconstruction for both players (maximum absolute difference zero).
It did not enter PPO. The report is `completed`; shutdown return code is zero,
errors are empty, owned cleanup is verified and protected inputs are unchanged.
No coordinator, capture worker, native client or replay process was running at
inspection. No newer launch, including one dated November 9, is registered.

`./run-100-games.command --check` passed in five seconds after auditing the
publication/evidence chain and freshly restoring the full 1,758 state. The next
unallocated reservation starts at offset 440 (engine base 1,980,000,440; policy
base 1,982,000,880), with no outstanding diagnostic. Approximately 793.4 GiB was
free. Private preflight output is `/tmp/heroes-status-20261109-preflight.log`;
durable evidence remains the run's `report.json`, `shutdown.json`,
`batch-train-000/publication.json` and diagnostic receipts. This inspection
performed no game, update, reservation or artifact deletion. The strong color
imbalance merits behavioral analysis; neither it nor one decisive diagnostic
establishes improvement over 1,757. This status check does not mark the run as
retention-reviewed; its artifacts remain available for that analysis.

## 2026-11-09 — Diagnostic 1,758 versus 1,757: analysis plan

Question: how did both 1,758 seats play, what sequence caused Blue's victory,
and which observations support or contradict improved behavior versus 1,757?
Analyze existing finalized histories only: no game, replay, training update,
seed allocation or experimental rule change. Analysis starts from `3d9641a`.
Bind the 1,758 histories to their existing verified receipts; reconstruct
decisions, resources, army/hero changes, town ownership and battle sequences.
Reuse the existing offline behavior metrics and the same four-action window
and movement-error definitions as the earlier comparison. Store scratch
analysis privately inside the current run so normal retirement removes it.

Acceptance: explain the terminal event from both players' evidence, distinguish
observed choices from inferred strategic intent, and record useful behaviors
and remaining failures. The 1,757 raw histories were deliberately retired;
comparison must use their preserved journal findings, without pretending to
recompute them. The map changed from v2 to v3 (prebuilt taverns and disabled
calendar neutral spawning), and seeds/initial economic observations differ.
Both seats use the same checkpoint within each game. A single self-play result
cannot isolate a weight-update effect or establish stronger play.

## 2026-11-09 — Observed behavior of diagnostic 1,758 and comparison with 1,757

The existing offline history audit and behavior metrics completed successfully.
Both player-history byte counts and SHA-256 hashes match the already validated
receipt; sampled choices match 147 Red and 420 Blue submitted actions. Existing
receipts confirm native replay and exact reconstruction of both policies with
zero maximum absolute difference. No native replay, inference, new gameplay or
training was launched. All results below refer to the diagnostic engine seed
1,980,000,439 and policy seeds 1,982,000,878/879, with checkpoint 1,758 in both
seats. Private evidence is inside the run at `analysis-1758/` (`analysis.json`,
`travel.json`, `critical-decisions.json`, `comparison.json` and local analysis
code/log), so the next normal retirement also removes this analysis material.

Blue won naturally on day 18. The sequence is directly corroborated by both
player histories, own-army snapshots, visible object visits and town ownership:

- Both initial heroes started with 25 pikemen; both town garrisons were empty.
  First visible resources differ: Red has 10 wood/10 ore/5,000 gold, Blue has
  5 wood/5 ore/5,500 gold. Blue saw the enemy town by day 2; Red saw it by day 6.
- On day 4 Blue's initial hero won a neutral battle and still had 25 pikemen.
  On day 6 Blue hired a second hero for 2,500 gold. This hero brought 20 pikemen
  and six archers as its starting army; no creatures were purchased separately.
- On day 10 Red selected the tile occupied by Blue's original hero at [0,7].
  Red was the attacker and lost all 25 pikemen and its original hero. Blue's
  original 25-pikeman army survived with seven. Evidence: Red decision 1178,
  battle/hero-loss events 1202–1207; Blue defender/completion events 1631/1634.
- On day 13 Red hired a replacement for 2,500 gold: 19 pixies, three air
  elementals and three water elementals. Its second movement command targeted
  Blue's second hero at [0,12], and Red lost this replacement the same day.
  Blue defended with 20 pikemen and six archers, surviving with 15 pikemen.
  Evidence: Red hire decision 1353 and attack decision 1404/events 1422–1429;
  Blue defender/completion events 2276/2280.
- During days 14–18 Red made only five `open_town`, five `close` and five
  `end_turn` choices. It had no hero, an empty garrison and 7,375–8,875 gold in
  its recorded observations. The town overview still exposed the Tavern and
  recruitment controls, but the policy opened neither panel. Specific refreshed
  hero offers were not inspected, so their post-loss availability is not assumed.
- Blue hired a third hero on day 15 and retained all three. On day 18 its second
  hero (15 pikemen) explicitly targeted Red's visible town at [4,7] from [0,10].
  There was no siege battle: the town was undefended. Blue's roster changed from
  one town to two, Red's from one to zero; Red already had zero heroes, so the
  engine declared loss/win. Evidence: Blue decision 3810, visit 3824, ownership
  update 3825 and terminal 3829; Red final ownership update 1721 and terminal
  1723. The final winning movement has no ordinary result because the terminal
  supersedes it; this is not a failed capture or missing verification.

No building, recruitment, upgrade, stack rearrangement or garrison-swap action
occurred for either player. All observed town garrisons stayed empty. Blue's
advantage came from surviving two defenses, retaining the extra army acquired
through early hiring, and eventually entering the undefended enemy town. The
learned policy chose adventure-map movements; built-in BattleAI controlled the
battles (`ExplorerRL::getBattleAIName()` and captured BattleAI load messages).
In particular, the first 25-versus-25 battle does not demonstrate learned combat
tactics, and no specific RNG/tactical explanation of that auto-battle is claimed.

There were continuing inefficient-looking patterns. Blue knew the enemy town
from day 2, made 80 movement choices while it was visible, and targeted its tile
only once, on day 18. It revealed 377/400 tiles; Red revealed 301/400. Blue used
303/420 choices (72.14%) on `open_hero`, `open_capability`, `close_capability` and
`close`; Red used 89/147 (60.54%). The overlapping four-action
open-capability/close-capability/open-capability/close-capability sequence appears
37 times for Blue and ten for Red. Window-family counts do not mean every such
choice was useless, but the recorded repeated sequences remain present.

Comparison uses the preserved November 7 analysis of 1,757; its retired raw
histories were not reread or reconstructed. Identically defined descriptive
metrics for both seats combined are:

| Metric | 1,757 diagnostic | 1,758 diagnostic |
| --- | ---: | ---: |
| End day / outcome | 365 / day-limit cutoff | 18 / Blue win |
| Enemy towns captured | 0 | 1 |
| Sampled policy choices | 17,537 | 567 |
| Four window-action kinds | 12,783 / 17,537 (72.89%) | 392 / 567 (69.14%) |
| Movement `no_route_known` | 130 / 3,176 (4.09%) | 3 / 109 (2.75%) |
| Recruit actions | 54 | 0 |
| Observed nonzero town-to-hero transfers | 3 | 0 |
| Heroes remaining | 0 | 3 (all Blue) |

Useful observed behavior includes earlier additional armies, survival of Blue's
heroes and conversion of an undefended town into a terminal win. This is not
evidence that development, army consolidation or risk assessment improved:
none of the development/transfer actions appeared, Red again lost its field
heroes, and after its second loss it stopped trying to recover despite visible
town controls and substantial gold. In 1,757 the hero-less Blue had no Tavern
and lacked the resources to build it; v3 directly removes that earlier obstacle.
Thus earlier hiring is confounded with the explicitly changed rules.

The strongest check against attributing decisiveness to the 1,758 update is the
preceding training cohort. Its immutable plan binds both learner and opponent
weights to the 1,757 digest
`11aebbe1228f108b3027d54273f120e7d1e2d72becfae5bb82e78739356a19ee`;
PPO started only after all 100 captures. Already on v3, this older policy yielded
83 engine-reported natural terminals (69 Blue wins, 14 Red wins) and 17 day
cutoffs. Those training outcomes lack independent replay/reconstruction, as
agreed; they nevertheless describe the pre-update captures, not 1,758 play.
The color imbalance predates the update and its cause is not established here.

Conclusion: the diagnostic demonstrates a valid completed game and a successful
town capture, with a few better descriptive rates. It does not establish a
strength improvement over 1,757. Different map rules, seeds, economic starts,
short versus long game exposure and changing both seats prevent that inference.
A strength claim requires repeated comparisons under identical rules with
matched seeds and swapped seats against a fixed opponent. No such experiment
was authorized or run during this analysis.

Retention-reviewed run: `run-20261107T161908Z-3d80d63a`.
The requested analysis is now recorded. Raw artifacts are still present; only a
subsequent manual launch or explicit cleanup will retire them under the agreed
single-run policy. No source/configuration change or artifact deletion occurred.

## 2026-11-09 — Authorized next 100-game cycle from 1,758: plan

The user explicitly requests that the assistant prepare and launch one more
100-game training cohort from checkpoint 1,758 with the same restrictions,
then run the checkpoint diagnostic and analyze whether behavior improved.
This supersedes the earlier user-launch-only preference for this attempt.
Reuse the existing `./run-100-games.command --run`: it already performs exactly
100 captures, one complete PPO update and one fully validated diagnostic,
within the unchanged ten-hour overall deadline. No new launcher machinery or
gameplay/training configuration change is needed.

Pre-launch acceptance: current complete checkpoint is 1,758 (57,659 optimizer
steps; 7,222 rollouts; 1,771,426 decisions), source at preparation is `3f309a2`,
no native/capture/coordinator process is active, the full restore/preflight
passes, and the completed prior run is explicitly retention-reviewed above.
The same v3 map, starting taverns, disabled calendar neutral spawning,
no-artifact/army restrictions, hero-relative support and cooldowns remain.
Collection stays at ten rolling actors, 365 days and the existing choice,
memory and disk limits. PPO/Adam and the native runtime remain unchanged.

The next reservation is offset 440: 100 training games plus one diagnostic,
engine seeds 1,980,000,440–1,980,000,540 and policy seeds
1,982,000,880–1,982,001,081. These are planned, not yet allocated at this entry.
Before starting, the launcher must copy and exactly restore the current full
1,758 pair and then retire the analyzed old raw run under the agreed policy.
Record the actual launch/source/reservation and all failures or results after
the pinned run stops; do not edit the checkout during execution.

Completion criteria: all 100 captures admitted to one update, exact full-state
restoration and publication of 1,759, verified native replay and both policy
reconstructions for the fresh 1,759 diagnostic, and owned process cleanup with
protected inputs unchanged. Analyze the new game's causal sequence and compare
it with the journal's 1,758 findings using the same metrics. A single diagnostic
can show behavior, not establish a strength improvement; no additional paired
evaluation campaign is authorized here. Preserve the new run until analysis.

## 2026-11-10 — 100 games from 1,758 captured; PPO stopped at user request

The planned November 9 attempt actually ran via
`./run-100-games.command --run`, with source pinned to clean commit
`3c859137dfc1ccee7496a3871e139ca61373da1f`. Private evidence is under
`/Users/orsi/Projects/Heroes-artifacts/selfplay-ten-then-check/run-20261109T141819Z-14d160b4`:
`launch.json`, `batch-train-000/plan.json`, `report.json`, `shutdown.json`,
`coordinator-deadline.jsonl`, `progress/ppo-batch-train-000.json`,
`power-sleep-evidence.log` and `post-stop-preflight.log`. The existing reusable
launcher needed no source/configuration change. The checkout stayed unchanged
until the supervisor and its owned processes had exited.

Runtime remained `local-v8-taverns-static-neutrals-v1`; the v3 map SHA-256 is
`2ddd04177e17f588922cc7733ec4fa049be10bb0f020f7387fd7285ea1604056`.
Both starting Taverns, disabled calendar neutral spawning, artifact/army rules,
hero-relative controls and cooldowns remained unchanged. Collection used ten
rolling actors, 365 days, 158,720 ordinary plus 5,120 reserved choices per
player, the same PPO/Adam/native configuration, 30 GiB disk reserve, 48 GiB
memory headroom and the shared 36,000-second wall deadline.

The actual reservation was offset 440, width/count 101: training engine seeds
1,980,000,440–1,980,000,539, diagnostic engine seed 1,980,000,540, and policy
seeds 1,982,000,880–1,982,001,081. All remain consumed, including the unstarted
diagnostic. The launcher preserved and exactly restored the complete 1,758
pair before deleting the preceding reviewed run's raw artifacts. Parent policy
SHA-256 is `38933a5073f44a562aecacd339db8b1dc24108c4e89517e3efc1b01cc67cbe4c`,
resume SHA-256 is `ea0ed7551b06f54bf090f1670d38f8fabd18829fde173803d49875ffbfbbea37`,
and weights SHA-256 is `475ba121c6da24c055c756e1b26c460c3ed5ccca264e48b434e7586417a70d08`.

Launch was November 9 at 16:18:19 Kyiv (14:18:19 UTC). All 100 attempts reached
`captured` after approximately 35 minutes: 66 engine-reported Blue wins,
16 Red wins and 18 day-limit cutoffs, with zero capture failures/interrupted
games. As configured, these training outcomes have no independent native
replay or policy reconstruction. Both seats used the frozen parent 1,758;
the outcomes are not games played by a newly updated model.

PPO began after collection, but macOS power logs show closed-lid sleep starting
at 19:53 Kyiv, brief maintenance wakes and a full lid/HID wake at 23:14.
The existing `caffeinate -i` asserted prevention of idle sleep, not prevention
of closed-lid sleep. The wall deadline continued advancing while the monotonic
progress timer excluded sleep. The final launcher display read 06:37:47,
whereas launch-to-shutdown wall time was 09:53:32. Sleep accounts for a material
part of that difference; it does not establish the cause of every later change
in optimizer throughput. Power was subsequently connected, and the user
confirmed the lid was open. No deadline, system power setting, PPO setting or
source binding was changed to bypass the run's limits.

Before the 02:18:19 deadline, the user explicitly requested stopping for the
night and trying again later. SIGINT was sent only to the identity-checked
current supervisor; its owned cleanup completed at 02:11:51 Kyiv on November
10. The supervisor exited 1, recording `KeyboardInterrupt`; the coordinator
return code is -9 after owned shutdown. All four previously identified
supervisor/coordinator/guard/caffeinate PIDs were gone, and shutdown evidence
reports `owned_cleanup_verified=true` and `protected_inputs_unchanged=true`.
This was a user-requested interruption, not an observed deadline timeout.

The last disposable PPO snapshot was 6,826/9,073 minibatches (75.23%), with
phase still `updating`; it is stale after process termination and is not a
checkpoint. The finalized campaign report is `stopped`, its training phase is
`failed`, admitted games/samples and completed optimizer updates are all zero,
and latest checkpoint remains 1,758. No 1,759 publication or diagnostic exists.
The complete durable state therefore remains 57,659 optimizer steps, 7,222
rollouts and 1,771,426 decisions. The partial in-memory update cannot be
continued through the current fresh-capture contract.

Post-stop `./run-100-games.command --check` passed without gameplay, seed
allocation or optimization, including exact model/Adam/RNG/progress/sampler/
opponent-pool restoration. The next fresh offset is 541 (engine base
1,980,000,541; policy base 1,982,001,082), not a replay of this reservation.
The observed cohort and interruption are now analyzed; there is no post-update
behavioral result or evidence of learning to compare with the 1,758 diagnostic.
The earlier completion criteria were not met. A future launch starts a fresh
100-game cycle from 1,758 with the same restrictions; it is deferred at the
user's request, and no restart or automation was scheduled.

Retention-reviewed run: `run-20261109T141819Z-14d160b4`.
Keep this run's private artifacts until the next authorized launch/cleanup;
the complete 1,758 pair and consumed seed ledger remain protected.

## 2026-11-10 — Authorized fresh 100-game retry from 1,758: plan

The user explicitly reauthorizes one fresh 100-game cycle from checkpoint
1,758 with the same restrictions, followed by its validated diagnostic, and
confirms AC power and an open lid. This supersedes the earlier stop/defer
instruction for the new attempt. The question is whether the unchanged cycle
can complete on an awake Mac and what behavior its resulting policy shows;
one diagnostic does not establish improved playing strength.

At preparation source `d1ff72e`, `./run-100-games.command --check` passed exact
model/Adam/RNG/progress/sampler/opponent-pool restoration at 1,758, with no
gameplay or optimizer update. `pmset -g batt` reported AC power, 100% charged.
The parent remains 57,659 optimizer steps, 7,222 rollouts and 1,771,426 decisions.
The preceding interrupted run is analyzed and retention-reviewed above.

Run the existing `./run-100-games.command --run` with one cycle and its unchanged
ten-hour shared deadline. Keep v3 starting Taverns, disabled calendar neutral
spawning, artifact/army restrictions, hero-relative controls/cooldowns, 365 days,
choice limits, ten rolling actors, PPO/Adam/native runtime, and disk/memory
reserves unchanged. Preserve and restore the complete parent pair before
retiring the reviewed old raw run. Pin the clean source after this plan is
committed and do not edit the checkout while execution uses it.

Planned fresh reservation: offset 541, count/width 101; training engine seeds
1,980,000,541–1,980,000,640, diagnostic engine seed 1,980,000,641, and policy
seeds 1,982,001,082–1,982,001,283. These are not yet allocated at this entry.
Never reuse the interrupted run's reservation or partial in-memory PPO work.
Completion requires all 100 captures admitted to one full update, publication
and exact restoration of 1,759, a fresh diagnostic with native replay and exact
two-policy reconstruction, and verified owned cleanup/protected inputs. Record
actual source, timing, outcomes, admission and validation separately after the
run ends; compare diagnostic behavior with the preserved 1,758 journal analysis.

## 2026-11-10 — Fresh 100-game retry completed; checkpoint 1,759 diagnostic

The authorized retry completed using `./run-100-games.command --run`, source
`68e36c0d1ca6d1c8bcbf1f8b4decdace8df1bc38`, unchanged v3 map/runtime and all
restrictions/configuration from the plan above. Private run:
`/Users/orsi/Projects/Heroes-artifacts/selfplay-ten-then-check/run-20261110T113903Z-94c3b975`.
The launcher retired the analyzed interrupted run only after preserving and
restoring the complete parent 1,758 pair. Actual reservation was offset 541,
width/count 101: training engine seeds 1,980,000,541–1,980,000,640, diagnostic
engine seed 1,980,000,641, policy seeds 1,982,001,082–1,982,001,283. The next
fresh offset is 642; every reserved slot remains consumed.

Launch was 13:39:03 Kyiv; checkpoint publication was 21:30:24 and clean shutdown
21:31:18 on November 10. The coordinator reports 28,327.12 seconds (07:52:07);
launch-to-shutdown wall time was 28,334.46 seconds (07:52:14). All 100 captures
were first observed ready after approximately 38 minutes 30 seconds. The
machine started on AC at 100%; a late check found battery power at 83%, with
lower observed PPO throughput. No closed-lid pause was observed in the progress
timing. These observations do not isolate the cause of the throughput change.
No runtime setting, source binding or deadline was changed during execution.

The 100 training captures produced 60 engine-reported Blue wins, 19 Red wins
and 21 day-limit cutoffs; capture failures/interrupted games were zero. These
outcomes remain capture-only, without independent replay or policy
reconstruction. Both seats used parent 1,758, so they do not describe 1,759.
All 100 games and 341,569 samples entered one completed update of 10,697 Adam
optimizer minibatches. Publication and exact checkpoint restoration passed:
1,758 -> 1,759, total optimizer steps 57,659 -> 68,356, rollouts 7,222 -> 7,322,
and decisions 1,771,426 -> 2,112,995. New identities are:

- Policy SHA-256: `a25fe82a215e567b9193b8e12301de015595168a0f7a1dcc7578b92356a67f19`.
- Resume SHA-256: `4efc10ec9297cbbe9d6c0836749127155a372939a9fd065f6417850164bf76b1`.
- Weights SHA-256: `5659d8978bd574cbd3e45f583fcb98acc568668cca4da0d411e6aab880b63eae`.

`report.json` is `completed`; `shutdown.json` reports return code 0, no errors,
owned cleanup verified and protected inputs unchanged. The supervisor exited
0 and was absent from the process table. Post-run
`./run-100-games.command --check` passed exact model/Adam/RNG/progress/sampler/
opponent-pool restoration at 1,759 without gameplay, updates or seed allocation.
Evidence: `launch.json`, `report.json`, `shutdown.json`,
`batch-train-000/publication.json`, `progress/ppo-batch-train-000.json` and
`post-run-preflight.log` under the private run.

The fresh diagnostic used engine seed 1,980,000,641 and policy seeds
1,982,001,282/1,982,001,283, with both seats frozen at 1,759. It ended with Blue
winning on day 19, Red losing, and 249/237 sampled choices respectively.
Native replay passed, and both policy reconstructions had maximum absolute
difference 0. Receipt and all bound evidence file sizes/SHA-256 values were
rechecked before offline analysis. No diagnostic sample entered PPO.
Evidence is in `evaluation-000/execution/episode-000001`, particularly
`training-receipt.json`, both `agents/*/history-v8.jsonl` files and
`training-verification/`. Offline bound-history analysis is saved privately in
`analysis-1759/{travel,metrics,critical-events,summary,comparison}.json`.

Observed causal sequence:

- Both heroes initially had 25 pikemen and both town garrisons were empty.
  Red recruited 14 pikemen into its town on day 4 (decision 449), but made no
  town-to-hero transfer. Its original hero kept 25 pikemen throughout the game.
- Blue first saw the opposing town on day 3. On day 12 it explicitly targeted
  Red's town at [4,7] from [0,4] (decision 1297). Blue's 25 pikemen fought the
  town's 14-pikeman garrison; Red's field hero was elsewhere at [9,14]. Blue won
  with 22 pikemen remaining and acquired its second town. Evidence: Blue visit
  1317, battle start 1318, ownership change 1321 and battle completion 1322;
  Red battle start 1593, ownership view 1595 and completion 1596.
- Red retained its living field hero and 9,385 gold after losing its town.
  Across days 13–19 it made 20 movement choices, none targeting a visible
  opposing town. It neither recovered its own town nor captured another.
  Blue hired a second hero on day 17 in the captured town (decision 2001),
  with 16 pikemen; this happened after the decisive town capture.
- On day 19 Red received `heroes.baseline.seven_days_without_town` at seq 2462,
  then loss at 2464. Immediately before loss its hero still had 25 pikemen at
  [0,6]. The following `hero_lost` at 2465 and empty roster at 2466 are
  elimination cleanup, not evidence that the field hero died in battle.
  Blue's corresponding all-opponents-defeated notice/win are seq 2389/2390.

Combat remained built-in BattleAI; the learned policy selected adventure-map
actions. The successful siege therefore does not demonstrate learned combat
tactics. Neither player built, upgraded or rearranged/transferred stacks.
Red's single recruitment made a real town garrison, but did not consolidate
the separate field and town forces. Blue survived with two heroes and two
towns. Both sides retained repeated window navigation patterns (20 overlapping
open-capability/close-capability/open-capability/close-capability sequences
each); the four window action kinds are not all presumed useless.

Comparison with the preserved 1,758 journal analysis (its raw run was already
retired) uses the same definitions and v3 restrictions:

| Metric | 1,758 diagnostic | 1,759 diagnostic |
| --- | ---: | ---: |
| End day / outcome | 18 / Blue win | 19 / Blue win |
| Enemy town captured on day | 18, empty garrison | 12, 14-pikeman garrison |
| Sampled choices, both seats | 567 | 486 |
| Four window-action kinds | 392 / 567 (69.14%) | 327 / 486 (67.28%) |
| Movement `no_route_known` | 3 / 109 (2.75%) | 5 / 101 (4.95%) |
| Recruit actions / hero hires | 0 / 3 | 1 / 1 |
| Town-to-hero transfers | 0 | 0 |

The earlier capture of a defended town and one useful recruitment are positive
descriptive observations; fewer window actions coexist with a higher no-route
fraction and no recovery response from Red after losing its only town. These
mixed observations do not establish stronger play or understanding of the
seven-day loss condition. The seeds/economic observations differ, both seats
changed policy, and there is only one diagnostic per checkpoint. No matched
seed, swapped-seat fixed-opponent evaluation was authorized or run. The cycle
completion criteria were met; the stronger-learning hypothesis remains unproven.

Retention-reviewed run: `run-20261110T113903Z-94c3b975`.
Keep this run for inspection until the next authorized launch/cleanup. Its
summary and causal findings are durable here; preserve the complete 1,759 state
and seed ledger before retiring raw artifacts. No further cycle was launched.

## 2026-11-10 — Requested detailed 1,759 diagnostic review

Rechecked the retained diagnostic receipt and all 18 bound evidence files by
size/SHA-256, then reread both raw player histories and the saved offline
metrics. The day-19 seven-days-without-town notice, terminal outcomes and action
counts agree with the causal analysis above. Compared with the preserved 1,758
journal analysis; its retired raw histories were not available for a fresh
reconstruction. No new game or training ran. Earlier defended-town capture and
one recruitment remain descriptive positives, alongside persistent window
loops, no army consolidation and no Red town recovery. Stronger play remains
unproven by these two differently seeded self-play diagnostics.

## 2026-11-10 — Prepared user-launched 100-game continuation from 1,759

The user requests another 100 games with the same restrictions followed by the
validated diagnostic, and explicitly retains launch control. No game or PPO
update was started during preparation. Reuse `./run-100-games.command --run`;
the existing launcher restores the latest complete checkpoint automatically.
At source `3574c71`, its read-only `--check` passed exact model/Adam/RNG/progress/
sampler/opponent-pool restoration at 1,759 and the disk reserve gate. Parent
progress is 68,356 optimizer steps, 7,322 rollouts and 2,112,995 decisions.

Keep the v3 starting Taverns, disabled calendar neutral spawning, artifact/army
rules, hero-relative controls/cooldowns, 365-day and choice limits, ten rolling
actors, PPO/Adam/runtime, memory/disk reserves and ten-hour shared deadline.
The preceding 1,759 run is analyzed and retention-reviewed. At actual launch,
preserve/restore its complete checkpoint state before retiring its raw data.

Planned reservation is offset 642, count/width 101: training engine seeds
1,980,000,642–1,980,000,741, diagnostic engine seed 1,980,000,742, policy seeds
1,982,001,284–1,982,001,485. No seeds were allocated during preparation.
Acceptance remains all 100 captures admitted to one complete update, exact
restoration/publication of 1,760, one fresh diagnostic with both policy
reconstructions and native replay, and verified cleanup/protected inputs.
Record actual launch identity, timings and results after execution. A single
diagnostic supports behavioral inspection rather than a strength claim.

## 2026-11-11 — User-launched 100-game cycle and checkpoint 1,760 analysis

Question: did the prepared cycle complete, and how does the fresh diagnostic
compare with 1,759? Acceptance is the preceding plan's 100 admitted captures,
one exact-restored update and independently verified diagnostic. Actual run:
`/Users/orsi/Projects/Heroes-artifacts/selfplay-ten-then-check/run-20261111T115614Z-5c919e30`,
source `d6127c277f1e7d710223ad5586a1475d9f2d651b`. The user launched the existing
script. v3 map/runtime, starting Taverns, disabled calendar neutral spawning,
artifact/army rules, horizons, PPO/Adam, ten rolling actors and resource/deadline
limits remained unchanged. The launcher retired the reviewed 1,759 raw run
after preserving its complete continuation state.

Actual reservation offset 642, width/count 101: training engine seeds
1,980,000,642–1,980,000,741; diagnostic 1,980,000,742; policy seeds
1,982,001,284–1,982,001,485. Diagnostic pair: 1,982,001,484/1,982,001,485.
Next fresh offset 743. Launch was November 11 at 13:56:14 Kyiv, publication
18:13:56, shutdown 18:24:38. Coordinator elapsed 16,098.96 seconds (04:28:18).

All 100 training games were captured: engine-reported Blue 65 wins, Red 16,
19 day cutoffs, zero capture failures/interrupted games. These unverified
training outcomes used frozen 1,759 weights, not the resulting 1,760.
100 games / 253,760 samples entered one completed PPO update of 7,956 optimizer
steps. Total optimizer steps became 76,312, rollouts 7,422, decisions 2,366,755.
Policy SHA-256 `d1ef96bf3fbe150810a1f8c34fc82065b336373cf731717507812f5e8423007e`;
resume SHA-256 `ec7646cd28e5f75b3a3d1b44589dc91d524e364ba5089cc3275fd1baa8d97b9f`;
weights SHA-256 `6147646c783e945cf07331632a76ffb751b91b24dc66726b2d6dceac78cbb215`.
Publication exact-state restoration passed. Campaign status is `completed`;
shutdown return code 0, errors empty, owned cleanup verified and protected
inputs unchanged. Read-only `./run-100-games.command --check` restored 1,760
including model/Adam/RNG/progress/sampler/opponent pool and allocated no seeds.

The diagnostic passed native replay and exact reconstruction of both policies
(maximum absolute difference 0), but was truncated at `game_day_limit`, with
no winner or terminal rewards. There were 365 action days; the final public
boundary refresh is labelled day 366, not an extra played day. Red made 5,664
choices, Blue 5,227. The receipt and every bound evidence file were rechecked
by streaming size/SHA-256 before offline analysis. Private outputs:
`analysis-1760/summary.json`, `{red,blue}-metrics.json`, `critical-events.json`
and `development.json`. Raw evidence is in
`evaluation-000/execution/episode-000001`; campaign/publication/shutdown files
bind the training outcome and saved state. No new gameplay or update ran
during analysis. The retired 1,759 game is compared through its journal record.

Observed sequence and reasons for the cutoff:

- Both original heroes started with 25 pikemen. Blue lost its original hero
  on day 7; Red lost its original on day 15. Both later hired replacements
  (seven Red hires, nineteen Blue hires); roster removals total eight Red
  heroes and eighteen Blue heroes. These are losses, not necessarily distinct
  hero identities because dismissed/lost heroes can reappear in offers.
- Red built twice (days 14 and 323; Castle special3 and marketplace), Blue
  twice (days 183 and 188; Castle special3 and upgraded level-one dwelling).
  Construction results are `built`; recruit results are successful 10/10 Red
  and 6/6 Blue. No explicit stack-upgrade action occurred.
- On day 104 Red's `swap_garrison` decision 9167 moved its visiting hero into
  the garrison and added all 162 town pikemen to the hero's 40 troglodytes and
  four harpies. The town army became empty. The day-110 swap released that
  hero to the field. It retained a substantial pikeman stack for much of the
  game but was eventually lost on day 357.
- Blue's day-326 swap (decision 46958) changed its hero's 17 pikemen into
  333 pikemen plus 91 halberdiers, taking 316 pikemen and 91 halberdiers from
  the town. A day-336 swap released it. This hero won the late battles on days
  348, 349 and 357, removing Red's last three heroes. At the cutoff Blue had
  this hero with 326 pikemen / 65 halberdiers and another with 14 pikemen /
  five archers. Red still owned its town, with 445 garrison pikemen and
  174,484 gold; Blue had 92,426 gold and its original town.
- No town ownership changed. Red selected the enemy town only four times
  among 1,040 movement choices (days 57, 142, 204, 344); three stopped at the
  movement limit and one ended `hero_unavailable`. Blue selected the enemy
  town once among 910 moves, on day 93, and lost that hero. After acquiring
  its large day-326 army, Blue never selected the opposing town as a movement
  target. Both eventually knew all 400 tiles, so lack of map exploration does
  not explain the continued absence of captures. This does not prove a
  particular attack would have won against the remaining garrison.

Metric limitation: generic `army_operations.across_armies` is zero because
`swap_garrison` does not use source/destination transfer arguments. That zero
must not be reported as no army transfer. Before/after own-army rows prove
the two substantial merges above. Six successful swaps are not six proven
nonzero troop transfers; some just change the hero's garrison/visiting status.
BattleAI still controls combat; late battle wins do not prove learned tactics.

| Metric | 1,759 diagnostic | 1,760 diagnostic |
| --- | ---: | ---: |
| Outcome / action days | Blue win / 19 | No winner / 365-day cutoff |
| Town captures | 1 | 0 |
| Choices | 486 | 10,891 |
| Four window-action kinds | 327/486 (67.28%) | 7,638/10,891 (70.13%) |
| No known route / movement choices | 5/101 (4.95%) | 88/1,950 (4.51%) |
| Successful construction actions | 0 | 4 |
| Successful recruitment actions | 1 | 16 |
| Hero hires | 1 | 26 |
| Observed substantial town-army merges into heroes | 0 | At least 2 |

Both seats retained repeated window loops: 503 Red and 440 Blue overlapping
open-capability/close-capability/open-capability/close-capability sequences.
Window actions are not all intrinsically useless, but repetition and the lack
of follow-through on town attacks persist. The strongest positive observations
are actual construction, recruitment and mobilization of accumulated town
troops. The weakest behavior is converting those forces and full map knowledge
into a sustained town-capture objective. Defense and field battles can continue
without satisfying the victory condition when both players retain a town.

This game demonstrates a broader observed action repertoire, not statistically
established learning or regression. A 365-day game offers far more chances for
rare actions than a 19-day game; seeds and both policies also differ. Shorter
training runtime reflects fewer samples/steps and does not establish smarter
play. No fixed-opponent paired-seed evaluation was run. The execution acceptance
criteria passed; stronger strategic play remains unproven.

Retention-reviewed run: `run-20261111T115614Z-5c919e30`.
Keep raw evidence until the next authorized launch/cleanup, preserving the full
1,760 state and seed ledger first. No further cycle was launched.

## 2026-11-11 — Configurable 10–1,000-game launcher and grouped dashboard: plan

The user requests a configurable game count (default 500), retaining the current
cell dimensions with multiple game markers per cell. They explicitly accept
PPO batches of at most 100 games and one diagnostic after the last update, and
authorize scaling the overall time allowance at ten hours per 100 games with
an explicit `--hours` override. Launch remains under user control.

Reuse the current coordinator, checkpoint publication, seed ledger and terminal
dashboard. Partition a requested total into balanced complete batches of 10–100
games, preserve checkpoints after each batch, and reserve fresh seeds for every
game plus the final diagnostic. Preserve historical launch/checkpoint/ledger
validation and pending-diagnostic recovery. Keep ten actors and memory/disk gates.
The UI should retain a bounded grid, show per-game outcome letters B/R/C/E and
calendar progress while playing, with totals and PPO/diagnostic status outside
the cells. Default 500 should display five games per old half-card.

Acceptance: offline checks for boundary/odd counts, seed offsets, batch progress,
metadata and terminal dimensions; existing relevant tests; read-only restore
of checkpoint 1,760 and configuration previews without games, training, seed
allocation or cleanup. Keep new validation artifacts private and outside Git.

Implementation reuses `run-100-games.command` with positional N or `--games N`
(default 500), plus `--hours`. Balanced partitions avoid a partial batch below
ten games (101 becomes 51+50). New launch manifests bind requested total and
batch sizes; per-batch plans/metadata/PPO caps bind the actual batch size.
Seed reservations consume N+1 slots, with a separate consumed block if a prior
diagnostic is outstanding. Historical manifests, metadata and ledger widths
remain supported. Checkpoint publication is unchanged after each full update;
only the final update gets the new launch's diagnostic. Capture memory stays
bounded by one batch and is released between updates.

New dashboard cells retain the half-card width and contain five individual
markers above 100 games. 500 has 100 cells, 1,000 has 200 and double the rows.
Full two-line cells fit 80x28 / 80x48 respectively. Compact rows, outcome colors,
day-progress digits and exact active-game details use the existing disposable
display channel. No training decisions use UI state.

Offline validation: every integer partition 10–1,000, invalid values, terminal
geometry/markers and cumulative multi-batch display passed. A simulated
coordinator covered 10, 101, 500 and 1,000 games, pending-diagnostic recovery,
seed offsets, final-only diagnostics and an interrupted third update retaining
the first two publications. These used mocked games/PPO, not native execution.
The initial coordinator fixture mistakenly wrote the private live latest
pointer because TEN_BASE was not isolated. Immediately reconstructed it from
the real, verified 1,760 publication and checked policy/resume hashes. Reran
with TEN_BASE isolated in temporary storage and verified the real pointer hash
did not change. Full live restoration remains required after committing source.

The initial 84-test existing suite had two stdin-mock errors in historical
controller tests; both reproduced against the unchanged pre-task controller.
No application stdin behavior or tests were altered to mask them. The other
82 tests and four contextual checkpoint tests passed (86 total in the focused
run). Contextual test imports required the existing tests directory on sys.path;
terminal fixtures were corrected to set TERM=xterm. Logs and synthetic previews
remain in /tmp, outside the published diff. No game, training campaign, seed
reservation or artifact retirement was authorized or intentionally executed.

The first clean-source preflight correctly rejected unregistered Python source
changes at the existing exact runtime-source gate. Added the four reviewed
before/after hashes (checkpoint batch metadata, training contract, controller
and dashboard) to the existing source-transition mechanism in
`tools/run_selfplay_one_game.py`, including native-validation source agreement.
Native engine/sampling/PPO kernels are unchanged; no gate was removed. The
evaluator suite passed 17 tests and exposed one older storage-inventory fixture
missing the starting-equipment dependency; reproduced the same failure against
the pre-task evaluator. A direct identity check passed exact source comparison
and then correctly stopped at the dirty-checkout gate before this pin commit.
Live read-only preflight and tamper rejection are the remaining checks.

Final validation at source `e6309f7` passed live read-only preflight and exact
1,760 model/Adam/RNG/progress/sampler/opponent-pool restoration for 10, 101,
500 and 1,000 requested games. Previews reported [10], [51,50], five hundreds
and ten hundreds, respectively, with exactly one final diagnostic. Default
500 allows 50 hours and previews a 501-slot reservation from offset 743; 10
allows one hour, 101 allows 10.1, and the 1,000 preview accepted `--hours 80`.
No live reservation was made. Hashes of the live latest pointer, seed ledger,
lineage and retained state were unchanged across all previews.

Actual policy save/load and the unmocked bound-plan validator also passed for
51-game and 50-game plans plus a one-game diagnostic, in a temporary private
namespace under Heroes-artifacts. Initial temporary paths were correctly
rejected by the no-symlink/private-root gates; the fixture was moved into the
allowed private root rather than relaxing those gates. Its directory was
removed afterward; no capture or PPO kernel ran and the live namespace hashes
remained unchanged. Exact source acceptance passed, while mutations of all four
reviewed source hashes and an engine hash were rejected. Logs:
`/tmp/heroes-configurable-preflight-*.log`,
`/tmp/heroes-configurable-real-plan-check.log`,
`/tmp/heroes-configurable-coordinator-isolated.log`,
`/tmp/heroes-configurable-focused-tests.log` and synthetic `/tmp/heroes-ui-*.txt`.
The three historical mock failures are documented above; no tests/fixtures or
validation tooling were published. The new 500/1,000-game campaign itself has
not been launched. The user retains launch control.

## 2026-11-11 — Keep training awake while allowing display sleep

The user requests unattended overnight computation with the screen off. No
training run was active: the latest run was completed with shutdown evidence.
Reviewed the local caffeinate(8) manual and current pmset settings/assertions.
The launcher already used `-i` for idle system sleep prevention; add `-s` for
system sleep prevention on AC power, keeping display-sleep prevention unset.
Current AC display timer is ten minutes. Do not change global pmset settings,
force the display off during preparation, launch training or add a daemon.
Document open lid, AC power and a running Terminal process; minimizing/locking
the screen is supported, but closed-lid/manual sleep is not overridden.
Acceptance: shell syntax, a short harmless caffeinate-wrapped command showing
sleep assertions without a display assertion and releasing them on exit, then
read-only checkpoint preflight from clean source. No gameplay or PPO required.

Shell syntax passed. A three-second `/bin/sleep` wrapped in `caffeinate -is`
showed both PreventUserIdleSystemSleep and PreventSystemSleep on AC power,
with no PreventUserIdleDisplaySleep assertion from that process. Both sleep
assertions disappeared when the harmless command exited. AC was connected at
100%. This verifies assertion lifetime, not an overnight physical sleep trial.

At clean source `19f2904`, `./run-100-games.command --check` passed exact full
checkpoint 1,760 restoration and the default 500-game preview without gameplay,
PPO, cleanup or seed allocation. Output: `/tmp/heroes-awake-preflight.log`.


## 2026-11-12 — Investigate interrupted 500-game run after low-power sleep

Question: identify the first failure, its relationship to host power events,
and which completed updates remain recoverable. Acceptance: inspect the final
report, first failed episode and shutdown evidence; correlate native timestamps
with macOS power logs; run read-only complete-state continuation preflight.
No new gameplay, PPO, cleanup or seed reservation is authorized by this audit.

The user-launched `run-20261111T173132Z-64575310` used source
`f30de3307681c6fbe374a3db74d232ec614788ae`, parent checkpoint 1,760,
the pinned v3 runtime, unchanged rules and five planned 100-game updates.
It reserved 501 slots at offset 743 (engine base 1980000743, policy base
1982001486); all reserved slots remain consumed. It began November 11 at
19:31:32 Kyiv and finished cleanup November 12 at 06:06:48 Kyiv.
Reported coordinator elapsed time is 37,195.08 seconds; do not equate that
monotonic duration with wall time across host hibernation.

Batches 000 and 001 each captured/admitted 100 games and published exact
restore-verified checkpoints 1,761 and 1,762. Engine-only outcomes were
Blue/Red/cutoff 59/13/28 and 70/21/9. They admitted 332,159 and 210,649
samples with 10,405 and 6,608 optimizer steps respectively (542,808 total
samples). These training outcomes were not independently replay-verified
and do not establish learning improvement.

Batch 002 stopped with 52 captured, one failed, nine interrupted and 38
unstarted games. None of this partial batch entered PPO. The first failure
was episode 62 (overall game 262), engine seed 1980001004, policy seeds
1982002008/1982002009: manifest failure_detail `blue: Policy socket timeout`.
The native log reports this at 06:04:17.213 Kyiv; the generic coordinator
message `Self-play capture failed; no retries or partial batch admission`
is the consequence. No diagnostic game ran. The other 200 planned games
were not started. Total completed captures: 252/500; admitted: 200/500.

macOS pmset evidence records battery power at 4% at 05:42:52, AC attachment
at 2% at 05:47:25, then Low Power Sleep at 05:48:17 (3%, AC), lasting 960
seconds, and Wake from Hibernate at 06:04:17 (4%, AC). The native timeout
coincides with that wake. This strongly identifies low-power hibernation as
the trigger; the native socket wait has a bounded timeout and no sleep recovery.
Both caffeinate sleep assertions remained alive during the incident and were
released only at coordinator exit, so assertion launch/lifetime did not fail.
The logs do not establish why external power had been unavailable or whether
a charger/cable/display was inadequate. This was not the configured 50-hour
campaign deadline. Do not claim a generic timeout increase fixes low-power sleep.

Shutdown return code 1 records owned cleanup verified, protected inputs
unchanged and no supervisor errors. Process inspection found no remaining
training/native workers. `./run-100-games.command --check` exited 0 and
restored the complete 1,762 continuation state; no campaign was restarted.
Evidence: final report/shutdown, batch 002 episode-000062 manifest and native
log; private `analysis-failure/power-events.txt` and `analysis-failure/preflight.log`.
The existing live-UI reconstruction and its actual-Terminal-access limitation
remain recorded under this run's `analysis-live-ui/`; that was not a screen
inspection. No production code or power settings changed in this investigation.

Retention-reviewed run: `run-20261111T173132Z-64575310`.


## 2026-11-12 — Prepare authorized 100-game continuation from 1,762

The user explicitly requests launching 100 training games followed by the final
diagnostic with unchanged rules. Intended acceptance: restore checkpoint 1,762,
complete one 100-game PPO update and independently validate the final diagnostic;
recover any pending diagnostic first under the existing continuation contract.
The default shared allowance is ten hours. No seed reuse is permitted.
Initial read-only preflight refused the uncommitted investigation journal
(`Selected checkout must be clean and committed`); commit documentation before
retrying, without changing runtime code. macOS currently reports battery power
at 99%, discharging. Ask the user to connect AC before the launch because the
previous run failed after low-power sleep. Launch remains pending that condition.


## 2026-11-12 — Completed checkpoint 1,763 and paired diagnostic analysis

Question: explain the 1,763 diagnostic Red victory and compare observed behavior
with the 1,762 recovery diagnostic. Acceptance: bind all four histories to their
verified receipts, reconstruct battles/town ownership and recruitment from
before/after observations, separate descriptive changes from learning evidence.
No additional gameplay or training was launched during this analysis.

Run `run-20261112T112858Z-6cd66b99`, source cc10b53, parent 1,762, unchanged
hero-management-v3 rules and native runtime, completed in 15,302.615 seconds.
The user explicitly instructed launch despite the observed 99% battery-power
state; AC charging was observed in subsequent snapshots. The launch first
recovered the pending 1,762 diagnostic, then captured/admitted 100 training
games (engine Blue 79, Red 19, day cutoffs 2, failures 0). PPO admitted 201,094
samples, performed 6,308 optimizer steps and published exact restore-verified
checkpoint 1,763. Shutdown return code 0, owned cleanup verified, protected
inputs unchanged. Diagnostic engine/policy seeds: recovery 1980001244 /
1982002488,1982002489; final 1980001445 / 1982002890,1982002891. Both diagnostics
passed native replay and exact two-policy reconstruction (maximum difference 0).
Private evidence: report.json, shutdown.json, diagnostic training receipts and
`analysis-1763/{1762,1763}-{red,blue}.json`. All four history SHA256 bindings
were independently checked while loading the recorded frames.

1,763: Red won on day 34, with 735 Red and 390 Blue choices. Red hired four
heroes (days 15,16,18,29), recruited once, built nothing; Blue hired twice,
never recruited or constructed. On day 17 Red's original 25-pikeman hero
attacked and killed Blue's original 25-pikeman hero, retaining nine. Blue hired
16 pikemen that day; this hero attacked on day 19 and died. Red hired a
one-goblin hero on day 18, put it in garrison, recruited 42 pikemen directly
into that hero's army on day 19, then released it through swap_garrison.
This is not a transfer of 42 troops from a separate town stack.

On day 32 Red attacked Blue's last hero with 12 pikemen and four archers and
lost. Blue retained 12 pikemen and four archers after that battle. Another Red
hero attacked with 42 gnolls, won, and retained 37. Blue now had no heroes.
On day 34 Red's one-goblin/42-pikeman hero targeted and captured Blue's empty
last town; no siege battle occurred. The 42 recruited pikemen did not win the
decisive field battle. Red finished with four heroes, two towns and 6,143 gold;
Blue with zero heroes/towns and 12,375 gold. Blue never targeted the enemy town.
Red targeted it on days 22,24,31 (movement limits) and 34 (capture/terminal).
Earlier approaches did not immediately continue to capture, so do not describe
this as a proven coherent long-range plan. The final action has no ordinary
result frame because the natural terminal follows; independent replay passed.

1,762 recovery: 365-day cutoff without town captures, 4,915/9,765 policy
choices. Red hired 25 heroes and lost 25, recruited nine times, constructed
once. Blue hired 16, recruited 13 times and constructed four times. Final Red
had one hero (31 troglodytes, five harpies, three beholders), one empty-garrison
town and 48,683 gold. Blue had four heroes with 800 total pikemen and 21 archers,
one empty-garrison town and 70,721 gold. Blue targeted the enemy town six
times and failed to capture it; Red targeted twice. This prior checkpoint
already used recruitment and substantial field armies, so those capabilities
are not new in 1,763.

For a more comparable first-34-day window, 1,762 Red hired two heroes, recruited
zero times, lost one hero and issued 123 moves; 1,763 Red hired four, recruited
42 pikemen in one action, lost one hero and issued 170 moves. However 1,762
Blue recruited three times and constructed once within that window, while
1,763 Blue did neither. Same checkpoint controls both sides in each diagnostic;
the seeds differ and these are not head-to-head or matched-seed evaluations.

Four window-action kinds (open_hero/open_capability/close_capability/close)
account for 67.91% Red /75.52% Blue in 1,762, versus 66.80%/61.28% in 1,763.
Combined: 72.98% versus 64.00%. This is less window activity but remains large;
opening panels is not inherently useless. No-route moves: 39/855 Red and 84/1680 Blue
in 1,762; 1,763 has 6/170 Red and 11/78 Blue. Avoid a general pathfinding-improvement claim.
Observed positive: earlier Red mobilization and conversion to actual city capture.
Limitations: losing Blue underused economy, one weak Red attack, much window
activity, no construction or troop upgrades, stochastic same-policy self-play.
Combat is built-in BattleAI; these results concern adventure-map decisions.
No demonstrated playing-strength improvement follows from this pair alone.

Retention-reviewed run: `run-20261112T112858Z-6cd66b99`.


## 2026-11-12 — Restore native neutral spawning for a manual 200-game cycle

The user chooses restoring native calendar neutral spawning, retaining starting
Taverns and all other rules, and explicitly retains control of the launch.
Prepare 200 training games from checkpoint 1,763, two 100-game PPO updates,
then one final diagnostic, with the default 20-hour shared allowance.
Hypothesis: broaden encountered map states without simultaneously changing
starting economy or action support. Acceptance: v4 differs from v3 only by
removing allowMonthlyMonsters=false and descriptive text; retain the existing
runtime's pinned native positive/negative calendar acceptance; verify map and
scenario binding rejection controls and read-only full-state continuation.
Do not launch training, reserve seeds or retire artifacts during preparation.
The historical acceptance receipt remains available, but its raw diagnostic
folder has been retired; distinguish inherited native evidence from fresh tests.


Implementation and validation: v4 SHA256
`a076b4dca6410b5dc16388dc47333b036323a94c8fe412500cd9f31b78bcc833`.
Only the map, existing scenario selector and documentation changed. No engine,
policy, PPO, resource limit or default game-count change. Both starting Taverns
remain. The selector checks the original acceptance and its successful default
spawn/replay control, binds v4 independently and rejects any other semantic map
change. The run identity records this derivation and fresh_native_capture=false.
At clean source f9dba8e, `./run-100-games.command 200 --check` passed complete
checkpoint 1,763 restoration and two 100-game batches; preview only, 201 fresh
slots from offset 1446 (engine 1980001446, policy 1982002892), no reservations.
Default allowance: 20 hours. Evidence: `/tmp/heroes-v4-200-preflight.log`.
Two existing hero-scenario tests passed. Direct selector validation selected v4
and passed native launcher preflight; a changed map hash was rejected, and an
unrelated hero-level-limit change was rejected even with a matching temporary
hash override. These local controls created no native games or training and
were not added to the repository. The actual 200-game run remains user-launched.


## 2026-11-13 — Remove cycle wall deadlines and support suspend/resume

The user explicitly requests removing time limits and surviving Mac sleep.
Current v4 200-game run remains pinned to be4e87a with its original 20-hour
wall deadline; do not change its source or immutable launch records in place.
Prepare changes in isolated worktree Heroes-sleep-resume. At investigation,
checkpoint 1,764 was published and the second PPO update was in progress.
System logs show Clamshell Sleep at 01:12 on battery and full wake at 12:52;
PPO resumed but the existing absolute deadline still includes the sleep.

Acceptance: default cycles serialize absent campaign/phase/capture deadlines,
retain guardian parent/owner-pipe cleanup, allow explicit bounded legacy runs,
and run ordinary/native policy and diagnostic replay waits without wall timers
in the unlimited mode. A suspended process retains its RAM state; this is not
recovery after reboot, process death or power loss. Verify cancellation, real
protocol/engine failures, finite historical deadlines, exact source bindings,
new frozen runtime and bounded unscored capture/replay controls. Test pauses
longer than the former 30-second waits without publishing test fixtures/tools.
No new training campaign is authorized. Existing day/action, RAM/disk and
checkpoint admission constraints remain. The current in-flight PPO cannot be
retrofitted through an existing control API; do not promise it is rescued.

The old run `run-20261112T155949Z-e563c702` subsequently stopped at its
original deadline. The second PPO update reached 7,858/8,637 minibatches and
rolled back; 1,764 remains complete and 1,765 was not published. Shutdown
reported owned cleanup verified and protected inputs unchanged. No automatic
replacement campaign was launched. After shutdown, the isolated implementation
was copied to the main checkout; no pinned active source was changed.

Implemented paired absent deadlines, retained parent/owner guardian cleanup,
unlimited native bridge/action/replay waits only for unlimited cycles, and
explicit unlimited-wait manifest evidence. Native client/server/ExplorerRL/replay
rebuilt successfully (179 build steps); all 22 existing process-deadline tests
passed. The exact reviewed source transition was checked. Full frozen-runtime
acceptance and suspend/resume controls are still pending at this commit.

The unlimited owned-process control survived SIGSTOP/SIGCONT for 35 seconds,
then exited normally; explicit cancellation also verified group cleanup. The
78 existing headless tests and 12 training tests passed. Controller suite:
35 passed, two known old fixtures still fail because their fake subprocess
omits the registration stdin pipe. No test or probe changes are published.
Review also corrected failed-capture evidence cleanup for absent deadlines.

Fresh runtime freeze: `runtime/local-v8-sleep-resume-v1`, built from b893b81,
exact recipe checked, nine binaries and 592 resources. Private equipment
acceptance attempt e8beab failed in its restored historical probe's obsolete
Tavern artifact-count assertion (native excludes book/machine slots). Attempt
acba6e passed all nine pairs, but its obsolete mutant check expected exit 2/3;
the native mutant correctly rejected with exit 1 and an init game-state digest
divergence. Restore the already used v8 hire proof and correct return-code
expectation; preserve both failed attempts and rerun with fresh reserved seeds.

Equipment attempt `214a03` passed: 220 controls, nine paired initializations,
all three random-bonus branches, both hired-hero restrictions, a rejected
wrong-setting mutant, 64/62 frozen choices, exact policy reconstruction and
native replay. Elapsed 30.23 seconds; no optimizer updates or admitted samples.
The receipt is pinned as the new runtime's native acceptance.

Scenario attempt 01 exposed the restored probe's obsolete direct close action
from a recruitment overlay, rejected before transmission. Attempt 02 uses
existing panel unwind actions. It deliberately repeats the November 7 mechanics
seeds 2,020,000,032/33 as deterministic regression fixtures, with fresh prefix
seed 2,032,000,002 and policy seeds 2,033,000,000/1; none enter training.
Four mechanics captures/replays passed, including initial Taverns and hiring,
calendar spawning enabled/disabled and town weekly growth. Actual new-object
packets occur in the positive calendar control and not the negative control.

The additional checkpoint-1,764 prefix delayed the first policy response for
35 seconds with bridge timeout zero. It continued to an unscored 53/64-choice
cutoff, reconstructed both policies with maximum absolute difference zero,
and passed independent native replay. That replay was also SIGSTOP-paused for
35 seconds with absent guardian deadlines, resumed via SIGCONT, and completed
with owned cleanup verified. Scenario acceptance is pinned at
`runtime/local-v8-sleep-resume-v1/scenario-acceptance.json`; private harnesses,
raw captures, failed attempts and weights are excluded from Git. Existing
native shutdown-order checks also passed. These are process-pause controls,
not a fresh physical closed-lid sleep/wake experiment or reboot recovery proof.

All implementation and runtime acceptance checks are complete; the manual
launcher preflight is next, after committing its exact runtime pins. No new
training campaign was launched.

Final manual preflight `./run-100-games.command 200 --check` passed on 0acca57
in four seconds. It restored checkpoint 1,764's exact model, Adam, RNG, progress,
sampler and opponent pool; campaign, attempt and phase time allowances are null.
It previews two batches of 100, the outstanding 1,764 diagnostic and one final
diagnostic (202 physical games), with no games, optimizer updates, seed
reservations or artifact retirement. Next unreserved cycle offset is 1,647.
Evidence: `/tmp/heroes-sleep-200-preflight.log`. Default unlimited phase
propagation, strict JSON serialization and rejection of mixed absent/present
deadlines passed local checks. Training remains manual.

Correction to the preceding validation note: the native shutdown-order unittest
was discovered but skipped by its existing environment guard; it was not an
executed pass. The reported 112 executed passes comprise 22 deadline, 78
headless and 12 training tests; controller tests add 35 passes and the two
previously described fixture errors. Native process cleanup was exercised by
the actual diagnostic runs and the separate unlimited cancellation control.

## 2026-11-13 — Complete the interrupted-run retention review

The user's manual 200-game launch stopped before retirement with the missing
retention-review marker. The earlier sleep-resume work recorded the interruption
but omitted this required completion marker. Its `--check` preview did not
exercise the retirement gate; describing that preview as full launch readiness
was incomplete. No new training, seed allocation or deletion occurred at this
failure. Complete the actual review without bypassing the retention guard.

Reviewed `selfplay-ten-then-check/run-20261112T155949Z-e563c702` launch, report,
shutdown, checkpoint publication and both batches. Source be4e87a; runtime
`local-v8-taverns-static-neutrals-v1`; v4 map with starting Taverns and restored
native calendar spawning; parent 1,763. Reservation offset 1,446 consumed 201
slots (engine base 1,980,001,446; policy base 1,982,002,892), including the unused
final diagnostic slot. Both batches captured 100/100 games without capture
failures. First-batch engine outcomes: Blue 61, Red 33, six day cutoffs; second:
Blue 63, Red 26, eleven day cutoffs. These are unverified training outcomes,
not replay-verified wins or evidence of learning.

Only the first batch was admitted: 100 games, 248,320 samples, 7,785 optimizer
steps and fully published checkpoint 1,764. The second update rolled back at
the old wall deadline after 7,858/8,637 minibatches. No 1,765 or final diagnostic
exists. Reported coordinator elapsed time is 31,212.76 seconds; the original
72,000-second wall allowance also included overnight sleep. Shutdown verifies
owned cleanup and unchanged protected inputs. A fresh manual `200 --check`
restores the exact complete 1,764 model, Adam, RNG, sampler, progress and pool;
no seeds are reserved, no files retired, and default time allowances are null.

The important results and failure are now recorded. The previous raw run may
be retired at the user's next manual launch, only after the existing retirement
code copies and restore-tests the complete checkpoint pair in retained storage.
The outstanding 1,764 diagnostic remains pending and uses fresh seeds; the
interrupted batch is not reused. Next unreserved offset remains 1,647. Evidence:
`/tmp/heroes-retention-fix-preflight.log` and the private run files above.

Retention-reviewed run: `run-20261112T155949Z-e563c702`.

Validation also invoked the actual `cycle_retire` worker/path/journal guards
under the activity lease, checked protected inputs, then deliberately stopped
at the first pre-copy boundary using a private in-process test hook. All these
gates passed; no checkpoint copies, deletion, seed reservation or training ran.
Evidence: `/tmp/heroes-retention-gate-check.log`. Only this journal correction
is published; production guards and private validation tooling are unchanged.

## 2026-11-13 — Isolated PPO profiling and exact one-map experiments

The user requested PPO profiling and acceleration where possible without losing
quality, explicitly preserving the active training run and authorizing one-map
experiments. Question: does independent validation, full recurrent-prefix work,
or gradient computation dominate? Acceptance: use identical copied captures and
full parent state, attribute kernel phases, and require exact weights, final
gradients, Adam and all RNG before accepting a numerically unchanged candidate.
No new campaign, game seed reservation or official sample admission is permitted.
The experiment plan and chronological journal were written privately before
execution. This documentation is published from a separate worktree because
editing the active checkout would violate its clean-source launch contract.

The active run is `run-20261113T124117Z-f8262954`, source 9ca85df, parent 1,764,
v4 rules and `local-v8-sleep-resume-v1`. Its recovery diagnostic passed with a
Blue win. That entire diagnostic took approximately 5m10s, including 3m13s
between game completion and verification receipt; native replay itself reported
26.54s. First-batch collection/preparation took approximately 29m20s; PPO reported
13,750 seconds, 8,061 optimizer steps and 257,260 admitted samples, publishing
1,765. Training-game native replay and probability reconstruction were already
disabled. Thus removing independent verification cannot explain away this PPO
cost. The earlier suggestion that prefix recomputation dominates was a hypothesis,
not a measured conclusion. The second training batch continued throughout this
investigation and reached 100 engine outcomes without reported capture failures.

Private evidence root:
`/Users/orsi/Projects/Heroes-artifacts/run-investigations/2026-11-13-ppo-profile`.
`plan.json`, `plan-long.json`, `research-log.md`, `summary.json`, per-trial result
JSONs, state digests, cProfile data, native OS samples and ownership receipts
remain local. Source was archived from 9ca85df; production checkpoints/captures
were copied and hash-bound, never edited. The original native learner manifest
SHA is 53970605c4560bbe14b655442df373d3577efef3de0c5f824b95cfa22a74b573.
Workers ran sequentially at nice 19 under 300-second ownership guardians.
The private build used one low-priority compilation job. No production process
was sampled, signalled, restarted, reconfigured or replaced.

Primary workload: completed training episode 17, learner Red, 995 decisions,
10,702 history frames, copied full checkpoint 1,764 including Adam and RNG.
Each complete disposable PPO update used the original one epoch, 16 decisions
per window and two windows per minibatch: 32 optimizer steps. The archived
trajectory is reused only for numerical experiments, not readmitted to training.

| Experiment | Kernel seconds | Exact complete numerical result |
| --- | ---: | --- |
| Original 16 workers, cProfile enabled | 46.687 | Reference |
| One worker, including a short OS sample | 175.310 | Yes |
| Four workers | 43.613 | Yes |
| Eight workers, first / repeated | 36.275 / 36.730 | Yes / Yes |
| Original 16 workers, OS sample enabled | 37.167 | Yes |
| Original 16 workers, first / repeated, unprofiled | 35.830 / 36.358 | Yes / Yes |
| Autograd multithreading disabled, 16 explicit native workers retained | 36.585 | Yes |
| Shared current-weight prefix text cache, private native build | 39.125 | Yes |

Every complete trial matched model, final gradients, Adam, minibatch-generator,
Torch/Python RNG and numerical PPO metrics exactly. Profiling/sampled runs are
excluded from the unprofiled thread comparison. Eight workers initially looked
promising against the instrumented baseline; repetition against unprofiled
16-worker controls did not confirm an improvement. No thread-setting change is
recommended from these data. The autograd scheduling switch also did not help.

For the unprofiled 16-worker reference, backward took 24.553s (68.5%), no-grad
prefix recomputation 5.592s (15.6%), window forward 2.580s (7.2%), and Adam
0.020s (0.057%). Native sampling put the main thread predominantly inside the
outer PyTorch autograd engine, with parallel text-gradient tape work, waiting
workers and frequent small allocations. Idle-worker sample counts are not wall
time percentages. The sample does not isolate one allocator or operator as the
sole cause. Ordinary Torch operator profiling was avoided because its callbacks
would change eligibility for the optimized native tape path.

Length control: the first official batch's learner counts have median 1,429.5,
mean 2,572.6 and maximum 11,136. Copy episode 100, learner Blue, with 5,975
samples and 55,784 frames. Retain its complete trajectory and original shuffled
window schedule, but stop the disposable profile after 16 complete optimizer
steps. This partial run took 25.065s: prefix 10.842s (43.3%), backward 11.595s
(46.3%). Setup/encoding/compilation took another 23.939s outside the timed kernel;
peak process RSS was approximately 4.0 GiB. Thus prefix work is also a major
bottleneck for long episodes. Do not generalize the medium-map proportions to
all 100 games or shorten recurrent history solely on that initial measurement.

The exact-cache prototype precomputed unique text encodings once per current-
weight no-grad prefix and seeded worker-local caches with immutable tensors.
Caches were discarded before subsequent weight updates. It changed neither
history length nor PPO settings. Its private native manifest SHA is
744cc3b9ef80b593f1fe80715d242dcfe30668e237843b6aeebd70119281330c.
The complete medium-game update preserved all numerical outputs, but prefix
cost was 6.041s and total 39.125s: no observed benefit. Reject integration;
one trial does not establish a statistically significant slowdown. The patch,
build and diagnostic harnesses are not published.

Conclusion: profiling completed, but no faster production change was accepted.
Further exact optimization should target execution of the main backward graph
and duplicate no-grad prefix computation, retaining current-weight semantics
and gradient accumulation order. Memory truncation, larger PPO minibatches or
stale hidden states change the learning contract and require separate quality
experiments. These low-priority one-map measurements ran alongside production;
they are not uncontended full-batch throughput estimates. In particular, several
windows from one map can share prefixes differently from a 100-game batch.
All eleven private trials completed with owned cleanup; ten were full updates
and one was partial profiling. Zero official updates, admitted samples or game
seeds were produced by this study. The active checkout and its settings remain
unchanged. Only these compact findings are published on the documentation branch.

## 2026-11-13 — Follow-up exact PPO experiments and Metal feasibility

Question and scope: find real backward/history-recomputation speedups without changing the learning result. The user explicitly authorized isolated experiments and prohibited changes to the active run. Before each experiment, hypotheses and acceptance criteria were recorded in the private investigation journal. Acceptance requires exact model, final gradients, Adam state, minibatch generator, Torch/Python RNG and numerical PPO report, plus repeated unprofiled timing comparisons. A one-map timing signal is not sufficient to publish an optimization into the active campaign.

Source/runtime identity: immutable reference source `9ca85df0721c99605a466f23eefe932b196011f4`; copied checkpoint 1764 policy/resume pair, existing `plan.json` and `plan-long.json` bind hashes, PPO settings, seeds/RNG and episode inputs. No new game seeds were allocated. Same episode 17 (995 decisions, 10702 frames, 32 optimizer steps) and episode 100 (5975 decisions, 55784 frames; intentional stop after 16 steps) as the preceding investigation. CPU FP32, ATen intraop 1, native workers 16, nice 19, individual 300-second owned-worker bounds. Private source copies/builds never replaced the production library or acquired campaign locks. Host: Apple M4 Max, 16 CPU cores, 40 GPU cores; PyTorch 2.9.1.

Hypotheses and implementations, all private:

- `slots`: compile reusable storage slots for nonoverlapping backward input-buffer lifetimes. Preserve ATen nodes, hooks, eligibility checks, graph validation, execution schedule and addition order.
- `chunks`: assign 32 consecutive prefix frames per worker scheduling claim to improve locality and reduce scheduling overhead. Retain current-weight recomputation, per-worker canonical caches, their existing 65536-entry clearing threshold, and chronological recurrent transitions.
- `constants`: compile immutable byte-index/numeric FP32 input tensors once per history, preserving the original double-to-float conversion. No weight-dependent state is cached across optimizer updates.

Observed unprofiled kernel times (seconds; setup excluded):

| Trial | Kernel | Prefix | Backward |
| --- | ---: | ---: | ---: |
| `slots-medium-a` | 41.999 | 6.493 | 28.365 |
| `follow-control-a` | 43.274 | 7.037 | 28.695 |
| `chunks-medium-a` | 41.328 | 5.685 | 28.780 |
| `slots-medium-b` | 41.963 | 6.323 | 28.459 |
| `follow-control-b` | 42.386 | 6.128 | 29.014 |
| `chunks-medium-b` | 41.837 | 5.697 | 29.099 |
| `constants-medium-a` | 42.286 | 6.230 | 28.553 |
| `follow-long-control-a` | 27.812 | 12.281 | 12.504 |
| `chunks-long-a` | 26.485 | 10.616 | 12.604 |
| `constants-long-a` | 28.160 | 12.292 | 12.687 |
| `chunks-long-b` | 25.326 | 9.936 | 12.375 |
| `follow-long-control-b` | 28.171 | 12.236 | 13.105 |

All 12 trials completed with owned cleanup. Seven full medium updates matched the reference model, final gradients, Adam, all recorded RNG state and numerical PPO report exactly; five deliberately partial long trials matched the reference state after 16 steps exactly. No checkpoint was admitted or published. All 18 existing replay-tape tests passed for slots; all 16 existing native learner kernel tests passed for chunks. Tests/probes/builds/weights remain private and are not part of the published diff.

The chunk candidate reduced mean full medium update time from 42.830 to 41.583 seconds (2.91% less time; 1.030x speed), and mean partial long update from 27.992 to 25.905 seconds (7.45% less time; 1.081x speed). Prefix time fell 13.55% and 16.17%, respectively. The long comparison included the reverse-order pair: control/chunks, then chunks/control. This is a repeatable small signal on these copied histories, not a measured full 100-game speedup. Only two repeats per condition, concurrent official training, low-priority scheduling and different history sharing in real minibatches limit extrapolation. No substantial backward speedup was established. The slots variant's small overall difference is not attributable robustly to backward; the constants variant did not improve the long case. Keep all candidates isolated; no production source or execution manifest was changed.

Metal feasibility: a separate synthetic FP32 GRUCell probe used the actual model dimensions (input 96, hidden 64), 128 sequential transitions plus backward, seed 43161, identical weights/inputs, one warmup and three timed repeats per device. MPS synchronization bracketed timed GPU work. Median CPU 10.082 ms versus MPS 47.648 ms: this naive small sequential workload was 4.726x slower on GPU. Outputs and gradients were not bit-identical; maximum observed absolute difference 4.768e-7. This does not establish worse learning quality, nor predict a custom fused Metal implementation or whole PPO performance.

A full Metal rewrite has no measured end-to-end speedup yet. The CPU-specific native graph/tape cannot be switched by changing a device string. Batching/fusing independent work, reducing dispatches, retaining device-resident data and preserving required recurrent/gradient ordering need a separate implementation and evaluation. Apple explicitly recommends batching work and reusing buffers; PyTorch does not guarantee CPU/GPU bitwise identity. Under the present strict equality criterion, no Metal speedup can currently be accepted.

Conditional planning arithmetic, not a Metal forecast: the earlier first 100-game batch spent 13750 seconds in PPO and approximately 1760 seconds collecting/preparing. If all PPO became 2x or 5x faster with the other work unchanged, that collection/preparation+update portion would improve 1.80x or 3.44x. The final diagnostic is excluded. These factors do not establish that either PPO acceleration is achievable; the measured naive MPS probe points in the opposite direction.

Primary references: [Apple MPS tuning hints](https://developer.apple.com/documentation/metalperformanceshaders/tuning-hints), [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html).

Evidence: private `Heroes-artifacts/run-investigations/2026-11-13-ppo-profile/` contains `followup-summary.json`, individual result/state/progress files, source snapshots, patches, build manifests, ownership logs, `slots-tests.log`, `chunks-tests.log`, and `metal-probe.py/json`. Existing plans bind the source parent, exact checkpoint pair, RNG and copied game inputs. Native manifest SHA256 values:

- `slots`: `61931bdbd64045c5eda8984279aec73af8f358f0514d2e09f11d49376b669c99`.
- `chunks`: `e46ea2b67f72ddb9d2ea15a0f8a7bd4dffc7258a74fcb94b3614816a731acc2f`.
- `constants`: `38c9afb57aad285d4dd09690f0203c28ed02cf8c3e1f7853d95b9b007c833ebe`.

At completion the active checkout remained clean at 9ca85df; the original run remained running, first batch admitted 100 games and checkpoint 1765 verified, second batch still updating. No active processes, launch configuration, source, weights or checkpoints were changed. Documentation is synchronized from a separate worktree/branch only.


## 2026-11-13 — Isolated FP32 Metal PPO rewrite and bounded acceptance

**Question and preregistered criteria.** The user requested a fivefold complete
PPO update speedup using the M4 Max GPU, subsequently accepting small FP32
numerical differences while retaining PPO and checking quality. Hypotheses and
attempts were recorded before execution in the private investigation journal.
Batch independent semantic-tree, map and policy-head operations, keep weights,
gradients and Adam moments on MPS, and fuse recurrent/tree operations in Metal.
Keep full current-weight prefixes, 16 decisions/window, two windows/minibatch,
one epoch, losses, clipping, learning rate, sampler order and parent unchanged.
Require finite outputs/gradients, identical action supports, compatible CPU
checkpoint state, and report preparation plus synchronized update time. A
kernel-only speedup does not satisfy the fivefold acceptance criterion.

**Identity and scope.** Isolated worktree `Heroes-metal-ppo`, branch
`codex/metal-ppo`, based on `08184ad`; CPU source/control `9ca85df`. Hardware:
M4 Max, 16 CPU cores and 40 GPU cores, 128 GiB; local PyTorch 2.9.1 MPS shader API.
Private Python runtime: `Heroes-artifacts/python/next-launch-numpy235-v1/venv`.
Native reference/recipe validator: `remove-telemetry-v1-attempt1/manifest.json`,
SHA-256 `53970605c4560bbe14b655442df373d3577efef3de0c5f824b95cfa22a74b573`.
Parent 1764 and immutable copied episode 17 (995 decisions, 10,702 frames) and
100 (5,975 decisions, 55,784 frames), with original bound RNG/rollout identity
from the November 13 PPO profile plans. No new seeds, games, training admission
or official checkpoint publication. Workers ran sequentially at nice 19 with
owned deadlines and a 25% recommended-MPS-memory cap. The active official CPU
campaign shares hardware, so timings are observations, not isolated-machine
performance guarantees.

Private evidence root:
`Heroes-artifacts/run-investigations/2026-11-13-metal-ppo/`.
Each named timed attempt has a log, owned-process evidence and, on completion,
`result.json` plus disposable state. Later attempts also snapshot source hashes;
production-candidate attempts snapshot all changed numerical source. Commands
used the private `run.py NAME 16`, adding `--gpu-optimizer` for prototypes,
`--long --full-long` for the 187-step update, `--steps=N` for explicitly partial
probes, and `--production` for the real candidate API. These workers, validation
scripts, captures and weights remain private, outside the published diff.

**Implementation and corrections.** The candidate batches DAG nodes by depth,
uses CSR pooled-child kernels and a single gradient arena, pads independent
text/ordered sequences without truncation, and preserves their true endpoints.
Persistent Metal GRU kernels use FP32 operations; a synchronized eight-kernel
microcomparison selected the 16-lane reduction for long single-history prefixes
(13.995 ms per 10,000 transitions versus 56.006 ms for the initial kernel).
The single-SIMD-group alternative was slower (75.979 ms) and was rejected.
Cross-window deterministic encodings share work only at current weights.
Static map tensor recipes are reused exactly, while own-observation semantic
traversal and recurrent states remain frame-specific. Exact recipe comparison
passed all 13,679 nodes and 10,702 frames; 4,488 map frames used 2,449 unique
static-map inputs. Prefix CNN work is sliced to required maps; immutable plans
are cached, not encoded values or recurrent states across updates.

The first prototype failed before an optimizer step because empty and singleton
child buckets collided and produced nonfinite pooling. The existing finiteness
gate caught it. A later first-step gradient discrepancy (relative L2 0.005437)
localized to embedding padding rows: a local torch 2.9.1 MPS probe accumulated
row-zero gradients where CPU excluded them. Explicitly excluding that row
reduced first-step relative gradient error to approximately 5e-6; this is a
local-version observation, not a claim about every MPS release. The first full
long GPU attempt hit the 26.88 GiB allocator cap after 175 completed steps.
A map-dedup variable shadowed the plan-cache key and defeated eviction; this was
fixed, along with delayed context release. Subsequent full runs completed with
bounded live tensor allocation (about 150–181 MB in prototype probes). An initial
prefix-slicing probe took 16.187 s for four steps because attribute offsets were
constructed quadratically; linear cumulative sums corrected that mistake.

The internal `_update(use_metal=True)` candidate requires checked prepared
histories and the pinned native validator. It restores parameters, gradients
and Adam tensors to CPU and restores the previous foreach setting even when
an update raises. The ordinary CPU default and every training launcher retain
their previous selection. This branch is not an accepted campaign runtime.

**Attempt ledger.** Times below separate update and setup; partial, profiled and
cold-cache probes are not full-update speed comparisons. Earlier prototype
setup omitted the C++ recipe constructor; final candidate setup includes it
and owned immutable inputs. Failed attempts above have no accepted result.

| Attempt | Steps | Update seconds | Setup seconds |
| --- | ---: | ---: | ---: |
| `cpu-diagnose` | 1 | 1.197 | 5.174 |
| `cpu-first-step` | 1 | 1.165 | 5.217 |
| `cpu-long-full-a` | 187 | 394.104 | 27.247 |
| `metal-allgpu-first` | 1 | 1.678 | 4.826 |
| `metal-allgpu-full-a` | 32 | 10.486 | 4.847 |
| `metal-batched-full-a` | 32 | 33.845 | 4.935 |
| `metal-csr-first` | 1 | 0.973 | 4.821 |
| `metal-csr-full-a` | 32 | 15.022 | 5.193 |
| `metal-device-fullplan` | 4 | 3.063 | 25.985 |
| `metal-device-partialplan` | 4 | 2.343 | 25.421 |
| `metal-diagnose` | 1 | 1.550 | 4.910 |
| `metal-first-step-b` | 1 | 6.272 | 4.964 |
| `metal-four-profile` | 4 | 4.836 | 5.039 |
| `metal-four-steps` | 4 | 6.969 | 5.132 |
| `metal-fused-first` | 1 | 1.277 | 4.790 |
| `metal-joint-first` | 1 | 0.909 | 7.687 |
| `metal-joint-full-a` | 32 | 9.896 | 4.758 |
| `metal-long-full-b` | 187 | 103.204 | 26.151 |
| `metal-mapreuse-full-a` | 32 | 8.601 | 4.719 |
| `metal-padding-fixed` | 1 | 1.374 | 5.055 |
| `metal-production-failure-cleanup` | 1 | 1.281 | 5.470 |
| `metal-production-long` | 187 | 88.285 | 26.834 |
| `metal-production-medium` | 32 | 7.763 | 5.112 |
| `metal-simd-first` | 1 | 0.940 | 4.740 |
| `metal-sliced-long` | 187 | 86.654 | 25.953 |
| `metal-sliced-prefix` | 4 | 16.187 | 26.298 |
| `metal-static-first` | 1 | 0.730 | 5.063 |
| `metal-tree-first` | 1 | 1.604 | 4.786 |
| `metal-tree-full-a` | 32 | 26.405 | 5.109 |
| `metal-warp-long-a` | 187 | 136.102 | 26.101 |
| `metal-warp-medium-a` | 32 | 10.005 | 4.908 |
| `metal-wide-full-a` | 32 | 12.587 | 4.679 |

**Complete candidate results.** The actual candidate API completed all 32 steps
in 7.763 s plus 5.112 s setup (12.875 s total), versus two CPU controls of
48.351 s and 47.386 s total: 3.68–3.76x end-to-end, 5.46–5.57x kernel-only.
The 187-step candidate completed in 88.285 s plus 26.834 s setup (115.119 s),
versus CPU 394.104 + 27.247 = 421.351 s: 3.66x end-to-end and 4.46x kernel-only.
The preceding private prefix-sliced implementation measured 112.608 s total
(3.74x); do not select that lower number as a production guarantee.
**The requested fivefold complete-update target is not met.** No claim is made
about an entire 100/200/500-game campaign. Sequential nonlinear recurrence and
CPU preparation remain limits; changing PPO minibatch size or truncating
history to increase utilization would change the experiment.

**Numerical and behavioral validation.** Five GRU cases, including a 1,024-step
sequence, plus same-history/cross-history head/tree/map cases passed: maximum
output error <=4.77e-7 and relative gradient L2 <=5.27e-7; padding rows stayed
zero. Full-update CPU/GPU RNG states, Adam step counts and optimizer settings
matched exactly, and parameters/moments remained finite. Model relative L2
was 1.65e-5 after 32 steps and 1.63e-4 after 187. Maximum long-update parameter
difference was 0.01241 in the final coordinate-logit scalar bias (a common logit
offset); maximum coordinate weight difference was 0.002107. This is not bitwise
training equality and must not be hidden behind first-step tolerances.

Offline held-out comparison evaluated both resulting policies through the same
Metal inference implementation, separately from CPU/Metal operator parity.
After training on episode 17, held-out episode 100 used 128 decisions and 223
conditionals: argmax 223/223, mean KL 4.38e-14, max KL 4.29e-13, value RMSE
2.37e-7. After training on episode 100, held-out episode 17 used 128 decisions
and 227 conditionals: argmax 227/227, mean KL 2.37e-11, max KL 3.47e-10, value
RMSE 4.30e-6. Both passed preregistered matching-support, >=0.999 argmax,
mean KL <=1e-6, max KL <=1e-5 and value RMSE <=1e-3 criteria. Evidence:
`production-numerical-checks.json`, `production-state-comparison.json`,
`heldout-after-medium.json`, `heldout-after-long.json`. This is bounded offline
policy agreement, not proof of preserved playing strength over future updates.
Paired gameplay and campaign-scale memory/performance acceptance remain open.

**Existing checks and failure cleanup.** All 40 existing tests passed in 6.404 s
(`tests.test_native_learner_kernels`, `tests.test_native_replay_tape`,
`tests.test_contextual_ppo`), using the explicitly pinned native manifest; no
new tests were published. `metal-production-failure-cleanup` deliberately raised
a private stop exception after one completed optimizer step, returned all model
and Adam tensors to CPU, restored foreach configuration and verified owned
cleanup (1.281 s update, 5.470 s setup). This
confirms device/config cleanup, not rollback of an already applied optimizer
step. Complete medium/long updates made the same CPU restoration assertions.

At 2026-11-13T19:12:18+00:00,
the protected checkout remained clean at `9ca85df`; the read-only official
report still showed `running` and latest published checkpoint 1765. No official
process, checkpoint, capture, runtime selector or launcher was changed by this
investigation. Candidate code and compact documentation are intended for the
explicit experimental branch only; promotion, paired gameplay and the fivefold
end-to-end criterion remain unfulfilled.


## 2026-11-13 — Follow-up: complete-update fivefold target

The user authorized continued isolated experiments, retaining the active run.
Baseline is experimental `b9dbcf4`; official checkout remains `9ca85df`.
Reuse the same parent 1764, episode17/100 copies, PPO configuration, RNG state,
private runtime and CPU native manifest above; no new seeds or admission.
Acceptance remains at least fivefold setup-inclusive speed, with output/gradient,
full-update state and held-out policy checks. The original CPU controls imply
9.67 s for medium and 84.27 s for long. GPU utilization alone is not acceptance.
Hypotheses and all attempts were recorded before execution in the private
`2026-11-13-metal-ppo/research-log.md`; probes and raw evidence remain private.

Synchronized profiles, rather than enqueue-only cProfile times, identified
prefix encoding and backward as the remaining costs. Source inspection led to
first-use slicing of semantic DAG rows, length-limited text/ordered GRUs, a fused
padding-aware byte-gradient scatter, direct window-vector PPO loss assembly,
and shared categorical projections. These keep full current-weight history and
PPO minibatch/window/loss settings. The full flags-only delta-CNN attempt still
had 8,734 bases for 8,740 long-history maps: moving heroes also occur in the
object list. The next candidate groups exact terrain/road bytes, encodes unique
object descriptors at current weights, uses signed per-cell multiset changes
plus all five flag changes, recomputes affected radius-one/radius-two CNN cells,
and updates pooled features. Loss-window CNN/backward stays on the previous
contiguous path. Sparse metadata is precomputed with NumPy; 14 CPU cases matched
the earlier metadata byte-for-byte before the object-delta extension.

Rejected attempts and limits: combined CPU map-component/own-observation caches
were exact but 19.4% slower in alternating checked-compile trials (2.599 versus
3.105 s median); they were reverted. Channels-last CNN output parity passed but
its loss-window gradient check failed at relative L2 0.1624. A medium timing
had already been dispatched before that redirected failure was inspected; it
is diagnostic only and rejected. Channels-last remains only in no-grad prefix
work, with the original contiguous loss path. Nondivisible 6x5 adaptive pooling
still raises the existing MPS restriction; initial dedicated validation hit
that boundary, and its corrected expectation records the unchanged limitation.

Object/flag delta checks covered movement, additions/removals, descriptor
changes, duplicate objects at a cell, edges, empty objects, five flag channels,
and changed weights: CPU global max error 8.20e-8, ordinary MPS max 5.22e-8,
recurrent prefix max 1.79e-7. Seven existing numerical probe cases passed after
combination. Sixty existing tests passed after PPO vector assembly, before the
subsequent serialization change. None of these local test/probe sources are
part of the published change.

Exact serialization is another candidate: bounded digest-local immutable tile
fragments, generated with the same stdlib JSON encoder, preserve schema
validation and all omission rules. All 10,702 medium frames and 21 v2-v8/Unicode/
integer-bound cases matched exact canonical bytes and digests. Three alternating
trials reduced medium digest median from 1.3167 to 0.5320 s. Default public
canonical-frame encoding remains unchanged; digest caching emits identical bytes.

Intermediate complete results are not final acceptance: object/flag CNN plus
vector/head/backward changes completed long 187 steps in 72.738 s + 25.976 s
setup = 98.714 s, approximately 4.27x against the prior CPU control. Medium was
8.074 + 5.043 = 13.117 s. A contemporaneous previous-Metal control measured
10.051 + 5.232 s medium versus its earlier 7.763 + 5.112 s, showing meaningful
runtime variation. A phase-only setup overlapped briefly with a CPU canonical
byte check; that setup time is not accepted performance evidence. AC power and
no recorded thermal/performance warning were observed; system settings were not
changed. Final timing, parity and source identity are recorded below after the
remaining candidate checks, not inferred from these intermediate measurements.


**Dispatch bottleneck and final candidate.** Direct measurement isolated
`Tensor.index_copy_`, rather than semantic GRU or pooling arithmetic. For a
100,000x32 tensor and 5,000 unique row writes, local MPS took 28.508 ms versus
0.0179 ms for a direct Metal write, with identical bytes. A 20,000-row table
measured 5.481 versus 0.0181 ms. The compiler already guarantees unique row
indices. The candidate replaces those writes inside the tree's existing manual
backward boundary and in explicitly no-grad prefix globals/own-feature tables;
loss-window autograd writes retain the ordinary operation. Empty writes are
handled without launching a zero-size grid. A prefix encoding with gradients
enabled is rejected. This is a measured implementation-specific bottleneck,
not a claim about every MPS operation or version.

Rejected lower-value alternatives remain private: pooled-child parallel R4
kernels were numerically sound and improved a large raw pooled operation by
about 0.04 ms; R8/R16 regressed, and none were integrated. Actual semantic GRUs
were only approximately 0.3-1.8 ms per group, so their further tuning could not
explain the tree phase. Gradient clipping and Adam were about 0.12 s each per
16 steps, too small to explain the remaining gap. The private trace/profiling
workers are diagnostic artifacts, not published dependencies.

The final long candidate completed 187/187 optimizer steps twice:
44.034 + 21.718 = 65.752 s, and 43.312 + 21.647 = 64.959 s, versus the original
CPU 421.351 s. These are **6.41x and 6.49x setup-inclusive speedups** for the
5,975-decision/55,784-frame update. CPU return and optimizer configuration
restoration assertions passed. The first final medium run measured
8.440 + 4.236 = 12.676 s, approximately 3.81x: the fivefold threshold is achieved
on the long full update, not uniformly on both history sizes. These measurements
are not a claim of fivefold campaign/game throughput, and the CPU control is
identified above rather than silently relabelled as a contemporaneous rerun.

All follow-up attempt result files are indexed below. Partial profiles, failed
numerical variants and kernel-only measurements must not be pooled with full
accepted timing trials. Detailed phase/source hashes are private result records.

| Attempt | Steps | Update s | Setup s |
| --- | ---: | ---: | ---: |
| `follow-channels-medium` | 32 | 11.577 | 6.338 |
| `follow-control-medium` | 32 | 10.051 | 5.232 |
| `follow-delta-long` | 187 | 75.762 | 26.623 |
| `follow-delta-medium` | 32 | 10.245 | 5.328 |
| `follow-final-long-a` | 187 | 44.034 | 21.718 |
| `follow-final-long-b` | 187 | 43.312 | 21.647 |
| `follow-final-medium-a` | 32 | 8.440 | 4.236 |
| `follow-gru-shapes` | 4 | 3.005 | 21.656 |
| `follow-length-medium` | 32 | 10.259 | 6.393 |
| `follow-object-long` | 187 | 72.738 | 25.976 |
| `follow-object-medium` | 32 | 8.074 | 5.043 |
| `follow-object-phase` | 16 | 9.138 | 26.127 |
| `follow-object-phase2` | 16 | 8.301 | 26.000 |
| `follow-phase` | 16 | 10.713 | 48.931 |
| `follow-tree-long-phase` | 16 | 8.790 | 27.513 |
| `follow-tree-medium` | 32 | 10.388 | 5.343 |
| `follow-vector-loss-medium` | 32 | 8.407 | 5.159 |


**Final validation and timing boundary.** The subsequent fresh medium CPU
control measured 48.380 s update + 5.139 s setup;
the final medium repeat measured 9.490 + 4.612 s,
which is 3.80x
against that fresh CPU control. Short-history fivefold acceptance remains unmet.
The repeated long candidate meets fivefold on the complete fixed long-history
update only; promotion to a multi-game campaign remains separate.

All 118 existing selected native/PPO/model/canonical-cache/history-contract tests
passed in 8.468 s after the final changes (`follow-final-existing-tests.log`).
Full-state comparison (`follow-final-state-comparison.json`) found exact sampler,
Torch/Python RNG, Adam step counts and optimizer configuration, with finite
parameters/moments. Model relative L2 against CPU was 2.16e-5 medium and 1.36e-4
long; maximum absolute differences were 0.001193 and 0.010361, respectively.
These checks establish bounded FP32 agreement, not bitwise identity.

Held-out checks passed both preregistered criteria: after medium training,
223/223 argmax agreement, mean KL 4.07e-14,
max KL 4.49e-13, value RMSE 2.54e-07; after long training,
227/227 agreement, mean KL 2.36e-11,
max KL 3.46e-10, value RMSE 4.29e-06. Each uses 128
held-out decisions and identical inference implementation for both resulting
policies. This bounded evidence does not establish future playing-strength
preservation. Evidence: `follow-final-heldout-medium.json` and
`follow-final-heldout-long.json`; source hashes match the repeated long-run
snapshot for every final numerical source file.

At 2026-11-13T19:46:08+00:00,
the official report remained running with latest published checkpoint 1765;
its checkout remained clean at 9ca85df. No launch/runtime selection, process,
weight or checkpoint was changed. Only the separate experimental branch is
published; tests, benchmarks, probes and raw artifacts remain private.

Final exception cleanup (`follow-final-failure-cleanup`) deliberately stopped after
one optimizer step and again verified model/Adam CPU return, original foreach
configuration and owned-process cleanup (0.952 s update,
4.494 s setup). This is cleanup, not rollback of the applied
private step. No official update or checkpoint was published.


## 2026-11-13 — CPU campaign termination and authorized 100-game Metal cycle

**Question and authorization.** The user requested a live completion audit and,
once the previous run ended, a full 100-game Metal cycle measurement. The
previous process has exited; it did not complete its planned 200-game campaign.
The next run will use the latest published complete checkpoint, 1,765, retain
all gameplay/PPO settings, recover its outstanding diagnostic first, then play
100 fresh training games, perform one complete Metal update and verify one final
diagnostic. Existing leases, seed reservations, source/runtime checks, full
batch admission, rollback, checkpoint restoration and artifact retirement stay
active. CPU remains the default; the experimental branch gets explicit opt-in.

**Observed previous run.** `run-20261113T124117Z-f8262954`, source `9ca85df`,
parent 1,764, captured all 200 games. The first batch admitted 100 games and
257,260 samples; 8,061 optimizer steps published and exactly restored 1,765.
Engine-reported outcomes: Blue 58, Red 34, eight cutoffs; training replay and
policy reconstruction were not performed. Collection/preparation took about
1,760 s and PPO progress recorded 13,750 s; plan to publication was about
15,569 s. These are retrospective phase boundaries, not a paired Metal control.
The second batch captured 100 games (Blue 55, Red 32, 13 cutoffs) and contained
330,327 samples, but PPO stopped at 5,226/10,347 steps after 9,816 s. None of
that batch was admitted; no 1,766 checkpoint was published.

Coordinator PID 1725 aborted at 2026-11-13 20:31:30 UTC (22:31:30 Kyiv), return
code -6. macOS report `python3.13-2026-11-13-223132.ips` records
`___BUG_IN_CLIENT_OF_LIBMALLOC_POINTER_BEING_FREED_WAS_NOT_ALLOCATED`, with
`UnsafeSplitBackward0` destruction through `native_text::Leaf::release`,
`Scope::close`, and `NativeModel::end_text_replay`. This locates the failing
CPU cleanup path; the initial memory-corruption cause is not established. It
does not demonstrate a deadline, power or OOM failure. The Metal update does
not use that CPU text backward/replay path; this is not a fix claim for CPU.
`shutdown.json` confirms owned cleanup and protected-input preservation, no
supervisor errors. No game/training/caffeinate process remained at the audit.
The only diagnostic was the recovered 1,764 Blue win, with both policies and
native replay verified. Checkpoint 1,765 still requires its diagnostic.

The 402 reserved slots beginning at offset 1,647 remain consumed, including
unused slots; preview must recheck next offset 2,049. Policy SHA256
`17efd802d78bc8d3b3f5a36fd4c2360a3956a7da0d2c9022e36222c679f5fa67`;
resume SHA256 `ced4a88b3ac5672391959bc980a3329f09f2625baf7525deabe45fbb88396f7d`.
The existing retirement path must copy and fully restore that pair before
removing the reviewed run. Evidence before retirement: report/launch/shutdown,
`progress/ppo-batch-train-{000,001}.json`, first publication and the macOS crash
report. Meaningful findings are preserved here.

Retention-reviewed run: `run-20261113T124117Z-f8262954`.

**Prospective acceptance.** Measure recovery diagnostic separately from the
100-game collection/preparation, complete PPO (including device transfers),
checkpoint save/restore and final diagnostic; also report total elapsed wall
time. Success requires 100/100 admitted games, one published and restore-tested
update, verified final diagnostic and owned cleanup. Do not infer learning or
paired speedup from differently seeded games. Keep the existing 25% recommended
MPS allocator cap and CPU/system reserves. Bound device-resident immutable
prefix-plan caching so the single-history result is not extrapolated to 100
uncapped histories. Cache eviction/reupload must preserve exact constant bytes
and existing numerical acceptance; no minibatch, sequence truncation, reward,
loss, seed or optimizer change is allowed.

Implementation and validation outcomes, exact launch identity and measured
results will be appended after observation. No new games have been launched
at this entry.

**Implemented and local validation.** Explicit `--metal` is bound into the
existing cycle configuration before retirement or seed consumption; checked
native recipes select Metal only for training preparation/update. Actors and
diagnostic inference stay CPU. The existing transaction forwards the opt-in and
preserves rollback. Phase/backend fields and wall durations distinguish recovery
from the new measured cycle; per-game preparation overlaps collection.

The new immutable-plan cache keeps packed CPU constants and uses a model-owned
4 GiB LRU for uploaded prefix constants. Eviction/reupload preserves exact bytes;
oversized plans execute without retention. This bound does not include host
recipes or transient activations; the existing 25% MPS allocator cap, 48 GiB
physical headroom and system-pressure checks remain. Private forced-eviction,
recency, oversize and close-release controls passed. Seven numerical cases passed
(max output error 4.77e-7, gradient relative L2 5.27e-7, zero padding gradients).

The actual transaction was checked on the saved medium history: an injected
failure after one optimizer step restored exact model/Adam/gradient/RNG state,
CPU devices and foreach configuration. A subsequent complete 32-step update
passed (6.797 s after warmup, setup 3.664 s). This is integration/rollback
validation, not a new cold timing comparison or official admission.
118 existing numerical/native/model/history tests passed in 7.016 s. Of 72
existing controller/training/checkpoint/transaction tests, 70 passed; two legacy
controller fixture errors (`SimpleNamespace` lacks `stdin`) reproduced unchanged
on detached control `b9dbcf4`. They are recorded rather than counted as passes;
no gate or runtime ownership check was removed. Shell syntax/help/argument
forwarding and explicit backend forwarding checks passed. Evidence stays in
private `run-investigations/2026-11-13-metal-ppo/campaign-*` files; no test, probe
or benchmark source/artifact is included in the published diff.

The unmodified CPU checkout read-only preview restored 1,765 exactly, confirmed
100 training plus two diagnostics and next fresh offset 2,049 (202 reserved
slots required, none reserved by preview). AC power and 759 GiB free disk were
observed. Final candidate preflight and launch are still pending at this entry.

A private synthetic preparation integration also passed for CPU and Metal:
actual admission, encoding, native compilation/recipe validation and backend
selection ran on eight frames; parameter flags and receipt registration/file
contents were checked. Only capture loading/completion envelope were mocked.
This is wiring evidence, not a native game or official training receipt.

**Preflight attempt 1 (no launch).** Committed candidate `a3a5f89` ran
`./run-100-games.command 100 --metal --check` and stopped before reservation or
retirement: the historical source allowlist attempted to archive the newly
added Metal files from its old base commit. The complete source gate must be
extended with an exact reviewed transition for the changed/new numerical and
integration files, preserving final inventory/hash equality and existing native
acceptance. This is a missing release registration, not evidence that the
Metal numerical update failed. Evidence: `campaign-metal-preflight.log`.

The source-registration correction pins exactly 14 implementation paths from
CPU source `9ca85df` to candidate `a3a5f89`, including seven new Python/Metal
files. It extends the existing transition tables; final complete inventory/hash
equality is retained. Reused native and scenario receipts report that native
capture was not repeated. The actual current source and both accepted receipt
inventories pass; 84 private mutated/missing-file controls (14 files, two
mutations, three gates) all reject. The existing native-transition test passes.
Evidence: `campaign-source-transition-check.log`. No seed was consumed.

**Preflight attempt 2 (no launch).** Candidate `5d6277d` passed the source
transition but failed the temporary-map acceptance comparison because the
pinned receipt names the main checkout's map path while this separate worktree
has identical map bytes at another absolute path. The validator now accepts
that relocation only with the same exact filename, SHA256 and byte count,
while retaining receipt SHA, runtime identity, all native controls and semantic
v3-to-v4 map comparison. No gameplay, retirement or reservation occurred.
Evidence: `campaign-metal-preflight-2.log`.
Six private map-relocation controls passed: exact relocated bytes reached the
following source gate; wrong hash, size, filename, extra or missing binding key
were rejected. The source gate was a test-only sentinel; production checks stay
active. Independent review found no actionable issue.

**Live attempt 1 — dependency identity failure, no training admission.**
Candidate `ad23fdf` passed the full Metal preflight and was synchronized to
GitHub. `./run-100-games.command 100 --metal` created
`run-20261113T211303Z-e1815b91` at 21:13:03 UTC, parent 1,765; reservation offset
2,049/count 202. Retirement preserved/restored the initial and 1,765 checkpoint
pairs before deleting the analyzed CPU run. Recovery diagnostic seed
1,980,002,049 played to day 10 with a raw Blue terminal, but capture finalization
correctly rejected `missing_source_identity`: this isolated checkout lacked its
ignored `vcmi/` dependency directory. The client itself returned zero; this is
a technical capture failure, not a verified win. PPO never ran; zero games or
samples were admitted; checkpoint 1,765 and its diagnostic obligation remain.
The coordinator ended at 21:13:11 UTC; owned cleanup and protected-input checks
passed, supervisor errors empty. All 202 slots stay consumed, including the
unstarted100; retry must preview offset 2,251 and reserve fresh slots.
Evidence: the run's report/shutdown/worker-00.log and `campaign-live.log`.

Retention-reviewed run: `run-20261113T211303Z-e1815b91`.

A temporary symlink to the existing dependency was rejected by the existing
preparation containment guard (`Path escapes its project directory: vcmi`);
the newly created symlink was removed. The canonical `tools/prepare_vcmi.py`
workflow is preparing a real ignored dependency checkout at pinned upstream
`6901839ae9760dfcd1d5ff2c4aa4ef6968fee752`, with the root-owned patches/overlays.
No engine binary, canonical engine source or main-checkout dependency is changed.
The preparation/check result is required before another launch.
The canonical fresh dependency preparation and subsequent explicit
`python3 tools/prepare_vcmi.py --check` both passed. The ignored candidate
checkout now has the pinned VCMI revision, initialized upstream submodules and
exact root patch/overlay agreement. The accepted runtime binaries are reused
unchanged; no rebuild or new native mechanics claim is made.


## 2026-11-14 — Full Metal100 attempt: MPS limit, rollback and timing boundary

**Identity and actual execution.** The preceding prospective acceptance applies.
After the dependency correction and successful fourth read-only preflight,
`./run-100-games.command 100 --metal` ran from the clean, synchronized
`codex/metal-ppo` checkout at
`2fc67101b5b8d0ff75bedc91800f2e972d5dbcb9`. Run
`run-20261113T211610Z-a9566f61` started at 2026-11-13 21:16:10.620279 UTC;
parent checkpoint 1,765, parent optimizer steps 115,479. The accepted native
manifest remained `remove-telemetry-v1-attempt1/manifest.json`, SHA256
`53970605c4560bbe14b655442df373d3577efef3de0c5f824b95cfa22a74b573`.
Python 3.13 / PyTorch 2.9.1 / NumPy 2.3.5 ran on the M4 Max, 40 GPU cores,
128 GiB RAM. CPU actors and diagnostics, ten rolling actors, one thread each,
hero-management-v4, starting taverns, restored native calendar neutrals and the
365-day horizon remained unchanged. PPO retained 16 decisions/window, two
windows/minibatch, one epoch, its original losses and Adam settings. Only the
explicit Metal update backend and previously documented implementation changed.
The 25% recommended-memory MPS cap and existing resource guards remained active.

Reservation offset 2,251/count 202/cycle width 101 was consumed, including
unused slots. Recovery used engine seed 1,980,002,251 (policy base
1,982,004,502); training began at engine seed 1,980,002,352. Retirement preserved
and restored 1,765 before deleting the reviewed first live attempt. No source
or run configuration changed during the timed attempt; no concurrent profiling
or experiment was injected.

**Observed recovery and captures.** The pending 1,765 diagnostic passed exact
two-policy reconstruction and native replay: Blue won on day 87. Recovery
collection/preparation/verification took 209.487082 s (3m29s), separately from
the new 100 games. All 100 training games completed and were prepared in
1,871.353404 s (31m11s). Engine-reported outcomes were Blue 66, Red 27 and seven
cutoffs, with zero capture failures. Training replay and policy reconstruction
were intentionally not performed. The prepared cohort contained 273,913
learner samples and planned 8,583 optimizer steps. Per-game preparation summed
to 1,491.634949 s but overlapped collection; it must not be added to 1,871 s.
The small progress summaries report final maximum player-day median 106.5,
mean 130.54 and range 7–366; these are calendar positions, not verified game
lengths or evidence of improved play.

**Failure and preservation.** The coordinator stopped at
2026-11-14 03:15:41.350942 UTC. PPO progress recorded 6,022/8,583 completed
steps, phase `rolled_back`, elapsed 5,329 s (1h28m49s). The exception was
`MPS backend out of memory`: MPS allocated 26.05 GiB, other allocations
203.55 MiB, maximum allowed 26.88 GiB, attempted new allocation 1.99 GiB.
This establishes an allocator-limit failure; it does not show that all physical
RAM was exhausted or identify the precise tensor operation. The exception's
generic suggestion to disable the high-watermark limit was not applied.
The transaction reported exact model, Adam and RNG restoration. There was no
published update, no admitted game/sample and no 1,766 checkpoint or diagnostic.
The earlier recovered 1,765 Blue game must not be reported as a post-update test.

`shutdown.json` records return code 1, owned cleanup verified, protected inputs
unchanged and an empty error list. No owned training/caffeinate/VCMI process
remained. The retained 1,765 policy/resume hashes still match their launch
bindings. The subsequent existing command
`./run-100-games.command 100 --metal --check` passed exact restoration of model,
Adam, RNG, progress, sampler and opponent pool without gameplay or reservation.
It reports next fresh offset 2,453, 101 slots for a future 100 games plus final
check (no pending recovery), `reserved=false`.

**Timing and system conditions.** Coordinator monotonic elapsed time was
7,411.102674 s (2h03m31s); wall elapsed was 21,569.648805 s (5h59m30s), including
the recovery diagnostic. Their difference was 14,158.546132 s (3h55m59s).
macOS logs confirm sleep/dark-wake intervals and a lid/HID wake at 04:38:43
Kyiv. The same process resumed and advanced afterward. AC power was observed
at launch; after wake the Mac reported battery power (92%, later 85%). These
observations validate survival of this sleep episode, not recovery from process
exit or reboot. Monotonic elapsed is not GPU kernel time and may include awake
maintenance intervals or contention. The failed update has no final
`update_seconds` field; 5,329 s is the saved PPO progress clock, not a completed
update measurement. No fixed-power, controlled speedup can be inferred here.

The prior CPU100 admitted 257,260 samples in 8,061 steps, with about 13,750 s of
PPO progress time. This attempt used different seeds, parent weights, history
composition and sample count, and did not finish. Therefore neither a full-cycle
speedup nor the fivefold complete-update target is established. The earlier
6.41–6.49x result remains limited to its paired single-history offline trial;
it cannot be extrapolated to this 100-history workload. Playing strength and
post-update quality were not evaluated because no update was published.

**Read-only source investigation and next experiment boundary.** The 4 GiB LRU
bounds uploaded immutable prefix constants only. It does not bound transient
tree/GRU/CNN activations. Inspection found no demonstrated unbounded ownership
in that LRU. `Tree.forward` retains manual GRU backward intermediates until
return even for no-grad prefixes; `metal_map_delta.encode` materializes complete
neighborhood patch tensors for the selected prefix. Manual `ctx.saved` also
outlives managed saved-tensor cleanup while previous PPO graph references remain.
These are candidates for memory reduction, not a proven location of this failure: the stopped report
contains no allocation stack or failing shape. Different histories also require
separate current-weight prefix reconstruction; single-history cache locality
and map composition do not represent the full cohort. No speculative numerical
fix, guard relaxation or further campaign was launched after this failure.

All 100 binary captures remain for a targeted, isolated memory investigation.
The current launcher has no supported resume of already captured games after
process exit: completion/admission ownership is process-local, and a new cycle
retires reviewed artifacts, reserves fresh seeds and recollects. Reconstructing
private receipt registries or bypassing admission is not an authorized recovery
mechanism. Keep this cohort until the next authorized launch under the existing
retention policy; a targeted investigation should extract its reproduction before then.

Retention-reviewed run: `run-20261113T211610Z-a9566f61`.

Private evidence: the run's launch/report/shutdown, recovery verification,
`progress/ppo-batch-train-000.json`, training attempts and retained checkpoint;
`run-investigations/2026-11-13-metal-ppo/campaign-live-2.log`,
`campaign-metal-preflight-4.log`, `campaign-post-failure-check.log` and private
research notes. Public changes after the stopped attempt are documentation only;
no tests, probes, benchmarks, weights or raw captures are published.


## 2026-11-14 — Authorized preserved-cohort Metal memory correction (prospective)

The user authorizes diagnosing and correcting the Metal allocation failure, then
measuring a complete update on the same 100 saved games, targeting fivefold
speedup. The immutable source cohort is `run-20261113T211610Z-a9566f61`, parent
1,765, 273,913 learner samples and 8,583 planned optimizer steps. No new games,
seeds, retirement or official checkpoint publication are part of this offline
comparison. Hash-bound restricted loading and pure capture validation supply
diagnostic numerical inputs without fabricating live admission ownership.

Before experiments, distinguish active tensors from allocator-reserved heap
capacity; test immutable-cache fragmentation, unneeded no-grad auxiliary
retention and sparse CNN patch peaks. Preserve the 25% MPS memory cap, PPO/Adam,
full current-weight histories, minibatch order and all resource guards. Require
reproduction evidence, FP32 numerical/gradient checks and a complete paired
CPU/Metal update with setup, transfer, update and restore timing separated.
A partial update or extrapolation cannot establish the 5x target. Experiments
and raw results remain private; publish only necessary production changes and
compact findings. The user explicitly chose battery-powered measurements;
record power state and elapsed/active time rather than implying an AC benchmark.

## 2026-11-14 — Preserved 100-game update: memory completion, 6.945x timing, numerical failure

**Scope and identity.** This is the authorized offline investigation, not a new
training campaign. All runs restore the complete original checkpoint 1,765 and
consume the same 100 captures from `run-20261113T211610Z-a9566f61`: 273,913
samples, 17,165 windows and 8,583 optimizer steps. The source base is
`0d8b9549ca55723996e51861e4062210ce8e345d`; numerical sources equal the failed
campaign's `2fc6710` source. The initial-state file SHA256 is
`d663f65fb6493b3addade2316b27bd3f2ac463b612a2683864427683f3b778bf`;
the cohort-manifest SHA256 is
`cb1e878d5c283ab5a4387948d9a04fb937b09d1ad690528ce333c940794376c5`.
The comparison retains one PPO epoch, 16 decisions/window, two windows/minibatch,
Torch threads 1, native prefix threads 16, nice 19, the 25% MPS hard cap and the
4 GiB immutable-constant LRU. No new seed, game, admission, official update or
checkpoint publication occurred. Checkpoint 1,765 and next seed offset 2,453
remain unchanged. All experiments and raw results stay outside Git.

**Memory reproduction and correction.** The original failure requested another
1.99 GiB at the allocator limit. The original minibatch 6,023 contains window
ordinals 612/4,017 from episodes 4/27, with prefix boundaries 54,943/60,749.
A fresh isolated reproduction did not exhaust memory, but its allocation trace
identified a 2,136,644,352-byte sparse CNN neighborhood gather and a contiguous
copy with shape `[3709452,16,9]`, matching the failed request size. This is direct
allocation evidence, not proof that every retained byte belonged to that tensor.
The old isolated map path reached 8,581,425,920 active MPS bytes.

The private candidate fuses no-grad sparse CNN gathering/weight application and
pooled accumulation, range-checks compact int32 indices while retaining 64-bit
address arithmetic and an int64 fallback, removes redundant prefix-only dense
metadata, avoids unused no-grad tree auxiliaries, and clones small prefix
boundaries instead of retaining views of complete state storage. Grad-enabled
loss-window semantics and memory guards remain. An attempted destructive clear
of saved tree intermediates broke repeated backward with `retain_graph=True`;
it was reverted before full runs, and the repeated-backward control passed.
The compact actual-window control reduced observed map active peak to
723,980,032 bytes; CPU-plan constants for these two histories fell from
1,458,134,272 to 471,600,444 bytes. These are local measurements, not whole-run
memory claims. Candidate versus original Metal maximum prefix/output differences
were 2.09e-7/2.38e-7 and gradient relative L2 4.98e-8. Existing CPU-relative prefix
error around 3.40e-4 persisted. Both earlier actual-window attempts,
`fused-memory-failing-window` and `baseline-failing-window-control`, failed their
original 1e-4 absolute CPU prefix/output assertions and exited with errors;
owned cleanup passed. After the unchanged original Metal baseline also failed,
the subsequent local regression criterion explicitly added direct candidate
versus original comparisons, retained the 1e-4 CPU gradient-relative limit, and
bounded CPU output error by the measured baseline plus 1e-5. The compact control
passed this revised local criterion, not the original absolute CPU criterion.
This scoped revision did not change any full-update policy bounds or establish
full training equivalence. Seven synthetic GRU/tree/head/map numerical cases
and padding-gradient checks passed; their small scope does not establish full
training parity. An allocator soft-watermark control showed no benefit and was
not adopted. Offline shader compilation lacked MetalToolchain; actual Torch
runtime compilation succeeded without installation.

**Attempts and intermediate controls.** Trial A was deliberately interrupted
before PPO because the private loader incorrectly forced a full GC after every
game, unlike production. Exact rollback and cleanup passed. It was not an OOM.
Only the private forced collections were removed; ordinary automatic GC remained
unchanged, with passive timing instrumentation added identically to both later
workers. Trial B completed all 8,583 steps without OOM, with peak process RSS
67,940,974,592 bytes, but the user switched Low Power to Automatic during it.
Its combined 7,007.632081 seconds cannot establish a controlled speed ratio.

Subsequent 200-step controls on episodes 4/27 tested Metal matrix-kernel selection
and graphless indexing/accumulation. They used 13,453 samples, 842 windows and
an exact rollback after the partial update; their shuffle and normalization
differ from the full cohort. With matching Battery/Automatic endpoints, the
matrix preference improved 57.207140 to 50.810228 seconds (1.126x), and graphless
operations alone to 45.536845 seconds (1.256x). The combined control measured
40.821217 seconds, but Battery changed to AC between its endpoint observations:
the initially stated 1.401x ratio and matching-power assertion were explicitly
withdrawn. Its numerical result remains valid. Both compared snapshots were
Metal, not CPU; all 450 recorded conditional argmax choices matched, maximum KL
was 3.07e-12 and worst value RMSE 9.48e-7. Seven synthetic numerical cases and the
actual failed-window control also passed their registered baseline-relative
criteria. Loss-window padding, larger caches and GC-policy changes were not
adopted.

**Complete paired timing.** Trial C uses the frozen overlays
`graphless-gather-scatter`, `prefix-pruned`, `memory-candidate` and
`map-candidate/fused-int32`, with `PYTORCH_MPS_PREFER_METAL=1` in a fresh process.
Its exact source inventory is retained privately. The sequential CPU control
uses no overlays and unsets this preference. Both reuse identical worker/loader,
initial state, cohort, PPO settings and native acceptance. C exited and completed
owned cleanup before CPU started. Both runs observed AC/Automatic, powermode 0
and low-power false at start/end; manual observations during CPU also stayed AC.
The assistant did not alter power settings. Endpoint observations are not
continuous power-profile proof. Active/wall times agree within milliseconds.

| Full 100-game numerical update | CPU | Metal C |
| --- | ---: | ---: |
| Preparation, seconds | 1,072.344527 | 1,192.313358 |
| PPO, seconds | 18,781.578800 | 1,666.522696 |
| Combined, seconds | 19,853.923327 | 2,858.836054 |
| Peak process RSS, bytes | 31,978,635,264 | 48,270,114,816 |
| Samples / optimizer steps | 273,913 / 8,583 | 273,913 / 8,583 |

The measured combined ratio is **6.944757569x**; PPO alone is **11.269920801x**.
This is one same-input sequential pair, with no confidence interval, excluding
capture and final-game verification. It is not a measured complete campaign
speedup. Both workers returned zero with owned cleanup, exact initial/prepared
state, unchanged source bindings, finite CPU-restored tensors (39 model,
117 Adam, 31 gradients), restored optimizer settings and exact private save/load
roundtrips. C finished with zero active MPS bytes and 7,681,294,336 driver bytes;
coarse progress samples do not measure every within-step peak. The previously
observed native CPU abort on another cohort did not recur; this does not prove
that older issue fixed.

**Full-update numerical acceptance failed; no adoption.** The comparator passed
exact initial state, final RNG/sampler/progress/opponent-pool identity, Adam
configuration/step counts, finite tensors and conditional support checks. CPU/C
model relative L2 was 0.0242810851, maximum absolute difference 0.0704162493;
Adam moment relative L2 errors were 0.126554886/0.0435645293. Final gradient
relative L2 was 0.110466695, but these gradients were evaluated at different
trajectory states immediately before their final optimizer step and cannot
diagnose a backward implementation error directly.

Both policies use identical original reference Metal inference, with matrix
preference absent, on eight fixed 16-decision windows from each of earlier
episodes 17/100 (seeds 1,980,001,864 / 1,980,001,947). These fixtures are outside
the current cohort but already contributed to parent 1,765; they are not
never-trained evaluation data. The predeclared bounds remain agreement >=0.999,
mean/max conditional KL <=1e-6/1e-5, value RMSE <=0.001, and exact supports.

| Fixture | Matching conditional argmax | Mean / maximum KL | Value RMSE |
| --- | ---: | ---: | ---: |
| Episode 17 | 224/227 | 9.74293e-5 / 8.87122e-4 | 0.045416482 |
| Episode 100 | 210/223 | 6.90846e-5 / 0.002145940 | 0.027895365 |

Overall agreement was 434/450, not a count of games or entire decisions. Both
fixtures failed all four numerical bounds. The comparator persisted results,
then returned 1 for the failed assertion; cleanup passed. No bounds were relaxed.
This establishes numerical divergence, not poorer playing strength. The timing
target passed, but the combined speed-and-quality objective remains unfulfilled.
Production integration has not passed numerical acceptance.

Saved B also failed against the same CPU final state: 436/450 argmax agreement,
model relative L2 0.0237884390 and fixture value RMSE 0.04027854/0.03299335.
B/C model relative L2 was 0.00683829364. B already contains the memory corrections:
these comparisons exclude C's later indexing/kernel preference as the sole cause,
but do not exonerate the memory corrections or prove inevitable FP32 drift.
B's mixed-power timing remains excluded.

**Bounded divergence investigation.** Static independent reviews found no input,
normalization, sample-order, PPO formula or obvious gradient-formula mismatch.
CPU prefix/replay contributions are folded in deterministic order despite worker
threads. A same-gradient control repeated one saved final CPU clipped-gradient
vector (31 non-None tensors) for 8,583 Adam steps without PPO feedback; it did not
replay the actual gradient sequence:
CPU foreach=True equaled original None exactly; MPS foreach=True equaled False
exactly. MPS/CPU final model relative L2 was 4.29e-7, with moment errors
5.61e-8/1.47e-8. This does not explain the much larger full trajectory difference
and provides no reason to change optimizer settings.

Ten real-PPO gradient-only comparisons use common parent and completed-CPU
weights with data selected by original minibatch ordinals 1, 100, 1,000, 6,023 and
8,583. These are data positions, not historical weight snapshots. All 100 captures
were decoded for per-game GAE and exact global advantage normalization; only
nine histories were compiled. Seven of ten prefix comparisons exceeded the diagnostic
maximum-absolute 1e-4 bound;
maximum prefix/log error was 3.40e-4/5.63e-4. Nine gradient-relative errors ranged
1.22e-5 to 8.39e-5; the completed-CPU-weight / step-1,000 selection reached
2.37e-4. All clipping masks matched. Model/Adam/RNG and source checks passed;
zero optimizer steps occurred. These local discrepancies do not yet attribute
full drift. Next controls compare the original pre-memory Metal stack and a
forward-only CPU-order GRU overlay against the same bound vectors. A separate
no-contraction shader variant is prepared but unexecuted. Further full updates
require evidence of a material local improvement; the original policy failure
remains in force.

**Private evidence.** All paths below are relative to
`/Users/orsi/Projects/Heroes-artifacts/run-investigations/2026-11-14-metal-memory`:
`cohort-manifest.json`, `failure-minibatch.json`, the append-only private journals,
`metal-full-100-{a,b,c}`, `cpu-full-100-a`,
`comparison-full-100-{b,c}/comparison.json`, `full-b-versus-c-state.json`,
`adam-same-gradient-control`, `common-state-minibatches-c`, and each overlay's
manifest/diff. The frozen full-worker/loader SHA256 values are
`625384fca3dc812c9eac329b555ed0dc4312773cecbfdcbf593284d8bc95cd8f` /
`d846c385f353ef907f23578a76a7564c5f266de9be4636250c60a52adf6277a6`.
All raw state, captures, probes and diagnostic infrastructure remain private;
this public update records evidence and limitations only.

## 2026-11-14 — GRU expression order and FP32 contraction: revised full trial D

The original pre-memory Metal stack was evaluated on the same ten common-weight
PPO cases. Its CPU discrepancies were essentially the same as C's: direct
original/C maxima were prefix 4.17e-7, log probability 4.77e-7, value 1.19e-7 and
gradient relative L2 3.38e-7. All clipping masks matched. These sampled local
errors predate the memory corrections; they do not prove the cause of the full
trajectory divergence. This compares whole stacks, including absent/present
matrix-kernel preference. The control completed in 214.888 seconds, with exact
bindings and cleanup.

A first private shader variant aligns recurrent-bias grouping and hidden-state
interpolation with CPU expression order. All ten gradient errors decreased, but
five prefix comparisons regressed and maximum prefix error increased at both
common weight snapshots. Its 217.810-second control was therefore insufficient
to justify another full update by itself. Backward code was unchanged.

A separate variant additionally disables FP contraction with
`#pragma STDC FP_CONTRACT OFF`. The pragma covers all four forward and both
backward GRU kernels; this second variant is not forward-only. Shader SHA256:
`7f934168448120e9798474f3e847b3ccb0063d36dd6e87617e85970e881a6111`.
All ten common-weight cases improved versus C: prefix errors fell 6.24–33.77x
and gradient-relative errors 2.48–25.27x. Maximum prefix/gradient errors were
1.81794e-5 / 2.76056e-6 at parent weights and 2.36034e-5 / 2.16353e-5 at
completed-CPU weights. All diagnostic 1e-4 flags, clipping masks and gradient
presence checks passed. Maximum value errors were 1.10e-6 / 4.72e-6. The control
completed in 210.808 seconds, including 195.427 seconds of preparation; runtime
shader compilation and cleanup passed. No optimizer steps occurred.

This is material local improvement at both weight sets, not full-update quality
acceptance. The subsequent existing seven-case numerical suite also passed:
maximum output error 4.77e-7, gradient relative L2 5.31e-7, zero padding-row
gradients, with verified cleanup. All probes and candidate code remain private.

**Prospective full D measurement.** The predeclared next trial restores the exact
original 1,765 state and all 100 immutable captures, retaining all 8,583 steps,
PPO/Adam, shuffle, guards, threads, priority and cache settings. It reuses the
unchanged full worker and loader. The no-contraction overlay precedes the four
frozen C overlays; only the GRU shader's source content differs from C.
`PYTORCH_MPS_PREFER_METAL=1` and `PYTORCH_MPS_FAST_MATH=0` are passed before fresh
Python startup. The existing completed CPU control will be reused; this is not
another independent CPU replicate. Timing comparability and the original fixed
recorded-policy bounds remain required, without post-hoc relaxation. Production
integration remains pending this evidence.

D launched with unbounded ownership supervision and `caffeinate`, on observed
AC/Automatic, powermode 0, low-power false. Independent read-only audits verified
all 74 frozen input bindings and all 60 resolved startup source bindings. Its
initial-state file is byte-identical to CPU/C. The unchanged worker does not
record FAST_MATH in its own environment field; the retained launch and owner
argv explicitly record the executed zero value. No new game, seed, admission or
official checkpoint is involved. Results are pending at this entry.

Private evidence, under the same investigation directory:
`common-state-original-10`, `common-state-gru-order-10`,
`common-state-gru-no-contract-10`, `gru-cpu-order-no-contract/manifest.json`,
`full-d-plan.md`, `full-d-input-bindings.json` (SHA256
`60e5ceafa74c78df89a22aada5777ad18fe2e6c662751e82bcfc896179c004bc`) and
`metal-full-100-d`. The production transition proposal is private and unapplied;
it would bind both precision/kernel settings in existing startup and source
validation paths, with no new orchestration or published diagnostic tooling.

**Full D outcome and unchanged acceptance failure.** D completed all 100 histories,
273,913 samples and 8,583 steps, with supervisor return code 0 and verified owned
cleanup. Preparation took 1,157.210402 seconds, PPO 1,636.999826 seconds, combined
2,794.210228 seconds (46m34s). Against the same completed CPU control, the measured
combined ratio is **7.105379234x**, PPO-only **11.473170920x**. These timings
exclude capture and final-game verification. D is another Metal measurement
against the existing CPU control, not an additional independent CPU replicate.
Start/end power signatures match AC/Automatic, powermode 0 and low-power false;
intermediate observations stayed AC. Active and wall durations agree within
microseconds. Peak process RSS was 48,158,474,240 bytes. Initial/prepared identity,
all 67 final source bindings, finite CPU-restored tensors (39 model, 117 Adam,
31 gradients), optimizer-setting restoration, private save/load roundtrip and
owned cleanup passed. Final private state digest:
`330cd22f5fa3f8a7c7ac0a5b333669fa7976ee5d529a90cbf5cad08780839276`.

The unchanged comparator then ran in a separate guarded process, with both
policies evaluated by the same original reference inference. Prefer-Metal,
FAST_MATH and allocator environment overrides were explicitly absent.
The full comparison again passed state/RNG/sampler/pool identity, exact Adam
configuration/step counts, finite tensors and conditional supports, but failed
all four fixed numerical policy bounds on both fixtures:

| Fixture | Matching conditional argmax | Mean / maximum KL | Value RMSE |
| --- | ---: | ---: | ---: |
| Episode 17 | 224/227 | 7.41553e-5 / 0.001042062 | 0.042081047 |
| Episode 100 | 215/223 | 6.59979e-5 / 0.002282997 | 0.033871096 |

Overall conditional agreement was 439/450 (97.5556%), compared with C's 434/450.
Model relative L2 fell from 0.0242810851 to 0.0170347106, but its maximum absolute
error increased to 0.0777812991. Adam first/second-moment relative errors were
0.1747195395 / 0.0211049101; last pre-step gradient relative error was
0.0837634599 at different trajectory states. Both maximum KL values and the
second fixture's value RMSE worsened relative to C. Thus the local precision
improvement did not establish full-update numerical equivalence or improved
playing strength. The comparator persisted its complete failure report, returned
1 for the failed policy assertion, and verified cleanup without errors.

The fivefold timing target has been exceeded and the corrected memory path has
completed this cohort repeatedly under the original cap. The combined speed and
preserved-quality objective is still unfulfilled. No acceptance threshold is
relaxed, no production candidate is adopted, and no official checkpoint changes.
Full evidence is in `metal-full-100-d/result.json`,
`comparison-full-100-d/comparison.json`, their source inventories and shutdown
records. No further full update is scheduled on the basis of these results.

Before the final bounded diagnostic: reuse the frozen V2 common-state helper
with exact D arithmetic, the same nine histories, two common weight snapshots,
five data selections and two repeats. Inject the saved exact native CPU prefix
boundary into each loss window, while retaining the separately computed GPU
boundary for comparison. Compare outputs and real-PPO gradients to native CPU,
and compare the repeats directly after cleanup. This separates prefix error from
loss-window/backward error and measures local GPU variation; it does not test
full-trajectory reproducibility or waive the failed policy bounds. No optimizer
step or game is involved. The prospective record and outputs are under
`common-state-d-cpu-prefix-repeat2` in the same private investigation directory.

## 2026-11-14 — Localizing repeat variability in current-weight prefixes

The exact-CPU-prefix control completed 20 cases (ten selections/weight pairs,
two repeats), with all no-update/state/source checks and owned cleanup passed.
Injected boundaries, repeated logs, values, losses and clipping norms were
bitwise equal. Gradient repeat variation was at most 1.41e-7 relative; native CPU
gradient errors fell to maxima 1.54e-6 / 4.01e-6 at parent/completed-CPU weights.
However, the separately computed GPU boundaries varied between repeats by up to
1.14e-5 / 3.33e-5. This provides an actionable local repeatability issue, not a
proof that it explains the failed full-update comparison. The maximum difference
was episode 40, completed-CPU weights, boundary frame 50,755.

A single-history localization freezes each stage's operands. Its first attempt
failed the existing native compile gate because the private probe omitted
`train().requires_grad_(True)` preparation. It stopped before any GPU stage,
returned 1 and cleaned up. The separately hashed correction adds only that
existing preparation; measured operations remain no-grad. The corrected attempt
completed in 28.67 seconds with unchanged inputs/model/Adam/RNG/source and no
gradients. Three fixed-input sparse CNN pooling calls differed by up to 3.28e-7
before the global-map projection. Three complete GRU calls on the same encoded
input `[1,46356,96]` produced bitwise-identical full sequences and boundaries.
This directly locates variation in the pooled atomic additions for this case;
it is not an exhaustive audit of all preceding operations.

A private pool-only candidate stably groups affected cells by map/bin and gives
each bin/channel one writer, retaining anchor-first, separately area-divided
additions. It preserves checked int32/int64 handling and introduces no im2col or
large activation tensor. Fifty-four synthetic map comparisons passed the existing
2e-6 CPU tolerance (maximum error 9.69e-8); 24 nonempty sparse CSR dispatches
actually exercised both index variants. Frozen pooling repeats and int32/int64
results were bitwise equal. Empty bins, selected-prefix offsets, zero rows and
no-grad checks passed, followed by the seven existing regressions. Those seven
regressions alone do not cover sparse pooling. All bound inputs and cleanup
checks passed; the small sampled memory values are not true peak measurements.

Six ordinary episode-40 prefix passes (parent/completed-CPU weights, three each)
showed that pool-only correction was insufficient. Parent boundaries were
bitwise repeatable, but completed-CPU boundaries still varied up to 8.11e-6;
pooled/map outputs also varied slightly. CPU-error ranges overlapped earlier D
draws, and the parent's maximum error was slightly worse. No general accuracy or
full-update claim follows, and no full run was launched from that result.

The next frozen-input control isolated two object scatters, both anchor
convolutions and adaptive pooling. It completed in 30.43 seconds with all
state/source/no-gradient and cleanup checks passed. The remaining variation was
directly reproduced in `object_delta.index_put_(accumulate=True)`: maximum
1.19e-7 with identical values and zero destinations. All changed rows belonged
to 32 of 119 targets receiving three heterogeneous contributions; single- and
double-hit targets were identical. The anchor scatter, convolutions and adaptive
pool were bitwise repeatable in these controls. This is sampled attribution,
not a claim that no other history can expose another source.

The next private candidate will replace only the demonstrated delta-object
scatter with a stable per-target sum over the already gathered/scaled FP32
values, alongside deterministic pooling. Anchor scatter remains unchanged.
It must retain partial-prefix boundaries, checked index widths, zero outputs,
resource limits and original PPO. Before considering another complete update,
require direct captured triple-hit checks, synthetic/regression gates, then the
same real-PPO common-weight cases with ordinary computed prefixes and repeated
execution. Repeatability and CPU numerical agreement are separate criteria;
neither waives D's failed full-policy bounds. Production code and checkpoint
1,765 remain unchanged.

Private evidence: `common-state-d-cpu-prefix-repeat2`,
`prefix-localization-d-{one,two}`, `deterministic-pool-validation`,
`pool-prefix-repeat-candidate-one/all-D-boundary-comparison.json`,
`upstream-map-localization-one/scatter-collision-analysis.json`, their plans,
source manifests, results and shutdown records. All are under the previously
identified private investigation directory; no probe or raw tensor is published.


The combined pool/delta candidate passed its three prerequisite gates before
real-history testing. The full captured 161,319-row scatter and all 119
three-contribution targets matched explicit source-order CPU sums bitwise in
three repeats for both index widths. Actual GPU operands passed readback
immutability checks. All 54 map comparisons passed the unchanged 2e-6 tolerance
(maximum 9.69e-8), including 24 nonempty pooling and 16 object-sum launches;
partial-prefix, offset, zero-row and no-grad controls passed. The seven existing
regressions also passed. All three processes exited zero, verified owned cleanup,
and preserved all 91 bound inputs. These are local operator results, not a
full-update equivalence claim. Evidence: `deterministic-pool-delta-validation`;
candidate manifest SHA256
`2e9e65a1cc64d355c49477d7b5cf0846934109bdc7666ea08416df36682a2aee`.

Before the next real-history control, freeze V3 helper SHA256
`2e4a7052026506a82a8bf3c116754556ea89262ff2f4ffa6b9dbe5424bdb4215`
and its source/cohort/control bindings. Recompile the same nine selected
histories using the combined candidate; evaluate the same ten common-state
real-PPO cases with ordinary computed prefixes three times. Reuse original
100-game advantage normalization and saved native CPU controls; perform no
optimizer steps. Compare direct repeatability separately from CPU agreement.
D's injected-CPU-prefix runs supply additional computed-boundary draws only;
their losses and gradients cannot serve as ordinary-prefix D controls. Use
D's ordinary-prefix run for that comparison. GPU allocation was granted after
the prerequisite cleanup; no new full update or game was scheduled.


The combined real-history control completed all 30 evaluations (ten cases,
three repeats) in 224.66 seconds, including 197.38 seconds preparation. All
existing 1e-4 local diagnostic flags passed, with exact clipping masks, unchanged
state/source bindings, zero optimizer steps and verified cleanup. All ten cases
had bitwise-identical computed boundaries, logs, values, losses and clipping
norms across repeats. Backward repeat variation remained small: maximum gradient
relative L2 6.29e-8 at parent weights and 1.56e-7 at completed-CPU weights.

This removed sampled forward variation but did not generally improve CPU
agreement. Maximum CPU prefix errors were 1.3471e-5 / 2.7418e-5 at the two weight
sets. Three case errors exceeded every observed D boundary draw, only one was
below all D draws, and six overlapped. Ordinary-prefix gradient error improved
in seven of ten cases and worsened in three; the worst completed-CPU step-6023
case increased from 2.16353e-5 to 2.22308e-5 relative L2. Consequently no full
trial E was scheduled. Evidence: `common-state-pool-delta-repeat3`, its frozen
bindings, comparison report and cleanup record. Sampled repeatability does not
waive the failed full-policy bounds or establish retained playing strength.

Before another narrow diagnostic: separate deterministic but differently rounded
map algebra from ordinary dense Metal arithmetic. Inspect the worst boundary in
the completed-CPU step-6023 case and define a bounded nonzero-start frame slice
before it. Compare native CPU map-global features with dense and sparse Metal
features for identical raw maps and weights; keep tree/GRU error separate. The
existing dense path must receive a freshly constructed dense plan, never a pruned
sparse plan reinterpreted as dense. Freeze the selection and helper before any
numerical execution. No optimizer, new game or full update is part of this
control, and slice timings cannot predict a complete dense-prefix update.


A complementary control reuses the already frozen episode-40 encoded sequence,
initial hidden state, CPU-final GRU weights and saved repeated GPU outputs from
`prefix-localization-d-two`. Run only CPU `torch._VF.gru_cell` sequentially for
each vector with one thread and no gradients, matching native `transition`;
a batched `nn.GRU` would change the reference input-affine arithmetic. Compare
full sequences and the final boundary with preserved hashes and owned cleanup.
No history preparation or fresh GPU execution is needed. This isolates recurrent
arithmetic on common GPU-produced input, not the episode-4 map discrepancy.


The frozen-input CPU GRU control completed without errors: 46,356 per-vector
transitions took 0.663 seconds, with 1.154 seconds total worker time, zero fresh
GPU execution, unchanged inputs/source/RNG and verified cleanup. Against the
identical-input saved Metal sequence, maximum absolute error was 1.66297e-5
(relative L2 2.75961e-6); the final boundary error was 1.45435e-5 (relative L2
3.72050e-6). All three saved Metal repeats and their ordinary call were byte
identical. Thus a systematic CPU/Metal recurrent-arithmetic difference remains
independently of map encoding. This is the episode-40 frozen sequence, not the
episode-4 map case, and does not attribute the error to one GRU sub-operation.
Evidence: `frozen-gru-cpu-reference` and its plan, bindings and shutdown record.


The final bounded map control completed in 29.08 seconds, including 25.93 seconds
preparation, with exit zero and verified cleanup. The predetermined episode-4
interval `[54879,54943)` contained 31 map-bearing jobs and 16 unique maps.
Its ordinary sparse-prefix comparison retained all 18,157 required unique maps
and their original anchors; the dense path was independently built for the
64-frame slice. Maximum map-global feature differences were 3.57628e-7 for
native CPU versus dense Metal, 5.36442e-7 for CPU versus sparse Metal, and
2.98023e-7 between the Metal paths. All 87 bound inputs, model/Adam tensors and
parent/global RNGs remained unchanged, with no gradients or optimizer steps.
Timings covered different amounts of work and are not a speed comparison.
These local map differences do not attribute the complete recurrent drift.
Evidence: `map-arithmetic-one`, `map-arithmetic-preparation`, their frozen helper,
source bindings and shutdown record.

Investigation outcome: the memory-corrected candidates completed the original
100-game update under the original resource limits, and full trial D exceeded
the fivefold complete-update timing target at 7.105x. Near-exact final-policy
reproduction failed, and subsequent deterministic-prefix corrections did not
generally improve CPU agreement. No full trial E, new game, training admission
or checkpoint publication followed. No production code was adopted. The
preserved-quality objective remains unresolved: the recorded-state numerical
gate is not a playing-strength test, and neither retained nor reduced playing
strength has been demonstrated. A future strength evaluation needs prospectively
specified fresh games, balanced seats and statistical criteria; small smoke games
or a post-hoc relaxation of the failed numerical bounds cannot supply that proof.
Only this research record and README are synchronized; all diagnostic code,
private candidates, tensors, weights and raw captures remain local.

## 2026-11-14 — User-owned execution outside Codex

The user establishes the first rule in `AGENTS.md`: project scripts and all
training are launched personally outside Codex by default. The agent prepares
scripts and commands; an assistant launch needs an explicit exception for that
run. The rule also covers saved-data PPO updates, profiling, benchmarks and
diagnostic experiments, including detached, delegated and remote execution.
Earlier launch permissions do not carry forward. Requested read-only status
inspection and lightweight status instrumentation remain possible, while
unattended waiting or repeated monitoring is not the default. Runtime and
validation gates remain in force. README now reflects the same policy.

This is a documentation-only change, applied to the working checkout and the
current Metal experiment checkout. Validation is limited to reviewing the text
and Git diffs; no project script, experiment, training or monitoring loop runs.
No model, checkpoint, seed allocation, runtime configuration or launcher changes.

## 2026-11-14 — Consolidate main and adopt measured Metal PPO

The user requests one current `main`, removal of obsolete branches and working
copies, clear navigation to core implementation/research findings, and GitHub
synchronization. They also explicitly choose Metal and accept the observed
CPU/Metal numerical differences. This supersedes the earlier non-adoption
decision; the failed numerical comparisons remain recorded and do not become
passes. No playing-strength equivalence is inferred from this decision.

GitHub `main` was `d785587` (October 30). The current accepted line `6050886`
contained all 118 subsequent commits, including the user-owned launch policy.
Local `main` fast-forwarded from `989dcd0` to that line; merge `78e3788` preserved
the separately committed policy history from `2a64930` with no content changes.
The only merge conflict was a duplicate journal entry already present in the
latest journal; all research records were retained. Existing published tests and
validation assets are inherited unchanged, not newly authored for this cleanup.

Before cleanup, local branch/remote/tag refs and worktree inventory were saved
under private `Heroes-artifacts/repository-archive/2026-11-14-main-consolidation`.
`branches.bundle` verified as a complete Git history; editor checkpoint refs
were excluded, and this backup is not published. Deletion requires checking
active processes, ignored state, runtime links and unique changes first. Existing
hash-bound source, validation and historical-launcher paths are retained rather
than breaking continuation through cosmetic relocation.

Adopt precisely full trial D's six numerical files, identified in the preceding
measurement, plus its existing-controller environment/source-identity transition.
Do not include the later unmeasured deterministic pool/delta variant. The manual
`run-100-games.command` now defaults to Metal PPO, with explicit `--cpu` fallback;
gameplay and diagnostic inference remain CPU. All PPO settings, 25% MPS cap,
4 GiB immutable cache, resource guards, full-state restoration and exact source
checks remain. This adoption does not launch training or publish checkpoint 1,766.

Validation for this integration is static only under the user's launch policy:
verify exact measured file hashes, inspect diffs and source-gate transitions,
parse syntax without importing project code, and prepare the existing preflight
command for the user. Previously recorded numerical/full-update evidence applies
to the exact D files; a newly executed integrated preflight or test suite is not
claimed. No project script, numerical experiment or training is run by Codex.

Cleanup outcome: 63 obsolete worktree directories were removed after preserving
43 ignored/private-state entries by same-volume rename and one tracked-change
patch. One already-missing worktree registration was pruned. The main checkout
and two detached dependency snapshots remain: `Heroes-metal-ppo` is referenced
by the saved Metal investigation, and `runs/perf-worktree` supplies historical
runtime links. Neither is an active development branch. All 65 unique original
worktree HEADs are preserved in verified `worktree-heads.bundle`; branch tips
and tags are preserved separately in `branches.bundle`. The private
`cleanup-result.json` records exact source/destination paths and preserved inodes.
Checkpoint 1,765, seed lineage, locked Python, accepted runtime/native builds,
saved 100-game cohort and open replay/game applications were not removed.

The concise README now leads to `docs/architecture.md`; dated README chronology
is retained in `docs/project-history.md`. The restriction register identifies
v4 calendar spawning as restored and starting taverns as still enabled. Static
checks passed for six Python ASTs, seven pinned source digests, both launcher
shell syntax checks, local navigation targets and whitespace. No project code
was imported to perform these checks. GitHub publication and branch retirement
are recorded below after remote read-back.

Publication and retirement completed: commit `8185957` was pushed explicitly to
`origin/main` and its exact SHA read back from GitHub. All 76 other remote heads
were removed in one atomic, per-ref lease-guarded push after verifying each tip
against `branches.bundle`; 81 non-main local branches were then removed against
the same archived inventory. Remote and local branch listings now contain only
`main`. The previously open PR #30 was closed because its head was already
included in main's history through the accepted integration; its historical
embedded-engine proposal was not reapplied. No GitHub comments were sent.
Private `branch-retirement-plan.json`, `branch-retirement-result.json` and command
logs preserve the exact retired refs. No weights, raw artifacts, Git backup
bundles or editor checkpoint refs were pushed. The two detached dependency
snapshots remain unchanged and the root checkout is the single development
location. The final publication records this outcome in a documentation-only
follow-up; no launch, preflight, test suite or numerical workload was executed.

## 2026-11-15 — One authorized diagnostic from the consolidated main checkout

Question: how does the latest published policy play one fresh map under the
current restrictions? The user explicitly authorized the assistant to launch
this one game. This exception does not authorize training, PPO updates, another
game or retirement of the saved 100-game investigation cohort.

Read-only preflight on source `f5405308a7ebe42f5a9c7fc5855782c726ff98df`
passed: `select_cycle_mode()` followed by `ten_inspect()` restored checkpoint
1,765 with model, Adam, RNG and progress; no pending diagnostic remained.
Weights digest: `5e88fc9d236204e57fd32f73cd003f19b675a7f6f9d0c25ef6ce376d04cda35e`.
The accepted runtime is `local-v8-sleep-resume-v1`, with the locked
`next-launch-numpy235-v1` Python and `remove-telemetry-v1-attempt1` native learner.
The selected map is `hero-management-v4.vmap`, digest
`a076b4dca6410b5dc16388dc47333b036323a94c8fe412500cd9f31b78bcc833`.

Existing standalone CLI launchers still select historical checkpoint 1,753;
the campaign CLI would include training and retention. Therefore invoke the
existing current-cycle inspection and single-game collection/verification
functions directly, without changing those launchers or bypassing source,
runtime, checkpoint, seed or replay checks. Both frozen players use 1,765;
preserve hero-relative policy support, own-army information, artifact prohibition,
starting taverns and native calendar spawning. Preserve the current 365-day
horizon and 158,720 ordinary plus 5,120 reserved choices per player.

Acceptance: exactly one fresh diagnostic capture, exact reconstruction of both
policies, verified native replay, unchanged protected checkpoint inputs and
closed owned processes. The standalone evaluation namespace preview is engine
seed 1,960,000,000 and policy seeds 1,962,000,000/1,962,000,001; reserve durably
under the machine activity lease before capture. Store the bound plan and raw
evidence privately under `selfplay-one-game-2x`, with zero training admission or
optimizer updates. Record actual seeds, timing, result and validation afterward.
Describe observable adventure-map behavior; combat remains BattleAI. One
self-play game cannot establish improvement or validate Metal training quality.

Observed: the single attempt `selfplay-one-game-2x/run-20261114T220914Z-19c6bccb`
ran from clean source `4ede04e`, using the previewed seeds (standalone reservation
offset 0). Its private `plan.json` binds the exact checkpoint, runtime, rules,
source and protected inputs. Existing `collect()` and `verify_capture()` APIs
ran under the machine activity lease and `caffeinate -is`, with guarded native
processes and parallel native replay. The legacy single-game API requires finite
deadlines, so this diagnostic used its 7,200-second maximum; this does not alter
the unlimited default for training campaigns. No source code or launcher changed.

The game ended naturally with Blue winning on day 72. Both players used the
same frozen 1,765 weights. Red made 1,674 policy choices and Blue 1,524. Capture
and preparation took 87.556 seconds (manifest engine wall time 65.497 seconds);
the complete attempt including verification and cleanup took 137.963 seconds.
Native replay took 16.794 seconds within the overlapping verification phase,
not an additional duration to add to the total.

`report.json` passed; `episode-000001/evaluation-policy-reconstruction.json`
records zero maximum absolute difference for both reconstructed policies,
including 3,040 Red and 2,611 Blue conditional distributions. Native replay
checked all 20,276 client packets, 6,400 messages, 3,806 requests, 3,198 responses
and 41,120 transactions, with zero unchecked entries and no divergence. Both
history streams ended naturally on day 72. `shutdown.json` confirms unchanged
protected inputs/plan and clean owned-process shutdown. An independent read-only
audit rehashed all 686 protected files and confirmed no owned game/replay process
remained. There were zero PPO updates, admitted games or admitted samples; the
official checkpoint remains 1,765 and the saved training cohort was not retired.

Behavioral observations from the sealed player histories: Blue acquired Red's
remaining town on day 65; Red's day-72 notice explicitly records
`heroes.baseline.seven_days_without_town`, while Blue's terminal notice records
`heroes.baseline.all_opponents_defeated`. This is a native victory, not a cutoff
or a validation error. The policies issued six/five hero hires and five/three
troop recruitments for Red/Blue, but only two/zero construction actions. Much of
the action stream is interface opening/closing, so the victory does not establish
efficient economy or coherent long-term planning. These are observations of the
existing policy, not evidence of improvement since the previous checkpoint or
of learning from the newly adopted Metal backend. Full training after source
consolidation and the test suite were not run in this task.

The decisive sequence is more specific than a general army-strength advantage.
On day 65 Blue attacked the town at (4,7) with 63 pikemen and four lizardmen
against 42 halberdiers, capturing it with 38 pikemen remaining (Blue history
sequences 13,879–13,895; Red 16,043–16,046). On day 67 Red destroyed that last
Blue field army at (0,16), retaining 60 pikemen and one griffin (Red
16,254–16,259; Blue 14,146–14,151). Blue recruited 42 halberdiers in the captured
town on day 67 and hired a replacement hero on day 68. Red submitted no
`move_to` targeting a town between that battle and the day-72 terminal; on day
69 its hero passed (3,8), near the lost town, before moving away. Red still had
a substantial army when the seven-day loss rule triggered. This identifies a
failure to act on the town-loss deadline in this trajectory; it does not prove
that a recapture attack would have won. Battle results came from BattleAI.

Useful troop-management behavior was also observed within the restricted action
space: Red combined troops into a 66-pikeman hero army through a garrison swap
on day 37. Blue bought 70 pikemen on day 35 and moved them out of the garrison
on day 47; this army later took the town. The twelve-day mobilization delay is
another reason not to equate successful actions with efficient planning.

## 2026-11-15 — Prepare an isolated three-framework PPO comparison

Question: how much time does a complete current-policy PPO update take with
the existing optimized PyTorch/Metal implementation, Python/MLX and Swift/MLX?
The user requested the separate branch `codex/ppo-framework-comparison`, based
on main `d131d92100363b213787e1673163d11773135649`, and will launch all scripts.
The user clarified that collection/inference should be shared across all three
updates and that the PyTorch reference is the current optimized Metal backend,
not a new stock-MPS implementation. No experiment execution, merge or push is
authorized. Publication requires an explicit instruction; journal work does not
grant publication permission.

Design: collect ten fresh native games from complete checkpoint 1,765 on v4,
with the existing 365-day/158,720+5,120-choice rules and balanced learner seats.
Reuse the real capture registration and capture-only preparation path, with a
private experiment launch and independent consumed seed ledger (engine namespace
1,990,000,000+, policy namespace 1,992,000,000+). Do not register this experiment
in the official training lineage, retire existing captures or run a final game.
Each optimizer trial restores identical weights, named Adam moments/counters
and exact minibatch order; all targets and complete histories are preserved.
The numerical kernel may update disposable benchmark copies only. No trial
publishes 1,766 or remints official admission from saved evidence.

Implementation is isolated under `experiments/framework_comparison` and
`native/framework-comparison`, with manual preparation and three PPO launchers.
MLX ports include the complete semantic tree, map CNN, recurrent memory,
conditional heads, PPO loss, gradient clipping and restored Adam. Prefixes are
rebuilt with current weights and detached at the original window boundary.
Python MLX 0.31.1 and Swift MLX 0.31.4 share core version 0.31.1; SwiftPM pins and
CPython 3.13/macOS-arm64 wheel hashes are fixed. Dependencies/build artifacts and
results go to private storage. The production numerical sources are unchanged.

Prospective acceptance: complete ten-game capture/preparation; exact bound input
identities; first scheduled minibatch finite with maximum log-probability/value
differences <=0.02/0.002 against CPU behavior before any Adam step; all declared
samples, epochs and optimizer steps completed; finite final weights/moments;
unchanged official checkpoint/protected inputs and clean owned-process shutdown.
Initial sanity is not gradient or quality equivalence. Report any failed gate
separately, preserve its evidence, and never time a partial update as successful.

Timing boundaries: report collection/preparation, portable export, backend
preparation, synchronized optimizer work, saving and whole process separately.
The portable export preplans MLX windows; the unchanged Metal path still plans
inside PPO. Charge conversion to MLX when deriving complete pipeline times and
retain the actual shared-workflow total. MLX dense prefix maps and the optimized
Metal sparse implementation compute equivalent model functions using different
algorithms; timings cannot isolate language overhead. One run per backend cannot
establish stable acceleration or training quality. See `docs/framework-comparison.md`
for the complete user-run protocol and artifact layout.

Validation performed so far is static only: Python AST parsing, shell syntax,
source review of tensor layouts/GRU/PPO/Adam and dependency metadata. Swift is
being compiled without running the executable; record its final result below.
No game, inference probe, replay, PPO update, checkpoint publication or benchmark
has been launched by the assistant for this experiment. Source changes remain
local and uncommitted pending the user's publication boundary; the existing
clean-committed-checkout gate remains intact.

Static validation outcome: all four Python modules, embedded setup Python and
four shell launchers parse successfully; local documentation links and whitespace
checks pass. The full Swift release build compiled and linked MLX and the
experiment executable. Initial compiler errors in a large report expression and
the default-device API reference were corrected before the successful build;
no executable was run. Cross-review checked GRU gate equations, map tensor
layouts/pooling, hierarchical heads, loss clipping and named Adam inactive-gradient
semantics. This is source/compile evidence only, not a numerical pass.

A further static deployment check found that SwiftPM did not build the required
Metal shader library. `xcrun metal --version` also reported the optional Metal
Toolchain missing; no toolchain download or system upgrade was attempted. The
pinned Python `mlx-metal` 0.31.1 wheel contains `mlx/lib/mlx.metallib`; the Swift
vendored core matches official MLX v0.31.1 revision
`ce45c52505c8158ea48d2a54e8caae05efd86bfe`. Its native loader searches beside the
executable for `mlx.metallib`. User setup now verifies the installed wheel RECORD,
copies that same-core precompiled library beside the Swift executable and writes
a size/hash receipt; launch checks bind the binary, receipt and library. The
macOS-26 wheel inspected without installation had library SHA256
`198488eb61359e953580a9c4530400feee1a06dd2f28a930a6ffa58aec66a597`.
The actual setup command and GPU loading remain unexecuted in this task.

The user subsequently authorized exactly one local commit of the prepared
experiment, launchers, dependencies and documentation on the experiment branch.
This authorization explicitly excludes merge, push and execution. The final
static checks and compile-only Swift result are included in that local snapshot;
main and GitHub remain at their preceding revision. The first numerical results
are still pending the user's launches.

## 2026-11-15 — Correct the comparison dependency lock location

The user's first manual attempt on experiment commit
`59f0aa5515700926ecec022b5347f3854948dd04` installed MLX/MLX-Metal 0.31.1 and
NumPy 2.3.5 and built the Swift release executable successfully (165.26 seconds,
per the supplied terminal output). The colocated Metal resource setup completed.
Collection then failed inside `ten_inspect()` before creating a study or reserving
seeds: `verify_runtime_sources()` requested
`dependencies/framework-comparison-mlx.lock` from historical base commit
`b47c1b9aed1bbe8d2c52d87a02bcdac3e4931617`, where that new file does not exist.
The subsequent Metal PPO command failed because no shared cohort existed.
These are preparation failures, not game outcomes or numerical failures: zero
games and zero optimizer steps ran in these attempts. The private experiment
directory contains setup outputs but no study pointer or experiment seed ledger.

Cause and proposed correction: the existing production source inventory includes
every file in `dependencies/*`. The new isolated MLX lock was placed there even
though production capture uses the unchanged Torch environment. Move the lock,
without changing versions or hashes, to
`experiments/framework_comparison/mlx.lock` and update the setup reference.
Keep all production inventory, historical hash and clean-checkout gates intact.
Acceptance for this correction is unchanged lock content, exact agreement of the
production inventory with its reviewed historical sources, and valid launcher
syntax. No game, PPO update, replay or numerical probe is authorized here.
Installed dependencies and the existing Swift build can be reused.

Local validation passed: the unchanged `verify_runtime_sources()` accepted all
225 current source hashes when evaluated with the existing inventory functions
and constants, without importing launch controllers or neural modules. A changed
PPO hash was still rejected. The relocated lock is byte-identical to the file in
`59f0aa5`; shell syntax checks for all four launchers and `git diff --check`
passed. Independent source review found no other comparison files entering the
production inventory. This establishes the reported source-gate correction only;
collection and numerical execution remain untested. The fix is local and
uncommitted. A further local commit needs fresh user authorization before the
user can pass the unchanged clean-checkout gate; no merge or push was performed.

The user then explicitly authorized one additional local commit of this lock
relocation, setup reference and documentation. Merge, push and workload execution
remain outside that authorization. The next manual step is the ten-game
preparation command; repeating successful dependency setup is unnecessary.

## 2026-11-15 — Analyze the first user-run three-framework comparison

Question: does the observed Metal timing advantage reflect equivalent PPO work,
and which implementation differences could explain it? This analysis reads
existing artifacts and source only; no new game, replay, probe or PPO workload
was authorized or launched. Evidence is under
`/Users/orsi/Projects/Heroes-artifacts/framework-comparison/study-20261114T225228Z-f35d0fb9`:
`launch.json`, `report.json`, `bundle/manifest.json`, `results/*/trial.json`,
`results/*/backend/result.json`, and `comparison.json`. All trial source commits
are `8d502e0ec26f13960cd6063b56699214468b3e7c`. The common manifest SHA256 is
`30368956be6e0db22d6787c89ef1dc1d1235d542a66fbbb4db65447dfdc5e73d`;
initial weights SHA256 is
`5e88fc9d236204e57fd32f73cd003f19b675a7f6f9d0c25ef6ce376d04cda35e`.
Manifest, result and final-weight artifact bindings were checked during analysis.

The user collected ten games from complete checkpoint 1,765, with experiment
seed offset 0 (engine base 1,990,000,000; policy base 1,992,000,000), v4 rules and
the existing 365-day horizon. All ten were captured: six Blue wins and four Red
wins, with zero cutoffs/capture failures. These are engine-reported capture-only
outcomes, not replay-verified evidence or measurements of the updated policies.
Collection/preparation took 140.375 seconds and portable export 19.761 seconds.
Each disposable update completed 10,885 samples and 343 Adam steps. Correction
to the earlier design description: actual `tools/run_selfplay_training.py:PPO`
specifies one epoch, and all three implementations used that same configuration;
the earlier statement of two epochs came from the class default and was wrong.

All trials passed, with protected inputs unchanged and clean owned-process
shutdown. Host snapshots agree on Apple M4 Max, 128 GiB RAM, AC power and
Automatic mode. Runtime versions were Torch 2.9.1, Python MLX/MLX-Metal 0.31.1,
Swift MLX 0.31.4 with core 0.31.1. In Metal/Python-MLX/Swift-MLX order, preparation
times were 5.855/1.352/0.670 seconds, update times 65.009/202.825/251.318 seconds,
full update times 70.864/204.177/251.988 seconds, and supervised process times
75.422/207.902/258.925 seconds. Full-update ratios against Metal are 2.881x and
3.556x. There is one trial per backend; no component profile, thermal trace or
randomized repeated comparison establishes attribution or stable speedup.

Initial prediction checks covered 32 samples before Adam: maximum log-probability
and value differences against CPU behavior were 4.77e-7/5.96e-7 for Metal and
3.81e-6/3.64e-6 for both MLX ports. Aggregate value losses were
0.500390732441/0.500391440547/0.500391431426; approximate KL values were
0.002290436242/0.002290355779/0.002290353677. Final model relative L2 differences
against Metal were 0.000245365 (Python MLX) and 0.000236256 (Swift MLX), with
maximum absolute parameter differences 0.0191512 and 0.0184053. Close aggregate
metrics and a first-minibatch forward check do not establish full-gradient or
playing-strength equivalence. No official checkpoint was published.

Source audit found no concrete discrepancy in the reviewed model/PPO equations,
GAE, schedule, restored Adam or full current-weight prefix/detach contract.
Execution strategies differ: Metal uses fused GRU forward/backward, custom tree
gradient kernels and sparse current-weight prefix-map deltas; MLX uses composed
operations, explicit recurrent loops and dense map CNNs without graph compilation.
These are plausible speed mechanisms, not measured per-component attributions.

The timed work is also unequal beyond intentional kernel optimization. Both MLX
ports check all model weights and Adam moments after every minibatch; Metal does
that full-state check after its update timer (while retaining per-step loss and
gradient-norm checks). MLX prints/writes progress 343 times versus Metal's 35.
Swift recreates/parses prefix NumericPlans each minibatch, while Python caches
converted plans; Swift reads each finite-check scalar separately, while Python
combines the checks into one scalar. These costs prevent interpreting the result
as an intrinsic framework or language ranking. Their time contribution is unknown.
Any follow-up should first align validation/reporting scope and plan caching,
then profile recurrent/map/tree execution and evaluate MLX compilation separately
on the same cohort with unchanged PPO and explicit numerical checks. This is a
recommendation, not an implemented change or launch authorization. Only local
documentation was updated; no commit, merge or push was made.

## 2026-11-15 — Align optimized MLX execution for a two-game comparison

The user explicitly authorized implementation changes and assistant execution of
a two-game comparison. This is a scoped exception to the default manual-launch
rule, not permission for an official campaign, checkpoint publication, merge or
push. Hypothesis: the previous timing gap partly reflects missing fused kernels,
dense prefix-map recomputation and unequal host/validation overhead in the MLX
ports. Reuse the first balanced pair from the completed ten-game cohort (learner
Red then Blue) with parent 1,765, identical Adam/RNG and one PPO epoch. These are
offline disposable updates from already captured histories; no new game, seed
reservation or official admission is needed. The ten-game source cohort remains
immutable and bound in a new derived-study report.

Implementation target: reuse the current Metal GRU/tree/map kernel algorithms
through shared MLX shader definitions in Python and Swift; keep full current-weight
prefixes and the original gradient boundary. Align prefix-plan caches and progress
frequency, and retain all per-step MLX full-state finite checks by adding the same
checks to the isolated Torch trial. Do not weaken production source/runtime gates
or modify the campaign numerical implementation. Run bounded local forward and
gradient checks before the complete two-game updates. Acceptance requires finite
outputs/gradients, declared tolerances recorded with each check, all selected
samples/steps completed, identical source data and protected checkpoint inputs,
clean child shutdown and a report of any remaining implementation differences.
Record failed/interrupted checks as well as successful ones; speed alone is not
learning-quality evidence. No additional local commit is authorized yet.

Implementation and bounded-validation progress: both MLX ports now use shared
production-derived GRU/tree/map shader bodies with custom first-order gradients,
sparse prefix maps and cached immutable plans. Dynamic sequence/map sizes are
runtime uint32 buffers rather than template arguments, avoiding a shader
specialization for each prefix length. Per-step full-state checks remain in MLX
and were added to the isolated Torch Adam wrapper; progress cadence is aligned.
MLX functional scatter buffers still differ from Torch's mutable arena, so do not
claim identical allocation or dispatch costs. The campaign code is unchanged.

Private checks under `framework-comparison/local-validation/` passed: three GRU
forward/backward shapes (including H64/T300 and padded H32), a real 793-node,
19-group tree, and sixteen sparse-map cases with current and perturbed weights.
Maximum errors against Torch/MPS were approximately 5.96e-7 for GRU gradients,
5.7e-7 for tree gradients, and 8.20e-7 for map outputs (map bound 2e-5). Initial
compile/numerical failures identified scalar Metal builtin syntax, signed scalar
comparison and byte-embedding gradient width (8, not the tree feature width 32);
these were corrected and the affected checks repeated. The full Python MLX first
minibatch (32 samples; recurrent prefixes 1,412/1,039 frames) passed against the
current optimized Torch computation: gradient max absolute 6.71e-7 and relative
L2 4.49e-7; after perturbing GRU/CNN weights the relative L2 was 4.06e-7.
See `fused-ops-parity.py`, `fused-ops-check.log`, `full-minibatch-parity.json`
and their recorded source hashes. A later Python result-report-only change is
explicitly recorded as a post-validation source review in the private report.

Swift release compilation and its first GRU/tree checks passed, but extending
the private probe exposed an intermittent tree mismatch and then a Metal GPU
address fault, before the new full-minibatch section. With Metal API/GPU validation
enabled, the same probe passed, including first-minibatch prediction errors below
6e-7. This is an unresolved failure at this point, not a successful Swift benchmark.
Preserve `swift-optimized/probe-full-first-minibatch*.log` and
`swift-optimized/probe-metal-validation.log`; diagnose and retest normal execution
before any complete PPO timing. No full update has run in this task yet.

Failure correction: pooled/embedding atomic outputs were declared explicitly,
but this alone did not resolve the Swift mismatch. Joint gradient evaluation
and removal of redundant zero-plus-first-gradient operations also did not resolve
it. The final diagnosis identified an input-ABI collision in the shared kernel
adapter. In pinned MLX core 0.31.1, `custom_kernel.cpp` selects a Metal `constant`
pointer for input arrays smaller than eight elements and a `device` pointer for
larger arrays; scalar inputs also change pointer/reference syntax. Reusing one
kernel name across these signatures triggers `clear_library()` when generated
source changes. The pinned device uses command buffers with unretained references
and releases cached pipelines on library invalidation, consistent with the
observed timing-dependent address faults and validation-layer masking.

Both adapters now include input dtype, rank, small-array address-space class,
runtime-parameter names and output dtype in the pipeline cache key/name. Runtime
dimension values remain ordinary uint32 buffer arguments. After this correction,
Swift passed 20/20 fresh normal-mode processes containing the GRU/tree forward
and backward cases and a full real first minibatch with both sparse prefixes.
No GPU validation layer or per-kernel synchronization was used in those repeats.
See `swift-optimized/probe-abi-cache-repeated.json` and its twenty logs. Python's
final full-minibatch check against current source hashes passed for original
and perturbed weights: gradient relative L2 4.31e-7 and 4.46e-7, respectively;
maximum absolute gradient differences 5.66e-7 and 5.07e-7. Earlier failures remain
in the private logs and are not counted as successful trials.

The official experiment `--setup` completed a fresh Swift release build in
18.55 seconds and copied the pinned Metal resource. Its source and shared-kernel
hashes match the final source. Python AST, JSON parsing, all launcher syntax and
whitespace checks pass. Prospective timing protocol: derive the balanced pair
once, then run three fresh-process trials per backend in rotating order
(Metal/Python/Swift, Python/Swift/Metal, Swift/Metal/Python), restoring the same
parent for every trial. Report the median and range and keep compile/setup outside
the timed processes; first-use shader compilation inside a process stays included.
No additional games or official training updates are part of this protocol.

Final timing review found that Metal repeats the full source/capture verification
inside its child, in addition to the common runner verification. Keep this gate,
but record its duration as `child_preflight_seconds` before starting numerical
preparation. The full supervised process duration still includes that work.
This aligns the numerical timing scope without removing any validation. Final
staged review found no further numerical or ownership defects; source hashes of
the numerical implementations and the Swift setup receipt remain unchanged.

The user explicitly approved one local commit of the prepared optimizations,
shared shaders, two-game comparison mode and documentation, followed by the
scoped comparison. No merge or push is authorized. Private validation scripts,
compiler products and raw evidence are excluded from the staged change.

## 2026-11-15 — Completed optimized two-game PPO comparison

Executed the authorized protocol from clean local commit
`8da9caac68904c2c616948eaf961c6f854879e7f` on
`codex/ppo-framework-comparison`. No merge or push occurred. Preparation command:
`./prepare-ppo-comparison.command --two --study /Users/orsi/Projects/Heroes-artifacts/framework-comparison/study-20261114T225228Z-f35d0fb9`.
Derived study `study-20261114T234306Z-1e6adbe7` preserves original indices [0,1],
balanced learner colors and parent 1,765; no new game, seed or admission was
created. Its manifest SHA-256 is
`3b8a498e70a5a51a5d12fee8436c917940a778caa71a2b536fcfe62079ce6561`.
The unchanged starting weights hash is
`5e88fc9d236204e57fd32f73cd003f19b675a7f6f9d0c25ef6ce376d04cda35e`.
The pair contains 801 + 288 = 1,089 samples, 69 windows and 35 optimizer steps,
one epoch. Torch 2.9.1 / NumPy 2.3.5 / Python 3.13 and the pinned MLX 0.31.1 core
runtimes were retained; Swift uses mlx-swift 0.31.4 / Swift 6.2. The M4 Max with
128 GiB RAM was on AC power in Automatic mode throughout.

Ran each existing `run-ppo-{metal,python-mlx,swift-mlx}.command` with the explicit
derived `--study` path. Round orders were Metal/Python/Swift,
Python/Swift/Metal and Swift/Metal/Python, each restoring the same parent state.
All nine attempts completed and passed; none failed or was interrupted.

| Round | Metal full update | Python/MLX full update | Swift/MLX full update |
| --- | ---: | ---: | ---: |
| 1 | 9.760424 s | 2.152762 s | 2.268684 s |
| 2 | 8.625214 s | 1.680775 s | 1.525859 s |
| 3 | 9.159370 s | 1.535182 s | 1.496446 s |
| Median | 9.159370 s | 1.680775 s | 1.525859 s |

Median supervised process durations were 17.581609, 5.945088 and 5.808817
seconds. These include post-run verification and Metal's repeated child
preflight, so they are not GPU execution times. Ready-bundle full-update ratios
are 5.449x and 6.003x, but MLX export precomputes GAE, normalized advantages,
window order and immutable prefix/joined minibatch plans; Metal rebuilds them
inside its timed update. Portable export cost 2.012082 seconds. Charging that
entire cost to each MLX median gives 3.692857 / 3.537941 seconds and ratios
2.480x / 2.589x. These are conservative accounting estimates, not measured
integrated production-pipeline durations: export includes experiment-only file
serialization and hashing. No profile assigns seconds to individual GPU kernels
or host operations. The MLX memory peaks were approximately 200 MB (Python) and
212–215 MB (Swift), well inside the unchanged reserve.

Saved-state audit passed for all 39 parameter tensors / 74,344 FP32 values.
All values/moments were finite, final RNG matched the exported schedule exactly,
and Adam counters increased by the exact active counts (+35 ordinary tensors,
+29 coordinate heads, +5 quantity heads). Initial prediction errors were at most
1.43e-6 for log probability and 7.15e-7 for value. Across all nine trials,
value loss was 0.6403089476–0.6403090209 and approximate KL
0.000406017918–0.000406020359; clip fraction matched at 0.001836547291.
MLX final weight relative L2 difference from the first Metal trial was
1.348e-5–2.125e-5, equivalent to 0.159–0.250% of the actual update norm.
The largest absolute difference, 0.001684, was `coordinate.2.bias`, a shared
coordinate-logit offset that cancels in exact-arithmetic softmax. Excluding only
that tensor, cross-backend final-weight relative L2 was 2.217e-6–3.128e-6,
comparable to Metal's own repeated-run range of 2.899e-6–3.058e-6. Remaining
differences were at most 0.0368% of the reference update norm. Adam first/second
moment relative differences were at most 2.189e-6 / 1.797e-8. This supports close
numerical agreement on the pair, not bitwise reproducibility or equal strength.

No skipped optimizer work or deferred GPU work outside the timer was found:
both MLX implementations evaluate parameters/moments and check finiteness every
step, then synchronize before recording completion. All trial source, input,
result and backend-asset bindings passed; cleanup was verified with no errors.
An independent audit rehashed all 686 protected files successfully. Checkpoint
1,765 and the original ten-game cohort remain intact. Numerical and timing
analysis used saved artifacts only and did not run additional training.

Evidence lives under the derived private study: `comparison.json`,
`timing-analysis.json`, `numerical-analysis.json`, and all nine `results/*`
directories. Only the requested implementation was committed; these post-run
documentation updates remain local and uncommitted. The original comparison was
uneven in optimization coverage; the aligned ports reverse its observed ranking
on this small pair. Remaining limits: different MLX buffer allocation and standard
framework operators, preparation placement, driver/filesystem cache effects,
three repetitions and only two games (about one tenth of the original samples).
Do not infer 10/100-game speed or stronger play. No official checkpoint was
published, and no branch was merged or pushed.

## 2026-11-15 — MLX production migration and fresh 100-game timing target

The user explicitly requested migration to MLX, authorized assistant execution
and sustained optimization until PPO on 100 games takes no more than ten minutes,
and requested progress updates every twenty minutes. They then clarified that
the acceptance run must use **100 new games**, continuing from the latest saved
checkpoint, rather than a benchmark on a previously saved cohort. Latest complete
state is checkpoint 1,765 / 115,479 optimizer steps, independently hash-checked.
No merge or push was authorized. Previous diagnostic/seed/admission/retention
requirements remain; the next official unused reservation offset is 2,453.

Hypothesis: the optimized Python MLX implementation can retain its small-cohort
advantage in the live cycle by building plans in memory, without intermediate
NPY/JSON export. Move the neural network kernels/model into the canonical learner,
retain PPO equations, all current-weight history, the complete one-epoch schedule,
Adam state and sampling RNG. Migrate live neural inference as well, with the
existing CPU checkpoint/tensor/sampling boundaries retained for compatibility;
this is not a claim that all PyTorch dependencies have already been removed.
Do not assume gameplay becomes faster. Both capture and diagnostic reconstruction
must explicitly select the same bound inference backend and keep exact replay.

Acceptance: all 100 fresh games complete under the existing rules and enter one
ordinary admitted update; no truncated training schedule or smaller loss set.
The existing production `phase.update_seconds` / `update_wall_seconds` boundary
must be at most 600 seconds, including admission checks, transaction snapshots,
conversion, GAE/plan construction, current-weight prefix reconstruction, backward,
Adam, synchronization and import back to the compatible checkpoint state. Report
collection/preparation, checkpoint save and the final replay-verified diagnostic
separately. Require finite states, exact checkpoint roundtrip, protected original
inputs and verified owned-process cleanup. If the first new run misses the target,
record the actual denominator/time and diagnose before the next authorized attempt;
do not reinterpret a saved-data microbenchmark as a successful fresh campaign.

Bounded implementation/rollback/replay probes may reuse private saved inputs;
they are validation only, not the requested final timing run or fresh admission.
Install a new hash-pinned combined environment at `python/mlx-v1/venv`, leaving
the old runtime intact. The source gate will retain the historical exact source
transitions and bind the complete new MLX code, shaders and dependency lock.
Implementation and numerical validation are in progress; no fresh game or
official update has started at this entry.

Integration validation: the new in-memory MLX adapter completed the disposable
1,089-sample / 35-step pair in 2.7657 seconds including GAE, plan building,
initialization, numerical update, CPU state import and cleanup. Counts, minibatch
RNG and named Adam counters matched. An injected exception after two steps passed
the unchanged transaction's exact model/Adam/RNG/gradient-flag rollback. The first
standalone harness attempt stopped before GPU work because it omitted the required
single ATen thread; setting the same thread limit as the campaign corrected the
harness. Maximum weight difference from an earlier MLX repeat was 0.000731235;
maximum Adam-moment difference was 6.519e-9. Prior MLX repeats already differed
by 0.000415–0.000634, predominantly in the common coordinate bias. Do not claim
bitwise equivalence from these observations. Reports are private
`framework-comparison/mlx-production-{adapter,rollback}-probe-20261115.json`.

The MLX policy adapter passed both complete saved histories: 9,711 frames,
1,089 decisions, identical sampled CPU/MLX actions under the probe seeds, hidden
maximum error 3.87e-6 and value error 7.75e-7. More importantly, the existing
session plus exact reconstruction gate reproduced all 1,089 actions/values and
1,861 conditionals with maximum difference zero after a cold MLX cache reset.
Serial inference took 15.51 seconds on MLX versus 5.12 on CPU; this is a measured
slowdown for incremental policy calls, not the concurrent game-cycle timing.
Evidence: `framework-comparison/local-validation/mlx-policy/result.json` and
`replay-full.log`. No reconstruction tolerance was weakened.

The existing targeted unit run executed 97 tests: 94 passed and three errored.
All three errors reproduced on an archived clean `8da9caa` source tree: two old
controller mocks lack the stdin registration channel, and one historical source
fixture omits a previously required starting-equipment dependency. No test was
changed or published, and these are not new MLX regressions. Preserve
`python/mlx-v1/unit-checks.log` and `baseline-failures.log`; do not call the whole
suite green. The first baseline confirmation accidentally ran in the current
directory and was corrected by rerunning from the actual archived source.

Canonical production MLX code has no dependency on comparison orchestration.
The optional comparison-import refactor was preserved privately in
`python/mlx-v1/comparison-canonical-imports.patch` and restored out of the working
diff; old experiment files stay unchanged. Production shader copies live under
`native/learner/mlx`. The prefix cache now accounts for actual normalized
FP32/int32 arrays, including expanded uint8/bool inputs and separately converted
aliases. This changes accounting only, not the validated neural equations.

Retirement audit found that the earlier full-100 Metal investigation still read
the original run directory. Before ordinary retirement, preserve only its 406
required inputs (8,325,497,121 bytes) as independent APFS clones under
`run-investigations/2026-11-14-metal-memory/preserved-cohort/run-20261113T211610Z-a9566f61`.
Every original/archive hash and size matched the pinned cohort manifest. Receipt
`preserved-cohort/preservation-manifest.json` records the original/archive prefix
mapping; later use of the frozen absolute-path loader requires explicit restore
or remapping. Original files remain untouched until standard retirement. This
keeps the completed research reproducible while ordinary old campaign artifacts
can be removed and exact checkpoint state retained.

Final source review: all sixteen MLX migration entries match current bytes;
the actual 234-file runtime inventory and both historical native/scenario
acceptance chains pass exact transition checks. Deliberately altered hashes and
an omitted MLX policy file reject. Evidence: `python/mlx-v1/source-transition-check.json`.
Prepare one focused local production commit so the unchanged clean-checkout gate
can admit the authorized fresh cycle. No test, probe, benchmark-script change,
raw result, checkpoint, merge or push is included. The historical comparison
implementation is retained unchanged; production owns its canonical MLX modules.


## 2026-11-15 — First fresh production MLX cycle; scaling target missed

Authorized assistant execution continued on local branch
`codex/ppo-framework-comparison`, source
`4210efdef3f3bf851e049dd33a9e3a7ce480494d`, using
`./run-100-games.command 100 --mlx` and the pinned combined
`python/mlx-v1/venv` runtime. Run
`selfplay-ten-then-check/run-20261115T000548Z-0add1c7f` continued the complete
checkpoint 1,765 (115,479 optimizer steps; weights
`5e88fc9d236204e57fd32f73cd003f19b675a7f6f9d0c25ef6ce376d04cda35e`).
Training engine seeds were 1,980,002,453–1,980,002,552, with 50 learner seats
of each color; diagnostic seed was 1,980,002,553. MLX performed live neural
scoring and PPO under the unchanged scenario, PPO, admission and resource rules.
The user authorized continued work when disconnecting power. Every actual
recorded power observation through completion still reported AC power and
Automatic mode; this is not a battery benchmark. Observations are in
`python/mlx-v1/fresh-100-power-observations.jsonl`.

All 100 captures completed without capture failures: engine outcomes were
64 Blue wins, 32 Red wins and four cutoffs. These training games deliberately
did not receive replay/policy reconstruction. They contributed **259,090 eligible
PPO samples** from 259,097 sampled decisions (seven bootstrap-only decisions),
not 259,097 training samples. The complete update executed all **8,119 steps**.
Collection plus overlapping preparation took 2,570.5726 seconds. The complete
production update phase took **1,137.2352 seconds (18m57s)**; the numerical adapter
accounted for 1,068.3779 seconds, including 162.9678 seconds of plan building,
35.0831 seconds of post-state validation and 8.6288 seconds of cleanup. Timing
subfields overlap with the adapter update field and must not be added to it.
The approximately 68.86-second outer difference includes admission/transaction
work; its exact breakdown has not yet been profiled. The 600-second criterion
was missed. The earlier two-game comparison did not predict this larger workload
reliably. Different cohorts also prevent treating the old 1,637-second Metal
measurement as a controlled same-data speedup for this run.

The cycle published and exact-roundtrip-verified checkpoint **1,766**, now
123,598 total optimizer steps, with weights
`e6b29324db6375ebb315b2f2376469c89d05626f96b53b60c0cb7a0cab7e55cc`.
Reported PPO metrics were policy loss 0.00169306, value loss 0.4619262,
approximate KL 0.00387454 and clip fraction 0.0285422. The final diagnostic
ended naturally with new-policy Red defeating frozen batch-start Blue.
Both policies reconstructed exactly: 497/413 actions and values, 903/742
conditionals, maximum difference zero; native replay also passed. One outcome
is not evidence of improved playing strength. The complete cycle took
3,781.8988 seconds (63m02s) and ended at 01:08:52 UTC. Final report/shutdown
confirm exit zero, owned cleanup and unchanged protected inputs. No repository
source changed while the campaign was active. No merge or push was performed.

Retention-reviewed run: `run-20261115T000548Z-0add1c7f`.
The sealed captures remain available for the following bounded numerical
investigation until the next reviewed launch; the checkpoint pair must survive
ordinary retention.

Next question: which prefix, planning, backward or optimizer component prevents
the full phase from reaching 600 seconds without changing PPO or truncating
current-weight histories? Private candidates prepared during the live run
retain only requested no-gradient GRU boundary states, and compile fixed Adam
arithmetic/finite checks while preserving individual active-parameter counters.
Neither candidate ran during the campaign. Synthetic selected-state equivalence,
real-history numerical/state comparisons, memory guards, unchanged inactive
parameters/counters, and transaction rollback precede adoption. A saved-data
probe is diagnostic only; final acceptance requires another **fresh 100-game
continuation** from the latest complete checkpoint.

After PPO and checkpoint publication, a CPU-only private preparation overlapped
the final diagnostic, without changing source or performing GPU computations.
It restored parent 1,765, bound sealed episodes 1 and 42, and built native
histories for 10,038 PPO samples / 314 planned steps in 35.0771 seconds.
Episode 42 has 1,790 sampled decisions but 1,789 PPO samples. This selected
long/near-median pair and its 314,183,601-byte native payload are in
`framework-comparison/local-validation/mlx-scale/cohort-1765-long-median`;
they are not live admission capabilities. The private profiling harness records
inclusive host versus synchronized-attribution timing separately; resource
checks run at least once per second and their overhead has its own field.
GPU probes begin only after the completed campaign's owned cleanup.

The user subsequently reserved the **next fresh training launch for themselves**.
Continue the authorized isolated optimization/validation work and prepare the
next command, but do not start another game campaign. Therefore the 600-second
fresh-100 acceptance remains pending the user's next launch, even if an offline
probe becomes faster.

Initial synchronized attribution on the selected pair completed 32 bounded steps
in 5.2560 seconds; the private CPU parent model/Adam/RNG stayed unchanged.
Inclusive recurrent-prefix work took 4.0693 seconds: main GRU 2.1335 seconds,
initial construction of two full prefix plans 1.1731 seconds, sparse map 0.3948
seconds, tree 0.1249 seconds. Forward/backward took 0.2608, joined planning
0.1595, Adam 0.0745, post-state checks 0.1100 seconds. Synchronization changes
scheduling, and these nested fields cannot be summed. Evidence:
`mlx-scale/synchronized-32/result.json`. This identifies serial prefix GRU as
the largest warm component on this pair, not an optimizer-dominated workload.

The selected-state-only GRU candidate passed all 56 exact synthetic comparisons,
nine invalid cases, and empty/mixed prefix wiring. It failed its speed hypothesis:
at 10,000 frames, median baseline was 13.876 ms versus candidate 14.974 ms.
The extra per-step selection branch is a hypothesis for this slowdown; it was
not separately measured. Do not adopt this candidate as a speed improvement.
Evidence: `mlx-scale/prefix-selected-candidate/synthetic-result.json`.
Next private candidates vary the GRU reduction width or keep all hidden
coordinates within one SIMD group to reduce synchronization. They must preserve
full nonlinear recurrence and accepted FP32 agreement; no truncation or
cross-step caching of learned states is permitted.

The corresponding 32-step host-submission profile took 5.0574 seconds and
confirmed the same unchanged private CPU parent state. Its prefix total was
3.8898 seconds. These host rows include deferred GPU work in later checks and
are not independent kernel timings (`mlx-scale/host-32/result.json`).

The compiled-Adam candidate completed the entire selected pair from identical
parent state: baseline 33.6557 seconds versus compiled 33.4811 seconds,
10,038 samples / 314 steps in each, including all adapter planning/import/cleanup.
A 1.0052x single-pair ratio provides no meaningful speed evidence, so this
additional optimizer complexity is not selected for production. Individual
Adam counters and minibatch/Torch/Python RNG states matched exactly. Maximum
weight difference was 0.0017891 (common coordinate bias); relative L2 was
2.309e-5. Maximum moment difference was 5.821e-9. These results do not establish
playing quality or bitwise equivalence. Evidence: `mlx-scale/native-baseline-01`,
`native-compiled-01`, and `native-optimizer-comparison.json`. The drafted private
failure/inactive-state harness was not executed because adoption was rejected
for lack of performance benefit.

Seven no-gate recurrent variants were tested privately on the actual Apple M4
Max with 128 GiB unified RAM, still on AC power. R8/R4/R2/R1 reductions and
single-SIMD register/streaming designs all passed the bounded FP32 gates but
were slower than the existing R16 implementation. At 10,000 frames, baseline
was 13.850 ms, no-gate R16 13.307 ms, R8 15.118 ms, R4 20.235 ms, R2 24.303 ms,
R1 38.807 ms, register SIMD32 27.635 ms and streaming SIMD32 41.666 ms. At
40,000 frames, baseline was 54.690 ms versus no-gate R16 52.575 ms (about 1.04x).
The R16 no-gate control was bit-exact above 256 frames; all other measured maximum
errors were at most 8.95e-8. Removing synchronization by increasing serial work
per lane did not improve speed. Evidence: `mlx-scale/gru-parallel-candidate/`
`length10000-result.json` and `full-matrix-result.json`.

Next private hypothesis: pack the two independent history prefixes in one GPU
dispatch while retaining each original projection GEMM shape, full individual
sequence length, recurrence arithmetic and short/long dispatch threshold.
This should overlap independent work rather than cache outdated hidden states.
The analytical probability of different histories in a uniformly shuffled
two-window minibatch is 97.8995% for the 100-game cohort but 29.3542% for the
selected long/median pair. These are window-count expectations, not observed
schedule frequencies; a pair benchmark will underrepresent this opportunity.

Packed-prefix probes confirmed that MLX already overlaps independent lazy
GRU dispatches. Equal 40,000-frame sequences took 57.44 ms with the baseline,
55.03 ms with separate no-gate kernels and 55.34 ms packed. The plain 32-step
real-pair timing was 5.3806 s baseline versus 5.2922 s packed, with a higher
packed GPU peak. Packing is therefore not selected for production. The initial
real comparison rejected nonidentical full-encoder recomputations at step 0;
repeated baseline prefixes themselves varied by 4.41e-6 and 5.13e-6 due to the
existing nondeterministic sparse-map reductions. The corrected diagnostic
shared identical encoded vectors and passed bit-exact boundary comparisons
on all 32 steps. Both failed and corrected reports remain in
`mlx-scale/gru-packed-candidate/`. Two cached history plans occupied
134,997,188 bytes, so these pair probes do not establish 4 GiB cache pressure.

Next implementation candidate overlaps GPU prefix evaluation with CPU
construction of the same minibatch's joined plan. Prefixes still use the
current weights and full histories, are passed once to that step, and are
never cached across optimizer updates. Boundary keys are checked at the
consumer. The existing synchronized path remains the default for direct
backend callers; production submits via MLX async_eval and later loss/finite
evaluation waits on its dependencies. Validate full private state/counters/RNG
and exception rollback before retaining this scheduling change. This does
not change epochs, minibatches, losses, recurrent formulas or resource caps.

The batch-admission implementation now validates the shared exact plan and
source before and after all individual receipts instead of hashing the same
common inputs 100 times. Every registered capability, plan path/hash/JSON,
schedule position, on-policy weight identity, unused slot, receipt JSON and
capture envelope hash remains checked; final common validation precedes any
consumption. Other launch/completion/preparation rechecks remain unchanged.
Fifteen private CPU-only mocked tests passed, including final validation
failure retaining all capabilities (`mlx-scale/admission_batch_tests.log`).
This changes the sampling frequency for common files during admission; it
does not prove detection of every transient change-and-revert. No cache or
unchecked caller-supplied plan crosses batches. Actual phase-time benefit
will be measured on the user's next fresh run.

The canonical candidate now removes unused pooled segment IDs and text/ordered
row/scale fields, freezes CPU-only metadata tuples, and performs tensor
conversion with imports outside recursive scalar visits. Five real-plan CPU
cases passed exact consumed-buffer/metadata/slicing checks and dtype/index
rejection checks. Long-prefix host conversion fell from 58.86 to 2.03 ms,
near-median from 15.34 to 1.19 ms, and joined-window conversion from 1.09–1.24
to 0.67–0.73 ms. Actual two-plan upload accounting fell from 128.74 to 127.31 MiB;
the cache cap remains 4 GiB. Evidence: `mlx-scale/planning-candidate/cpu-01`.
Only the Python production plan is compacted; the comparison/Swift schema
and code remain untouched.

The retained no-gradient main-GRU kernel removes unused backward-gate stores
while returning every hidden state. It preserves the original R16 recurrence
above 256 frames and the original short-sequence path otherwise. The slower
selected-state, alternate-width, SIMD32 and packed candidates stay private.
The asynchronous adapter explicitly clears prepared-prefix dictionaries both
after consumption and on failure, then drains submitted GPU work before
clearing allocator caches. Two canonical failure probes passed: an exception
after exactly 3 completed steps and a joined-plan exception immediately after
a nonempty async submission before any completed step. Model/Adam/RNG, mixed
gradient flags, an existing gradient, modes and progress/checkpoint restored
exactly; prepared/native/MLX caches cleared. Evidence:
`mlx-scale/canonical-async-failures-01/{after-three,joined-plan}/result.json`.

A full selected-pair trial on this candidate completed 10,038 samples / 314 steps
in 31.7434 seconds versus 33.6557 for the earlier baseline (1.0602x; one pair
comparison, not a 100-game result). The async-only intermediate was 32.3056 s.
Final individual Adam counters and all RNG states matched. Final loss/KL
metrics agreed within about 1e-8; clip fraction was identical. Maximum weight
difference was 0.00414595 in the softmax-invariant common coordinate bias; the
next largest was 0.000260708 in coordinate output weights. Maximum Adam-moment
difference was 7.509e-9. This is accepted FP32 variation, not bitwise training
equivalence or playing-quality evidence. GPU peak was 795,073,736 bytes.
The private timing excludes official admission; its common-plan optimization
will need measurement on the next fresh run. Evidence:
`mlx-scale/native-final-01` and `native-final-comparison.json`.

After the planning changes, the existing live-session/reconstruction numerical
harness passed 256 saved decisions and 445 conditionals with maximum
reconstruction difference zero. It launched no game and applied no update
(`mlx-scale/policy-final/result.json`).

One remaining bounded hypothesis uses FP32 SIMD-group 8x8 matrix multiply for
the recurrent hidden projection, with no reduced precision or truncated
history. Preloaded weight tiles produce all 192 gate values; two SIMD groups
then compute the 64 recurrent coordinates, retaining two barriers per time
step. This may improve work distribution relative to scalar dot reductions,
but may also be slower or unsupported. Require bounded synthetic and real
vector agreement, then a complete private update before any adoption.
No new training game is authorized for assistant launch.

The FP32 SIMD-matrix kernel compiled through MLX and passed synthetic gates
through 40,000 frames, but failed the long-prefix speed hypothesis. Nine-repeat
medians were 1.629 versus 1.503 ms at 1,000 frames, 14.112 versus 14.693 ms
at 10,000, and 55.909 versus 56.407 ms at 40,000 (baseline versus MMA).
Maximum full-sequence difference was 8.95e-8, RMSE 8.31e-9. It remains private
and did not proceed to PPO or production. Reports are
`mlx-scale/gru-mma-candidate/{initial-1000-10000,followup-40000}.json`.
The optional offline Metal toolchain/headers were unavailable; no component
was installed. The existing MLX runtime compiler supports the FP32 APIs
documented in [Apple's SIMD-matrix example](https://developer.apple.com/videos/play/tech-talks/10858/).

Final targeted existing checks: all 24 tests in selfplay_training,
selfplay_checkpoint and selfplay_behavior passed in 1.610 seconds. This does
not erase the three pre-existing errors from the earlier broader suite.
No test, probe, diagnostic script, benchmark, fixture or raw artifact is
added to the production commit. All candidate experiments remain private.
The exact runtime source allowlist is updated only to the reviewed current
production bytes; its inventories and rejection behavior are unchanged.

A second complete candidate repeat passed the same 10,038 samples / 314 steps
in 31.8512 seconds (first 31.7434), with exact expected minibatch RNG and
individual Adam counters and a 798,757,324-byte GPU peak. Against the single
33.6557-second baseline this supports a modest roughly 6% local improvement,
not a 2x gain or the 600-second fresh-100 criterion. The next full campaign
remains user-launched from checkpoint 1,766. No new game or official update
followed the completed first campaign.

Prepare one local production commit containing only the reviewed learner code,
shader, exact operational source pins and documentation. Keep every private
experiment, validation script, raw result and weight out of the staged diff.
Then run the read-only 100-game MLX launcher inspection against that clean
commit and checkpoint 1,766 before handing over the command. No merge or push
is authorized by this handoff.

The read-only launcher inspection passed on the clean production commit:
checkpoint 1,766 restored exact model/Adam/RNG/progress/sampler/pool; MLX is
selected for policy and PPO; the requested next cycle is 100 training games
plus one diagnostic, with one update and no wall-clock deadline. It performed
no gameplay or optimizer update and did not reserve seeds (next offset 2,554).
Evidence: `mlx-scale/next-100-preflight-before-journal.log`. After recording
this result in a documentation-only amendment, repeat the same read-only
inspection and retain `mlx-scale/next-100-preflight-final.log` for the handoff.


### 2026-11-15 — User-run optimized MLX fresh-100 result (1766 to 1767)

Question: do the prepared prefix, plan and admission optimizations improve the
complete production update on 100 new games? Acceptance remains a complete PPO
phase at or below 600 seconds, with unchanged admission/checkpoint/diagnostic
gates. The user launched this campaign; this review only read existing files
and did not launch any workload.

Source: `13e6bf49a0d7e5d3bd7b038ad6cd335f30373ecf`, clean at launch;
Python 3.13.5, Torch 2.9.1, MLX/MLX-metal 0.31.1, MLX policy and PPO.
Evidence: private `selfplay-ten-then-check/run-20261115T111633Z-27357989/`
`launch.json`, `report.json`, `shutdown.json`, and diagnostic reconstruction.
Parent 1766 weights `e6b29324db6375ebb315b2f2376469c89d05626f96b53b60c0cb7a0cab7e55cc`;
training engine seeds 1980002554–1980002653; diagnostic 1980002654.
Ten rolling actors and existing scenario/rules were retained.

All 100 captures were admitted, with 238,862 PPO samples and 7,487 optimizer
steps. Engine outcomes were Blue 71, Red 26, three day-limit cutoffs, zero capture
failures. These training outcomes remain unverified by replay/reconstruction.
Collection plus overlapping preparation took 2,140.7244 s (35m41s), complete
production PPO 729.1366 s (12m09s), checkpoint 1.1598 s, and diagnostic including
verification 213.8603 s. Entire cycle: 3,090.1456 s (51m30s), completed at
12:08:04 UTC. Shutdown returned zero; owned cleanup and protected inputs passed.

Compared with the preceding fresh-100 MLX run: collection fell from 42m51s,
PPO from 18m57s, and total from 63m02s. PPO elapsed fell 35.89% (1.560x), but
samples also fell 7.81% and optimizer steps 7.78%. Throughput normalized by
samples or steps improved approximately 1.438x. This is encouraging evidence,
not an isolated causal measurement: histories, checkpoint, seeds and possibly
power/thermal conditions differ. Recurrent prefix cost is not proportional to
sample count. No claim that the entire observed gain comes from code alone.
The 600-second criterion is still missed by 129.14 s; reaching it from this
result requires about 17.7% less elapsed time.

Adapter full_update was 725.4271 s versus 1,068.3779 s previously; nested plan
building 97.0012 versus 162.9678 s, state validation 16.7983 versus 35.0831 s.
Production-minus-adapter overhead fell from about 68.86 to 3.71 s, consistent
with the common admission-check change. Nested timings must not be summed.

Checkpoint 1767 passed exact model/Adam/RNG/progress/pool restore, with weights
`b67131a563931d1c26be1a587090c57f9b0726215edd3243def9106cd7055e12`.
PPO approximate KL 0.00600706, clip fraction 0.0351291, policy loss 0.00237569,
value loss 0.412056. Different data prevent treating metric changes as quality
improvement. Diagnostic Red (new 1767) lost naturally to Blue (frozen 1766).
Both policies reconstructed exactly: Red 1,823 actions/values and 3,204
conditionals, Blue 1,456 actions/values and 2,476 conditionals; maximum difference
zero for both. Native replay passed. One loss does not establish regression,
just as the preceding diagnostic win did not establish stronger play.

Retention-reviewed run: `run-20261115T111633Z-27357989`.
No artifacts were removed during this review. No commit, merge or push performed.


### 2026-11-15 — Prepare user-run CPU-inference / MLX-PPO fresh-100 comparison

Hypothesis: the existing CPU policy path reduces small-request inference overhead
and complete cycle time relative to MLX policy inference. The user requested
100 new games continuing complete checkpoint 1767, followed by the usual fresh
validated diagnostic, and will launch personally. No assistant launch is allowed.
Acceptance: retain PPO/settings/seed/admission/checkpoint/replay gates, record
collection, complete update and cycle times plus sample/step denominators; compare
against the preceding 35m41s / 12m09s / 51m30s cycle. Different fresh histories and
parent weights limit causal attribution and no strength improvement is assumed.

Implemented: new `--mlx` cycles select CPU policy and unchanged MLX learner.
Existing actor and diagnostic configuration propagation already selects this
CPU path. Stored MLX launches accept the two explicit supported policy values
(cpu or mlx), retaining their recorded choice; exact source identity, runtime
version and policy binding checks remain. Updated the controller's exact source
pin to its current bytes, without changing inventories or other gates. No PPO,
network, engine or optimizer code changed. No seed allocation, game, replay,
benchmark, numerical probe or optimizer update was launched. Syntax inspection
only; full clean-checkout preflight is pending a local production commit.
Prepared user command: `./run-100-games.command 100 --mlx`.


Read-only launcher inspection passed on local commit `f5d7397`: parent 1767
restored exact model/Adam/RNG/progress/sampler/pool; configuration reports CPU
policy, MLX PPO, 100 training games plus one diagnostic and one update, with no
wall-clock deadline. No gameplay, optimizer update, deletion or seed reservation
occurred; next unreserved engine seed is 1980002655 (101 games). Evidence:
private `framework-comparison/local-validation/mlx-scale/cpu-policy-1767-preflight.log`.
Only documentation is added in the following amendment; repeat inspection on
the resulting clean commit before handoff. No merge or push is authorized.


### 2026-11-15 — First hybrid launch failed before gameplay; correct plan gate

The user-run `run-20261115T122358Z-b5fb6e5d` on source `8cdb953` stopped
with exit 1 after 0.6524 seconds. Reported error: `ValueError: MLX capture and
update must share the pinned policy/runtime contract`. Configuration correctly
selected CPU inference / MLX PPO, but `selfplay_training._plan` still required
MLX inference whenever MLX PPO was selected. The previous read-only launcher
inspection never exercised this capture-plan gate; its success did not establish
that the hybrid campaign could start. This was an incomplete implementation.

No games completed, zero samples admitted, zero optimizer updates, no diagnostic.
Latest complete checkpoint remains 1767 with weights
`b67131a563931d1c26be1a587090c57f9b0726215edd3243def9106cd7055e12`.
Owned cleanup and protected inputs passed; shutdown errors were empty. No timing
comparison with the successful 51m30s MLX-inference cycle is valid. Seeds reserved
for this failed attempt remain consumed (engine range 1980002655–1980002755).
Evidence: private run `report.json`, `shutdown.json`, `launch.json` and
`batch-train-000/attempts.json`.

Correction: the capture-plan gate explicitly accepts CPU or MLX inference with
MLX PPO, keeping the exact pinned runtime, source, schedule, receipt and other
validation gates. Updated only the changed production source's exact hash pin.
No network/PPO math/settings changed. Syntax parsing and diff checks are allowed
file inspection; no game, replay, profile, numerical probe or benchmark was run.
The hypothesis and acceptance criteria of the preceding hybrid experiment remain
unmet pending a new user launch. A successful read-only preflight is limited to
launcher/state/configuration inspection and must not be reported as a successful
capture-plan or end-to-end execution. No merge or push is authorized.

Retention-reviewed run: `run-20261115T122358Z-b5fb6e5d`.


### 2026-11-15 — Successful user-run CPU-inference / MLX-PPO fresh-100 cycle

Question/acceptance follow the preceding hybrid preparation: compare complete
cycle and collection time, retain exact checkpoint and final diagnostic gates,
and report denominators instead of inferring either causal speedup or strength
from different fresh games. The user launched; this review only reads artifacts.
Source `71923b18403960fd4fcf24eb89d6a2af5f69710c`, run
`run-20261115T123640Z-aa5cd62b`, parent checkpoint 1767 weights
`b67131a563931d1c26be1a587090c57f9b0726215edd3243def9106cd7055e12`.
CPU policy / MLX PPO, MLX and mlx-metal 0.31.1, same scenario and rolling actors.
Fresh training engine seeds 1980002756–1980002855, diagnostic 1980002856.
Evidence: private run `launch.json`, `report.json`, `shutdown.json`, and
`evaluation-000/execution/episode-000001/training-policy-reconstruction.json`.

Status completed, exit zero, zero capture failures, 100 admitted games,
228,930 samples and 7,179 optimizer steps. Engine outcomes: Blue 64, Red 30,
six day-limit cutoffs; training captures remain not independently replay-verified.
Collection and overlapping preparation: 1,390.0022 s (23m10s); complete production
PPO: 600.1246 s; checkpoint 1.0899 s; diagnostic plus verification 70.3122 s.
Whole cycle 2,065.8846 s (34m26s), ended 13:11:07 UTC. Owned cleanup and protected
inputs passed. Adapter full_update 596.9127 s; nested update 588.8830 s,
plan_build 84.2245 s, state_validation 15.1744 s, cleanup 7.6810 s.
Do not sum nested timings. The strict <=600 s complete-production criterion is
still missed by 0.1246 seconds, despite the adapter finishing below 600 seconds.

Against the preceding successful MLX-policy run (`run-20261115T111633Z-27357989`):
collection 2,140.7244 to 1,390.0022 s (-35.07%, 1.540x); complete PPO 729.1366 to
600.1246 s (-17.69%); total 3,090.1456 to 2,065.8846 s (-33.15%, 1.496x), saving
17m04s. Samples decreased 4.16%, steps 7,487 to 7,179. Per eligible sample,
collection throughput increased 1.476x and PPO throughput 1.164x. Collection
also includes engine/opponent work and preparation, so this normalization does
not isolate inference. Different parent weights, seeds, full trajectories and
uncontrolled power/thermal conditions limit causal attribution; PPO arithmetic
was not changed by the CPU-inference switch. The complete-cycle result supports
keeping CPU policy with MLX PPO, but does not establish an exact backend-only
speed multiplier. Diagnostic duration also depends on the different game.

Checkpoint 1768 published and passed exact model/Adam/RNG/progress/pool restore;
weights `e7d559efa7ea1e56e7abc6e5debb78dbc4bb0c698bda3b0ee917e9098e70161d`.
PPO approximate KL 0.00417767, clip fraction 0.0264753, policy loss 0.00128614,
value loss 0.367145. Red (new 1768) beat Blue (frozen 1767) naturally. Native
replay passed; both policy reconstructions had maximum absolute difference zero:
Red 1,050 actions/values and 1,880 conditionals; Blue 345 and 606. One diagnostic
win is not evidence of improved strength. No new workload was launched by the
assistant and no commit, merge, push or artifact deletion occurred in this review.

Retention-reviewed run: `run-20261115T123640Z-aa5cd62b`.


### 2026-11-15 — Detailed checkpoint-1768 diagnostic behavior review

Question: explain Red's win from recorded actions, events and army transitions,
and distinguish useful behavior from evidence of general improvement over 1767.
Acceptance: trace decisive events to both player histories, count effective
management actions separately from UI navigation, and do not run a new game,
replay, numerical comparison or benchmark. Read-only artifact analysis only.
Evidence is the successful `run-20261115T123640Z-aa5cd62b` diagnostic's
`agents/{red,blue}/history-v8.jsonl`, `neural-rollout.json`, plan and exact
reconstruction. Source/runtime/seed/parent identities are in the preceding entry.
Red is 1768, Blue frozen 1767, CPU inference for both; terminal day 33.

Chronology: both start with one hero and 25 pikemen. Red first hires another
hero on day 9 (11 pikemen, six archers, three griffins), and recruits 28 pikemen
into the town garrison. Red wins a neutral battle on day 18 without an observed
army-count loss, then visits/captures the wood mine on day 19 (notice +2 wood/day).
Day 20 adds 14 garrison pikemen and a hero with 14 pikemen/five archers/three
griffins. Day 22 Blue attacks Red town at (4,7) with its original 25 pikemen;
Red has 42 garrison pikemen, loses nine, wins, and Blue loses its hero. This
links actual troop purchases to a decisive defensive result without asserting
that the policy anticipated the attack. Blue recruits no creatures at any time.

Day 24 Red recruits 14 more pikemen (garrison 47), hires a hero with 13 pikemen,
and swaps it into the garrison: the resulting hero has 60 pikemen and town
has none. The hero stays garrisoned through day 25; a day-26 swap releases it.
Blue hires a replacement on day 25 with 12 pikemen. On day 30 Red's 60-pikeman
hero moves to (9,13), attacks that Blue hero, wins with no observed casualties,
and Blue again has no hero. On day 33 Red's day-9 hero explicitly targets the
Blue town at (15,7), takes it without another battle and ends the game. This
was a successful defense followed by removal of the last hero and capture of
an empty town, not a decisive defended siege or demonstrated battle tactics.
BattleAI controls combat, the policy controls adventure choices.

Within this one match, Red hired five heroes (six total), recruited four times
(28+14+14+14=70 pikemen), won all three battles, and captured the wood mine and
opponent town. Blue hired one hero (two total), recruited zero, won zero of two
battles, collected a 500-gold resource and ended with 15,000 gold and no assets.
Red ended with 2,846 gold, six heroes and both towns. No construction action
was submitted by either policy. Army-exchange dialogs opened repeatedly but
no inter-hero stack-transfer action was submitted; the garrison swaps are the
observed effective troop consolidation. The original Red hero retained its
25-pikeman stack throughout the game.

Limitations remain visible: 397 opens + 397 closes account for 794/1,050 (75.6%)
of Red decisions, versus 241/345 (69.9%) for Blue. Red issued 207 movement
commands, 139 targeting x=0; Blue 69 with 36 targeting x=0. This is a pronounced
left-edge target bias, not proof of a software defect or strategic intent.
Red leaves some heroes with movement at seven end-turn decisions. First hire
only on day 9, first neutral victory day 18, first mine day 19; macro planning
is still weak. More heroes also create more UI actions; decision volume alone
is not a quality metric.

This match provides positive behavioral examples (spending on relevant troops,
actual consolidation, exploiting an undefended town), not statistically valid
improvement evidence. The prior 1767 diagnostic was a loss against 1766, but
its raw artifacts were retired under retention; do not invent a detailed
cross-game behavior comparison from its aggregate outcome. Current 1767 Blue
is directly observed, but sides, seed, stochastic choices differ, and prior
1767 diagnostic used MLX inference whereas this one uses CPU. No paired-seed,
seat-swapped quality evaluation was run. Exact reconstructions and replay
establish execution fidelity, not strength. Documentation remains local;
no commit, merge, push, deletion or new workload in this review.


### 2026-11-15 — Prepare restored troop management from checkpoint 1768

User request: restore troop-slot management for 100 new games from 1768, keep
other restrictions, CPU inference and MLX PPO; user launches personally.
Hypothesis: restoring split/merge/swap/transfer/inspection/upgrade actions allows
useful troop consolidation beyond whole-hero garrison swaps. Acceptance: restore
exact checkpoint 1768, version the support change, preserve historical semantics,
retain hero-relative movement, command/town cooldowns and interaction budgets,
keep map/taverns/artifact ban/BattleAI/PPO unchanged, and prepare 100 games plus
one exact-reconstruction/native-replay diagnostic without assistant gameplay.
Quality must not be inferred solely from outcomes after expanding action support.

Implementation adds `visible-interaction-budget-hero-relative-stack-management-v1`
with the same relative coordinates and cooldown tracking as the prior version,
but outside the recruitment-only exclusion set. New cycle launches select it;
per-batch policy wrappers, published metadata, reconstruction and native plans
carry the exact selected support. Metadata restores accept both historical and
new cycle variants. No network shape, weights or optimizer initialization changes.
Exact source pins include the changed guard/model/checkpoint/controller bytes.
Validation is pending; all test/probe artifacts stay private and unstaged.
No game, training, replay, profiling or numerical benchmark launch is authorized.


Existing tests passed: 48 checks across recruit_only_guard, interaction_guard,
selfplay_checkpoint and selfplay_training (1.893 s). They preserve historical
support and checkpoint behavior; no game or production optimizer ran.
Private non-numerical support validation covered all seven restored action
kinds using existing army/management/upgrade fixtures, unchanged native offers,
unchanged other initial guard fields and exact support serialization. Attempt 1
failed in the validation harness because it passed the legacy armies fixture to
an incompatible action enumerator (TypeError, before its assertions); attempt 2
used that fixture's explicit transfer action and passed. Evidence: private
`mlx-scale/stack-management-{existing-tests,support-checks,support-checks-02}.log`.
No test or diagnostic code is included in the production commit. Full fresh
quality and mandatory-dialog behavior remain unobserved until a user game runs.


Read-only launcher preflight passed on clean local source `ccde10e`, checkpoint
1768 exact model/Adam/RNG/progress/sampler/pool, CPU policy, MLX PPO, 100 games
plus one held-out diagnostic, no deadline, new stack-management support. Next
engine seed 1980002857, count 101, reserved=false. Evidence:
`mlx-scale/stack-management-1768-preflight.log`. Additional non-numerical plan
validation used temporary copies of existing training and diagnostic plans with
current exact source/implementation identities and the new support, without
reserving seeds, changing captures, saving checkpoints or executing games.
Both actual `_plan` gates and checkpoint metadata validation passed, and both
rejected unknown support and an unpinned MLX runtime. Evidence:
`mlx-scale/stack-management-plan-gates.log`. This exercises the gate missed by
the prior hybrid handoff; it is still not an end-to-end game or quality result.
Documentation-only amendment follows, then repeat read-only preflight on the
final clean commit for the user. No merge or push performed.

### 2026-11-15 — Authorized production consolidation into main

The user explicitly requested CPU policy / MLX PPO in main and clarified that
the experimental comparison branch must be preserved. Consolidate production
code/configuration, restored troop management and research findings from
`534f947`, excluding comparison runners, Swift/Python framework prototypes,
comparison-only dependency files and launch scripts. Keep those on the local
`codex/ppo-framework-comparison` branch, without publishing its tooling.
Production learner source bytes are unchanged from the prepared source; existing
48 tests and plan/preflight checks are recorded above, not rerun during gameplay.
No tests, probes or benchmarks are added to this publication.

A user-launched campaign `run-20261115T131956Z-07d9b15c` is currently active
in the canonical checkout. Therefore prepare this focused main tree with a
private Git index; do not switch the running checkout, modify its files/index,
stop/restart its process or change its HEAD. Publish main separately, preserving
the experimental branch and active source identity. Defer switching the working
checkout to main until that campaign finishes. Checkpoint and runtime assets
remain private and unchanged. This records consolidation, not new training or
validation results.


### 2026-11-15 — Investigate restored-stack capture failure; 1000 games deferred

The user withdrew the requested next 1000-game cycle and requested investigation
of the failed current campaign. No launch/restart or numerical probe is authorized.
Evidence: `run-20261115T131956Z-07d9b15c`, source `534f947`, parent 1768,
CPU policy / MLX PPO / restored-stack support. Campaign stopped after 1160.779 s
(19m21s), exit 1. Episode 37, engine seed 1980002893, failed during capture:
manifest `failure_detail` is `Fixed-weight diagnostic failed: ValueError:
Conditional action construction disagrees with support`. It reached day 227,
last Red decision seq 100717. This is a policy action-construction failure before
PPO, not evidence of a Metal update failure. Native -9 is recorded during failed
capture shutdown; it alone must not be labeled GPU OOM.

Report status denominators differ: engine outcomes record 58 naturally completed
games (42 Blue,16 Red), whereas attempt states contain 50 captured, nine
interrupted, one failed and 40 not started. Eight of the interrupted attempts had
completed engine games but preparation was interrupted. Zero samples admitted,
zero PPO updates, no checkpoint 1769, no final diagnostic. Latest remains complete
1768, weights `e7d559efa7ea1e56e7abc6e5debb78dbc4bb0c698bda3b0ee917e9098e70161d`.
Campaign-owned cleanup and protected inputs passed. No partial batch may train.

Read-only bounded extraction from the 7.1-GiB partial-policy-evidence.json shows
Red successfully split four creatures on day 227: source army-slot-369 to
army-slot-364, subject object-39, submitted seq 100711, split result accepted
seq 100716, then decision seq 100717. Afterward source count is nine and target
count four. Blue's last decision remains day 226. The failed newly sampled action
is not serialized, so do not claim its exact count was independently recovered.

Code inspection identifies a matching definite defect: split/transfer count
sampling ignores the immediately repeated quantity, while the final guard rejects
that command. The existing recruit/trade path already excludes repeated counts.
A repeated count of four for the same slots at this next decision therefore
passes native quantity range checks but fails final guarded support. A second
related defect filters an entire slot pair when count one is forbidden even when
other counts remain legal. These branches were inaccessible under the old
recruitment-only restriction and were not covered by the initial restored-action
checks. Exact stochastic reproduction has not run.

Local correction on main reuses the existing quantity representative and excluded
count-tree logic for split/transfer; keep the final legality check and all cooldowns.
Update the exact changed model-source pin. A non-neural control-flow check passed
10,912 count-tree paths (maxima 2..32, each excluded quantity and each allowed
target) and representative checks for split/transfer count-one exclusion. It
performed no model forward, optimizer, game or replay. Full model reconstruction
and end-to-end validation remain pending user-authorized execution. No new test,
probe or benchmark files are published. No commit or push performed for this fix.

The prior user-authorized main consolidation is already on GitHub (`1b4bd95`).
With the campaign stopped and no active training processes, switched the canonical
checkout to main as previously deferred. Preserve the experiment branch and all
failed-run evidence; generated native/framework-comparison build assets remain
local. Do not retire this failed run before preserving the bounded failure inputs.


### 2026-11-15 — Authorized saved-failure reconstruction and validation

The user explicitly authorized assistant execution of checks on the saved failed
party, followed by commit/GitHub sync only if checks pass, and preparation of
100 fresh games from 1768. The user will launch training; no new campaign is
authorized. Acceptance: restore the exact 1768 weights, reconstruct Red's saved
history and original CPU sampling RNG, reproduce the old support failure at
seq 100717/day 227, then sample and independently evaluate a valid fixed action
from the same state/RNG. Also run relevant existing tests and read-only launch
inspection. No partial failed campaign is admitted to training.

A private streaming reader extracts Red decisions/history from the 7.1-GiB
partial evidence, preserving compact failure inputs and checkpoint wrappers
outside Git before retention can retire old raw captures. It compares the
original distribution methods from published `1b4bd95` against the working
correction without changing weights or launching an engine. Evidence/scripts
remain private under `run-investigations/2026-11-15-stack-repeat/`; none are
publication scope. This entry is appended while extraction is in progress;
results and any failed validation attempts follow below.


Validation completed on the saved failed Red prefix: all 100,718 frames through
seq 100717 were reconstructed with exact 1768 weights, CPU inference and seed
1982005786. All 12,726 stored preceding sampled actions matched the original
methods from `1b4bd95`; maximum value and log-probability differences were both
zero. Original methods then reproduced the exact support-construction exception
at seq 100717/day 227. Instrumentation recovered the previously unrecorded failed
candidate: split_stack, object-39, army-slot-369 to army-slot-364, count 4.
The original failure diagnosis is now directly confirmed, not just inferred.

Resetting to the identical pre-choice RNG/state, the fixed distribution selected
count 5 for the same split. Native offers and guard both permit it; independent
evaluation reproduced its total and individual conditional log probabilities
exactly. A further 512 seeded samples on this state all passed support and exact
sample/evaluate agreement. All native quantities 1..9 for this pair were checked:
4 is excluded and evaluation rejects it; the other eight remain available.
Differentiable evaluation of those actions produced finite gradients in 19
parameter tensors, with no optimizer step or weight publication. Existing tests:
54 passed in 2.842 s across recruit_only_guard, interaction_guard,
contextual_management_model, selfplay_checkpoint and selfplay_training. These
are targeted checks, not a claim that the full historical suite is green.
Reconstruction phase took 69.68 s; no game, training update or native replay ran.

Private evidence: `run-investigations/2026-11-15-stack-repeat/{result.json,
final-state-result.json,existing-tests.log,check-saved.log,check-final-state.log}`.
Preserved compact Red history (about 81 MiB), decisions, full failing frame,
guard, recurrent/RNG state, both policy wrappers, plan, failure manifest and
finalization, plus private reproduction scripts. `preservation-manifest.json`
records their sizes/hashes. This bounded archive remains outside campaign
retention and Git; it permits subsequent investigation after ordinary retirement.
Complete parent checkpoint 1768 remains governed by the existing retention gate.

Retention-reviewed run: `run-20261115T131956Z-07d9b15c`.

The user authorized committing/syncing this verified production correction.
Publish only the model fix, exact source pin and documentation, without tests,
validation scripts, captures or weights. Prepare a new user-run 100-game cycle
from complete 1768 with fresh seeds, CPU policy / MLX PPO, restored troop
management and all other existing restrictions. Launcher preflight follows the
clean commit; full fresh-campaign success remains unobserved.


A separate transfer-window fixture passed all remaining quantities after a
count-one repeat, rejected the repeated quantity, and passed 128 exact
sample/evaluate checks (`transfer-check.log`). The validated candidate model SHA
is `3f0d74ee3ad47c7cf9eb0f4b6db9f525bc7ff00c8e0024bdace5aeca809d81bf`.
Together the saved-state and fixture checks cover both corrected branches.


Clean-source read-only preflight passed on local `d070595`: exact complete 1768
restore, 100 training + one diagnostic, CPU policy, MLX PPO, restored-stack
support, one update, no wall-clock deadline. Fresh engine base 1980002958,
101 reserved candidates previewed but reserved=false (no seeds consumed).
No gameplay, update or artifact cleanup occurred. Evidence:
`run-investigations/2026-11-15-stack-repeat/retry-100-preflight.log`.
After this documentation-only amendment, repeat inspection on the final commit
and retain `retry-100-preflight-final.log`, then sync that exact main to GitHub.
User command remains `./run-100-games.command 100 --mlx`; this is a fresh
100-game continuation from 1768, not resumption/admission of the failed batch.


### 2026-11-15 — First successful restored-stack cycle after repeat-count fix

User requested a current-run status/error check only. Read existing reports;
no workload, replay or optimizer was launched. `run-20261115T140446Z-0a723861`
on source `10ddb9f94586e99beb9d9032efeb8f691fe0a0fe` completed from parent 1768
to 1769: 100 captured/admitted games, zero failures or interruptions, 466,381
samples, 14,598 optimizer steps. CPU inference / MLX PPO, restored-stack support,
MLX/MLX-metal 0.31.1. Engine training outcomes: Blue 73, Red 25, two cutoffs;
training captures do not receive independent reconstruction/native replay.
Collection/preparation 2,217.5668 s (36m58s), production PPO 1,791.9264 s
(29m52s), final diagnostic/verification 154.0003 s, full cycle 4,170.1019 s
(69m30s), completed 15:14:18 UTC. The strict 600-second PPO target is not met.
Larger histories/action counts prevent a direct backend speed comparison with
the previous restricted 228,930-sample run.

Checkpoint 1769 passed the report's complete restore verification; weights
`733002cd969ca2a4b9d151d71a37027d2e29a07bd20fafc3404d17e88a7d8e27`.
Exit zero, empty shutdown errors, owned cleanup and protected inputs passed.
No recurrence of the support-construction failure was reported in 100 training
games or the diagnostic. This establishes a successful campaign, not immunity
to every possible future management state.

Diagnostic Blue (frozen 1768) beat Red (new 1769). Both policies reconstructed
exactly: Red 1,737 actions/values and 4,102 conditionals, Blue 2,825 and 7,145;
maximum difference zero, native replay passed. Restored actions were exercised:
Red/Blue split 88/65 times, merged 26/41, swapped stacks 554/1,513; Red also
upgraded twice. These are submitted-action counts, not a claim that every
operation was strategically useful or improved playing strength. Numerous
rearrangements warrant behavioral analysis before claiming learning.
Evidence: run report/shutdown and diagnostic neural-rollout/reconstruction JSON.
No commit, push, new launch or artifact removal performed during this check.


### 2026-11-15 — Detailed 1769 diagnostic: useful consolidation and slot cycles

Question: explain Blue's win, distinguish newly available management actions
from learned improvement, and identify whether rearrangement is productive.
Acceptance: trace events/armies and compare exact slot states within consecutive
submitted swap runs; do not infer strength or intent from counts. Read-only
analysis of existing player histories and rollout audit, no engine/model job.
Evidence: `run-20261115T140446Z-0a723861/evaluation-000/execution/episode-000001/`
`agents/{red,blue}/history-v8.jsonl`, `neural-rollout.json`, and reconstruction.
Source/backend/weights identities are in the preceding successful-cycle entry.
Red is 1769, Blue frozen 1768; both use CPU and restored-stack support. Day 31
natural terminal, Blue wins. Exact policy reconstruction and replay passed.

Red clears the nearby neutral on day 1 and buys 14 pikemen, builds upgraded
level-1 dwelling on day 5, hires a hero and combines it with town troops through
garrison swap (25 pikemen), upgrades that stack on day 6. A second hired hero
on day 16 brings 20 pikemen; a cross-hero swap passes the 25 halberdiers to it,
and the other hero upgrades its 20 pikemen on day 18. Red builds marketplace
day 19. Useful actions exist, but troops are scattered: before decisive fights,
25 pikemen on original hero, 22 halberdiers on another, four on another and
19 in town. Only 14 creatures purchased all game, three additional heroes hired.
Red sells two units of resource 5 for 100 gold on day 19, then buys one unit of
that same resource for 5,000 gold on day 23; ends that day with 105 gold. No
later construction uses it in this game. This is an observed poor exchange
sequence, not a claim about unobserved policy intent.

Blue buys 28 pikemen day 14 and 14 day 16, hires three heroes over days 14–15.
Repeated cross-army movement ultimately yields useful consolidation on day 18:
28 purchased town pikemen plus 12 from a hired hero form 40, then a merge into
the original hero's 25 produces 65 (Blue history seq 10673). Blue wins neutral
combat and captures the wood mine day 19. Day 20 kills Red's one-gnoll hero.
Day 22 Red original hero kills one Blue hero, falling from 25 to 15 pikemen;
day 23 Red's 22-halberdier hero attacks Blue's 65-pikeman hero and dies, Blue
retains 54. Day 27 Blue's weaker 15-pikeman/six-archer hero fails attacking
Red town; its 19-halberdier garrison falls to five. Day 28 Blue's 54-pikeman
hero captures that town, retaining 53. Day 29 Blue kills the four-halberdier
hero, day 31 its other hero (three griffins +21 goblins before combat) kills
Red's final 15-pikeman hero and wins. Blue's consolidation and Red's dispersed
forces explain the decisive material mismatch. Failed day-27 assault also
shows Blue's play was imperfect. Combat choices are BattleAI's, not the policy's.

Slot analysis uses every submitted swap_stacks matched to its result and the
latest preceding/following own_armies observation. Exact layout signatures retain
subject, slot, creature and count; inventory signatures aggregate creature counts
per subject, ignoring slot. A run ends at any other submitted action. These
metrics are descriptive, not a learned usefulness classifier.
Red: 554 swaps across 264 runs; 552 preserve every subject's inventory, 72 return
to a layout already seen in that run, 31 are two-step returns; longest run 12
(day 26, seq 12187–12264). Blue: 1513 swaps across 304 runs; 1498 preserve each
subject's inventory, 442 return to an earlier run layout, 80 are two-step returns;
longest run 30 on day 1, seq 114–317. Thus 2050/2067 swaps (~99.2%) only change
slot layout, and 514 swap completions return to an earlier full army layout in
the same consecutive-action run. Counts are returns, not distinct cycles, and
must not be added as disjoint categories. No swapped result was rejected.

Swap share of all decisions: Red 554/1737=31.9%, Blue 1513/2825=53.6%. Red also
splits 88 and merges 26 times, Blue splits 65 and merges 41. Pure reordering can
affect battle deployment, so unchanged inventory alone does not prove uselessness;
repeated reversals and returns with no intervening travel/battle are direct
behavioral-loop evidence. Most reordering has no observed lasting contribution,
whereas a small number of cross-hero operations demonstrably build useful armies.
The command cooldown compares ordered command arguments, so A-to-B and B-to-A
swaps are different commands; it neither recognizes equivalent inverse swaps nor
remembers visited army layouts. The existing 32-management-action budget bounds
bursts, not all possible later loops. Terminal-only rewards with gamma=1 have no
explicit per-action cost; this is a plausible contributor, not a proved causal
training diagnosis. Do not introduce penalties or new restrictions without a
separate user decision.

Positive Red examples relative to the previous recorded 1768 diagnostic are
first-day neutral combat, construction and upgrades, but upgrades/stack operations
were forbidden in that older game. The frozen 1768 Blue in the current game also
uses restored actions and constructs the winning consolidated stack. Therefore
new management behavior is not evidence that the 1769 update learned it. Red
makes fewer swaps both absolutely and as a fraction, but one unmatched game with
different positions/armies is not a reliable improvement/regression measure.
Current Red move targets still favor x=0 (96/132 versus Blue 76/157); Red ends
17 turns with some hero movement remaining, Blue zero. Overall no clear strength
improvement is established; useful behaviors coexist with costly cycles.
No code changes, simulations, training, commit, push or artifact removal in this
review. Research findings are appended locally.


### 2026-11-15 — Prepare cycle-aware swaps and user-run 1000 games from 1769

User request: fix slot-rearrangement loops while preserving useful troop
management, prepare 1000 games from 1769; the user launches training personally.
No game, optimizer, numerical probe or benchmark launch is authorized here.
Hypothesis: a history-derived visited-layout mask prevents exact rearrangement
cycles without removing native split/merge/transfer/upgrade actions or legitimate
new formations. Acceptance: version the new support, keep old snapshots and
checkpoint restore valid, block returns in both swap directions across window
reopening, reset on meaningful world progress, preserve the observed 65-pikeman
consolidation and mandatory exits, run existing tests and non-neural fixture/
recorded-history checks, and inspect a clean 1000-game preview (ten 100-game
updates, one final diagnostic, CPU policy / MLX PPO, no deadline).

Implementation adds `visible-interaction-budget-hero-relative-stack-cycle-v1`.
Tracker records semantic complete owned-army slot layouts at decisions, scoped
to the existing visible-progress epochs. A decision snapshot carries only the
canonical unordered subject/slot pairs whose swapped result has already been
seen. Slot refs/revisions and UI navigation cannot evade the mask. The existing
progress reset covers travel/world changes, a new day, combat and successful
recruitment/construction/upgrades; closing/opening windows does not reset it.
Only swap_stacks gains the visited-layout restriction. Split/merge/transfer,
quantity exclusion, garrison changes and upgrades retain existing legality and
budgets. Mandatory dialogs bypass optional restrictions as before. No reward,
model shape, engine, precision, PPO hyperparameter or backend change.

This removes exact cycles, not every strategically unhelpful novel permutation;
unchanged historical rules remain available for reconstruction. Broader changes
to action cost, exploration or formation budgets are outside this change.
Validation results follow. All validation code/evidence remain private and out
of any published diff.


Validation: 54 existing tests passed in 2.675 s (recruit_only_guard,
interaction_guard, contextual_management_model, selfplay_checkpoint,
selfplay_training). Private non-neural fixtures passed inverse and longer-cycle
rejection, window reopen persistence, novel-swap allowance, travel-progress
reset, tracker-copy isolation, strict snapshot roundtrip/rejection, historical
snapshot shape, all seven restored management kinds, optional close and a
mandatory policy-selected level-up choice. No numerical probe or game ran.
Evidence: private `run-investigations/2026-11-15-stack-cycles/`
`existing-tests.log` and `check-rules-final.log`.

Recorded-history annotation on the 1769 diagnostic passed 4,562 decision snapshot
roundtrips. It would reject 200/554 Red swaps and 637/1,513 Blue swaps (837 total),
while all 2,495 non-swap chosen actions retained their original permissibility.
Blue's useful merge to 65 pikemen remains allowed. Attempt 1 checked the wrong
submitted sequence number for that merge and failed its coverage assertion;
no action-preservation assertion failed. Inspection identified submitted seq
10668, result seq 10673; attempt 2 corrected the harness and passed. This also
corrects the earlier detailed-review reference to seq 10671. Evidence:
`check-recorded.log`, `check-recorded-02.log`, `recorded-result.json`.
These are annotations of fixed histories, not a counterfactual game trajectory,
measured speedup or evidence that all remaining permutations are useful.
All validation scripts/results remain private and outside the staged diff.

Retention-reviewed run: `run-20261115T140446Z-0a723861`.

Prepare a local production commit and read-only 1,000-game inspection from
complete 1769. The existing launcher already partitions 1,000 into ten batches
of 100 with one final diagnostic and twice the 500-game dashboard height; no
new launch wrapper or training job is added. Keep CPU policy / MLX PPO, starting
taverns, native neutral spawning, no artifacts, automatic combat, existing
cooldowns and native management legality. No wall-clock deadline, no new seed
reservation until the user launches. No merge or push is authorized by this
preparation request; keep the experimental comparison branch unchanged.


Clean-source 1,000-game read-only preflight passed on local `551fce7`: exact
complete checkpoint 1769 restore, CPU inference, MLX PPO, new cycle-aware support,
ten batches of 100, ten optimizer updates and one held-out diagnostic (1001
physical games). No deadline, seed reservation, cleanup, gameplay or optimizer
execution occurred. Next unreserved engine base 1980003059, count 1001; policy
base 1982006118, count 2002. Evidence: private `stack-cycles/next-1000-preflight.log`.
Actual training and diagnostic `_plan` gates plus metadata checks also passed
on temporary read-only copies carrying current source identities/new support,
and unknown support was rejected (`stack-cycles/plan-gates.log`). No checkpoint
was saved by these checks. A documentation-only amendment follows; repeat preview
on the final clean source (`next-1000-preflight-final.log`) before user handoff.
Production source/configuration only is committed locally; validation scripts,
raw captures and test changes are absent. GitHub publication is not performed.


### 2026-11-16 — 1000-game training saved; final cleanup not certified

User asked whether `Process cleanup deadline exhausted; closure is not verified`
is a problem. Read-only inspection, no workload, kill, recovery, receipt rewrite
or gate bypass. Run `run-20261115T230642Z-da5480b8`, source `9e85dd2`, CPU
policy / MLX PPO / cycle-aware stacks, parent 1769. All ten batches are
checkpoint_verified, 1000 captures/admissions, 10 PPO updates, 4,892,383 samples.
Zero capture failures or interruptions; engine outcomes Blue 681, Red 284,
35 cutoffs. Latest complete published checkpoint 1779, weights
`30fec04b42477288e6f0d94b311196deaae464e94718df1e8d657e82ca886ca3`.
Final paired restore verification is true, policy/resume files exist and their
SHA256 values match the publication; latest-checkpoint.json points to this pair.
Collection/preparation sum 30,515.8663 s, complete PPO sum 20,788.0849 s;
overall elapsed 51,661.9329 s (14h21m02s), coordinator ended 13:27:45 UTC.

Final diagnostic evaluation-009 passed: Blue wins, Red loses; exact
reconstruction has max difference zero for both policies (Red 4047 actions/values,
9495 conditionals; Blue 3875,9182), native replay passed. Training captures are
not independently replay-verified. The coordinator returned zero, but final
supervision recorded owned_cleanup_verified=false and marked report stopped.
Protected inputs are unchanged. Thus training publication and diagnostic success
must be distinguished from an uncertified shutdown; do not call the entire
campaign cleanly completed or discard its saved updates.

Supervisor finally uses close_owned_process(timeout=5) and then audits each
execution deadline receipt with cleanup_from_evidence(timeout=5). The identical
TimeoutError can arise at either step and does not name the offending group;
the existing log cannot conclusively identify the original timeout's exact group
or cause. Coordinator guardian recorded parent_exited at 13:27:47 UTC; shutdown
file was written about 19 seconds later. Current ps shows no training coordinator,
VCMI client or process_deadline worker. Do not infer proof of historical closure
solely from that current command-name search.

Read 2003 deadline receipts (2003 launches,1991 distinct recorded PGIDs); one
historical numeric group currently maps to an unrelated Codex node_repl. Its
PID/PGID 70187 was the native worker for batch 004 episode 71 at 04:56 UTC and
its guardian recorded parent_exited at 04:57. Current node process start is
15:47:42 Kyiv (13:47 UTC), after this run's cleanup failure. This demonstrates
numeric PGID reuse and why historical groups must not be signalled; it does
NOT establish the cause of the original timeout, since this current process
started afterward. No process was killed. A future repair/reconciliation must
preserve identity-aware ownership and fresh closure evidence. The current
owned_cleanup_verified=false blocks ten_inspect and therefore another launch;
do not manually flip the flag or bypass the gate. No retirement marker for this
run is added before shutdown reconciliation.

Correction to earlier diagnostic interpretations: the evaluation plan binds
BOTH learner and frozen-opponent weights to 1779's identical digest. The name
`evaluation-009-frozen` does not mean checkpoint 1778. Source inspection also
confirms the preceding published controller (`10ddb9f`) passed state.model and
previous_pair['policy'] AFTER its update when constructing the final diagnostic.
Earlier journal/chat assertions that 1769 Red played against 1768 Blue (and that
similar final diagnostics compared adjacent checkpoints) were incorrect. Those
within-game behavioral observations remain valid, but they are self-play of the
same current checkpoint, not direct evidence for an adjacent-checkpoint win/loss.
Do not reuse those outcome-based strength comparisons. No new paired-seed quality
evaluation was performed. Evidence: run report/shutdown/latest publication,
evaluation plan/reconstruction and archived controller source.


### 2026-11-16 — Authorized cleanup fix and fresh closure reconciliation

User requested correcting final cleanup and verifying process completion so the
next launch is unblocked, preserving complete checkpoint 1779 and launching no
training. Acceptance: retain fail-closed behavior for live/ambiguous original
processes, never signal historical numeric groups, distinguish reused process
identities, batch receipt inspection, preserve original failure evidence, and
set closure verified only after a successful fresh audit. Verify exact model/
Adam/RNG/progress through read-only launch inspection afterward; no game or PPO.

Checkpoint 1779 policy and complete resume plus original report/shutdown/launch,
latest pointer and consumed-seed ledger have hash-verified private backups under
`run-investigations/2026-11-16-cleanup/` (`preserved-inputs.json`). Original run
and checkpoints remain in place. The known timeout's original offending group
is not identifiable from its old error; do not present PID reuse as proved cause.
The previous check did independently demonstrate a reused PGID after the run.

Implementation keeps guardian ownership/signalling unchanged. Receipt cleanup
accepts a batch, validates each launch receipt, uses one current process-table
snapshot per pass, and checks matching private-session membership. A matching
numeric session leader is excluded only when its process birth is provably
later than the recorded creation upper bound (with a one-second precision
margin). Missing legacy identity data or uncertain/live originals still block.
Existing direct-child cleanup receives a creation upper bound from its Popen.
Supervisor writes explicit closure-audit evidence and batches historical receipts
under one bounded 30-second deadline rather than running ps for each receipt.
These are existing cleanup functions, not a replacement orchestration/recovery
system. Private process-control validation runs only benign helper subprocesses;
no game, model or optimizer execution. Results and reconciliation follow.


Validation passed: 22 existing process_deadline tests (final run 2.547 s), plus
seven private lifecycle/identity cases (0.503 s). Private checks cover different
sessions with reused PGIDs, newer same-ID private-session leaders, live original
and birth-time-ambiguous leaders, surviving descendants with no leader, unresolved
receipts, a 2003-receipt single-snapshot batch, and a real benign owned helper
that remains blocked while alive and closes only after ownership release. No
historical numeric group was signalled by receipt inspection. Tests and private
scripts stay outside the production commit. Original deadline-exhaustion and
late-scan rejection tests remain green. Evidence: private cleanup investigation
`process-tests-final.log`, `private-tests-final.log`, `check_cleanup.py`.

Under the existing exclusive activity lease, a fresh audit matched every expected
worker/native/coordinator receipt against the complete report (2003), checked all
ten checkpoint pairs' policy/resume hashes, and rechecked protected inputs and
the original latest pointer/seed ledger. The corrected audit verified all 1991
distinct groups in 0.0801 s with zero original live members. Current PGID 70187
was safely excluded by its different session (SID 1). This confirms present
closure; it does not identify the cause of the original finalization timeout.
Evidence: run `closure-recheck.json`, a private copy, and `reconcile.log`.

Only after the audit passed, shutdown/report were atomically updated to closure
verified and completed, retaining `initial_errors` / `initial_supervisor_errors`
and a binding to the new audit. Original report/shutdown remain hash-bound in
private backup. Coordinator return code and original end time are unchanged.
All checkpoint, latest pointer, launch and seed ledger hashes remain unchanged.
No training or replay ran; no samples were newly admitted and no weights updated.
No artifacts were deleted and the unrelated reused-PGID process was not killed.
This is fresh evidence reconciliation, not manually bypassing a failed gate.

Prepare a local production commit with only cleanup code, exact source pins
and documentation, then run the normal read-only launcher check against 1779.
No new launch, merge or GitHub push is authorized by this cleanup request.


Final hardening keeps a member that disappears between ps and getsid unresolved
until another whole-table snapshot, because it may have forked before exiting.
A private regression requires that second snapshot. Final validation: 22 existing
tests passed in 2.562 s and eight private checks in 0.583 s; logs are
`process-tests-release.log` and `private-tests-release.log`. No validation files
are staged. A second fresh full audit with the release implementation passed
2003 receipts /1991 groups in 0.2873 s, zero original live members, same safely
excluded different-session PGID. `closure-recheck-final.json` binds the preceding
audit and the release implementation; report/shutdown point to this later proof.
Checkpoint, pointer, launch and seed ledger hashes are unchanged again.

Normal read-only launcher inspection passed on clean local `ee5ce83`: complete
1779 model/Adam/RNG/progress/sampler/pool restored exactly, no pending diagnostic
or cleanup block. It performed zero gameplay/optimizer updates, reserved no seeds
and removed no artifacts. Evidence: private cleanup investigation
`ready-1779-preflight.log`. This check revalidates the full saved publication
chain, not only the manually repaired report fields. Record this result and the
last conservative race hardening in a local amendment, then repeat the read-only
inspection against its clean final source (`ready-1779-preflight-final.log`).
No GitHub push, new training campaign, native replay or evaluation game performed.


### 2026-11-16 — Correct missing retention review before the next user launch

User reported launching another 100 games from 1779 and requested status. There
is no new campaign directory, consumed seed reservation or active training
process. The latest run remains the completed/reconciled 1000-game campaign;
its shutdown is closure verified with no current errors. Inspection identifies
a separate prelaunch blocker: cycle_retire requires the latest run's exact
Retention-reviewed marker, which was intentionally withheld before closure
reconciliation and inadvertently not added after reconciliation succeeded.
The previous handoff therefore incorrectly implied full launch readiness:
`--check` does not exercise this journal/retirement prerequisite.

The 1000-game results, checkpoint identities, timings, diagnostic verification,
limitations and diagnostic-opponent correction are already recorded above.
Original full run report/launch, closure evidence, complete checkpoint 1779
policy/resume and seed ledger are preserved outside run retention under
`run-investigations/2026-11-16-cleanup`. Rechecked live policy/resume SHA256 and
completed/verified closure before marking retention reviewed. No rule is removed
or bypassed; this supplies the documented review after actually completing it.
Normal next user launch may retire old raw captures while preserving complete
continuation state. No run artifacts were deleted by this status check.

Retention-reviewed run: `run-20261115T230642Z-da5480b8`.

Commit this documentation-only correction locally so the checkout remains clean,
then inspect the 100-game continuation without gameplay, updates or seed
reservation. The user launches personally. No runtime code change, training,
replay, process restart, merge or push is performed.

### 2026-11-18 — Commit the staged retention review that blocked the next launch

The user's next `./run-100-games.command 100 --mlx` attempt stopped before
gameplay with "Selected checkout must be clean and committed". Cause: the
2026-11-16 entry above, which supplies the exact Retention-reviewed marker for
`run-20261115T230642Z-da5480b8`, was staged but never committed. `cycle_retire`
reads the working-tree journal, so its marker check (the cause of the earlier
"Analyze the latest run and record its Retention-reviewed run marker" refusal)
now passes, but `checkout_identity` in `tools/benchmark_copied_training.py`
correctly refuses any uncommitted change because every launch binds the clean
source commit. Both refusals are prelaunch gate results, not game outcomes.

Read-only inspection found no side effects from either refusal: no new run
directory, seed reservation, retained checkpoint copy or deletion under
`selfplay-ten-then-check` since 2026-11-16, no game or learner process running.
Correction: commit the 2026-11-16 entry unchanged together with this note. No
guard, marker rule or runtime code is changed.

The "Process cleanup deadline exhausted; closure is not verified" message is the
initial shutdown error of that same 1000-game run. It is already superseded by
the verified closure recheck recorded above: the run's `shutdown.json` keeps it
under `initial_errors`, with empty `errors`, `owned_cleanup_verified: true` and
`protected_inputs_unchanged: true`, which is what the next launch requires.

The next normal cycle launch will copy checkpoint 1779 into `retained/` and then
delete the reviewed raw run (about 94 GiB); its report, launch, closure evidence,
checkpoint pair and seed ledger copies remain under
`run-investigations/2026-11-16-cleanup`. The assistant ran no `--check`,
gameplay, training, replay or benchmark; the user launches personally. No merge
or push is performed.

### 2026-11-18 — 1000-game MLX cycle stopped by the owned-process memory guard

User launched `./run-100-games.command 1000 --mlx` at source `a8637ca`
(run `run-20261118T003841Z-91a203dd`, parent checkpoint 1779, ten planned
batches of 100 plus one final diagnostic, 1001 consumed training seeds from
ledger offset 4060, engine base 1980004060, policy base 1982008120). Collection
used ten actors, CPU policy inference and MLX PPO. The user reported the run as
failed; this entry records read-only analysis of its private artifacts.

Observed: five complete batches each passed one PPO update and checkpoint
publication, 1779 -> 1784 (500 admitted games, 3,224,647 learner samples).
Engine outcomes are unverified training captures, not learner results:

| Batch | Engine Red / Blue / cutoffs | Samples | Collection s | Update s |
| --- | --- | --- | --- | --- |
| 000 | 17 / 78 / 5 | 500,788 | 2,319 | 2,135 |
| 001 | 31 / 66 / 3 | 586,267 | 2,618 | 2,746 |
| 002 | 21 / 76 / 3 | 636,201 | 2,918 | 3,439 |
| 003 | 14 / 82 / 4 | 755,229 | 3,292 | 4,474 |
| 004 | 23 / 72 / 5 | 746,162 | 3,511 | 4,341 |

Batch 005 began collection at 09:29:12Z. About 12 minutes later the supervisor's
`require_cycle_memory_budget` stopped the run with "Cycle memory budget reached"
(owned-process resident memory above physical RAM minus the 48 GiB reserve, i.e.
80 GiB on this 128 GiB Mac). At that point 13 games were captured, 10 running
games were interrupted and 77 had not started; no batch-005 update or publication
occurred. Shutdown records `owned_cleanup_verified: true`,
`protected_inputs_unchanged: true`, process closure verified over 1047 groups,
and coordinator return code -9 (killed by the supervisor after the guard), so
`report.json` has no `ended_utc` or final diagnostic. The latest pointer is the
complete publication chain at checkpoint 1784; the next launch must first run
the pending diagnostic recovery.

Evidence for the cause, which is inferred rather than measured because the guard
does not log per-process memory: four of the ten concurrent games (episodes 7, 9,
10 and 11) had run since batch start and reached days 245–253 with 23k–26k
learner choices each, far above the batch-004 mean of about 7.5k samples per game;
each had already written about 5.6 GB of temporary contextual replay. A capture
worker returns its whole episode from `run_episode`, so its Python history grows
with game length, while the coordinator held only 13 newly prepared games. The
guard therefore most likely tripped on several simultaneous very long games, not
on the PPO update: the 755k-sample update in batch 003 completed under the same
guard. Samples per batch rose by about 50% over five batches (the previous
1000-game run peaked at 661k), so recurrence is plausible. This is not evidence
of learning or regression in either direction.

No guard was changed. Whether to keep ten actors (a recorded decision) or reduce
concurrency is left to the user. Compact evidence (report, launch, shutdown,
closure, deadline log, progress files, seed ledger, lineage, latest pointer,
batch-004 publication, checkpoint 1784 policy/resume with matching SHA256, the
batch-005 plan/attempts and logs of the interrupted episodes) is preserved under
`run-investigations/2026-11-18-memory-stop`. The assistant ran no gameplay,
training, `--check`, replay or benchmark; the user launches personally.

Retention-reviewed run: `run-20261118T003841Z-91a203dd`.

### 2026-11-18 — Remove the cycle owned-process memory guard (user decision)

After the analysis above, the user decided to remove the cycle memory guard
completely rather than reduce the ten actors. History: the guard was added on
2026-11-05 (`cf92730`) together with one-game-at-a-time history compilation,
after an earlier stopped run had shown a memory warning caused by unbounded
concurrent full-history retention (that stop recorded no measured peak). It ran
every two seconds in the supervisor and stopped the whole cycle when the summed
resident memory of the supervisor's process tree exceeded physical RAM minus
48 GiB, or when `memory_pressure -Q` reported less than 20% free memory.

Implemented: `require_cycle_memory_budget`, its supervisor call and the
`memory_reserve_gib=48` launch-configuration field are removed from
`tools/run_selfplay_training.py`; README updated. Unchanged: ten actors, the
30 GiB disk reserve, deadlines, seed/checkpoint/source/cleanup gates and MLX's
own 25%-of-working-set allocator cap with its PPO MemoryError check.

Expected consequence: concurrent very long games may now drive macOS into swap
or compression, slowing the run, and in the extreme the OS may terminate a
process; a terminated worker or coordinator still stops the run through the
existing failure paths and preserves the last published checkpoint. Validation:
the module compiles; no test, `--check`, gameplay or training was run by the
assistant. The user launches personally.

Correction (same day): the first relaunch after `fadbdc9` stopped before any
gameplay with "Evaluation changed an existing learner, adapter, engine or
replay source". `verify_runtime_sources` in `tools/run_selfplay_one_game.py`
binds the cycle controller to an exact reviewed SHA256 (the `after` pin of
`tools/run_selfplay_training.py` in `MLX_PPO_SOURCES`), which the guard removal
changed. Following the precedent of `d0f0fb0`, that single pin is updated to the
new controller digest `8b1fa8d1c215a3c779b59ce8e41d84a22629d7587f10745ed3ec1fe70b5fcd24`;
no other source pin or check changes. The refusal reserved no seeds and changed
no run artifacts.

### 2026-11-18 — Replace the training pipeline with one small package (refactor branch)

**Goal (user request).** Throw away dead and overbuilt code and reduce the
project to a simple core that is quick to change and to optimize, while keeping
what learning needs: the engine binding, the policy, PPO and checkpoint
continuity from 1,784. Work on branch `claude/project-thread-tlvy67`; nothing
was merged to `main` and the user's checkout was not edited.

**Acceptance criteria, set before the rewrite.** (1) The moved policy
recomputes recorded decisions of captured games bit for bit. (2) The MLX update
of the new layout hands its GPU backend the same inputs as the old one for the
same games. (3) Checkpoint 1,784 imports into the new format with identical
weights, Adam state, shuffle generator and counters. (4) The loop runs end to
end against a scripted engine: play, update, save, resume. (5) A first real run
on the Mac, launched by the user. Criterion 5 is open.

**Implemented.** The loop is now the `heroes/` package: 34 Python files and
8,823 lines in the repository, down from 466 files and 121,882 lines.

- `heroes/engine/`: process launch, wire protocol, a two-seat environment that
  classifies each game as completed, truncated or failed.
- `heroes/actor.py`: plays one game with one set of weights on both seats and
  builds the learner seat's trajectory.
- `heroes/learner/`: PPO targets and loss windows shared by a plain autograd
  reference update and the MLX update.
- `heroes/checkpoint.py`: one file per update plus `latest.json`, and a one-time
  import of the previous `policy.pt`/`resume.pt` pair.
- `heroes/train.py`: restore, collect a batch with a pool of actor processes,
  update, save, repeat.
- `heroes/model/policy.py`, `heroes/game/contract.py`, `heroes/game/guard.py`
  and `heroes/learner/mlx/` are the existing policy, typed contract, interaction
  guard and MLX learner, moved and trimmed.

Removed, and still in Git history up to `5efe811`: launch plans, capture
bindings and receipts, evidence envelopes, source-hash pins, the supervisor and
its fourteen launch scripts, retention and cleanup audits, the final diagnostic
game with two-policy reconstruction and native replay, the CPU native and
PyTorch/Metal learner backends, the replay inspector app, map generators,
recording tools, `validation/`, `reports/`, `configs/` and the previous test
suite. `engine/` is unchanged.

**Behaviour that changed.** These are deliberate simplifications, not
measurements.

1. *Truncation.* A decision of either seat past day 365, or after 163,840
   answered decisions, stops the game at once. The learner keeps every completed
   action whose following decision its policy saw, and that decision's value is
   the bootstrap. The previous paired cutoff with bounded continuation and
   dialog draining is gone. The separate 5,120 reserved choices are folded into
   the single action limit.
2. *No proof frames.* Histories no longer end with `finalized`,
   `cutoff_finalized` or `failure` frames. The policy never read them.
3. *Failed games are replaced.* A failed game is logged and another is played.
   A run stops only when more games failed than the batch needs, or when the
   first ten all failed. Previously one failed capture stopped the cycle.
4. *No final diagnostic.* No game is reconstructed or natively replayed. The
   diagnostic that the interrupted run of 2026-11-18 still owed is not carried
   over.
5. *Seeds.* Engine and sampling seeds are drawn fresh per game and written to
   `log.jsonl`. The seed reservation ledger is gone.
6. *One guard version.* Only
   `visible-interaction-budget-hero-relative-stack-cycle-v1` remains
   executable. One model schema remains, `heroes-context-model-v6`.
7. *Memo validity.* Encodings reused while acting are tied to the parameters'
   version counters instead of a bit comparison of the weights on every frame.
8. *Every checkpoint is kept.* Nothing retires old files.

Unchanged: PPO settings (λ 1, clip 0.2, value clip 0.2, value coefficient 0.5,
gradient norm 0.5, one epoch, windows of 16 decisions, two windows per
minibatch, Adam 3e-4, normalized advantages), ten actors, 100 games per update,
both seats on the batch-start weights with the learner seat alternating, the
engine settings and environment, the scenario, `no-artifacts-v1`, and the MLX
network, kernels and first-minibatch sanity limits. Engine replay logging stays
on. An earlier text of this entry offered a switch for it; the review entry
below corrects that.

**What actually ran.** Everything below ran in a cloud sandbox on Linux x86-64
with CPython 3.13 and Torch 2.9.1. No VCMI, no Metal and no MLX kernel ran, and
nothing ran on the Mac. The inputs were read-only copies of the checkpoint 1,784
pair and of three captured games (episodes 000006, 000011 and 000019 of
`run-20261118T104905Z-ebe300b5`, batch 000) with that batch's `learner.pt`.
No PPO update was run on captured games.

- *Policy regression.* For both seats of the three games (19,667 frames, 2,580
  recorded decisions), the old code's recomputed log-probabilities, values and
  final recurrent state were hashed at `5efe811`. The new code reproduces every
  hash, before and after the trim.
- *Trajectory and recipe.* For the three learner histories (1,433 samples) the
  new sample builder returns the same decisions, log-probabilities, values and
  reward as the stored samples, and the compiled recipe has the same digest as
  the old compiler's. The only difference is the old trailing proof frame, one
  empty recipe entry.
- *MLX boundary.* The GPU backend class was replaced by a recorder in both the
  old and the new update, so no numerical step ran. For the three games the two
  updates produce identical manifests, whole-game prefix plans, 46 minibatch
  plans with their old log-probabilities, values, advantages and returns, and
  the same shuffle generator state afterwards.
- *Checkpoint import.* The 1,784 pair imports with weights digest `6f5f6592…`
  as recorded, progress 1,784 updates / 9,822 games / 12,926,530 decisions /
  406,884 optimizer steps, and survives a save and load unchanged.
- *Tests.* 25 tests pass. They use `tests/fake_vcmi.py`, a scripted stand-in
  that speaks the real wire protocol over real sockets, with random weights.
  They cover natural, terminal-closed, truncated, crashed and silent games, PPO
  targets and windows, the reference update, the acting memo, checkpoints
  including a synthetic legacy pair, recipe compilation, and the whole loop
  with resume and legacy import.

**Limitations.** The scripted engine is not VCMI: it plays end-turn-only games
on a one-tile map. Engine behaviour under the new environment (startup, the
stop at a limit, shutdown, ten concurrent processes) is untested until the first
real run. The MLX path is verified up to the backend boundary only; its
first-minibatch check against recorded behaviour is the remaining safeguard.
None of this says anything about learning or playing strength.

**Next.** The user launches a small trial into a scratch state folder, for
example `./train.command 10 --games-per-update 10 --state
~/Projects/Heroes-artifacts/training-trial`, then the real run. The assistant
ran no gameplay, training, replay or benchmark on real data beyond the
forward-only regression above.

### 2026-11-18 — Review of the refactor branch: thirteen findings, one correction

**Question.** Does the new pipeline contain defects that a real run on the Mac
would hit and that the scripted tests and the three-game regression cannot see?

**Method.** Separate reviewer sessions, which did not see the author's
reasoning, each compared one area of the new code with the pipeline at `5efe811`
and with the C++ engine side: the engine binding, what is trained on, the
trimmed policy/guard/contract, and the trainer with its process pool and
checkpoints. A second reviewer then tried to refute each finding. A second pass
reviewed the trainer again and the commit that fixed the first pass. Reviewers
ran only the scripted engine with random weights, differential tests of old
against new code on synthetic inputs, and the three existing read-only
regression checks. No real engine, no MLX and no update on captured games ran.
The first pass was cut short by a usage limit: its trainer review and the
challenge of the learning and model findings did not run. The second pass
covered the trainer, and the author reproduced the learning and model findings
with tests.

**Correction to the entry above.** That entry and the first README of this
branch offered `--no-replay-log` to stop the engine's replay logs. This would
have broken every naturally finished game. In evidence mode the client shuts
down only on the server's terminal barrier, and the server builds that barrier
from the replay recorder (`engine/patches/0013`, `0015`, `0029`); without the
recorder the call throws. This was established by reading the patches, not by
running the engine. The previous binding refused that configuration. The switch
is removed and `VCMI_EXPERIMENT_REPLAY=1` is always set. Removing the logs
needs an engine change and a rebuild.

**Findings of the first pass, fixed in `bac2a10`.**

1. The replay switch above.
2. A relative state folder reached the engine unresolved, although the engine
   runs in its binary's folder. Every game would have failed at startup.
   Reproduced with the scripted engine. The folder is now resolved.
3. An engine that exits before either seat connects was reported as a reader
   timeout after five seconds, hiding its exit status and log marker.
   Reproduced. Readers now stop waiting once the engine has exited.
4. A dead actor was noticed only after a second without any message from any
   actor, so not while others reported progress. Reproduced.
5. An actor killed while idle left its inbox queue locked, so its replacement
   never received a game and the batch hung. Reproduced. For 4 and 5 every
   actor now has its own pipe: its death is an end-of-file the trainer sees at
   once, and a replacement gets a new pipe.
6. Game numbers restarted from the count of usable games, so failed and
   discarded games shared numbers with later ones.
7. The acting memo stayed valid after weight changes that do not bump a
   parameter's version counter (`assign=True`, a replaced `.data`). No code path
   of the package does this. The memo key now includes the storage address and
   loading a state dict drops the memo. In-place writes through
   `parameter.data` remain undetected and are documented as unsupported.

**Findings of the second pass, fixed in the following commit.**

8. A state folder with checkpoints but without `latest.json` counted as empty:
   the run re-imported 1,784 (or started from random weights with `--fresh`)
   and replaced later checkpoints one by one. Nothing in the pipeline removes
   the pointer, so the trigger is manual. Such a folder is now refused and the
   message says how to restore the pointer.
9. The actor's watch for a dead trainer read the parent's process id inside the
   child. An actor still starting when the trainer was killed never noticed, and
   with a silent engine it stayed for good. The trainer now passes its own id.
10. `--workers 0` waited forever without output. Arguments are now checked.
11. Orphaned actors of a hard-killed trainer could delete the game folders of a
    run restarted within about a second, because in-flight game numbers were
    reused. The challenger could reproduce this only with a scripted engine
    that ignores SIGTERM and judged it unreachable with the real one. It is
    closed anyway: a `job` record is written to `log.jsonl` when a number is
    handed out, and numbering continues after the highest such record. The
    challenger's reproducer no longer collides.
12. The new startup cleanup of `games/` deleted folders an earlier run had kept
    with `--keep-game-files`. Kept folders now live in `kept-games/`.
13. The new actor-death test could misread a progress message under extreme
    load. Not observed without injected delays; the test now waits for the
    message it means.

**Changes prompted by reviewer notes that were not findings.**

- The environment waits for engine events one second at a time, so an actor
  that is told to stop cannot sit out a silent engine. A wait without a timeout
  is interrupted by signals on Linux, where CPython's locks are semaphores.
  Where they are a mutex and a condition variable, which to the author's
  knowledge includes macOS, POSIX does not guarantee that. Not verified on
  macOS.
- The actor's SIGTERM handler fires once, so a second signal cannot interrupt
  the cleanup the first one started.
- `latest.json` is synced to disk before it is renamed into place.
- `train.command` attaches `caffeinate -is -w` to the trainer instead of
  wrapping it, as the previous launcher did, so Ctrl-C returns once the trainer
  has stopped its games. Checked on Linux with stand-ins for `caffeinate` and
  Python; not run on macOS.
- The message after Ctrl-C names the last update that was saved.

**What the reviews found to be equal to the previous pipeline.** These are the
reviewers' own reports of what they ran.

- The launch (command line, environment, settings, profile layout) equals the
  recorded production launch apart from paths and ports. Production also ran
  without an engine deadline.
- Six real seat histories of up to 4,980 frames round-trip through the new wire
  protocol frame for frame.
- On 200,000 random histories the old and the new code select the same trained
  decisions, reward and usable/unusable verdict. Cutting the three captured
  games at every day gave 116 of 116 correct truncated trajectories.
- On 1,400 synthetic histories (26,826 frame encodings) the old and the new
  encoder agree on every frame. With equal weights and seeds, 3,458 steps and
  3,912 sampled actions are bit-identical between old and new model.
- The single remaining guard version equals the old stack-cycle version on
  16,142 decisions and 1.2 million permission checks.
- The weights digest equals the old one on 400 random state dicts.
- The legacy pointer file and the 1,784 pair on the Mac snapshot import as
  expected (read only).
- With the scripted engine on Linux: SIGINT, SIGTERM, SIGKILL and SIGHUP in
  several phases leave no process behind; twelve actors killed at random during
  a 40-game run gave twelve `actor_died` records, 40 completed games and unique
  numbers; 200 parallel games over four ending shapes were all classified
  correctly.

**What actually ran after the fixes.** 32 tests pass, each new test fails
without its fix, and the three regression checks on captured games pass
unchanged (19,667 frames and 2,580 decisions; 1,433 samples; 46 minibatch
plans). The reviewers' reproducers for findings 9 and 11 no longer reproduce.

**Limitations.** Everything ran on Linux. Signal delivery, `flock`, process
spawning and `caffeinate` on macOS were judged by reading. How the real engine
reacts to a closed policy socket or to SIGTERM was read from the C++ side and
from the previous pipeline's production use, not run. The review cannot replace
the first real run, which remains open.

### 2026-11-18 — First real runs of the new pipeline (user-launched) and timing against main

**Question (user request).** How long does it take to train ten games on `main`
and on the refactor branch? No acceptance criterion was set in advance. This is
a read-only comparison of runs the user had already launched; the assistant
started no game, training, check or benchmark, and edited nothing on the Mac.

**What ran, all launched by the user on the Mac.**

- `main` at `5efe811`: one cycle of 100 games, run
  `run-20261118T104905Z-ebe300b5`, launched 10:49:05 UTC, parent checkpoint
  1,784 (weights `6f5f6592…`), CPU policy, MLX PPO, ten actors. It ended with
  status completed and exit code 0 and published checkpoint 1,785.
- The branch at `e46fe66` (checked out on the Mac at 21:28 UTC): two runs of ten
  games and one update each into the scratch folder
  `~/Projects/Heroes-artifacts/training-trial`, started 21:28:34 and 21:49:56
  UTC, CPU policy, MLX PPO, ten actors. Engine and sampling seeds of all 20
  games are in that folder's `log.jsonl`.

Evidence: `launch.json`, `report.json`, `shutdown.json`,
`batch-train-000/attempts.json`, the 100 episode manifests and
`latest-checkpoint.json` of the `main` run; `log.jsonl`, `latest.json` and a
read-only copy of `checkpoints/update-001785.pt` of the trial folder.

**Correction to the two entries above.** They call 1,784 the last checkpoint of
the previous pipeline and say that a diagnostic game was still owed. Both were
true for the snapshot the refactor was checked against, taken while the `main`
run above was collecting. That run then played the owed diagnostic (670 s),
trained, published **1,785** (9,922 games, 13,713,041 decisions, 431,486
optimizer steps, weights `6a41cead…`) and passed its own final diagnostic. The
previous pipeline's last checkpoint is therefore 1,785 and no diagnostic is
owed.

**Observed: criterion 5 of the refactor, the first real run.**

- The one-time import chose 1,785 from `latest-checkpoint.json`, not the older
  pair under `retained/`. The copy of `update-001785.pt` loads with weights
  digest `6a41cead7e35c388b94d60908affddf68760bd42ed650393fdaa0c404028ad3f`,
  the digest `main` published, and with the same counters.
- 20 games were handed out and all 20 ended naturally. No game failed, was
  stopped at a limit or had to be replaced. `games/` is empty afterwards.
- Both MLX updates completed and were saved. The update raises before its first
  optimizer step when its first minibatch disagrees with the probabilities
  recorded during play, so both passed that comparison.

  | Update | Games | Learner samples | Optimizer steps | Approx. KL | Clip fraction | Policy loss | Value loss |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 1,786 | 9,923 to 9,932 | 105,875 | 3,311 | 0.00278 | 0.0233 | 0.00040 | 0.341 |
  | 1,787 | 9,933 to 9,942 | 172,718 | 5,400 | 0.00450 | 0.0295 | 0.00096 | 0.465 |

- The second run resumed from the first run's `latest.json`. Weights after
  1,787 are `ef2ebef0…`.
- Not observed with the real engine: a game stopped at day 365 or at the action
  limit, a failed or replaced game, a killed actor, an interrupted run, two runs
  competing for one folder. Nothing has run in the real state folder
  `~/Projects/Heroes-artifacts/training`; it does not exist yet. The two trial
  updates live only in the scratch folder.

**Timings as recorded.**

| | `main`, 100 games | Branch, run 1 | Branch, run 2 |
| --- | --- | --- | --- |
| Decisions of both seats | 1,546,610 | 183,683 | 319,372 |
| Learner samples | 786,511 | 105,875 | 172,718 |
| Optimizer steps | 24,602 | 3,311 | 5,400 |
| Diagnostic games | 669.8 s owed + 306.1 s final | none | none |
| Collection | 3,532.4 s | 589.0 s | 1,004.2 s |
| Update | 4,798.8 s | 652.4 s | 1,692.0 s |
| Whole run | 9,317.9 s | 1,241.4 s | 2,696.2 s |

Inside the update, `main` reports 4,767.6 s for the steps including 275.2 s of
plan building and 54.0 s of state validation; the branch reports 651.4 s
(31.8 s, 7.1 s) and 1,690.1 s (48.1 s, 10.7 s). The `main` batch had 95 natural
ends and 5 games stopped at the day limit.

**The same quantities on one scale.**

- *Playing.* Decisions of both seats per second of one game's wall time
  (`wall_seconds` of the manifest on `main`, `seconds` in `log.jsonl` on the
  branch): 60.3 on `main` over 100 games, 105.3 on the branch over 20 games. By
  game size, from under 2,000 to over 45,000 decisions: 82.4, 72.7, 65.2, 60.8,
  58.6, 58.0 on `main` and 119.6, 119.1, 106.9, 105.1, 106.6, 103.5 on the
  branch, a ratio of 1.45 to 1.8. On `main` ten games were in play for the whole
  batch. In a batch of ten the number of games in play falls as games end, so
  the long games of the trial had fewer neighbours. The two smallest size
  classes ended while at least eight games were in play in both pipelines; their
  ratios are 1.45 and 1.64.
- *After a game.* `main` spent 2,981.0 s of `preparation_seconds` on 786,518
  learner decisions, 3.79 ms each, in the coordinator and one game at a time;
  capture receipts trailed the end of their game by 181 s on average and 406 s
  at most. The branch compiles inside each actor: 451.3 s for 278,593 learner
  decisions, 1.62 ms each. `main` also wrote 15.1 GB of `training-rollout.bin`
  for the batch.
- *Between the end of a game and its arrival in the trainer* the branch needs
  about 1 s for small games, 13.6 s for the game of 53,393 decisions and 25.8 s
  for the game of 89,123. This contains engine shutdown, removal of the game
  folder and the transfer of the pickled trajectory. Not broken down.
- *Update.* 5.16 optimizer steps per second on `main`, 5.08 in run 1, 3.20 in
  run 2; per learner sample 6.1 ms, 6.2 ms and 9.8 ms. The code is the same on
  both sides (the refactor moved it; equality was checked up to the backend
  boundary). The difference is the mix of game lengths: the sample-weighted mean
  game length Σ L² / Σ L is 16,120, 20,374 and 34,237 learner decisions.
  `MLXBackend.prefix_states` encodes each game of a minibatch from frame 0 to
  that minibatch's last window start at every optimizer step, so this part of a
  game's cost grows with the square of its length. A least-squares fit of
  `update = a·ΣL + b·ΣL²` over the three batches gives a = 2.79 ms and
  b = 2.02e-7 s and reproduces them within 0.2%, 12% and 0.7%; by that fit the
  one game of 63,390 learner decisions accounts for about 990 s of run 2's
  1,690 s. Three batches and two parameters do not establish the form; the code
  does. The five batches of the stopped `main` run earlier that day show the
  same growth, from 4.26 ms per sample at 500,788 samples to 5.92 ms at 755,229.

**Estimate, not a measurement: `main` on the same ten games.** No ten-game run
of `main` with the current policy exists. A ten-game batch on ten actors lasts
as long as its longest game. Taking the rate of `main`'s games above 45,000
decisions (55.7 to 62.6 per second), its preparation of long games (3.3 to
3.8 ms per learner decision), 60 s of batch setup and teardown seen in the
100-game run, the update time measured on the branch, and one final diagnostic
of 306 to 670 s:

| | `main`, estimated | Branch, measured |
| --- | --- | --- |
| The ten games of run 1 (longest: 53,393 decisions) | 33 to 41 min | 20 min 41 s |
| The ten games of run 2 (longest: 89,123 decisions) | 62 to 71 min | 44 min 56 s |

Not included for `main`: the launch itself, which first analyses and deletes the
previous run's folder.

**Limitations.** Different games, different weights (1,784 against 1,785 and
1,786), different times of day, uncontrolled machine load and temperature. The
`main` rates come from a full ten-actor load, the trial's long games from a
lighter one. Two ten-game runs differ by a factor of 2.2 on the same code only
because of the games drawn, so no single ten-game run on either side could
settle the comparison more precisely than the per-decision rates do. No part of
the difference in playing speed is attributed to a specific change. Nothing here
says anything about learning or playing strength.

**What a direct measurement on `main` would require.** The old launcher refuses
to start until the journal on `main` carries the retention marker for
`run-20261118T104905Z-ebe300b5`. It would then delete that run's folder and
train the previous lineage for real, 1,785 to 1,786. Neither was done; the
decision is the user's.

**Next.** The update's recurrent prefix is the largest remaining cost. Any
change to it changes what the update computes and is a decision about the
learning method, to be proposed separately.

### 2026-11-19 — Where a training cycle spends its time; reusable window states (optimization branch)

**Goal (user request).** Find and test ways to make training faster without
disturbing the user's own 1,000-game run. Work in a separate clone
(`~/Projects/Heroes-opt`, branch `claude/project-thread-mq0j5t`, from
`e46fe66`); nothing in `~/Projects/Heroes` or in the run's state folder
`training-trial` was written. The user authorized assistant execution in the
project thread. The permission check of the assistant's session nevertheless
refused every launch of a real game and every timing run, so **only tests
ran**; see "What actually ran".

**Observed in the user's run (read-only, from `training-trial/log.jsonl`).**
The run continues checkpoint 1,787 with the defaults: ten actors, 100 games per
update, windows of 16 decisions, two windows per minibatch, one epoch.

| Update | Games | Samples | Optimizer steps | Collect | Update | Per step | KL | Clip fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1,786 | 10 | 105,875 | 3,311 | 589 s | 652 s | 197 ms | 0.0028 | 0.023 |
| 1,787 | 10 | 172,718 | 5,400 | 1,004 s | 1,692 s | 313 ms | 0.0045 | 0.030 |
| 1,788 | 100 | 1,372,976 | 42,930 | 6,605 s | 16,756 s | 390 ms | 0.0642 | 0.105 |

The PPO update is 53%, 63% and 72% of these cycles. Of update 1,788's 16,756 s,
772 s is CPU plan building and 103 s is the finite check of weights and Adam
state. Batch 1,789 (100 games, 1,710,118 samples, 18 stopped at the day limit)
took 3 h 17 min to collect and its update was still running at the time of
writing.

*Why the update is slow.* Each optimizer step takes two windows and replays
both games from their first frame to the window with the current weights. The
replay length is the sample-weighted mean game length halved: about 10,200
decisions per window in update 1,786, 17,100 in 1,787 and 11,900 in 1,788,
against 16 decisions that are differentiated. A straight line through the first
two updates gives about 26 ms per step that does not depend on game length and
about 8.4 µs per replayed decision. That is a two-point estimate from whole
updates, not a measurement of the parts, and update 1,788 is slower than it
predicts (see the power finding below).

*The policy moves much further in a 100-game update.* The mean approximate KL
between the updated and the behaviour policy is 0.064 with 10.5% of the samples
clipped, against 0.003 to 0.005 and 2 to 3% in the 10-game updates. A 100-game
batch is one epoch of 42,930 Adam steps at 3e-4 on minibatches of 32 samples.
This is an observation about learning dynamics, not a speed measurement, and
nothing was changed because of it.

**The Mac was power-limited for most of the run.** `system_profiler
SPPowerDataType` reports a 65 W charger. `pmset -g log` shows the battery at
77% at 00:28, at 28% at 02:05 and at 8% at 02:18:40, all but the first "Using
AC", and still at 7% at 11:53. The display turned off at 02:18:51. The run
started at 01:52:37 local time. In `log.jsonl` the rate of answered decisions
per second of game time, both seats together, is:

| Games | Decisions per second and game |
| --- | --- |
| updates 1,786 and 1,787 (ten games each, the evening before) | 103.5 and 106.4 |
| batch 1,788, the first 50 games to finish (up to minute 27.3) | 98.7 and 94.3 per quarter |
| batch 1,788, the last 50 games (minute 27.6 to 109.6) | 44.8 and 32.8 per quarter |
| batch 1,789, all 100 games | 34.0 (33.2 to 34.9 per quarter) |
| games under 2,000 decisions: updates 1,786 to 1,788 / batch 1,789 | about 120 / 42.3 (median) |

The rate fell by about a factor of three within a minute of the battery
reaching 8% and stayed there for nine hours while the charge held at 7%, which
it can only do if the machine draws no more than the charger supplies. Short
games slowed as much as long ones, so the cause is not game length. CPU plan
building per optimizer step also doubled, from 9.6 ms and 8.9 ms in updates
1,786 and 1,787 to 18.0 ms in 1,788. The supported reading is that a 65 W
supply cannot carry ten games, that the battery covered the difference until it
was empty, and that macOS then limited performance to the charger. The display
turning off in the same minute is a confound these logs cannot separate, and no
intervention (a stronger charger, a display kept on) was tried. The user was
told. **Timings of this run after 02:18 are not a baseline for code changes.**

**Change: reusable window states.** `PPOConfig.state_refresh_steps` (default 1,
the behaviour so far). With a value above 1 the update replays a game once to
its last window, keeps the recurrent state before each of its windows, and
reuses these states for that many optimizer steps before it replays the game
again. A window can then start from a state computed with weights that are at
most that many steps old; the window itself is always computed and
differentiated with the current weights. Hidden states recorded during play are
still never used. A game expected in a minibatch less than once per refresh
interval (`refresh × windows per minibatch × its windows < all windows`), or
with a single window, is not kept: it is replayed up to the window at each use
with the current weights, as before, because replaying it to the end would cost
more than it saves. The rule is `Batch.kept` in `heroes/learner/ppo.py`; the
torch reference and the MLX update share it. `PPOReport.game_replays` and the
update record in `log.jsonl` count the whole-game replays.

This is a change of what is computed, not only of how fast, as soon as the
setting is above 1: the states are stale by up to that many optimizer steps.
Stored-state recurrent PPO as commonly implemented uses states from the rollout
for the whole update; this setting is between that and the exact replay.

**Hypotheses and acceptance criteria, set before any measurement.**

1. *Speed.* On a fixed batch of real games, MLX update time per optimizer step
   with `state_refresh_steps` of 1,024 no longer grows with game length, and
   the whole update of a ten-game batch like 1,787 is at least five times
   faster than with 1. Measured with `tools/perf/bench_update.py`, both settings
   on the same games, weights and shuffle seed, interleaved, on a Mac that is
   not power-limited and whose GPU is not shared with another update.
2. *Fidelity per step.* For the same minibatch and the same weights, taken along
   an exact update: the gradient computed from reused states has a cosine of at
   least 0.99 at the median and 0.95 at the 5th percentile with the exact
   gradient, and the log-probabilities differ by at most 0.02 at the 95th
   percentile (0.02 is the existing first-minibatch limit between play and
   update). Measured with `tools/perf/state_fidelity.py` for 64, 256, 1,024 and
   "never refresh", by fifths of the update, because staleness grows with the
   distance the weights have moved.
3. *A setting that fails 2 is not recommended as a default.* If only small
   values pass, the default stays 1 and the report says what the speed-up of
   the passing value is.

None of this can show that learning is unaffected; only training runs can.

**What actually ran.** `python -m pytest` on the Mac, against
`tests/fake_vcmi.py` with random weights in temporary folders: 41 tests pass
(32 before this branch). Nine are new:

- The MLX update is run in tests for the first time, on scripted games, and
  agrees with the torch reference in policy loss, value loss, KL and clip
  fraction within 2% (or 2e-4 absolute) after a five-step update; the same with
  kept states refreshed every third step, and on a scripted 20×20 map whose
  objects move every day, which takes the sparse prefix CNN.
- With a learning rate of zero the losses with states kept for the whole update
  equal those with a replay at every step (MLX: within 1e-5 relative; torch:
  exactly), and the replay counts are as the rule predicts.
- The scripts in `tools/perf` run end to end on tiny scripted games, and the
  fidelity probe reports zero error while the weights stand still and a nonzero
  one when they move.

No real game was played, no saved game was read, no benchmark was timed and no
checkpoint was written. The commands that could not run are in
`tools/perf/README.md`.

**Limitations.** The scripted games offer one legal action, so in these tests
the gradient reaches the weights through the value head only; the policy heads'
log-probabilities are constant. Agreement of the MLX update with the reference
on real games still rests on its first-minibatch check. The acceptance criteria
above are unmeasured. The two-point cost model is an estimate. The power
finding is an inference from two logs that agree in time.

**Tools added (`tools/perf/`).** `collect_games.py` plays real games into a
state folder of its own and keeps each `Game`; `bench_update.py` times the MLX
update on such a batch or on scripted games for several settings;
`state_fidelity.py` measures criterion 2; `profile_game.py` plays one real game
in-process and splits its wall time into engine wait, decoding, encoding, the
forward pass, sampling and recipe compilation.

### 2026-11-19 — Measured on real games: state reuse is fast but inaccurate; where play time goes

**Correction to the entry above.** It says the permission check refused every
launch, so only tests ran. Later the same day the user's own message reached the
assistant's session directly and the same launches were accepted. Everything
below ran after that, by the assistant, under the user's explicit permission in
the project thread ("you can run different parts of the project, or you can run
the whole learning loop"). The acceptance criteria of the entry above were
written before any of these runs and are judged here.

**Conditions of every measurement in this entry.** Clone `~/Projects/Heroes-opt`,
branch `claude/project-thread-mq0j5t`. Own state folder
`~/Projects/Heroes-artifacts/perf-opt-mq0j5t` with a copy of checkpoint 1,787
(SHA-256 `a70e7095…`, equal to the file in `training-trial`). M4 Max on the same
65 W charger, battery between 16% and 30% and charging, so not power-limited at
the time. The user's run was in its PPO update of batch 1,789 throughout: its
ten actors were idle, **its update shared the GPU with every update timing
below**, and at times two of the assistant's own GPU jobs ran together. Absolute
update times are therefore upper bounds; comparisons were run back to back or
alternating. Nothing was written to `training-trial`, and no checkpoint was
written anywhere.

**A fixed batch of real games.** `tools/perf/collect_games.py`, three actors,
seed 1: 12 games in 1,532 s, eleven finished naturally and one stopped at day
366. 135,373 learner samples, 1,033,919 learner frames (7.6 frames per trained
decision), games of 263 to 28,070 samples, 4,233 optimizer steps at the default
settings. The games ran at 94 to 165 decisions per second each.

**The update on that batch (hypothesis 1: confirmed).**
`tools/perf/bench_update.py`, same weights, games and shuffle seed for every run.

| `state_refresh_steps` | Run | Seconds per step | Whole update | Whole-game replays |
| --- | --- | --- | --- | --- |
| 1 (exact, the default) | 300 steps, twice | 0.360, 0.386 | 1,691 s and 1,785 s, projected | none |
| 1,024 | 300 steps, twice | 0.040, 0.044 | | |
| 256 | whole update | | 262 s | 185 |
| 1,024 | whole update | | 225 s | 55 |
| never refreshed | whole update | | 214 s | 12 |

The exact update was not run to its end for timing; its total is the 300-step
rate times 4,233 steps. With states reused the update is 6.5 to 8 times faster
and the time per step no longer depends on the length of the games. On scripted
games of 20,000, 5,000 and 5,000 decisions (180,003 frames) the same comparison
gave 0.162 and 0.182 s per step against 0.039 and 0.032. With states reused, 37
to 39 s of the 214 to 262 s is CPU plan building and 18 s the finite check.

**Accuracy of reused states (hypothesis 2: rejected for every value tested).**
`tools/perf/state_fidelity.py` ran the exact update over all 4,233 steps and, at
every 25th step (170 minibatches), compared what each setting would have used
for the same minibatch at the same weights.

| Refresh every | State error, median / p95 | Log-probability difference, median / p95 | Value difference, median / p95 | Gradient cosine, median / 5th percentile / lowest |
| --- | --- | --- | --- | --- |
| 64 steps | 0.063 / 0.244 | 0.009 / 0.112 | 0.047 / 0.452 | 0.998 / 0.514 / −0.104 |
| 256 steps | 0.132 / 0.365 | 0.022 / 0.179 | 0.089 / 0.455 | 0.990 / 0.320 / −0.104 |
| 1,024 steps | 0.213 / 0.501 | 0.034 / 0.280 | 0.151 / 0.647 | 0.951 / 0.004 / −0.862 |
| never | 0.417 / 0.683 | 0.066 / 0.445 | 0.449 / 1.299 | 0.381 / −0.275 / −0.956 |

State error is the relative L2 distance of the windows' starting states from
the exact ones; the differences are the largest over the minibatch's decisions.
The criteria were a cosine of at least 0.99 at the median and 0.95 at the 5th
percentile, and a log-probability difference of at most 0.02 at the 95th
percentile. Only the median cosine at 64 and 256 steps passes. The state before
a window moves by 6% within at most 63 optimizer steps and by 42% over an
update, far more than the weights themselves: this recurrent state depends on
the weights through the whole game before it. In the first fifth of the update
the error at a given age is larger than later (median 0.34 for ages up to 825
steps, against 0.19 to 0.21 for ages up to 1,023 later).

The exact update's own losses on this batch were policy loss 0.00155, value
loss 0.2803, KL 0.0047 and clip fraction 0.031. The updates that reused states
reported 0.0018 to 0.0020, 0.284, 0.0054 to 0.0055 and 0.033 to 0.037 (256 and
1,024 steps) and 0.0013, 0.2797, 0.0041, 0.026 (never). These averages look
close and hide the per-minibatch differences above; a state that is never
refreshed even reports a smaller KL, because it does not see the weights move.

**Decision by the criteria set beforehand.** `state_refresh_steps` stays at 1
and no other value is recommended. What "stored-state" recurrent PPO does for
the whole update, never refreshing, would here give gradients that mostly point
elsewhere (median cosine 0.38). The setting remains in the code, off, because
the measurement tools and the follow-up below build on it.

**Where a game's time goes.** `tools/perf/profile_game.py`, one game in-process
with the same checkpoint, both seats.

| Game | Decisions | Play | Engine wait | Policy forward | Decoding | Sampling | Encoding | Other | Compile after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed 1, ended naturally on day 12 | 1,483 | 9.0 s | 47.5% | 28.9% | 10.9% | 8.0% | 2.7% | 1.8% | 0.4 s |
| seed 4, stopped after day 90 | 14,801 | 136.1 s | 50.5% | 28.2% | 11.4% | 5.8% | 2.3% | 1.6% | 16.4 s |

Four later plain runs of the seed-4 game took 119 to 124 s (120 to 124
decisions per second) with identical digests, so a seeded game replays exactly.
"Engine wait" is the time the actor's main thread waits for the next message.
It includes the JSON parsing of that message, which runs in the reader thread.
The engine folder of the seed-4 game held 1.95 GB when it ended, 132 KB per
decision, almost all of it replay logs nothing reads. Under cProfile, which
inflates Python-heavy code, the largest items of the 205 s of play were the
observation features of the policy (69 s: its memo key 21 s, the map and tree
encoding of new observations 29 s, the memo's size accounting 16 s), decoding
and validating observations (27 s), the action walk (25 s, play and compile
together) and JSON parsing (21 s, of which 13 s the per-object duplicate-key
hook). In a 30-day game 12,779 observations arrived for 5,098 decisions, and
two in five repeated the observation before them exactly: the engine sends what
a seat sees as a view and again inside the decision that follows.

**A play-side change that was tried and dropped.** Reusing the decoded
observation when a decision repeats the preceding view, with shortcuts by object
identity in the policy memo, the guard and the recipe compiler. The same seeded
game gave the same digest and took 123.5 s against 120.8 and 123.8 s before:
the type-exact comparison of the raw observations costs what the reuse saves.
A first version compared marshal bytes of the whole observation and caught only
17.5% of the repeats, because those bytes depend on which strings the rest of
the message shares. The change is not in the branch.

**Changes kept.** In the MLX update, the plan of a minibatch is now built on a
helper thread while the GPU runs earlier steps, and the finite check covers the
arrays a step wrote and is evaluated with the step. Both leave the arithmetic
unchanged. One alternating measurement with states reused (400 steps, twice
each, three GPU jobs running) gave 0.127 and 0.125 s per step before and 0.115
and 0.118 after. That is a first indication under heavy contention, not a clean
number, and the effect on the default exact update is unmeasured.

**Observations that are not changes.**

- *Game length.* In the user's batches 1,788 and 1,789 a seat answers about 100
  decisions per game day (median 96.5), the mean game lasts 151 days, and the 27%
  of games that reach day 200 hold 60% of all decisions. With the guard's budget
  of 32 stack moves per hero or town and day, most of a long game is stack
  management. This multiplies both play and update time and is a question of
  the guard and the action space, not of code speed.
- *The engine.* About half of a game's wall time is spent waiting for the
  engine. Reading `engine/ExplorerRL` suggests where: the observation is rebuilt
  and serialized several times per decision, sent once as a view and again in the
  decision, and copied into the replay ledger, at 132 KB per decision. Removing
  that needs an engine change and a rebuild; it was read, not measured.
- *Minibatch size.* An update is 4,233 optimizer steps of 32 samples here and
  42,930 for 100 games. Each step replays two games whatever it does with them.

**Limitations.** One batch of 12 games from one checkpoint. Gradient cosines are
per minibatch of 32 samples, a noisy quantity; they show that reused states
change what a step does, not how learning would differ. The GPU was shared in
every timing. Nothing here measured learning.

### 2026-11-19 — A short replay does not rescue reused states: the policy remembers the whole game

**Question.** The entry above found that a window's starting state drifts too
far when it is reused across optimizer steps. Is the drift in what the state
remembers of the recent frames only? Then replaying a few thousand frames before
the window with the current weights, from a reused state or even from the
initial state, would be accurate and still cheap ("burn-in").

**Criteria, as before.** Gradient cosine with the exact gradient at least 0.99
at the median and 0.95 at the 5th percentile; log-probability difference at most
0.02 at the 95th percentile.

**What ran.** `tools/perf/state_fidelity.py --refresh 1024,1000000000 --burn-in
512,2048,8192 --from-zeros 2048,8192,32768 --steps 2200 --every 20`, the script
as committed in `ed004f4` with the learner code of `3908015`, on the same 12
real games, weights and conditions as the entry above: the first 2,200 steps of
the exact update, 110 minibatches compared. The
frames before each window were encoded with the current weights, and the
recurrent network was run over the last of them from the other starting state.
A window has 7.6 frames per trained decision, so 8,192 frames are about 1,070
decisions and 32,768 about 4,300. A test on scripted games checks that replaying
all frames reproduces the exact state.

| Starting state | Frames replayed | State error, median / p95 | Log-probability difference, median / p95 | Gradient cosine, median / 5th percentile |
| --- | --- | --- | --- | --- |
| reused, refreshed every 1,024 steps | none | 0.263 / 0.532 | 0.045 / 0.301 | 0.953 / −0.165 |
| | 512 | 0.148 / 0.425 | 0.039 / 0.291 | 0.960 / −0.129 |
| | 2,048 | 0.130 / 0.320 | 0.033 / 0.251 | 0.960 / −0.167 |
| | 8,192 | 0.092 / 0.284 | 0.022 / 0.217 | 0.982 / −0.304 |
| reused, never refreshed | none | 0.374 / 0.647 | 0.071 / 0.402 | 0.409 / −0.484 |
| | 8,192 | 0.146 / 0.351 | 0.031 / 0.364 | 0.811 / −0.487 |
| the initial state | 2,048 | 0.420 / 0.594 | 0.093 / 0.731 | 0.264 / −0.616 |
| | 8,192 | 0.329 / 0.560 | 0.067 / 0.420 | 0.325 / −0.640 |
| | 32,768 | 0.212 / 0.426 | 0.051 / 0.399 | 0.488 / −0.631 |

**What the evidence supports.** No variant meets the criteria. A state rebuilt
from the last 32,768 frames alone is still 21% away from the exact one at the
median, so the recurrent state at a decision carries information from more than
4,000 of the learner's decisions back, in practice from the whole game. That
is why it follows the weights so closely, and why neither a stale state nor a
truncated replay stands in for it. For this policy the exact replay from the
first frame is not an implementation luxury. Its cost can be shared, as with
`windows_per_game`, but not skipped.

**Limitations.** As above: one batch, one checkpoint, 32-sample gradients, a
shared GPU. The cosine criteria are the assistant's choice; a training run could
still show that learning tolerates some of this, and none was made.

### 2026-11-19 — Sharing the replay: grouped minibatches, and two small costs in the default update

**Question.** The two entries above leave the exact replay as it is. Can its
cost be shared between windows instead, and is anything in the default update
wasted outright?

**Corrections to the entry "Measured on real games".** The 12 collected games
ran at 94 to 136 decisions per second each (100.9 over all of them), not "94 to
165"; 165.7 was the single short game of the profile. These games ran on the
same Mac, charger and code as the user's batch 1,789, which ran at 34: the
difference is that the battery was charging while they ran. This supports the
reading of the first entry that the user's run was short of power, without
being a test of it.

**Change: `windows_per_game`.** `Batch.minibatches` now builds the schedule of
an epoch for both backends. With the default of 1 it is the same shuffle as
before, consuming the generator in the same way (checked by a test against
`torch.randperm`). With a larger value each game's windows are shuffled and cut
into groups of that size, the groups of all games are shuffled, and a minibatch
takes `windows_per_minibatch / windows_per_game` whole groups. A step then
replays each of its games once, to the latest of that game's windows in the
step, and all states are still computed with the current weights. What changes
is the optimization: an update makes fewer steps on larger minibatches. With 16
and 8 a minibatch holds 256 decisions from two games where it held 32 from two
games before.

**Hypothesis and criterion, stated before the timing.** Sharing one replay among
g windows should cut the replay work per sample by about (g + 1) / 2, since a
replay to the latest of g windows covers g / (g + 1) of a game on average where
one window needs half. The criterion was a whole update at least four times
faster than the default on the 12-game batch for g = 8.

**What ran.** Same batch, weights, shuffle seed and conditions as the entries
above (the user's update on the same GPU throughout, battery 49% to 56% and
charging). `tools/perf/bench_update.py` at `ed004f4` for the first table and at
`3ebcf3e` for the second.

| Windows per minibatch / per game | Optimizer steps | Seconds per step | Whole update | Policy loss | Value loss | KL | Clip fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 / 1 (default) | 4,233 | 0.369 and 0.364 (150 steps each) | about 1,560 to 1,580 s, projected | 0.0016 | 0.280 | 0.0047 | 0.031 |
| 16 / 1 | 530 | 1.12 (30 steps) | about 600 s, projected | | | | |
| 16 / 8 | 532 | 0.678 | 371 s, measured | 0.0027 | 0.288 | 0.0045 | 0.022 |
| 32 / 16 | 268 | 0.781 | 221 s, measured | 0.0026 | 0.290 | 0.0045 | 0.019 |

Projected totals are the steady rate per step times the steps, plus about 15 s
for building each game's plan once. The losses of the default are those of the
exact update in the fidelity run. The criterion is met, narrowly: 4.2 times
faster with eight windows per game, and 7 times with sixteen. Larger
minibatches without grouping gain less, and part of that only because 12 games
collide often; with 100 games they would not.

**Correction to the projections in the entry "Measured on real games".** Its
"1,691 s and 1,785 s, projected" for the exact update scaled a 300-step run up
with its one-time start-up costs, as the benchmark script then did. The steady
rates of those runs, 0.360 and 0.386 s per step, give about 1,540 to 1,650 s for
the 4,233 steps. Reused states were therefore 6 to 7.7 times faster, not 6.5 to
8. The script now projects from the steady rate. The conclusions of that entry
do not change.

*A batch of production size.* No 100-game batch was collected. `--copies 8`
uses each of the 12 games eight times: 96 games, 1,082,984 samples, 8.3 million
frames, the same length distribution. The copies share their CPU plans but
each has its own plan on the GPU, 9.1 GiB together against a cache of 4 GiB.
Only the first 80 to 120 steps of each setting were run.

| Setting | Seconds per step | Whole update, projected from that rate |
| --- | --- | --- |
| default, before the byte-count change below | 0.540, 0.556, 0.490 | 4.6 to 5.2 h (33,864 steps) |
| default, after it | 0.488, 0.472, 0.439, 0.439 | 4.1 to 4.6 h |
| 16 windows per minibatch, 8 per game | 0.748 | 53 min (4,256 steps) |
| 32 windows per minibatch, 16 per game | 0.898 | 32 min (2,140 steps) |

These are short runs on a shared GPU, taken at different minutes; the spread of
the default alone is 25%. They support "several times faster at production
size", about five and eight times, not a precise factor.

**Two costs found in the default update, and one that was not there.**

- *Counting a plan's bytes.* A cProfile of the default update on the
  production-size batch showed `MLXBackend.plan` spending 18.5 of 68.7 s in the
  recursion that counts the bytes of a plan entering the GPU cache: 40.9
  million calls for 84 cache misses, because it walked the planner's integer
  sequences, half a million Python ints for a long game, each worth zero.
  `plan_bytes` skips them; a test checks it against the plain walk. Alternating
  runs gave 0.556 and 0.490 s per step before and 0.488 and 0.472 after. The
  profiler inflates such code about fivefold, so the real saving is near 40 ms
  per miss, around 8% of a step at this size and nothing while all plans fit
  the cache, as they did for ten games.
- *Plan building beside the GPU, and the finite check.* Described two entries
  above. On the default update of the 12-game batch the step time is unchanged
  within the noise (0.369 and 0.364 s after, 0.360 and 0.386 before): there the
  plan was already built while the replay ran. It matters when a step is short,
  as with grouped minibatches or reused states.
- *The size of the GPU plan cache.* Raising it from 4 to 16 GiB, for the
  benchmark only (`--plan-cache-gib`), gave 0.425 and 0.411 s per step against
  0.439 and 0.439. A miss costs little once the byte count is fixed: converting
  the largest game's 267 MiB plan took 4 ms on the CPU and 10 ms to the GPU.
  The cache stays at 4 GiB.

**What was not done.** No training run used any of these settings, so nothing
is known about how grouped minibatches learn. Fewer, larger steps at the same
learning rate move the policy differently; in these single updates the mean KL
was the same (0.0045 against 0.0047) and fewer samples were clipped. The defaults
are unchanged. The assistant's benchmarks shared the GPU with the user's update
of batch 1,789 for about an hour and forty minutes in total and will have
slowed it by some part of that.

**Tests.** 45 pass (32 before the branch): the schedule and its step count, both
backends on grouped minibatches against each other, the byte count, the
command-line checks, and the measurement scripts on scripted games.

**Limitations.** As in the entries above. The production-size batch is a
multiple of 12 games, not 100 different ones. Timings are from a shared GPU.

**Evidence.** Private, not in Git:
`~/Projects/Heroes-artifacts/perf-opt-mq0j5t/results/*.jsonl` (one JSON line per
run), `logs/`, the batch in `batch/` and the two profiles
`results/game-seed4.prof` and `results/update-copies8.prof`.

### 2026-11-19 — A correct fast update: reused states fail even for 8 steps; the grouped update needs its learning rate

**Question.** The user asked for the fast mode of the entry above
(`--windows-per-minibatch 16 --windows-per-game 8`) to be a correct PPO update,
checked against the exact default on real games, and for an estimate of a
1,000-game cycle. His concern was that the larger minibatches lose information.
He authorized assistant-launched benchmarks and training games for this task.

**What the flag changes, and what it does not.** Every state in a grouped
minibatch is computed with the current weights, so each step is an exact PPO
step and nothing inside a minibatch is lost: all 256 decisions enter the loss.
What changes is the optimization. An update makes 8 times fewer Adam steps, each
on 8 times more decisions, at the same learning rate.

**First design, withdrawn: small steps on a shared replay.** To keep the
optimization of the default, `windows_per_replay` (commits `010acc3` to
`97b6a0e` on `claude/project-thread-aclj7d`, removed again in `980dff6`) made
an update a sequence of rounds: a round replayed each of its games once with
the weights it started from and then made 8 optimizer steps of the default
size, so a state was at most 7 steps old.

*Per step it looked good.* `tools/perf/state_fidelity.py` (in those commits)
compared each step with the exact one on the 12-game batch of the entries
above, checkpoint 1,787. With 8 steps per replay the state before a window
differed by 1.4% in the median and 6.3% at the 95th percentile, the
log-probabilities by at most 0.037 for 95% of decisions, and the gradient had a
cosine of 0.9999 in the median and 0.978 at the 5th percentile with the exact
gradient, with single outliers down to 0.001. With 16 steps the 5th percentile
of the cosine fell to 0.57.

*Whole updates did not.* A new script, `tools/perf/update_quality.py`, makes one
whole update per setting from the same checkpoint with its Adam state on the
same 12 games (135,373 decisions), then judges every set of weights with
forward passes only, on those games and on 12 games no update saw (150,188
decisions, played by the same checkpoint with another seed). It reports the
mean of (ratio - 1) - log ratio as KL, the share of decisions whose ratio left
the clip range, half the mean squared error of the value against the game's
result, the first-order estimate of the policy's gain, and the length of the
update's change of all weights with the cosines between such changes. No
numeric criterion was fixed before these runs. Every figure is read against two
runs of the exact update that differ only in the shuffle seed.

| Update | Steps | KL, trained / held out | Clipped, trained / held out | Value loss, trained / held out | Weight change |
| --- | --- | --- | --- | --- | --- |
| checkpoint, no update | | | | 0.396 / 0.464 | |
| exact default, seed 0 | 4,233 | 0.0067 / 0.0104 | 0.037 / 0.047 | 0.080 / 0.341 | 4.57 |
| exact default, seed 1 | 4,233 | 0.0144 / 0.0151 | | 0.077 / 0.266 | 4.38 |
| 8 small steps per replay | 4,233 | 0.0358 / 0.0209 | | 0.069 / 0.267 | 4.78 |
| 4 small steps per replay | 4,233 | 0.0078 / 0.0094 | | 0.054 / 0.244 | 4.67 |
| the 8-step order, but exact states | 4,233 | 0.0052 / 0.0076 | | 0.111 / 0.236 | 4.51 |

With 8 steps per replay the policy ended 2.5 to 5 times further from the one
that played the trained games than after either exact update, with no more
change of the weights. The same order of steps with states replayed at every
step behaved like the exact update, so the cause is the age of the states and
not the order in which a round visits its games. A state computed before the
earlier steps of a round still describes the policy as it was; the ratios the
clip sees are then too close to 1, and the clip stops the policy later than it
should on exactly the games the round keeps stepping on. A per-step gradient
cosine of 0.98 did not show this, because the error has the same sign step
after step. Four steps per replay stayed inside the spread of the exact
update on one run, at 1.5 to 2 times the cost of the grouped update below. The
mode is withdrawn, and with it `state_refresh_steps`, the switch of the entries
above that kept states for far longer.

**Why the grouped update moved less, and the learning rate that goes with it.**
At the default learning rate of 3e-4 the grouped update of the entry above
changed the weights by 1.99 and 2.00 (two seeds), the default by 4.57 and 4.38.
The cosine between the weight changes of two exact updates that differ only in
the shuffle is 0.54, and between any two updates measured here 0.36 to 0.76:
most of an update's change of the weights is noise of the minibatches, and for
Adam in that regime the distance grows with the learning rate times the square
root of the number of steps. Eight times fewer steps then travel the square
root of 8 times less far, 4.57 / 2.83 = 1.6 predicted and 2.0 measured, and
keeping the distance takes 3e-4 x 2.83 = 8.5e-4. This is the known square-root
scaling rule for Adam; here it was checked, not assumed.

**Hypothesis for the confirmation, stated before it ran.** The grouped update
at 8.5e-4 changes the weights as far as the exact default and moves the policy
no further. Measured with the code of `5028350` while the user's run was
collecting games, so the GPU was otherwise free:

| Update | Steps | KL, trained / held out | Clipped, trained / held out | Value loss, trained / held out | Weight change | Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| exact default, seed 0 | 4,233 | 0.0067 / 0.0104 | 0.037 / 0.047 | 0.080 / 0.341 | 4.57 | 986 to 997, projected from 300 steps |
| exact default, seed 1 | 4,233 | 0.0144 / 0.0151 | | 0.077 / 0.266 | 4.38 | |
| 16 / 8 at 3e-4, seed 0 | 532 | 0.0033 / 0.0064 | 0.018 / 0.025 | 0.082 / 0.236 | 1.99 | 248 |
| 16 / 8 at 3e-4, seed 1 | 532 | 0.0037 / 0.0031 | 0.017 / 0.013 | 0.078 / 0.249 | 2.00 | 252 |
| 16 / 8 at 8.5e-4, seed 0 | 532 | 0.0050 / 0.0070 | 0.037 / 0.045 | 0.061 / 0.232 | 4.14 | 250 |
| 16 / 8 at 8.5e-4, seed 1 | 532 | 0.0065 / 0.0066 | 0.035 / 0.032 | 0.067 / 0.277 | 4.89 | 248 |

The hypothesis holds on these games: the weight change (4.14 and 4.89) brackets
the default's, the clipped share is the same, the KL is at or below the lower
of the two exact runs, and the value error on held-out games is at least as
low. Against the exact update of seed 0, resampling the games gives 90%
intervals that include zero for every measure except the held-out value loss
of seed 0, which is lower (-0.109, interval -0.192 to -0.013). At 3e-4 the
grouped update is a correct but smaller update: about 0.44 times the weight
change and half the clipped share.

**What these runs cannot show.** The first-order gain estimate is within noise
of zero for the exact and the grouped updates alike (between -0.006 and
+0.0005 on the trained games), so a single update on 12 games does not show any of them
improving the policy, only how far each moves it and how well it fits the
value. Which setting wins more games is a question for training runs of many
updates.

**Change.** `980dff6`: the update is the exact one of the entry above, with
`--windows-per-game` and a new `--learning-rate` on the command line (default
3e-4; the help names 8.5e-4 for 16 windows). `state_refresh_steps`,
`Batch.kept`, `MLXBackend.prefix_table`, the `game_replays` report field,
`--refresh` in `bench_update.py` and `tools/perf/state_fidelity.py` are
removed. The defaults are unchanged, so a run without options is the update
of PR 32.

**Clean timings.** `bench_update.py` at `5028350`, 17:55 to 18:06 UTC,
the user's run collecting games on the CPU and nothing else on the GPU, 140 W
charger, battery full:

| Batch | Setting | Steps | Seconds per step | Whole update |
| --- | --- | --- | --- | --- |
| 12 games, 135,373 decisions | default | 4,233 | 0.230 and 0.232 (300 steps each) | 986 and 997 s, projected |
| 12 games | 16 / 8 | 532 | 0.448 | 248 s, measured |
| the 12 games 8 times, 1,082,984 decisions | default | 33,864 | 0.310 (120 steps) | 2.9 h, projected |
| the same | 16 / 8 | 4,256 | 0.538 (120 steps) | 38 min, projected |

The grouped update is 4.0 times faster on 12 games and 4.6 times at production
size. The figures of the entry above were taken beside the user's update and
are 1.4 to 1.6 times these.

**Training runs beside the user's run: one update, and two lost batches.** The
user asked for training games in both modes. Two runs of `980dff6` were started
at 18:34 UTC from copies of his checkpoint 1,789 in state folders of their own
(`arm-exact` with the defaults, `arm-flag` with `--windows-per-minibatch 16
--windows-per-game 8 --learning-rate 8.5e-4`), 4 actors and 10 games per update
each, while his run was collecting batch 1,790 with 10 actors.

- *The default run made one update.* Ten games, all ended naturally (red 2,
  blue 8), 159,436 learner decisions, collected in 2,526 s at 51 decisions per
  second per game beside 14 other actors. The update: 4,985 steps in 1,977 s,
  policy loss -0.0100, value loss 0.478, KL 0.111, clipped share 0.160. It was
  then stopped so that it would not collect beside the user's update.
- *The fast run made none.* A guard written for these runs stopped it twice
  when memory ran short: at 19:16, with 9 of 10 games played, when the kernel
  reported memory pressure level 2, and at 21:00, 2.5 minutes into its update
  on a second batch (10 games, 288,040 decisions, 5 of them stopped at day
  366), when swap grew beside the user's update of 1.77 million decisions. A
  stopped trainer discards its batch, so both batches are lost. Nothing aborted
  in those 2.5 minutes, which is all that is known about the first-minibatch
  check with 256 decisions per step in a real run.
- *Cost to the user's run.* About 3.7 GB went to swap between 19:18 and 21:00
  with these runs beside his, 0.3 GB of it during the fast run's update; very
  little was read back. His batch 1,790 took 1 h 54 min to collect (first job
  17:31, last game 19:25) at 63 decisions per second per game with these jobs
  beside it.
- *A mistake in the guard.* Its first stop signal never arrived: a process
  started in the background of a non-interactive shell ignores SIGINT. The run
  was ended with SIGTERM five minutes later, and later runs were started
  through a wrapper that restores the default handler.

**An observation outside the question: the update from checkpoint 1,789 moves
the policy about 20 times further.** The default update above reports a mean KL
of 0.111 and 16% clipped decisions over 4,985 steps. The same code from
checkpoint 1,787 on 12 games reported 0.0047 and 3% over 4,233 steps, and the
user's 10-game updates 1,786 and 1,787 reported 0.0028 and 0.0045. Between
them lie his two 100-game updates, 1,788 and 1,789, of about 43,000 and 53,465
steps, which themselves reported 0.064 and 0.116. Everything measured in this
entry about the learning rate was measured from checkpoint 1,787. Whether the
fast setting behaves like the default from 1,789 as well is not known yet; a
comparison of both on one batch of games from 1,789 is prepared and runs after
the user's update 1,790. Why the update moves so much further from 1,789 was
not investigated. One candidate is Adam's state after a long update, in which
most decisions end up clipped and the gradients shrink; another is a sharper
policy.

### 2026-11-19 — Faster game play: plan and acceptance criteria (play-speed branch)

**Question.** With the grouped update of the entry above, playing 100 games is
the larger half of a cycle: update 1790 of the user's run collected its 100
games in 1 h 54 min on ten actors (3.47 million decisions of both seats, 63.6
per second and game while two other runs shared the Mac; the two short runs of
2026-11-18 reached 103 and 106). How much faster can the same games be played,
and what would a further factor cost in training terms?

**What the earlier profile of one game says** (`5028350`, seed 4, 90 days,
14,801 decisions, 136 s): 51% of the play time waits for the engine, 28% is
the policy's forward pass, 11% decoding, 6% sampling; compiling the recipe
adds another 12%. Reading the engine source for this entry: every decision
record repeats the observation that the view record before it already
delivered, about 55 KB of JSON; the replay recorder copies and encodes every
message a second time for `contextual-replay.jsonl` and flushes each record,
about 130 KB per decision (5.9 GB after eight minutes of one game of the
update thread's run, about 450 GB per 100 games), and nothing reads that file.
On the Python side each of the two copies is parsed, decoded, validated and
keyed for the policy's memo separately.

**Changes under test, all meant to be exact** (the same game from the same
seeds, bit for bit in everything the learner reads):

1. Engine switch `VCMI_EXPERIMENT_LEDGER=0`: the recorder keeps its
   transaction order and terminal barrier and writes nothing.
2. Engine switch `EXPLORER_RL_DECISION_OBSERVATION=0`: a decision record
   carries no observation; Python gives the decision frame the observation of
   the seat's latest view, the same object.
3. Python: the policy recognises that shared observation by identity instead
   of by a key over all its contents, sizes a memo entry without walking it,
   the recipe compiler reuses the shared observation's map recipe and trees,
   and engine messages are parsed without a Python call per JSON object.
4. More actors than ten.

**Acceptance criteria, stated before any game ran.** (a) For each of 1 to 3,
the digest of `tools/perf/profile_game.py` for a seeded game is identical
before and after, on the old runtime and on the new one; a different digest
rejects the change. (b) `tools/perf/bench_play.py --games` gives identical
digests for every game of a ten-actor batch. (c) Speed is reported as
decisions per second of both seats over a window in which every actor is busy,
old and new alternating under the same load, with the charger and the other
runs on the Mac noted; games per hour follow from the mean length of the games
of the user's run (32,458 decisions over updates 1788 to 1790) and are labelled
as derived. (d) A higher actor count is recommended only if its window rate is
higher than ten actors' by more than the spread between repeated windows.

**Not part of this change.** Anything that alters what is trained on: the
guard's daily budget of 32 stack moves per hero and town (a seat answers about
100 decisions per game day), the day limit, playing the next batch during the
update. Those are measured and proposed, not shipped.

### 2026-11-19 — Faster game play, measured: the same games 1.6 times faster; what is left

All times are local (EET, UTC+2), 22:55 on 2026-11-19 to 00:16 on 2026-11-20.
Mac on the 140 W charger, battery at 100% throughout. The Mac was not free: the
user's run was in the PPO update of batch 1790 (one busy core and the GPU), the
update thread's comparison runs played two games at a time and later collected
games with `collect_games.py`, a browser process took up to three cores for
minutes, and a game ran under Wine. Every rate below is therefore lower than on
a free machine; versions were compared in turns under the same conditions.
Checkpoint 1789 (`weights_sha256` 74da41d9…), copied into
`~/Projects/Heroes-artifacts/play-speed-r3u8uh`; nothing was trained or saved.
"Base" is `8ad4fcc` with the runtime `local-v8-sleep-resume-v1`; "new" is
`2a0e5e4` with `local-v8-play-switches-v1`, built from this branch's engine
sources and frozen with `tools/freeze_runtime.py` (its `config` and `Mods` are
identical to the old runtime's). Results are in `results/` of that folder,
measurements ran from copies of the source tree (`git archive`).

**Exactness (criteria a and b): met.**

| Game | Version | Digest |
| --- | --- | --- |
| seed 4, to day 60, 14,208 decisions | base, twice | `fec36b049734ab94` |
| | new code, old runtime | `fec36b049734ab94` |
| | new code, new runtime, four times | `fec36b049734ab94` |
| | new runtime with the ledger forced on / the repeated observation forced on | `fec36b049734ab94` |
| seed 9, to day 60, 11,658 decisions | base / new | `f154a6e38260d5a5` both |
| ten games on ten actors, `bench_play.py --games 10 --max-days 60 --seed 3`, 91,868 decisions | base / new | all ten digests equal, game by game |

**One game alone** (`profile_game.py --max-days 60`, seconds):

| Seed 4 | Play | Waiting for the engine | Forward | Decoding | Sampling | Compile | Engine folder |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 149.7 | 76.6 | 43.8 | 15.7 | 8.3 | 17.1 | 2.24 GB |
| new code, old runtime | 135.4 | 72.1 | 33.9 | 16.1 | 7.3 | 16.3 | 2.24 GB |
| new runtime, observation still repeated | 117.3 | 53.9 | 34.5 | 16.1 | | 16.0 | 5.0 MB |
| new runtime, ledger still written | 112.9 | 61.0 | 26.7 | 12.1 | | 12.5 | 1.38 GB |
| new | 95.7 | 44.3 | 27.5 | 11.8 | 7.6 | 11.9 | 5.0 MB |

Seed 9: 114.7 s and 12.5 s to compile before, 72.4 s and 8.8 s after. Play and
compile together are 1.55 and 1.57 times faster. The first base run of seed 4
(147.4 s) had an 8-second `sample` of its engine attached and is not in the
table. The two "still" rows used wrapper scripts that set one switch back; the
runtime and the Python code are the new ones.

**Ten actors** (`bench_play.py --seconds 130 --warmup 20 --seed 7`, decisions
per second of both seats of all games, windows taken in turns): base 663 and
637; new (`52a9f5c`, the same play path as `2a0e5e4`) 1,058 and 999: 1.58
times. Two later ten-actor windows of the new code, in the actor-count runs,
gave 1,066 and 1,120. The ten-game batch to day 60 took 206.1 s before and
132.9 s after; an actor spent 12.64 ms per decision before and 7.98 ms after,
play and compile together: 1.58 times.

**Actor count** (criterion d), new code, the same windows:

| Actors | 8 | 10 | 12 | 14 | 16 |
| --- | --- | --- | --- | --- | --- |
| decisions per second | 819 | 1,058, 999, 1,066, 1,120 | 1,052 | 1,185, 1,255, 1,208 | 1,150 |

Every 14-actor window is above every 10-actor window, by 15% in the means;
the spread among the ten-actor windows is 12%. The 12-actor window coincided
with the browser's burst (load average 96 at its start) and the 16-actor
window was taken once, so the shape beyond 14 is not established; with 14
actors `top` showed no idle CPU. The default becomes 14. On a free machine the
best count may be higher or lower and should be measured again.

**Where the time went, from an 8-second `sample` of the base engine**: of the
engine's busy time 34% was the replay recorder (a copy of every message, its
encoding, a flushed write), 18% encoding and sending messages to the policy,
and 41% `captureContextual`, which builds the player's whole observation as a
`JsonNode` and compares it with the last one. `CClient::handlePack` runs it for
both players after every applied pack, and the plugin runs it again before
every result and every decision. A second sample, of the new engine, shows
that capture as 86% of what is left. The earlier entry's "the engine sends
every observation twice" named the smaller half of the engine's cost.

**What a decision costs now** (seed 4, new): 6.7 ms of wall time. In Python:
policy forward 1.9 ms, decoding 0.8, sampling 0.5, encoding frames 0.2, and
0.8 ms per decision of the game to compile the recipe afterwards. The engine's
CPU is not measured per decision; with 14 actors playing, `ps` showed the
engines at 30% of a core each and the actors at 50%, which puts the engine at
a little over 2 ms per decision, nearly all of it the captures.

**Hit rate of the whole-observation memo, measured before removing it**
(profile of seed 4 at `c52493f`+`9c1970d`): of 22,015 views looked up by
`_record_key`, 3,839 hit (17%); the key cost 14.6 s under the profiler and the
hits saved about 8 s. It is removed. The observation encoded last is kept and
recognised by identity, which serves all 14,208 decisions, and the map
features are kept under the recipe compiler's map key.

**A failed attempt, recorded.** The first actor-count run (23:07) reported two
failed games among nine. They were mine: I edited `heroes/learner/recipes.py`
in the tree the benchmark was running from, and an actor imports the recipe
compiler when its first game ends, so two actors imported a file that no longer
matched their loaded policy module. The same seeded games ran without a
failure three times afterwards, from the committed code and from copies. Its
rates are discarded; the perf guide now says to measure from a copy.

**What the learner chose** (the ten-game batch, 46,632 learner decisions, 435
of them `end_turn`, so 107 decisions per seat and day): `swap_stacks` 30.4%,
`split_stack` 8.2%, `merge_stacks` 4.8%; `open_capability` and
`close_capability` 8.2% each, `close` 7.4%, `open_hero` 6.6%, `back` 5.5%,
`open_stack` 5.3%; `move_to` 4.2%; `end_turn` 0.9%; recruiting, building,
hiring and upgrading together 0.5%. The games stop at day 60, so this is the
early game only.

**What this means for a cycle.** The user's update 1790 collected its 100
games in 1 h 54 min on ten actors under a similar load. At 1.6 times that is
about 1 h 10 min, and about 1 h with 14 actors; neither was run. A 1,000-game
cycle of ten such batches would spend about 10 hours playing instead of 19.
The target of one hour for the whole cycle is 10 to 20 times further away, and
no exact change found here closes it: the engine capture is about a third of a
decision's CPU and Python's share is spread over many small parts.

**Not changed, proposed to the user.** (1) Fewer decisions per game day: 43%
of decisions are stack moves and another 41% open or close a window; the
guard's budget is 32 stack moves per hero and town per day. A smaller budget
is a new guard version and changes what the policy can do. (2) Playing the
next batch while the update runs: the update needs the GPU and one core, the
actors the other cores; a cycle would take the longer of the two instead of
their sum, with games one update behind. (3) A day limit saves little (entry
above). (4) Rebuilding the engine's capture so that it is cheap to see that
nothing changed: exact, but a restructuring of the plugin's observation code.

**Limitations.** One evening on a busy machine; no full batch of 100 games and
no training run was made with the new code; the longest games measured ran to
day 60 (the windows run to 365 but last 150 s); the new runtime has started a
few hundred games in these measurements, most of them stopped after 150 s, and
has played no game of 365 days. `vcmireplay`
was not run against the changed recorder: with the ledger off there is nothing
to replay, and `--keep-game-files` restores it.

### 2026-11-20 — Review of the play-speed branch: four findings, fixed; the replay check is not run yet

The user reviewed PR 35 at `3814064` and left four comments. All four were
confirmed by reading the code; the second one was also reproduced with the
scripted engine before it was fixed. None of the fixes touches the play loop
or the update: a seeded game plays as it did.

**1. A kept game could not be verified.** The launcher sets
`EXPLORER_RL_DECISION_OBSERVATION=0` for every game, also with
`--keep-game-files`, so a kept ledger has decisions without the observation.
`vcmireplay` never set the variable and the adapter's default is 1, so a
replay from a fresh shell regenerated the observation in the first decision
and `ContextualEvidence::expect` rejected it. An older capture replayed from a
shell that still held `=0` failed the other way round.
*Fix, in `engine/ExperimentReplay/ContextualReplay.cpp` only:* before the
adapters are created, `ContextualReplay::run` reads the first recorded
decision of the tape and sets the variable to match, in both directions, as
`configureControllers` already does for `army_information`. Every regenerated
record is still compared whole with its recording. The summary of a verified
replay says which form was read (`decisions_repeat_observation`). A tape whose
decisions are not all of one form fails at the first that differs. The launcher
is unchanged: every game stays compact. `vcmiclient`, the plugin and the
recorder are not touched, so the digests of the entry above stand and the
default runtime stays `local-v8-play-switches-v1`; its own `vcmireplay` was
frozen before this fix and rejects compact captures, so a kept game is
verified with a `vcmireplay` built from this commit on.
Considered and not done: writing the form into the ledger's header. It would
change the recorder inside the plugin, which means a new play runtime and new
proof that games are unchanged, for no gain in strictness: the observation of
a decision is the one of the view record before it, and that record is
compared either way.

**2. Two benchmark calls removed each other's games.** `bench_play.py` named a
game's folder by its number under `STATE/games`, the number follows from the
seed, and an actor removes the folder before and after playing; the state lock
is taken only by `train`. Reproduced: two calls with the same folder and seed,
at once, on the scripted engine, gave `actor_error` in the unfixed code.
*Fix:* every call plays in a folder of its own inside STATE
(`bench-play-…`), removed at the end. It stays inside the state folder, which
is what the perf guide promises about these tools.

**3. A failed batch looked like a fast one.** With `--games`, a run in which
games failed printed its report and exited 0. *Fix:* the report is printed and
appended as before, and the exit status is 1 with the count on stderr when any
requested game has no trajectory. A game stopped at the day limit has one and
counts. The window mode is unchanged: it counts decisions, a failed game there
lowers the rate instead of raising it, and its report carries the failures.

**4. A failed freeze left a half-copied runtime at the final path.** It had an
executable `vcmiclient`, which is all `Engine.check` asks for, and the retry
failed on the existing `bin/AI`. *Fix:* `freeze_runtime.py` builds the runtime
in `NAME.partial-<pid>` beside the final place and renames it when `config`
and `Mods` are in; on any failure the partial folder is removed. An existing
`NAME` is refused before anything is copied, and the rename does not replace a
folder that has files in it.

**Tests.** 52 pass and 3 are skipped in the cloud container (Linux, no Apple
GPU; 38 and 2 before): 12 new ones, and the two play-benchmark tests now run
without MLX because only the update benchmarks need it. The three benchmark
tests fail on the code before the fixes. The freeze tests replace `otool`,
`install_name_tool` and `codesign` by a recorder. The changed
`ContextualReplay.cpp` was syntax-checked with g++ 13 (`-std=c++20
-fsyntax-only`) against the engine headers of the Mac's `vcmi` checkout and
this branch's `ExplorerRL` headers; a deliberate error was caught the same
way. It was **not built, linked or run**.

**Not verified: finding 1 against real captures.** That needs the engine and
the game data. `tools/check_replay.py` is the acceptance test: it plays one
short game with the ledger, then replays it from an environment without the
variable, replays it with the other form set, and replays it with one recorded
answer of the policy replaced by another answer of the same seat; it passes
only if the first two verify and the third does not. Run with the training
engine it checks a compact capture, and with `--vcmi` naming
`local-v8-sleep-resume-v1` a capture that repeats the observation; the
commands are in `docs/architecture.md`. The script itself has only run against
hand-written captures and stand-in verifiers (one that reads the form from the
tape passes, one that takes it from its environment fails, one that accepts a
changed answer fails). What it assumes about the real files and has not seen:
that a game stopped at the day limit leaves `engine/replay.jsonl` as `init`,
`start` and input/applied pairs with at most one trailing input, and that such
a game verifies with `--prefix` from its own profile folder. If one of these
does not hold the script stops with the reason and keeps the game.

### 2026-11-20 — The replay check on the real engine: both kinds of capture verify, a changed answer does not

The user approved the launch ("Yes, Claude runs it" on the question whether
Claude may run the replay check, then "I approve everything. Run as you will").
Everything ran in the clone `~/Projects/Heroes-opt` at `748a041` and in a new
state folder `~/Projects/Heroes-artifacts/replay-check`, which holds
`latest.json` and `checkpoints/update-001789.pt` copied from
`play-speed-r3u8uh` (same SHA-256). `~/Projects/Heroes` and `training-trial`
were not touched; the training run there (pid 42215) was going throughout.

**Build.** `prepare_vcmi.py` refuses to change a checkout it prepared earlier
("unexpected content": its own patches). The only file that differed from the
recipe was `ContextualReplay.cpp`, and the copy in `vcmi/` was byte for byte
the old one, so the new file was copied to its place by hand; `--check` then
passed. `cmake --build vcmi/build --target vcmireplay --parallel 2` compiled
one object and relinked `vcmireplay` in 6.4 s. `libvcmi.dylib`, `vcmiclient`,
`vcmiserver` and the AI plugins kept their timestamps of 19 November.

**Result.** `tools/check_replay.py`, seed 1, three days, `--replay` naming the
rebuilt `vcmireplay`; both calls exited 0 at the first attempt, with no change
to the script or the verifier.

| | engine that played | decisions in the capture | wall |
|---|---|---|---|
| a | `local-v8-play-switches-v1` (training's) | compact | 5.7 s |
| b | `local-v8-sleep-resume-v1` (before the switch) | repeat the observation | 4.6 s |

In both: `verified`, `form_read_from_capture`,
`verified_with_the_other_form_in_the_environment` and
`changed_answer_rejected` are true. The verifier's own summary named the form
that was in the capture each time, with the other form set in its environment
too. Checked per replay: 919 client packets, 673 messages, 241 requests,
335 responses, 2,630 transactions; the ledger has 7,429 records. Record 3763
(an answer of seat 0, replaced by another answer the same seat gave) made the
replay stop at record 3768, "Contextual replay differs", at the next message
of that seat after decision 133 was consumed. Both games were cut at the day
limit (`truncated:game_day_limit`, day 4) and replayed with `--prefix`, so the
two things the script assumed about a cut game's files hold. The lines are in
`replay-check/check-replay.jsonl`.

The two engines played the same game: choices `[202, 133]`, the same number of
records and of everything checked. That is the claim of the play-speed entry
(the switches change what is sent, not what is played) seen once more, on one
short game.

**Also on the Mac.** `tests/test_check_replay.py`, `test_freeze_runtime.py`
and `test_perf_tools.py -k "not update"`: 14 passed (macOS, with the pytest of
`perf-opt-mq0j5t/pydeps`; the training environment has none). The update tests
were left out so that no GPU work ran beside the user's training.

**Not done.** The full suite was not run on the Mac. One game of one seed and
three days was checked per engine, not a whole game; a whole game's ledger is
several gigabytes. `freeze_runtime.py` was not run for real, so no runtime
with the new `vcmireplay` is frozen: verification uses the one in
`vcmi/build/bin`, as `docs/architecture.md` says.

### 2026-11-20 — The replay check counted a crashed verifier as a rejection; corrected

A fifth review comment, on `tools/check_replay.py` at `92c1dc8`. The user
reproduced a false pass: a stand-in verifier passed both clean replays and
then ended with status 42 before writing a report for the capture with the
changed answer. The script's fallback for a missing report was
`{"verified": false}`, the third check asked only for `verified == false` and a
status other than 0, so the crash counted as "changed answer rejected" and the
whole check passed. Confirmed by reading the code and by a test that fails on
`92c1dc8`.

**Fix, in the checker only.** The changed answer counts as rejected when the
verifier's report exists, is a JSON object, says `verified: false`, carries the
error of `ContextualEvidence::expect` ("Contextual replay differs at record N")
with N greater than the number of the changed record, and the status is not 0.
The changed answer is itself read from the capture, so its effect can only show
in a record regenerated later; a difference at an earlier record is not caused
by it. A run without a readable report no longer gets a `verified` value at all;
its entry says the status and where the console log is. On any failed check the
work folder stays, as before, with every report and console log. The verifier
and the engine are not touched.

The rule is tied to the wording of that one error. If the wording changes, the
check fails and says what it got; it cannot pass by mistake.

**Tests** (scripted, 61 pass and 3 skipped in the cloud container; 52 and 3
before). Seven stand-in verifiers that pass both clean replays and then, on
the changed capture: crash with status 42 and no report, leave half a report,
write JSON that is not a report, stop with another error, give no reason,
report a difference before the changed answer, or report the difference and
end with status 0. Each fails the check; the first six passed it at `92c1dc8`.
One test runs `main` with such a verifier: exit status 1, the folder is kept
and named in the report. One test gives the rule the error text the Mac's
`vcmireplay` produced today (difference at record 3768 after the change at
3763): accepted. The earlier test with a verifier that takes the form from its
environment now also shows the second case: its difference at the first
decision is not counted as a rejection of the answer changed later.

**Not run.** The two real-engine checks of the entry above were not repeated
with the corrected script. What they recorded satisfies the new rule (the two
lines in `replay-check/check-replay.jsonl` carry "Contextual replay differs at
record 3768", changed record 3763, and `vcmireplay` ends with status 3 on that
error), but that is read from their output, not observed by a new run.

### 2026-11-20 — Faster game play, second round: the engine compares numbers, Python keeps what did not change

The user asked for the largest possible speedup of play, trying several ideas,
toward a 1000-game cycle in under an hour, and wrote "run it" in the thread for
measurements on the Mac. Branch `claude/project-thread-r4x7mn` on top of the
play-speed branch at `cb5fe99`, worked on in a clone of its own,
`~/Projects/Heroes-fast`; the training checkout was not touched. State folder
`~/Projects/Heroes-artifacts/fast-r4x7mn` (checkpoint 1789 copied from the
play-speed folder; `src-<commit>/` are the frozen trees every measurement ran
from, `measure-1.sh` to `measure-4.sh` the queues, `results/*.jsonl` the
reports, `logs/` the times of every start and end).

**Starting point and hypotheses, written before the measurements.** The entry
of 2026-11-19 left two large items: the engine rebuilt each player's whole
observation as JSON after every applied pack to see whether it had changed
(about 2 ms of a 6 ms decision), and the policy's forward pass (1.9 ms). The
map is 20 by 20, so each observation carries up to 400 tiles, about 30 KB of
its JSON, and the old CPU sample put more than two thirds of the capture's time
into the tile loop. Hypothesis 1: collecting tiles and objects as plain numbers
and comparing those finds the unchanged map without JSON, and a view that did
not change the map need not carry it; the engine's share should about halve.
Hypothesis 2: a view costs about 0.9 ms in Python because the passive army
statistics (17 fields for each of up to seven stacks of every hero and town)
are decoded, rebuilt as a semantic tree and walked for every view although one
army at most changed; doing that once per unchanged value should remove most
of it. Acceptance for both: the digest of everything the learner reads stays
the same on seeded real games, single and with the actor pool, versions
measured in turns from frozen trees.

**Implemented.**

- `7b39cfd`, engine and reader. `ExplorerRL::contextualObservation` fills a
  `VisibleMap` (per tile: position, terrain, road, blocked, visitable; per
  object: engine id, position, kind, subtype and the relative owner the policy
  is told) and builds JSON for everything else as before. Two maps are equal
  exactly when the JSON written from them is: object ids and handles
  correspond one to one, and the owner is stored as the relation, so two
  different neutral owner values still compare equal as they did. Handles are
  still taken while the tiles are scanned, so their numbering is unchanged.
  `captureContextualLocked` compares map and rest and returns as before when
  neither changed. `EXPLORER_RL_REPEATED_MAP=0`, which the launcher now sets,
  leaves `tiles` and `objects` out of a view when they are those of the
  player's previous view; the first view always carries them, and a decision
  that repeats its observation (the older form) is still complete.
  `Producer` hands `decode_observation` the previous view's tuples for such a
  view. `vcmireplay` reads the form from the capture (a view without tiles),
  as it does for decisions. `Observation.__post_init__` runs its tile checks
  once per tile tuple and map size.
- `ad363c0`, Python. `contract._decoded_once` keeps one decoded object per
  exact raw value (marshal bytes, the argument already used for tile lists)
  for own armies and their stacks, displayed armies, heroes, towns, town and
  capability panels. `policy._semantic` builds the tree of such an object once
  and keeps it as `_Kept`, a tuple subclass that remembers its hash, its repr
  (used to sort sets) and whether equality is identity inside it; bindings of
  two kept trees are kept as well. `_encode_tree` visits a kept tree once per
  version of the weights and adds the stored cache entries at later meetings;
  the recipe compiler's arena does the same once per game while the frame has
  added no leaf whose equality is not identity. The argument for exactness is
  in the docstrings: under the memo every cache entry is the exact encoding of
  its tree and the cache holds the subtrees of whatever it holds, so a visit
  leaves the cache with the tree's subtrees added.
- `1219d1d`, three small memos from the new profile: the guard's sort of all
  tiles per observation, the bounds check of a new tile tuple, and
  `army_operation_choices` asked several times about the same armies.

Runtime `local-v8-changed-map-v1`, built in `~/Projects/Heroes-fast/vcmi`
(`prepare_vcmi.py --check` passes) and frozen with `tools/freeze_runtime.py`;
it is now the default in `heroes/config.py`. An older runtime ignores the new
switch and an older reader never sets it, so the pairs still play.

**Tests.** 77 scripted tests pass (72 before). New in `tests/test_env.py`: a
view without a map keeps the tuples of the view before; an engine that repeats
the map plays the same game with an equal recipe digest; a first view without
a map, and a view with only one of the two parts, are refused; a checked tile
tuple is checked again for another map size. New `tests/test_policy_reuse.py`,
on hand-built observations of two heroes exchanging stacks with a town and
passive statistics, because the scripted engine shows no armies: unchanged
parts are the same objects; a kept tree equals, hashes and prints as the tuple
it was built as; a pass with all reuse gives bit for bit the states, values,
sampled actions, log-probabilities and per-decision cache entries of a pass
with autograd on and no reuse at all, while walking less than half the trees
of the earlier memo; kept encodings do not outlive the weights; the compiled
recipe is the same with and without reuse.

**Measured**, 2026-11-20 16:27 to 17:10 local. Charger 140 W, battery 100% and
not falling. The user's training run (PID 42215, `heroes train 1000` in
`~/Projects/Heroes`) was in its PPO update throughout, about 55% of one core
and the GPU; every figure is from a Mac that was doing that as well. Old
engine: `local-v8-play-switches-v1`.

One seeded game to day 60 per call (`profile_game.py`), in turns:

| seed, decisions | | PR 35 (`cb5fe99`) | PR 35 code, new engine | map (`7b39cfd`) | reuse (`ad363c0`) | memos (`1219d1d`) |
| --- | --- | --- | --- | --- | --- | --- |
| 4, 14,208 | play, compile | 90.5 s, 11.6 s | 74.6 s, 11.4 s | 66.3 s, 11.8 s | 48.6 s, 8.2 s | 47.2 s, 7.6 s |
| 9, 11,658 | play, compile | 66.5 s, 8.2 s | 54.9 s, 8.4 s | 44.4 s, 8.3 s | 35.5 s, 6.3 s | 34.9 s, 5.6 s |

The memos were measured in a later round against `ad363c0` again (51.6 s and
37.1 s that time), so their step is 9% and 6%, not the difference of the two
columns. Digests `fec36b04…` (seed 4) and `f154a6e3…` (seed 9) in every cell:
the same games. For seed 4 the wait for the engine went 42.6 s, 17.3 s, 15.6 s
(PR 35, map, reuse), the forward pass 25.8 s, 28.8 s, 15.4 s, decoding 10.6 s,
8.0 s, 6.6 s.

Fourteen actors on the same fourteen seeded games to day 60
(`bench_play.py --workers 14 --games 14 --seed 3`; 132,920 decisions, 61,734
samples): PR 35 139.9 s and 151.2 s, `ad363c0` 82.4 s and 89.5 s in turns,
`1219d1d` 79.6 s afterwards. All fourteen digests are equal in all five runs.
One actor's time per decision, compilation included: 8.55 and 9.38 ms, 5.17
and 5.56 ms, 4.89 ms.

Fourteen actors kept busy for 130 s after a 20 s warm-up, games to day 365
(`--seconds 130 --seed 7`), in turns: PR 35 1,209 and 1,145 decisions per
second, `ad363c0` 1,993 and 1,697. A 77 s run of the scripted tests overlapped
the first PR 35 window; the second was undisturbed. With the mean game of the
user's 421-game log (33,566 decisions), 100 games are about 48 minutes at
1,177 per second and about 30 at 1,845. That is derived, not a played batch.

**The replay check on the real engine** (`tools/check_replay.py`, 3-day game,
7,429 records): a capture of the new engine, whose views after the first leave
unchanged maps out, verifies with the new `vcmireplay`; so does a capture of
the old engine, which repeats the map; both reject the changed answer at
record 3768 after the change at 3763.

**Where the time is now.** CPU sample of the new engine during the seed 4 game
(`results/engine-sample-map.txt`, 20 s, with `7b39cfd`): the captures are 12%
of the game's wall time (over a third of that in `ownArmyDetails`, a fifth in
the tile visibility scan), writing JSON 5%. Python profile of the same game
with `ad363c0` (`results/reuse-seed4.prof`; cProfile inflates Python): the
walk that samples or evaluates an action is the largest block (about 56
`Action` objects are built and checked per decision, and the guard's `permits`
is asked about 30 times); tiles of views that did change the map (8,301 of
22,016 observations) cost a decode and a bounds check per tile; compiling the
recipe evaluates every trained decision's walk a second time. Each of these is
a few percent. No further change is claimed here.

**A training change, measured and not adopted.** The same fourteen seeds with
the guard's daily stack-management budget lowered from 32 per hero and town,
in copies of `ad363c0` with the constant edited (`src-limit8`, `src-limit4`):
budget 8 gave 83,702 decisions over 634 game days, 66.0 per seat and day
against 99.8; budget 4 gave 69,974 over 631, 55.4 per seat and day. These are
other games (the policy samples from another support), so only the counts
compare. With budget 4 the learner seats' actions are 71% opening and closing
windows and 7% moves. From the 421 games of the user's log, a day limit of 180
instead of 365 would keep about 72% of the decisions, and 66% of the games end
before it against 85%. The guard and the limits are unchanged; these numbers
were given to the user with two further options, training on both seats and
playing the next batch during the update.

**Limitations.** Timings are from a Mac that was also running a PPO update,
one sitting, two seeds for single games and one seed for the pool. The
exactness evidence is the digests of 16 seeded games to day 60, the scripted
tests and the argument in the code; no full-length game and no training run
was made with this code. The decoded-value cache and the kept trees hold up to
4,096 and 8,192 entries per actor process and the kept encodings 2,048; their
memory was not measured.

### 2026-11-20 — User stopped the 1,000-game run after update 1791

The terminal run in `~/Projects/Heroes-artifacts/training-trial` resumed
from 1787 with ten actors and MLX PPO. It saved four updates of 100 games,
1788 through 1791. At the user's request the coordinator received SIGINT
during collection for 1792. Three games had finished and ten were active in
that batch. Its trajectories were discarded before the PPO update.

The coordinator, ten actors, ten engine processes and both helper processes
exited. The state lock was available and the games directory was empty.
The checkpoint file and `latest.json` were byte-identical across the stop.
The checkpoint file SHA-256 is
`ca190c8a51257559c1243761420e68d6c26ef6dff8eb0bb0bbaf8879a0261fe9`.
The remaining 600 games can resume from 1791 in the same state directory.
The private `log.jsonl` records the jobs, outcomes and saved updates.

PRs #35, #34, #33 and #32 were merged into their successive bases,
ending at main commit `2363a6a`. Its tree matches reviewed head `cb5fe99`.
A separate main worktree at `~/Projects/Heroes-main` passed
`./train.command check --state ~/Projects/Heroes-artifacts/training-trial`
with the play-switches runtime, MLX and the saved progress for update 1791.

### 2026-11-20 — Play speed, third round: the settings that change training, a hero's step as two tiles, and what is left

The user's answer to the options of the entry above was "exhaust all choices,
try everything". Read as: build every training change as a setting, measure
each on real games, and continue with the pure speedups. During the work he
withdrew one: no shorter day limit, "with enough training games will become
shorter by themselves". A real training run was not started by the assistant,
also not in a scratch folder; that launch stays with the user. Same branch,
clone and state folder as the entry above; queues `measure-5.sh` to
`measure-7.sh`.

**Built** (`61bd38b`, then `ff5a2f9` removed the day-limit flag again). All
off unless given; a run without them plays and trains as before.

- `--both-seats`: `actor.play` also builds the opponent seat's trajectory
  (`Game.other`); the actor compiles both recipes and the update gets both.
  A seat whose history gives no trajectory is left out and the game counts
  with one.
- `--stack-moves N`, `--windows N`, `--decisions N`: daily budgets of the
  interaction guard, set through its name (`guard.version`,
  `VERSION+stack-moves-4+windows-6`). The name travels with every trajectory,
  so the recipe compiler and the reference update build the same tracker.
  `stack-moves` replaces the constant 32. `windows` counts a hero's or town's
  optional window navigation of the day (the same actions the existing
  repeat limit counts) and closes further navigation for that subject.
  `decisions` counts a seat's answers of the day; past it only closing,
  mandatory answers and ending the turn are permitted. The standard guard
  keeps its name and its behaviour; only one spelling of a name is accepted.
- `--play-during-update`: collection became a `Batch` that can be stepped; the
  update's progress callback steps the next batch with the weights the update
  started from, and after the update the rest of that batch is played with the
  new weights.

**Tests.** 86 scripted tests pass. New: guard names and their refusals, the
window budget closing one hero for one day and no other, the decision budget
leaving only the end of the turn until the next day; both seats giving two
trajectories with opposite rewards and twice the samples in the log; the next
batch's games handed out before the previous update is logged, and after it
without the setting; the flags reaching the run. The play-during-update path
has only run against the scripted engine and the reference update.

**Measured**, 17:15 to 18:47 local, 140 W, battery full. The user's training
run was in its PPO update until 18:32 and has been playing its own ten games
since; what was timed after 18:32 is marked. Trees `src-61bd38b` (settings)
and `src-a608dff` (round 2), runtimes `local-v8-changed-map-v1` and
`local-v8-tile-changes-v1`.

Both seats, the same 14 seeded games to day 60 with 14 actors, in turns
(`results/seats.jsonl`): one seat 74.3 s and 72.8 s, both seats 88.4 s and
89.4 s; one actor's time per decision 4.51 and 4.45 ms against 4.93 and
5.00 ms; recipe compilation 54 s and 52 s against 116 s and 116 s; samples
61,734 against 132,911. The learner seats' digests are the same in all four
runs: 2.15 times the samples for 11% more actor time.

Actor counts, 130 s windows after a 20 s warm-up, in turns
(`results/workers.jsonl`): 14 actors 2,055 and 2,191 decisions per second, 16
actors 2,027 and 2,186, 18 actors 2,204 and 2,287.

Daily budgets, each on the same 14 seeds to day 60 (`results/budgets.jsonl`).
Another support means other games, so these are counts. Decisions are of both
seats per seat and game day; moves per learner seat and day; the last columns
count the learner seats' recruitments, buildings and hirings in the 14 games.

| guard | decisions | moves | recruit | build | hire |
| --- | --- | --- | --- | --- | --- |
| standard (stack moves 32) | 99.8 | 4.15 | 44 | 16 | 61 |
| stack moves 8 | 66.0 | 4.23 | 48 | 19 | 60 |
| stack moves 4 | 55.4 | 3.52 | 37 | 17 | 45 |
| windows 12 | 81.1 | 3.96 | 41 | 13 | 48 |
| windows 6 | 73.8 | 3.86 | 38 | 13 | 45 |
| stack moves 8, windows 8 | 53.9 | 3.44 | 55 | 19 | 59 |
| stack moves 4, windows 6 | 43.2 | 4.37 | 41 | 21 | 63 |
| stack moves 4, windows 4 | 27.6 | 3.80 | 22 | 10 | 37 |
| stack moves 4, windows 3 | 29.2 | 4.12 | 24 | 4 | 43 |
| stack moves 2, windows 6 | 32.0 | 3.62 | 33 | 8 | 40 |
| stack moves 2, windows 4 | 25.3 | 4.09 | 30 | 11 | 43 |
| stack moves 2, windows 3 | 18.4 | 3.02 | 27 | 5 | 37 |
| stack moves 1, windows 3 | 19.5 | 3.14 | 26 | 5 | 43 |
| decisions 60 | 48.7 | 3.02 | 29 | 11 | 32 |
| decisions 40 | 33.4 | 2.00 | 7 | 7 | 15 |
| decisions 25 | 24.4 | 1.48 | 12 | 3 | 15 |

Read with care: 14 games per row, one seed, the first 60 days, and the small
counts move by chance (16, 19, 17 and 21 buildings in four rows that should
not differ). What the table supports: stack moves 4 with windows 6 removes
57% of the decisions with the useful actions where they were; a window budget
below 6 costs recruitments and buildings, which need three window steps each
in a town; the flat decision cap removes useful actions in proportion (at 40,
half the moves and a sixth of the recruitments), because a policy that samples
free actions at random reaches the cap before it moves. The stack-8 and
stack-4 rows repeat the counts of the edited copies of the entry above (83,702
decisions; 69,975 against 69,974, one answer of the opponent at the end of a
six-day game), so the setting does what the edited constant did.

Day limits on 14 whole games, seed 5 (`results/days.jsonl`, before the limit
was withdrawn): to day 365 539,195 decisions, 12 natural endings, 535 s; to
day 180 347,763 decisions (64%), 9 natural endings, 323 s; to day 120 261,996
(49%), 8 natural endings, 190 s. The estimate from the user's log in the entry
above (72% at 180) was too high: it assumed a game's decisions spread evenly
over its days, and later days hold more.

**Whole games match.** The same 14 games to day 365 with PR 35's tree and
engine: 980 s, and all 14 digests, statuses and decision counts equal those
of `61bd38b` with `local-v8-changed-map-v1`. One actor's time per decision
8.73 ms against 4.48 ms. This is the first evidence beyond day 60 that the
reuse of the entry above, with its bounded caches, changes nothing.

**Engine, round 2** (`a608dff`, runtime `local-v8-tile-changes-v1`, now the
default).

- A view that changed the map while the same tiles stayed visible carries
  `tile_changes`, the tiles that differ, and its objects
  (`EXPLORER_RL_TILE_CHANGES=1`); the reader replaces them in the previous
  view's tiles and refuses a change that fits no tile, repeats one, or changes
  nothing. A hero's step changes two tiles and is a view for each player who
  sees it: 8,301 of the 22,016 observations of the seed 4 game changed the
  map. `vcmireplay` reads this form from the capture as well.
- The policy socket's line is written straight from the JsonNode.
  `JsonNode::toCompactString` pretty-prints through a stream (the old lines
  carried the tabs); the new writer emits the values alone and escapes strings
  with JsonWriter's own rule. Checked with a temporary build that parsed every
  line back and compared it with its message: both seeded games ran to day 60
  with their digests; that build was not kept.
- The tile scan asks `getTile` alone, which tests visibility itself.
- In Python, displayed army slots are indexed once per tuple of armies.

In turns with `61bd38b` on its runtime, the user's run still in its update:
seed 4 43.8 s play to 37.1 s, seed 9 34.1 s to 27.8 s, digests equal; the
wait for the engine went 14.9 s to 9.8 s and decoding 5.4 s to 3.8 s (seed 4).
14 actors on the 14 games to day 60: 75.2 s and 77.3 s against 69.1 s and
62.5 s, all digests equal; one actor's time per decision 4.59 and 4.78 ms
against 4.33 and 3.83 ms. The four 130 s windows that followed ran after
18:32 beside the user's ten games (1,613 to 2,758 decisions per second with
no order between the versions) and say nothing. Against PR 35 the seed 4 game
is now 90.5 s to 37.1 s of play and 102.2 s to 43.9 s with compilation.

The replay check passes for the new runtime's `vcmireplay` on captures of all
three engines (changed tiles, unchanged maps left out, maps repeated): each
verifies from either environment and rejects the changed answer.

**Not built, with the reason.** Reusing the play-time encoded frames in the
recipe compiler (about 3% with both seats) and recording the sampling walk's
filtered choices for the compiler's second walk (about 3%): both add state
that must not be pickled or mistaken. Caching the passive army statistics in
the engine (about 4%): they come from bonus-system queries whose inputs have
no cheap version stamp, so a cache could not be shown exact. Batched or GPU
inference: the policy is a 64-unit GRU over trees encoded node by node; its
time is Python and small-tensor overhead, not arithmetic a GPU would take. A
compiled Python (mypyc, Cython) was not tried.

**What it adds up to, derived.** At about 2,120 decisions per second with 14
actors (measured before round 2) and the 33,566 decisions of the mean game in
the user's log, 100 games are about 26 minutes of play against 48 at PR 35's
rate; round 2 is about 15% on top by the 14-game batches. If the 57% fewer
decisions of stack moves 4 with windows 6 held over whole games, 100 games
would be about 10 minutes, and with both seats 1000 trajectories would come
from 500 games in under an hour of play. That is arithmetic on a 60-day
count, not a played batch, and says nothing about what a policy learns under
those budgets.

**Limitations.** No training run was made with any of these settings; the
update with two trajectories per game and the play-during-update path have not
met the real engine or the MLX update. Budget rows are single 14-game batches.
The last engine change was timed on two single games and two batch pairs only,
because the Mac was no longer free.

### 2026-11-20 — Review of the third play-speed round: three findings and the suggested launch, corrected

The user reviewed PR 36 at `de0720e`. He confirmed the collection speedup on
the saved measurements and reproduced three faults with the scripted engine.
All three were real.

1. **Play during the update stops the second MLX update.** Games begun during
   update N are played by the weights from before it. Before its first step
   the MLX update compares what it computes for the first minibatch with the
   log-probabilities and values recorded during play, and stops when they
   differ ("Initial MLX prediction sanity limits exceeded"). The tests of the
   entry above ran the reference update, which has no such check, so this went
   unseen, and the launch suggested in the PR would have failed at its second
   update. Fixed the small way: `train` refuses `--play-during-update` with
   the MLX backend before anything is played. The setting now works with the
   reference update only, which is too slow for training; supporting MLX
   needs each trajectory to name its behaviour checkpoint and the check to be
   made against those weights. Not built.
2. **A failing next batch lost the update that was running.** `Batch.step`
   raised from inside the update's progress callback, which unwound the update
   before `checkpoint.save`. The callback now keeps the error and stops
   stepping; the update finishes, is saved and logged, and the error is raised
   after that. A restart continues from the saved update.
3. **The two-seat benchmark hashed one seat.** `bench_play.py --both-seats`
   counted both seats' samples but its `digests` covered the learner seat
   only, so a changed or missing opponent trajectory was invisible. The report
   now also carries `seat_digests`, one per colour and game, `null` for a
   missing trajectory; `digests` is unchanged, so earlier runs still compare.
   The two-seat rows of the entry above therefore show that the learner seats
   were unchanged and say nothing about the opponent trajectories beyond their
   sample count.

**The suggested launch was wrong** in two ways besides the overlap flag. Its
state folder did not exist, so `restore` would have imported the earlier
pipeline's checkpoint 1785 instead of 1791; and it mixed the new settings into
what was meant as a speed comparison. Corrected in the PR description: copy
`update-001791.pt` into a new folder, write a `latest.json` that names it, and
read `./train.command check --state` before playing. Tried without playing:
the check prints `continues {'updates': 1791, 'games': 10342, 'decisions':
20791899, 'optimizer_steps': 652799}`, and the copied file has the SHA-256 the
entry of the pause records (`ca190c8a…`). The comparison run keeps the usual
settings.

**Tests.** 89 scripted tests pass (86 before): the refusal with the MLX
backend through `train` and through the command line, with no state folder
created; a first batch of two good games, every later game failing while a
slowed reference update runs, ending with update 1 saved, the failure raised,
and a restart reaching update 2 (without the fix the same test leaves only the
initial checkpoint); `seat_digests` equal across two runs, the learner's equal
to the one-seat digests, and a dropped opponent trajectory shown as `null`.

**Not run.** The two-seat benchmark was not repeated on real games with the
new digests: the user's 100-game baseline run on main was in progress and a
14-actor benchmark beside it would have spoiled its timing. A full 100-game
cycle including the PPO update has still not been measured with this branch;
the 26 minutes of the entry above are collection only.

### 2026-11-21 — Review of PR 36, continued: either seat admits a game; the two-seat benchmark on real games; one racy ending

**A fourth review finding**, posted after the entry above and seen only when
the user asked the thread to continue. With `--both-seats` the learner seat's
trajectory was built first, and a learner without one made the whole game
`failed` / `unusable_trajectory` before the other seat was looked at. He
showed it with the scripted engine and `Limits(max_days=1)`: the same game,
weights and seeds were kept with red as designated learner (one trajectory,
red's) and thrown away with blue. Real: the opponent's trajectory was treated
as an extra, not as an equal.

Fixed in `35ba545`. `actor.play` builds each seat's trajectory on its own;
`Game.trajectory` stays the designated learner's and `Game.other` the
opponent's, and either may be missing. The game is `failed` only when neither
exists, and the collector asks whether a game has any trajectory. The game's
`detail` names the seat that gave none (`blue: this seat completed no
action`). With one seat the rule is as before. `bench_play.py --both-seats`
counts such a game, lists `red` then `blue` in `seat_digests` whoever the
learner was, and exits 1 only when a game has no trajectory at all.

Tests, 90 scripted (89 before): the reviewer's case in both assignments, both
`truncated`, the same red trajectory by digest after compiling, as the
learner's in one game and as the opponent's in the other; one seat still fails
with blue as learner; the training loop over two such games admits both with 0
failed; the benchmark's report and exit status for the same games with one seat
and with both, and with both seats crashing.

**The two-seat benchmark on real games**, which the entry above left open.
2026-11-21 01:14 to 01:43 local, 140 W, battery full, no training run on the
Mac (the user's baseline cycle had ended at 01:02). Tree `src-35ba545`,
runtime `local-v8-tile-changes-v1`, the 14 games of seed 3 to day 60, queues
`measure-8.sh` and `measure-9.sh`, reports `results/seats2.jsonl` and
`seats3.jsonl`.

- Standard guard, 14 actors, one seat and both seats in turns: 70.3 s and
  66.3 s against 82.1 s and 83.2 s; one actor's time per decision 4.32 and
  3.99 ms against 4.52 and 4.54 ms; compilation 53 s and 52 s against 115 s and
  114 s; samples 61,734 against 132,911. The learner seats' digests are those
  of the one-seat runs and of the runs of 2026-11-20.
- All 28 seat digests are equal in seven two-seat runs with the standard
  guard, one of them on 7 actors; every game has both colours and the 28
  digests are distinct.
- With stack moves 4 and windows 6, ten two-seat runs: 27 of the 28 seat
  digests are equal in all ten. The blue seat of game 9, a natural ending on
  day 8 in which blue is the designated learner, has three digests: 7, 2 and 1
  times. Its game answers 161 and 210 decisions in eight runs and 161 and 211
  in two. The samples (61,204), statuses, days and the learner seats' actions
  by kind (30,452 actions) are the same in all ten runs. The one-seat run of
  2026-11-20 with this guard had the third of these digests for that game.

**Why that seat differs, from the code.** `ExplorerRL::gameOver` runs on the
network thread: it sets `terminal` and then writes the terminal under
`sendMutex`. The AI thread of the player who moved last reports the move's
result and then asks the next decision through `buildAndSend`, which takes the
same mutex and builds nothing once `terminal` is set. So the terminal comes
before the result, after it, or after the next decision, which it withdraws:
three ends of one history, as the comment in `buildAndSend` says the design
allows. `actor.learner_samples` gives the same samples for all three: the last
action owns the reward with a result or closed by the terminal, and a decision
withdrawn before its answer was used is no sample. The digest covers every
frame of the seat, so it has three values. This is the engine's behaviour from
before this branch; it shows here because this game's last move returns close
to the end of the game, not because of the budgets or the second seat. Not
confirmed by looking at the three histories: a script that would have printed
the last frames of this game was not run. `tools/perf/README.md` now says how
to tell this difference from a change of play.

**The user's baseline cycle, read from his log** (`training-trial/log.jsonl`,
update 1792 on main `c1c96ce`, 14 actors, `--windows-per-minibatch 16
--windows-per-game 8 --learning-rate 8.5e-4`, started 18:50 local on
2026-11-20, beside the last minutes of this branch's benchmarks and four runs
of its scripted tests): 100 games collected in 6,273 s, 6,901,403 decisions of
both seats, 1,100 per second; the update took 16,005 s for 3,467,583 samples
and 13,574 steps, 4,452 s of it building plans. 51 of the 100 games ran to
day 365; the batches of updates 1788 to 1791 had 4, 18, 15 and 24. A game had
65,700 decisions on average against 28,000 to 38,600 in those batches, and a
seat answered 136 decisions per game day against 101 to 120. The cycle was
6 h 11 min and the update 72% of it. This branch changes the collection only:
at about 2,100 decisions per second the same decisions are about 55 minutes
instead of 105, which would make the cycle about 5 h 20 min. Derived, not
played.
