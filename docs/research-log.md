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
