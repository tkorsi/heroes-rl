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
checkpoint1753 weights or Adam state were transferred. All126 protected original
files remain unchanged. Sixteen unique offline controls and an actual saved-tiny
C++/Python prefix parity check passed. A limited128-frame warm-prefix comparison
was1.18x faster; full gameplay/training speed and learning are unmeasured. The
[report](research/tiny-policy-2026-10-30.md) discloses test-selector/manifest and
metadata-gate diagnostic corrections, and [validation](../validation/tiny-policy-2026-10-30.json)
keeps raw initialization/checkpoints private. No gameplay or training started.
The tiny candidate is not accepted or selected by an existing training launcher.
