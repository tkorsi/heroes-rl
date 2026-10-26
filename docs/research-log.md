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
