# Curriculum v1 research record

The append-only [event log](../../validation/curriculum-v1-research-log.jsonl) records preregistration, unsuccessful candidates, validation and results.

Question: can changing only initial game states give unchanged update 146 both natural wins and losses under its own stochastic actions? No scripted rush trajectories are used. Stage 0 is a townless capture opportunity; farther/weaker starts and the original map are later hypotheses, not demonstrated curriculum achievements.

The configuration predeclares separate diagnostic and held-out seeds. Seat results will remain separate. Preparation does not consume or start the new two-hour campaign window. Native validation and do-nothing controls must pass before PPO. The outcome criterion is held-out win-rate change, not weight movement.

## Measured selection result

The bounded study completed 32 native games with unchanged update 146: 8 on the
original 20×20 townless candidate and 24 on the open 8×8 candidate. Each candidate
was tested in both seats. All 32 were natural losses on day 7, with 2490 sampled
choices and zero admitted training samples or optimizer steps. Native replay,
terminal/history validation and exact policy reconstruction passed for every
game. The 20×20 games sampled 80 moves and the 8×8 games 144; none targeted the
opponent town entrance. These are measured observations, not proof that victory
is unreachable. Sparse coordinate exploration is a plausible explanation.

The mixed-outcome gate failed, so no PPO campaign/session was created and no
new two-hour budget began. Do-nothing controls and held-out evaluation were not
run: selection already failed, and there is no trained checkpoint to compare.
No automatic-win or learned-strategy claim is made. Four scenario/seed tests
passed. Owned native cleanup and exact update 146 restoration were rechecked.

The two initial loader errors occurred before gameplay and are logged explicitly;
the canonical checkpoint metadata binding was corrected before the 32 games.
Stage 1 (farther/weaker start) and return to original difficulty remain untested
hypotheses. No failed diagnostic, previous scripted rush sequence or evaluation
was admitted to PPO. See [compact results](../../validation/curriculum-v1-selection-2026-10-26.json).


## Additional study: mixed outcomes, integrity stop

The separately authorized additional study ran for 172.7 seconds and stopped
on its first failed four-game report. The unchanged update146 policy produced
57 fully verified games: Red had 7 wins and 22 losses; Blue had 5 wins and
23 losses. Each counted outcome passed native replay and exact policy
reconstruction. These are results on the easier 8×8 initial state, not evidence
of learning or improvement on the original map.

The four controls passed: one visible town-capture move won naturally in each
seat, while waiting lost under the native seven-day townless rule in each seat.
Control wins are excluded from policy results. The offline target audit found
an initial joint capture-action probability of about 0.00512 in each seat;
reachability was established separately by the native controls.

The agent began editing campaign preparation while the diagnostic was running.
Adding `curriculum_training.py` and changing `builtin_first_run.py` changed the
bound capability implementation identity. Red block7 game2 completed gameplay
but failed its source-integrity check before outcome admission. Its outcome is
unscored. The remaining two Red entries failed before gameplay, and all four
Blue block7 entries were never started. This was an agent workflow error. It
must not be represented as a policy loss or repaired by retrospective admission.
The original preregistration contains stale top-level seed fields copied from
study1; its explicit 64-entry `seeds` schedule and the immutable runner/plans
correctly specify study2 seeds 810001/820001 plus block and episode offsets.

The mixed-outcome observation is positive, but the overall preparation gate
failed. No PPO, held-out campaign evaluation, release or two-hour session was
started, and no retry is being performed. Update146 restores exactly, including
Adam, RNG, progress and opponent pool. Owned-process closure was rechecked.
All raw evidence, failed and unstarted entries, audit and unfinished controller
work are privately preserved. The unpublished controller is retained as a
non-executable patch; it is incomplete and has not been validated for launch.
The runtime dispatch was restored and draft executable modules removed.

See [study2 evidence](../../validation/curriculum-v1-study2-2026-10-26.json).
Future work must keep all bound source files frozen throughout a diagnostic;
prepare a controller before the study or in a separate checkout.


## Renewed authorization and prospective frozen release v2

The user renewed authorization after the source-integrity stop: repair offline,
freeze the complete implementation, repeat clean native acceptance, then launch
one new two-hour local PPO session if all acceptance gates pass. Earlier studies
remain unchanged and diagnostic only. This supersedes their no-retry instruction
only for this newly declared preparation/campaign; it does not admit old games.

The completed controller uses a separate detached Git checkout for execution.
Its entire tracked tree must remain clean. Existing learner/capability identity
coverage remains intact; controller, acceptance runner, configuration and map
generator are explicitly bound as well. Research/status writes target the durable
artifact directory and the separate authoring checkout. No bound executable is
edited between acceptance and campaign completion. Further source work, if any,
must use another checkout and cannot change the executing release.

Acceptance study3 preregisters 64 fresh stochastic update146 games, four per
seat per block for eight blocks. Engine seeds start at1010001 and policy seeds
at1020001, adding100 per block and0–3 per game. The four capture/wait controls
use1030001/1030002 for Red/Blue. The whole study has1200 seconds. Every policy
game must have native replay, exact policy reconstruction, unchanged checkpoint,
natural termination and cleanup; each seat must contain wins and losses. Controls
must demonstrate capture wins and waiting losses separately. A material failure
blocks launch without replacement, retrospective admission or weakening checks.
Exact source/runtime/map/checkpoint bindings and schedules will be appended by
`freeze` before gameplay; paths remain under `local-curriculum-update146`.

The campaign uses `campaign-v2/session.json`, starts its new7200-second wall and
monotonic clocks only at actual launch after acceptance, and reserves the last
300 seconds for final evaluation/preservation/cleanup. Independent owned-process
guardians cover controller worker, batch supervisor and native processes. Training
starts from exact update146 (584 rollouts,40641 decisions,1475 Adam steps), with
unchanged model/Adam/RNG/pool, PPO and native runtime. Four-game batches alternate
Red and Blue on the open8×8 stage3 distribution. Training seeds start910001/920001,
adding4 per batch and0–3 per game. No harder-stage transition is included.

The same untouched fixed held-out seeds710001–710004/720001–720004 run separately
for each seat before PPO and after the last verified checkpoint, with periodic
checks after each five new updates while time allows. They supply no training
samples. Reward remains +1/-1 at natural win/loss and0 ordinarily;365 days and5120
choices, Nullkiller Easy/BattleAI, architecture and optimizer settings remain
unchanged. Only complete verified four-game training batches reach the existing
PPO worker. The outcome measure is descriptive before/after held-out wins per
seat; these four fixed games per seat cannot establish broad strategic progress,
and easier-start victories are not original-map victories.

The offline review added fail-closed acceptance, explicit historical state
restoration, failed-batch preservation, pre-run intent journaling, strict no-retry
namespaces, private backup restoration and wall/monotonic deadline checks. The
unfinished earlier patch remains retained for provenance. No embedded-engine or
PR29 path is used. A campaign correctness failure stops and preserves evidence;
there is no automatic restart, extension, overlap or cloud campaign.
