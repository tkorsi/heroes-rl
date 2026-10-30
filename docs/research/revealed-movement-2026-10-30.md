# Revealed movement and checkpoint 1585 continuation, 30 October 2026

## Status and completed-run evidence

The explicit policy-support intervention, runtime acceptance and real main-checkout
manual preflight **passed**. The final execution source is `14cb529`; later commits
pin its accepted evidence, clarify launcher help and publish results. Checkpoint
1585 is preserved exactly, with 1176 original-target games remaining and sequence
1226 next. No official training was started. The prospective protocol and every
failed diagnostic attempt remain preserved below; [compact results](../../validation/revealed-movement-2026-10-30.json)
bind the evidence.

The user-launched run `run-20261029T211050Z-3a3d748f` finished normally at
checkpoint **1585** after 193 complete updates and 772 admitted games. Its 771
natural losses and one valid collection-window cutoff are distinct outcomes.
All 32 evaluation games were excluded from PPO. The stop reason is
`training_game_target_reached`, worker return code is zero, owned cleanup and
the final full-state restoration passed, and no owned training process remained
in the read-only final audit. This is a completed goal, not a new engine,
adapter, replay, timeout or publication failure.

All 193 training admissions, sequences 1033 through 1225, were audited against
their plans, reports, checkpoints and durable backups. Model, Adam, global RNG,
sampler, progress and pool restore exactly from both final canonical files and
the backup. The final progress is 1585 updates, 6340 rollouts and 922041 decisions;
the cumulative optimizer-step count is 30855. The original 5000-game target has
3824 verified admissions and 1176 remaining; sequence 1226 is next. The prior
772-game goal is complete and must not be restarted or added again.

The full descriptor uses the final training plan `train-001225-seat1`, with its
initial dependency in `train-001224-seat0`:

| Artifact | SHA-256 |
| --- | --- |
| Final plan | `54f19c0c3b343769305d40b78d0ba46d02a71608dc947c483a7a71a6dbe82945` |
| Final report | `1da19d28299fc2bd3f1d82b6a86ccd44933ce3b332c15479808312bb8c3fabaa` |
| Final policy | `8b4ffa90b4e20c2d23f47cb30bf0719671cb378e8019f50186485f7d0e5cd80a` |
| Final resume state | `f983a04dd4b1c18e998ae172ca8f00a34c54a1eb423c51aa1e156acd2fe98411` |
| Initial policy dependency | `07edc5e95a94613b41aa588d649f45dee714d73fc78309f3de5f2a69511c09d9` |

The restored model-weight digest is
`679a58af36d2f8dabbae550214ca2ff947d616595b2345e7dfbd21efb4a55d60`.
Raw checkpoint tensors, captures and detailed audits remain in the private
artifact root. The read-only audit is under
`run-investigations/2026-10-30-final1585-analysis/integrity/`, with
`summary-confirmed.json`, `admission-audit.json`, `full-state-restore.json`,
`process-audit.json` and `cutoff1084-audit.json`. This local audit alone does not
certify publication or fresh remote readback of checkpoint 1585.

## Observed issue and explicit change

A read-only cross-tabulation of the 772 admitted games found 47712 movement
attempts. Of these, 41416 targeted coordinates absent from that decision's
current delivered known tiles. Every such attempt returned `no_route_known`
without changing the endpoint. The remaining 6296 targeted known tiles: 4642
reached the target, 1406 reached a movement limit, 105 performed an interaction,
136 returned `no_route_known`, and seven reported `hero_unavailable`.
This describes the recorded cohort; it does not prove the proposed intervention
will improve winning or learning efficiency.

`visible-interaction-budget-revealed-v1` restricts movement targets to tile
coordinates included in the selected hero's current delivered observation, on
that hero's map level. The historical navigation, management and no-effect
movement budgets remain in force. No hidden map state, blocked flag, native
route oracle, opponent strategy or future observation determines the new mask.
Known blocked or unreachable tiles are not removed merely for being blocked or
unreachable. Policy-controlled dialogs remain policy-controlled; mandatory
responses retain their existing support. The historical
`visible-interaction-budget-v1` remains supported for its original captures and
launchers.

The intervention changes the policy's available movement choices. It is
explicitly versioned in collection, independent reconstruction, PPO replay and
continuation metadata. It is not claimed to preserve the old action
distribution. The original map, model architecture, PPO settings, opponent,
four-game update, terminal-only +1/-1/0 rewards and all replay/integrity checks
remain unchanged. There are no demonstrations or reward shaping.

## Prospective validation protocol

The private protocol was recorded at `2026-10-30T07:21:57.557046+00:00` in
`Heroes-artifacts/run-investigations/2026-10-30-revealed-movement1585/protocol.json`,
SHA-256 `bd9d7547225f9c1e34922943701dfae385a70664973d9a3f0bb99c7cf8fc23fd`.
It binds the full final descriptor and 15 protected official/environment/native
inputs. The planned execution directory is
`run-investigations/pipeline-collection/revealed1585-attempt1/`. The planned full
suite evidence is `run-investigations/2026-10-30-revealed-movement1585/` under
`full-suite-attempt1.json` and `full-suite-attempt1.log`.

Run 16 bounded native diagnostic games from a copied, frozen checkpoint 1585:
eight with sequential verification and eight with pipelined verification. Each
arm has four games per seat. Both arms use the same newly disclosed revealed
guard; this is a validation of execution and support, not a policy-quality
comparison against the historical guard. Use fresh diagnostic reservations and
never reuse a consumed failed-attempt seed. Retain the complete raw evidence
privately. The whole native study has a 900-second deadline and verified owned
process cleanup.

Each completed game must pass actual native replay, exact policy reconstruction,
guard-history reconstruction and cleanup. Independently check every sampled
movement against the tile set delivered for that exact decision and require zero
unrevealed targets. Report actual movement counts and full choice counts rather
than substituting consumed PPO-target counts. Restore and compare the complete
parent model/Adam/RNG/progress/sampler/pool before and after; require unchanged
protected original inputs. Diagnostic capture receipts are revoked after audit:
zero samples admitted, zero optimizer updates, zero official reservations and
zero training. Do not weaken historical replay or native packet accounting to
obtain a passing result.

Check canonical engine/dependency agreement with `tools/prepare_vcmi.py --check`.
Run appropriate offline guard, rollout, reconstruction, acceptance, resource and
continuation regressions, followed by the full suite under an owned deadline.
Bind the actual accepted runtime and native learner bytes, source dependency
closure, launch/report/shutdown evidence and immutable Python environment.
Existing engine controls remain historical evidence for the unchanged engine;
do not relabel them as newly executed controls.

## Accepted performance provenance

Read-only `git fetch origin` confirmed `origin/main` and
`origin/codex/cpu-next-launch` at
`c289408c3bcfd57dcc5ba4f5f21b63f798737d65`. Claude's embedded-engine and
acceleration/phase-two branches, the four-collector integration and accepted Mac
native resource work are ancestors of main. The old prefix-performance branch
is patch-equivalent to an existing main commit; it is not a missing optimization.
Older unmerged contextual-engine alternatives and checkpoint-412 diagnostic
scaffolding are not pending accepted performance updates and are not merged here.

The accepted NumPy environment adds 2.3.5 while preserving Python 3.13.5, Torch
2.9.1, all existing locked versions, ABI and Torch library bytes. Its receipt is
`ce11a8abb87b091231a09a42552313ef0c3b88db7b9dd6466dbcacf536373964`.
The launcher uses `python/next-launch-numpy235-v1/venv/bin/python` under the
durable artifact root. This environment was prepared without modifying the
environment used by the completed run.

The accepted collection pipeline can verify a completed sealed game while other
games in the same four-game batch finish. Resource plan v2 requires at least
nine CPUs for four collectors, four verifiers and the coordinator; smaller
budgets retain sequential phases. It does not change the batch or begin PPO
before all four games pass. The accepted native manifest remains
`native-learner-builds/mac-utilization-v1-attempt6/manifest.json`, SHA-256
`919951d970eefa356ed19c901ddc3441e108035da186957b5cbe107b00ace623`.

The later C++ graph-compaction candidate was bitwise exact but 11.27% slower
across both production-thread comparison orders. It was rejected; its native
headers and PPO changes are excluded. The accepted native implementation remains
the previous build. The [CPU report](cpu-next-launch-2026-10-29.md) records the
contention, trajectory mismatch and timing limitations. Neither that work nor
this intervention establishes repeatable full-training speedup or CPU saturation.

## Continuation and validation status

`continue-1585.sh` uses the exact native manifest above, the prepared NumPy
environment, a default ten-hour cap and the explicit revealed guard. The
separate committed goal `configs/checkpoint1585-remaining1176.json` has SHA-256
`097e70bdacac96b456406577c1b6ba4cb30f6d1c4b1468f208b05264e8a7d917`.
Its completed-run schema binds the original final session, report and shutdown,
full checkpoint descriptor, progress and consumed sequence 1225. Subsequent
manual invocations follow verified descendants and subtract only complete
four-game admissions. They never select checkpoint 1585 over a later verified
descendant, apply an update twice or replay consumed reservations. The existing
772-game and 1000-game launcher semantics remain unchanged.

The focused continuation and mock-launcher selection passed 34 tests in 4.661
seconds, including unchanged old-goal behavior, exact completion, failure and
tamper rejection, consumed-seed protection, explicit guard forwarding and elapsed
runtime reporting. Shell syntax and whitespace checks passed. The new pure goal
also passed against actual accepted-main ancestry: 1176 remaining and sequence
1226 next. That is not the final accepted-runtime `--check`.

Two final-state audit report writers failed on mixed null/string outcome keys;
their original logs and partial outputs were retained. The corrected audit
reports the valid cutoff separately from natural losses. A first direct goal
probe supplied an evidence-only mapping instead of the continuation record and
failed before validation. A subsequent shared-source probe correctly refused
the unfinished new source/runtime closure. The successful pure-goal probe used
the independently audited accepted-main continuation; no official state was
changed by these probes.

Native acceptance and the real main-checkout check have passed. The manual
commands are:

```sh
cd ~/Projects/Heroes
./continue-1585.sh --check --hours 10
./continue-1585.sh --hours 10
```

The last command is for the user to launch. At this preregistration boundary it
is **not yet certified ready**. No new learning or playing-strength result can
be inferred from the proposed small frozen-policy validation cohort.

## Validation attempts retained before acceptance

The first full offline suite on `cc49ec0` ran 2045 tests in 328.834 seconds:
2012 passed and 33 platform or optional cases were skipped. Both native-test
manifest variables selected the unchanged accepted native learner. A subsequent
compiled PPO regression used 24 synthetic revealed-target movement samples in
one complete update with two minibatches. Python and actual C++ results matched
exactly for weights, gradients, populated Adam, minibatch RNG, global Torch and
Python RNG and the non-timing report. Its final 11-test focused selection passed.
An initial test-only attempt incorrectly left the synthetic PPO fixture at two
epochs while asserting one epoch's optimizer-step count; it stopped in the
reference branch before native execution. The corrected test explicitly uses
one epoch, matching the unchanged official setting. Both attempts are retained.

Native control attempt 1 used frozen `cc49ec0` and separately reserved seeds.
All 16 garrison controls passed with exact replay/history checks and owned
cleanup. The legacy group passed four controls, then its artifact control failed
at seed 1930100256: `Regression fixture must start with one backpack artifact`.
The remaining 23 controls did not start. The diagnostic fixture's singleton
assumption did not account for a native random starting bonus. The failed capture,
original artifact inventory and complete shutdown evidence are retained; no
sample or update entered official training. The entire reserved schedule is
consumed, including unstarted entries. This failed group is not runtime acceptance.
The one-item invalid-append coverage and exact ownership checks must be preserved
when repairing this diagnostic assumption; selecting a luckier seed is not a fix.

The retained failing panel showed authored Centaur Axe (ID 7) at slot 19 and
an additional Pendant of Free Will (ID 105) at slot 20; its neck slot was already
occupied. The additional item is consistent with the native random starting
bonus. The probe now temporarily transfers every extra backpack item to the
existing visible owned partner using only currently offered native actions.
Each transfer checks exact visible owner/slot/item layouts, including locked
reservations. It retains the singleton invalid-append assertion, both three-item
reorder directions and assembly/disassembly checks, then returns all extras and
requires the complete original layout. Missing items/offers or unrelated changes
still fail. Twenty focused fixture/probe/acceptance-identity tests passed.

Only the two authored diagnostic artifact maps now deliberately include ID 105
alongside ID 7, guaranteeing that fresh controls exercise this setup and return
on both seats. Random native bonuses remain possible and are separately accounted
for. Historical frozen maps/captures remain immutable. The official
hero-management-v2 map remains SHA-256
`ccfc8467847067a75091d68626e5b241839036ab729b5520ce4445a75a45b7b3`.

Native attempt 2 on frozen `71e14d4` used another fresh reserved schedule. Its
16 garrison cases passed. The legacy group again passed four cases before the
artifact control stopped, with 23 cases unstarted. This time parking, the
singleton check, both inverse reorders and assembly/disassembly ownership checks
all passed. Native disassembly legitimately placed all six components in backpack
slots 19–24 and the Centaur Axe at slot 25, while the Pendant remained on the
partner at slot 19. The new diagnostic restoration helper had incorrectly assumed
the actor was already back to the singleton layout. Its append assertion stopped
before issuing the return; no artifact was lost or invented. Owned cleanup and
protected-input hashes passed. This is a diagnostic setup/restoration defect,
not an engine failure or new policy-support failure.

The follow-up explicitly re-equips the observed components to their recorded
original slots using current native offers, verifies the exact layout and counts
after each move, and requires the recorded singleton layout before returning the
extra item. It preserves all prior gates. An artifacts-only, both-seat bounded
preflight will precede the next complete 44-control schedule. Failed reservations
and evidence remain consumed and immutable.

The final native preflight and full controls on `18aa3de` passed: 2 artifact
preflights, 16 garrison controls and 28 legacy controls, with owned cleanup and
all protected inputs unchanged. The first actual-policy study then completed
four native games but rejected the entire first batch at capture validation:
`Unknown planned policy support`. `builtin_gameplay._check_policy_support`
imported the historical VERSION alias and still allowed only that version.
This was a missed integration registration, not a timeout or engine failure.
No completion receipt, PPO sample or optimizer update was admitted. The eight
reserved paired starts remain consumed; the study is failed acceptance and its
report remains unchanged. A separate post-failure audit verifies exact original
and copied full checkpoint 1585 and all protected input hashes.

The capture gate now recognizes only None or an exact string in the registered
support versions and still requires exact plan/capture agreement. Real private-pipe
capture import tests cover both versions, historical None, all cross-version
mismatches and invalid types. Fifty-one focused tests passed. The initial test
fixture intentionally moved to an undisclosed tile and correctly failed under
the new guard; new capture tests now explicitly supply that visible tile, with
historical fixture defaults unchanged. An independent source audit found no
other stale production whitelist. The final native-policy study must use fresh
seeds and the current source; the full native controls must also be refreshed
because their complete dependency identity changed.

## Final accepted results and manual readiness

The corrected source `14cb5294930bba6a9034e9da33f3092c8e64186a` passed the
full offline suite: 2057 tests executed, 2024 passed and 33 skipped in 334.779
seconds. The accepted compiled learner was supplied to both native test routes.
The final acceptance/launcher selection passed 125 of 126 tests; one optional
case skipped. Counts overlap and are not independent experiments.

Fresh source-bound controls passed all 16 garrison and 28 legacy cases. Both
artifact seats retained the singleton invalid-append check, both three-item
reorder directions, exact equipment restoration and complete original inventory
return. Earlier successful cohorts and both failed fixture attempts remain in
the [native report](../../validation/revealed-movement-native-controls-2026-10-30.json).

Policy attempt 2 completed 16 games over eight newly reserved paired starts,
four per seat in each arm. Sequential and pipelined verification each
reconstructed 919 choices exactly; total 1838. Every one of 128 movement targets
was present in its decision's current revealed tiles; zero unrevealed targets
were selected. All 16 games ended in natural learner losses. All paired semantic
digests matched in this cohort, but concurrent offline tests and the fixed arm
order prevent a performance claim. These games demonstrate contract and
execution validity, not stronger strategy. All receipts were revoked, no PPO
data or optimizer update was admitted, and full parent/model/Adam/RNG/progress/
sampler/pool preservation, deadlines and owned cleanup passed.

Release `revealed-movement-1585-v1` is pinned to descriptor SHA-256
`877c1919964b5f9037b7f348015dcfa58e55c28f06ca58cceefb49c3000a35a6`.
Its historical `checkpoint_update=1392` binds the retained terminal-request
engine proof; `policy_checkpoint_update=1585` binds the new policy proof. It
does not select 1392 for training. No C++ engine or native learner code changed;
the actual accepted binaries passed the fresh controls and policy captures.
Canonical engine preparation checks passed in every frozen release.

The new immutable `machine-original-native-v5.json` selects that release and
the separate NumPy 2.3.5 environment. The real main checkout passed
`./continue-1585.sh --check --hours 10`: exact checkpoint 1585, 1176 remaining, sequence 1226,
16 available CPUs, four collectors/four verifiers, pipelining enabled and native
prefix budget 16. The NumPy/Torch bridge emitted no warning. Main preflight took
21.97 seconds and launched no gameplay/PPO. All 15 protected checkpoint/session/
ledger/environment inputs remain unchanged; no matching project workers remain.

The launcher stops when the verified 1176-game goal finishes or its ten-hour
window closes, prints total runtime and never automatically restarts. Full
CPU saturation, improved winning and new private-remote readback of 1585 are
not claimed. The original map, architecture, PPO/Adam settings, opponent,
four-game batch and terminal-only rewards remain unchanged.
