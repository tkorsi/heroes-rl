# Checkpoint 1392 terminal replay and reporting repair, 29 October 2026

## Outcome and scope

Both observed defects are repaired. A clean rebuilt runtime passed the captured
failure replay, 44 fresh native controls and eight frozen-checkpoint-1392 policy
games. The actual main-checkout continuation check initially refused a historical
cloud-import machine guard. The narrowly registered runtime transition passed
44 focused tests and independent review; the corrected final selection passed
182 tests. Main commit `9eb67debd056ea8426eb70331b153177cb46251d` then passed the
exact ten-hour continuation check in 10.42 seconds, with no gameplay or PPO.

No official training, seed reservation or optimizer update was launched by this
repair. The original map, architecture, PPO settings, Nullkiller Easy/BattleAI,
terminal-only rewards +1/-1/0 and existing `visible-interaction-budget-v1` are
unchanged. The [compact validation record](../../validation/checkpoint1392-repair-2026-10-29.json)
separates runtime acceptance, checkpoint preservation and manual launch readiness.

## Confirmed failure and preserved state

The user-started `run-20261029T195027Z-64238b2b` advanced checkpoint 1335 to 1392
through 57 complete updates and 228 admitted natural losses. Eight evaluation
games are excluded. It stopped at 20:24:52 UTC, after 34 minutes 26 seconds,
before the requested one-hour deadline. Batch 1032 completed four native games
but admitted no samples and performed no PPO update. Its seeds remain consumed;
1033 is the next unused sequence.

Episode 3 contains 423 authoritative inputs and 424 client requests. Final
input 423 was `MakeAction`, producing a natural terminal outcome. The native
opponent emitted `GarrisonHeroSwap` (type 189), request 424, while consuming
packet 2006 within the final 1988–2012 packet interval. This extra request has
no server application. Python rejected its unsupported type before native replay;
the C++ accounting excluded the same type. This is a replay-verifier coverage
failure, not evidence of an invalid learned action, engine crash, timeout or
publication failure. The exact AI caller and swap direction are not recorded.
The operation changes visiting/garrison hero placement in a town; it is separate
from artifact transfer and the earlier owned-mine dialog.

The collector then revoked all batch receipts while retaining `passed` status
for earlier verified histories. The strict supervisor rejected that contradiction
and replaced the visible cause with “Passed attempt lacks durable
post-verification completion.” Episode 4 independently passed native replay;
its propagated batch-aborted error was not another replay failure.

Checkpoint 1392 and its durable backup restore exactly: model, Adam, all RNG,
progress, sampler and opponent pool. Progress is 5,568 cumulative rollouts,
729,985 decisions and 24,626 Adam steps. The original 5,000-game target has
3,052 admitted games and 1,948 remaining. The interrupted 1,000-game goal has
772 remaining. The final 20:54:54 UTC closure audit checked 841 deadline files,
1,683 historical PIDs and 841 groups against 985 live processes: no owned process
matched, both locks were available and no signals were sent. All five official
pointer, ledger, import and original-machine bindings remained unchanged.

## Repairs and validation

Python and canonical C++ replay accounting now support `GarrisonHeroSwap` as the
sixth reviewed terminal-unapplied request type. Exact request bytes and order,
opposite-seat controller identity, contiguous request ID, final packet receipt,
natural terminal state and terminal barrier remain mandatory. Other unreviewed
request types still fail closed. No valid policy action is hidden and no
policy-controlled dialog is automatically closed.

Batch-rejected histories now receive `rejected` status atomically with receipt
revocation. Individual verification evidence and the original error remain
available. The supervisor's strict completion checks are unchanged. Metrics
classify rejected attempts separately and do not score their outcomes. The new
regression exercises the actual training worker, collection failure path and
report reader, verifying zero PPO calls, no serial retry and revoked capabilities.

All nine native targets were rebuilt from an empty directory with compiler cache
disabled. The build took 61.47 seconds. The separate frozen runtime is
`local-v8-terminal-replay-repair2`, release `terminal-replay-1392-v1`, built from
source `ba45482343feb39c6b9539ebc81c717fc4c47ef2`. Its preflight verified the actual
client/library load and found no donor-build dependency. Canonical engine checks
passed in the implementation worktree and synchronized main checkout.

| Validation | Observed result |
| --- | --- |
| Native request-accounting controls | 715 checks passed, including both seats and negative mutations; terminal cancellation controls passed |
| Supplemental replay of copied failed episode | 423 authoritative inputs, 424 client requests, 4,122 transactions and 2,012 packets verified; no divergence |
| Fresh garrison controls | 16/16 passed with replay, history checks and owned cleanup |
| Fresh legacy capability controls | 28/28 passed with replay, history checks and owned cleanup |
| Frozen checkpoint 1392 policy | 8/8 games passed verification; 1,136 choices reconstructed exactly; eight natural losses; four games per seat |
| Full offline suite | 1,936 tests: 1,864 passed and 72 platform/optional skips |
| Additional native learner suite | 41/41 passed with the existing accepted native learner binary |
| Final focused suite before the cloud-machine gate correction | 178/178 passed |
| Narrow cloud-machine gate correction | 44/44 passed; independent review completed |
| Final combined continuation/acceptance tests | 182/182 passed |
| Actual main-checkout continuation check | Passed in 10.42 seconds at `9eb67de`; exact checkpoint 1392, 772-game goal, next sequence 1033, no gameplay or PPO |

The supplemental replay changed only the verifier and its canonical accounting
source; captured client, libvcmi, AI and adapter identities stayed fixed. It is
explicitly diagnostic: `diagnostic_verified=true`, `verified=false`,
`optimization_eligible=false` and original cohort admission remains false. It did
not rerun the game seed or turn the failed batch into training experience.

Fresh diagnostic seed ranges were preregistered and checked against existing
plans and ledgers. Every diagnostic had bounded deadlines and verified owned
cleanup. Frozen-policy parameters and full parent state remained exact; all
diagnostics supplied zero training samples and optimizer steps. The additional
native learner tests used synthetic/disposable fixtures, not official state.
Test totals above overlap and must not be summed into a unique-test count.

## Failed preparation attempts and limitations

A launcher fixture initially disagreed on macOS `/var` versus `/private/var`
path normalization and was corrected. One focused test invocation named the
nonexistent `tests.test_cloud_import_application` module; the corrected selection
passed 44 tests. A later combined invocation named nonexistent
`tests.test_continue_772_games`: 94 tests passed and one import failed, with the
output retained. The corrected invocation passed 182 tests in 6.674 seconds.
These are test harness or test-selection failures; no native diagnostic game failed or was retried.

The first actual main command, `./continue-772.sh --check --hours 10`, exited 2
in 2.14 seconds. The cloud-import guard permitted only acceptance metadata changes
for its original historical machine, so it correctly refused the newly rebuilt
runtime. The correction registers this specific runtime transition by immutable
historical-machine and accepted-descriptor hashes, preserves all other machine
fields, and still requires the complete current native/policy acceptance gate.
It does not rewrite the historical machine, checkpoint lineage or imported claim.
The actual main retry passed with exact restoration, 772 goal games remaining,
1,948 original games remaining and next sequence 1033. It selected four collectors,
four verification workers and 16 native prefix/frame threads within the observed
16-CPU budget. The four-game PPO batch and phase dependencies remain unchanged;
this resource selection is not a claim of continuous 16-core saturation.

An inherited acceptance-description field still named the previous
`ExchangeArtifacts` repair. The original descriptor is retained as evidence; a
separately named `accepted-runtime-v2.json` descriptor preserves a supersedes
binding and corrects only the annotation, with explicit correction provenance.
It passed the full runtime acceptance gate. A new immutable machine-v3 selection
changes only creation time and the descriptor binding relative to machine-v2;
all native and policy proofs remain identical.

The eight fresh policy games did not reproduce a type-189 terminal request;
they contained five type-182 terminal tails. The new type-189 case is covered by
the exact failed capture and the native accounting controls. The policy games
are a small acceptance cohort and demonstrate no strategic improvement. They do
not establish long-run stability, broad map generalization or full CPU saturation. Allied-garrison behavior has no new native coverage in
this run. The exact post-terminal AI caller remains unknown.

## Private preservation and manual continuation

Private commit `a489c0e4ae133616bf69569db740a57c0e4c63cb` on
`codex/checkpoint1392-repair` contains the verified checkpoint package and failed
batch evidence. Fresh remote readback verified all 24 files and restored the full
checkpoint in 17.81 seconds. The failed archive covers 130 regular files
(150,677,409 uncompressed bytes) and four undereferenced Data symlinks; a separate
archive contains 1,378 run metadata files. Successful official-game raw histories
and intermediate checkpoint tensors remain intact locally and are explicitly
excluded from this package. The separate final diagnostic transaction, commit
`375b61c2f22f99c65552aabd654e3ca53e82948f`, passed fresh remote readback in 21.54
seconds on the same branch. It covers 1,694 regular files (407,726,793 bytes),
52 undereferenced symlinks, all new diagnostic captures, accepted runtime
descriptors, source reviews, failed test-selection logs and final main readiness
proofs. Its 15,588,658-byte archive excludes runtime binaries and game data.
This diagnostic-only transaction does not claim another checkpoint restoration;
the initial transaction already verified that independently. No copyrighted
game data or raw captures enter the public repository.

`./continue-772.sh --hours 10` is the prepared manual command. It verifies
checkpoint 1392 in the current official ancestry and the consumed batch-1032
reservation, then recalculates the remaining number of fully admitted games.
Repeated manual invocations finish the same 772-game goal rather than adding
another 772. Evaluation and failed games never count, the latest checkpoint is
never reset, and an already completed goal exits without work. The default cap
is ten hours, and total elapsed time prints after cleanup. The real main-checkout
`./continue-772.sh --check --hours 10` command passed without gameplay or PPO.
The user will launch official continuation manually.
