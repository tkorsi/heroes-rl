# Cloud checkpoint 1335 publication recovery, 29 October 2026

## Scope and current acceptance boundary

The user authorized repairing the failed private publication, recovering the
latest verified cloud state and preparing the Mac for a manually launched
1,000-game continuation. This work runs no official games, replays no consumed
training seeds and applies no official optimizer update. Raw evidence, checkpoints and
operational receipts remain in the private training-state repository and durable
local artifact directory.

**Accepted manual readiness:** checkpoint 1335 is installed on the Mac, including
complete state and the consumed-seed ledger. The actual main-checkout
`./train-1000.sh --check` passed with 2,176 original-target games remaining,
1,000 new games requested, sequence 975 next and a ten-hour cap. It launched no
game or update. The [compact evidence](../../validation/cloud1335-recovery-2026-10-29.json)
separates the original failed campaign from this certified recovery.

## Confirmed failure and retained evidence

The cloud session used frozen source
`c6ebc8327e8dfe65dfa384a3fe77da709343a4dd`. Its immutable private closure is
commit `89550dba6c481653ffc38cd78997438e846947f2`, under the original
`checkpoint1229-cloud-a08f55ff0c7548f58ff9d50ef5af8ba7` handoff. It originally
admitted 105 complete updates and 420 games, advancing checkpoint 1229 to 1334.
Batch 974 completed another four games and one PPO update, producing checkpoint
1335, but the campaign did not admit that candidate.

The failing operation was the publication verifier's
`git show HEAD:cloud/.../batches/train-000974-seat0/raw.part00000`, which exceeded
its 120-second timeout. The private push had succeeded. The pending checkpoint,
attempt report, raw archive and original failed campaign report all survived in
the immutable closure. The archive part is 5,661,006 bytes; its size does not
establish why the transfer stalled.

This was a publication readback timeout after a complete PPO update. The stored
evidence does not identify an engine failure, an invalid policy action, an
adapter-contract failure or a replay rejection. The original latest admitted
pointer remained at 1334, and the pending 1335 state stayed separate. Shutdown
recorded worker return code 1, verified owned-process cleanup and completion
within the hard deadline. The failed campaign remains `stopped_failure`; the
recovery does not rewrite it as a successful run.

The old publisher made a fresh blobless clone, then read each artifact through
individual `git show` commands. A missing object could trigger an implicit
promisor fetch inside a command treated as a bounded local read. This mixed
network transfer with verification and allowed repeated per-object fetches.
The observed failure proves the 120-second blocked read; the precise underlying
network slowdown is unknown. No claim is made that repository history or an
oversized archive caused it.

## Publication repair

[The publication helper](../../tools/linux_publication.py) now enumerates the
exact committed tree, explicitly fetches required blob objects in bounded
batches, and streams local blob contents into verification files. Local
verification disables lazy fetching and all Git transport. Fetches have a
separate bounded transfer allowance, and every operation remains constrained by
the caller's original deadline. A timeout closes the helper's owned process
group, including transport children.

The recovery API requires an exact commit, publication prefix and manifest
SHA256. It checks the expected entry set, file sizes and hashes, caller-preserved
source bindings and complete policy/resume restoration. It cannot overwrite an
existing label, create a replacement update or silently trust a moving `HEAD`.
Publication still uses an expected-head compare-and-swap push. A late readback
failure is recovered by verifying the already published transaction; it is not
re-published as a second training attempt.

The actual failed batch passed fresh remote readback through the repaired helper
in **9.58 seconds**, including complete checkpoint restoration. Two explicit
blob-fetch batches were used and no lazy fetch occurred during local
verification. The separate private return-to-Mac certificates were then
published and read back in **8.75 seconds**. That certificate-only publication
contains no new checkpoint pair; its receipt does not claim a second checkpoint
roundtrip. The final Mac receipts, selected pointers, runtime acceptance and all
new diagnostic captures were also published and freshly read back in 19.64
seconds on private commit `5ae4c57e4de91cfd1f9d80478ebb65d996024835`.
That branch retains the original verified checkpoint transaction in its ancestry.

## Recovered state and accounting

The audit validated all 106 published policy/resume pairs for sequences 869
through 974 against their manifests, plans, reports and continuous parent
lineage. The pending batch's archive passed an exact inventory audit of all
134 regular files, totaling 212,133,096 uncompressed bytes. Its four recorded
game-data symlinks were checked without following or extracting their targets.
The checks reject corrupt or missing files, unsafe archive paths, unexpected
members and altered checkpoint bindings.

Checkpoint 1335 restores the model, Adam state, generator, Torch and Python RNG,
progress, sampler and opponent pool exactly. Its saved state contains:

| Field | Verified value |
| --- | ---: |
| Complete PPO updates | 1,335 |
| Cumulative model rollouts | 5,340 |
| Cumulative decisions | 680,677 |
| Adam steps | 23,014 |
| Admitted games toward the original 5,000-game target | 2,824 |
| Remaining original-target games | 2,176 |
| Next unused batch sequence | 975 |

The cumulative model rollout count includes history outside the original
5,000-game campaign and is not its admitted-game denominator. Separately
certifying the already completed batch 974 adds exactly four games and one
update to the previously admitted cloud total: 424 games and 106 updates after
checkpoint 1229. Sequence 974 stays consumed. No game is rerun and no Adam update
is applied again. A subsequent 1,000-game manual session would target 250 fresh
complete updates, subject to its wall-clock safety cap.

The map, architecture, PPO settings, opponent and terminal-only rewards
`+1/-1/0` remain unchanged. The existing
`visible-interaction-budget-v1` support contract remains explicit. This repair
does not alter the engine or accepted native learner runtime, add demonstrations,
shape rewards, expose hidden information or introduce strategic masks.

## Cloud closure and Mac ownership boundary

The authorized recovery-only attempt to access the cloud chat failed because
the app reported that its cloud build had been deleted and could not be resumed.
Therefore no fresh cloud process snapshot, outer-supervisor PID absence or cloud
lock-mode change is claimed.

The narrow terminal-training certificate instead binds the exact app error,
unchanged consumed claim and results refs, frozen supervisor source, session,
shutdown and campaign bytes. In that frozen source, publication of the closure
occurs only after the training controller returns and its owned cleanup is
verified. Subsequent code can perform publication verification and return, but
cannot launch another training iteration. The same handoff cannot be relaunched
through the supported entry point while its consumed claim remains present.
This proves the relevant training boundary without pretending to observe the
current outer supervisor or infer the cloud provider's billing state.

[The import verifier](../../tools/cloud_state_import.py) retains the complete
local checkpoint-1229 ancestry, original moving-pointer snapshots and all cloud
reservations. Its cloud-evidence audit is bounded to 180 seconds. A committed
import profile binds the receipt bytes and official marker location. Deleting
the marker cannot restore access to the old checkpoint-1229 continuation path.
Later Mac descendants must retain the import boundary and every consumed
reservation, including failed attempts.

[The explicit application helper](../../tools/apply_cloud1335_import.py) defaults
to check mode. Application requires both existing campaign locks, their pinned
inode/owner/content identities and the recorded ownership fence. It journals
the exact old and new bytes durably before installing the marker and three
moving pointers. The real continuation verifier must accept the installed state
before recorded Mac launch permissions are restored. Partial failure stays
fenced, with the journal preserved for manual review; there is no automatic
rollback, retry or training launch. Independent review corrected a missing
parent-directory fsync before official writes.

## Runtime refresh and final validation

The first real main preflight after import rejected the old native acceptance
record: its Python dependency identity predated the merged performance work.
Earlier checks had stopped at the ownership fence. The runtime binaries and
engine recipe were unchanged; this was an acceptance-record mismatch, not a
new engine failure. The strict gate was retained.

Fresh diagnostics on the actual runtime passed all **16 garrison and 28 legacy
controls**, with independent replay/history checks and owned cleanup. Eight
fresh frozen-1335 policy games then passed replay and exact reconstruction of
**1,504 choices**. All eight were natural losses; no training samples or updates
were admitted. All five parent files were exact copies, full model/Adam/RNG/
sampler/progress/pool state was restored unchanged, and all owned processes
closed. The unchanged accepted native learner also passed all **13 exact-state
comparisons** on an archived disposable PPO workload, with 16 peak frame workers.
This is compatibility evidence, not policy improvement or a campaign speed claim.

The new `terminal-replay-cloud1335-v1` acceptance and
`machine-original-native-v1.json` are separate immutable artifacts. The original
machine remains untouched. Only `created_utc`, `runtime_origin` and
`runtime_release` differ; actual runtime, maps, learning settings, data, roots,
parent and seed namespace remain identical. Cloud ancestry re-audits the
receipt-bound historical machine and independently validates the new acceptance.
The diagnostic binds and rechecks all three moving pointers across worker lock
transfer, so an intervening launch cannot silently change the selected parent.

The full suite passed **1,861 of 1,908 tests**, with 47 platform/optional skips,
on `21c203f` before the final runtime-refresh extensions. On final implementation
`c42b5bb`, **289 of 290 focused tests passed**, with one platform skip. Canonical
engine preparation matched exactly in both checkouts. The actual native library
loaded under the updated source and binary gates. The real main preflight passed
in **6.81 seconds**, restored checkpoint 1335 and reported zero gameplay/updates.
The application journal proves installed continuation verification preceded
restoring both existing lock modes to `0644`. Historical failed evidence remains
unchanged, and deleting the import marker cannot roll the campaign back to 1229.

## Failed attempts and limits

Retained failures include: unavailable `pytest`; a stale apply-helper source pin;
reused historical PID numbers; a preview blocked by the owned native diagnostic
lock; a native comparison and intermediate full-suite run rejected during source
edits; the old runtime-acceptance mismatch; candidate preparation refused during
a final review edit; a read-only post-audit JSON-key error; and a mistyped focused
test module name. Each was corrected and rechecked. Failed preparation did not
launch games; no failed policy game was retried. All raw evidence stays private.

Process-generation checks compare UTC birth times with immutable historical
observations. A reused group number additionally requires a newer live group
leader and newer members. Missing, same-second or ambiguous identities and any
matching run path still fail closed. No unrelated process was signalled.

Stored Linux admission and replay evidence was audited, not recomputed as new
Mac gameplay. Complete restored state is exact; identical future trajectories
across architectures are not claimed. The successful remote readback is one
measurement, not a network reliability guarantee. The deleted cloud environment
prevents a fresh outer-supervisor inventory, as explained above.

Run manually from the synchronized main checkout:

```sh
cd ~/Projects/Heroes
./train-1000.sh --hours 10
```

The target is 1,000 additional admitted games (250 complete updates), with a
ten-hour safety cap. Evaluation games are excluded. The script prints total
running time; an early safe stop does not mean the 1,000-game target was reached.
No official training or automatic restart was performed during this recovery.
