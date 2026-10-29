# Cloud checkpoint 1335 publication recovery, 29 October 2026

## Scope and current acceptance boundary

The user authorized repairing the failed private publication, recovering the
latest verified cloud state and preparing the Mac for a manually launched
1,000-game continuation. This work runs no official games, replays no consumed
training seeds and applies no optimizer update. Raw evidence, checkpoints and
operational receipts remain in the private training-state repository and durable
local artifact directory.

**Draft acceptance status:** checkpoint 1335 has passed the separate publication
recovery and complete-state checks described below. Application to the official
Mac marker and moving pointers, final full-suite/native checks and the real
main-checkout launch preflight remain pending. A recovered checkpoint is not yet
a claim that the Mac is ready to launch. The final acceptance entry must record
those results before removing this qualification.

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
roundtrip.

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

## Validation, failed attempts and limitations

The initial focused validation passed 83 tests. The application-helper controls
have since expanded from 16 to 18, so the corresponding final focused total is
expected to be 85; that aggregate rerun is still pending in this draft. Controls
cover corrupt/missing blobs and raw data, explicit post-push recovery, conflicting
labels, denied lazy fetching, transfer-child cleanup, changed source/receipt
bindings, cloud closure, seed and ancestry protection, concurrent ownership,
partial pointer writes and re-fencing after unlock failures.

An initial test invocation could not start because `pytest` was unavailable;
validation continued through the repository's `unittest` route. The first real
import preview failed on an unreviewed apply-helper source hash after the helper
changed. Its source pin was corrected, with the failed receipt retained. The
second preview rejected historical PID-number collisions with currently
unrelated Chrome and system-service processes. No process was signalled and no
official state was changed by either preview. Resolution and a successful
preview must be recorded before application is accepted.

Stored Linux admission and replay evidence was audited; Linux-recorded behavior
was not recomputed as new Mac gameplay. Complete restored state is exact, but
identical future trajectories across architectures are not claimed. The
9.58-second readback is one successful recovery measurement, not a network
reliability guarantee. No policy-quality or campaign-throughput improvement is
inferred from this repair.

Final full-suite results, generated-engine agreement, actual native-library
loading, official application receipt, main-checkout synchronization and
`./train-1000.sh --check` remain pending here. The intended user launch after
those gates pass is `./train-1000.sh --hours 10`; the script reports total running
time and stops at the session's admitted-game target or its safety boundary.
