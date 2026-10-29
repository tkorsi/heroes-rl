# Manual session interrupted by Mac sleep — 29 October 2026

## Finding and scope

The user requested investigation of the latest failed manual launch. Run
`run-20261028T192106Z-44be2ecc` used frozen source
`f5a7f255a5ba52345c0034484549c67e6e70cf5d`, the accepted
`terminal-replay-parallel-v1` runtime and four collectors. Its one-hour wall
deadline expired while the Mac was in lid-close sleep. At wake the deadline
guardian sent SIGKILL, and the supervisor recorded return code -9 and unverified
closure within the expired budget. The saved checkpoint remains usable evidence;
the session did not finish cleanly.

This is a wall-deadline interruption caused by host sleep. There is no recorded
engine failure, unsupported action/query, replay rejection or checkpoint
publication failure at the stopping point. No engine, adapter or performance
code was changed for this investigation. No gameplay, PPO update, process signal,
seed reservation, cloud work or training restart was performed.

## Timeline

All events below occurred on 28 October 2026. Kyiv was UTC+02:00.

| Event | UTC | Kyiv |
| --- | --- | --- |
| Manual session started | 19:21:06 | 21:21:06 |
| Checkpoint 1,229 durably preserved | 20:10:11 | 22:10:11 |
| All four games of periodic Blue evaluation reached natural terminals | 20:10:27 | 22:10:27 |
| OS recorded `Clamshell Sleep` | 20:10:29 | 22:10:29 |
| Declared compute deadline | 20:16:06 | 22:16:06 |
| Declared hard deadline | 20:21:06 | 22:21:06 |
| OS recorded DarkWake; guardians recorded deadline expiry | 20:26:15 | 22:26:15 |

The OS reports 946 seconds of clamshell sleep in this interval and an earlier
18-second clamshell sleep at 19:36:08 UTC. Campaign wall elapsed time exceeds
monotonic elapsed time by 960.849 seconds. At the guardian's expiry event, its
wall deadline was 314.703 seconds past while its monotonic deadline still had
646.145 seconds remaining. The session's hard wall cap was exceeded by
309.703 seconds. OS durations are rounded and are not an exact measurement of
monotonic suspension.

`experiments/headless/process_deadline.py::_remaining` intentionally uses the
earlier remaining wall or monotonic allowance so suspend cannot extend the
authorized window. The campaign guardian is armed five seconds before the hard
deadline. Processes cannot execute their cleanup while the host is suspended.
`tools/run_local_training.py::supervise` starts `caffeinate -i -w <worker>`;
the corresponding idle-sleep assertion was active but did not prevent the
recorded clamshell sleep. The manual instructions already require an open lid.

## Exact interrupted stage

The pending batch was `periodic-001229-seat1`, a Blue evaluation cohort.
Its outer report was stale at `running` with attempts marked `not_started`.
The deeper worker report and captured files establish that all four games
finished: 70, 68, 40 and 47 sampled choices. The first three passed native replay
and exact policy reconstruction. The fourth was `pending_verification`, with
no completion receipt. The whole incomplete cohort remains excluded from the
accepted evaluation total; it supplied zero training samples or optimizer steps.

The worker's unpublished temporary report records a shared-budget timeout. Its
elapsed monotonic time was 7.193 seconds versus 951.998 seconds of wall time.
Its `within_shared_budget=true` compares against the relative phase ceiling;
it does not establish compliance with the earlier absolute session deadline.
The absolute deadline and admission gates rejected completion. There is no
evidence of 16 minutes of active verification work or a validation mismatch.

SIGKILL prevented the campaign worker's finalization from writing a final
restore/cleanup receipt or a failed-batch row. Consequently `failed_batches=[]`
and the stale `running` row do not mean the session succeeded. The supervisor's
`stopped_failure`, worker -9, `owned_cleanup_verified=false` and
`within_hard_deadline=false` remain the historical outcome. The unset final
restore and false initial-state certification are missing shutdown proofs,
not evidence that those checkpoint bytes were corrupted.

## Preserved training state

The session admitted 134 complete four-game batches, sequences 735 through 868,
advancing checkpoint 1,095 to **1,229**. All **536 admitted training games** were
natural losses. The accepted separate evaluation total was 20 losses. No final
evaluation completed. The original 5,000-game target has **2,400 admitted games
and 2,600 remaining**, with **869** the next unused training sequence.

The latest admitted batch is `train-000868-seat0`. All 134 admitted reports and
their preservation copies were checked; every sequence was reserved once.
Canonical checkpoint 1,229 and its durable backup were independently restored
with the exact preceding-policy dependency. Full model, Adam, Python/Torch/
minibatch RNG, sampler, progress and pool state matched. Progress is 1,229
updates, 4,916 cumulative rollouts, 627,977 decisions and 21,238 optimizer steps.
The full-state fingerprint is
`c5882092150d55b4c07df368534f63b0c742bdd74982f1228210007f67c5ad7e`.
The initial 1,095 inputs and official pointers, ledger and historical reports
were unchanged by the audit. No failed attempt was replayed or promoted.

A fresh process audit at 00:31:04 UTC on 29 October compared 978 processes with
2,803 recorded PIDs and 1,401 process groups from 1,401 deadline evidence files,
plus native/learner command patterns. It found no matches. This is a later
observation, not retroactive proof of cleanup before the original deadline.
Historical PIDs were not used as authority to signal any current process.

## Continuation status and limitations

The actual main-checkout command
`./continue-original-4720.sh check --hours 1 --collection-workers 4` returned
exit 2: `Continuation requires verified closed processes and an exact final checkpoint`.
This is the expected refusal in `original_continuation::closed`: the failed
session lacks its original final restore/cleanup proof and missed its deadline.
The audit receipts explicitly carry `authorizes_continuation:false`.

Checkpoint 1,229 should be retained. A separate reviewed interrupted-session
recovery path would need to bind the immutable failure evidence, exact complete
state restoration, admitted lineage, consumed seed ledger and current process
closure. This investigation does not implement that path, change historical
flags, grant a fresh deadline to the consumed session or bypass the launcher.
For a future separately prepared manual run, keep the Mac awake with its lid
open; an idle-sleep assertion did not prevent this recorded interruption.

The audit restores saved state and inspects existing evidence; it is not a new
native regression run or a proof of long-run parallel equivalence/performance.
Raw captures and full checkpoints remain private under
`Heroes-artifacts/local-script-training/original/run-20261028T192106Z-44be2ecc/`.
The new process/power evidence and audit receipts are under
`Heroes-artifacts/run-investigations/2026-10-29-1229-sleep/`.
Remote publication of this latest checkpoint/raw run was not verified here.
An initial attempt to redirect check-mode output failed before invoking the
script because the audit directory had not yet been created; the subsequent
actual command above produced the expected refusal. No diagnostic games ran.

[Compact source-bound results](../../validation/local-sleep-interruption-2026-10-29.json)
record hashes and audit scope. JSON/index parsing, report links, receipt hashes
and unchanged official-file hashes passed; `python3 tools/prepare_vcmi.py --check`
confirmed the unchanged generated engine matches the canonical recipe. This
report changes no scientific settings or runtime acceptance and supplies no
training data.
