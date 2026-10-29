# Checkpoint 1229 recovery and cloud lifetime limits — 29 October 2026

## Outcome

A separate, audited manual continuation is prepared from checkpoint **1,229**.
The original target has **2,400 admitted games, 2,600 remaining**, and training
sequence **869** next. The user still launches training. No official gameplay,
PPO update, seed reservation, process signal or automatic restart occurred.

The prior failure remains a **Mac sleep / wall-deadline interruption** during
periodic Blue evaluation. It was not a newly observed engine, adapter, replay or
publication defect. The historical failed shutdown and excluded evaluation
remain unchanged. See the [original investigation](local-sleep-interruption-2026-10-29.md).

## Recovery contract

Implementation `752b2ce6fd27a0e172e10c823d781ed8addf1a7e` adds a narrow recovery
profile for this exact interruption and an immutable private receipt. Receipt
creation and launch checks hold the existing campaign exclusion locks without
truncating their files. They revalidate all 134 admitted batches and preservation
copies, complete model/Adam/RNG/progress/sampler/pool state, current process
closure, the actual accepted runtime binaries and immutable failed-attempt
inventories. The full-state fingerprint remains
`c5882092150d55b4c07df368534f63b0c742bdd74982f1228210007f67c5ad7e`.

This does not rewrite the expired session or extend its deadline. A new manual
session gets its own bounded window. Arbitrary failed runs remain refused.
No partial evaluation or training batch becomes experience. The original map,
architecture, PPO, Nullkiller opponent, terminal-only +1/-1/0 rewards and
`visible-interaction-budget-v1` remain unchanged. No native engine changes or
new native gameplay acceptance are claimed.

Later clean descendants inherit immutable recovery evidence. A clean stop
before its first update can retain checkpoint 1,229 without blocking the next
manual session. Current seed reservations must extend every stored ledger
snapshot exactly, including reservation timestamps; failed and preparing
training attempts remain consumed. A later checkpoint is never reset to 1,229.
Review found and reproduced a gap in the first inherited-mode implementation:
removing an unadmitted later reservation could permit its reuse. The final
implementation rejects that case and has regression coverage for failed555,
failed/preparing869, changed reservation rows and valid append-to870.

## Validation and failed attempts

The compact [validation record](../../validation/checkpoint1229-recovery-2026-10-29.json)
binds test logs and actual commands. The targeted final suite passed 110 tests.
The first full suite ran while a reviewed runner change was landing; it retained
old in-memory source pins and produced three failures and seven subtest errors,
all rejecting the changed runner hash. That attempt is preserved. The full
suite was then rerun against stable committed source: **1,710 tests, 1,663
passed and 47 platform/optional skips**, with no failures or errors. No integrity check was relaxed.

Real check-mode commands passed in the implementation checkout, a newly prepared
frozen release, and the main checkout. Main passed both one-hour and ten-hour
limits with four collectors, exact checkpoint restoration, 2,600 remaining and
869 next. The old invocation without an explicit recovery receipt still refused
the interrupted session. `tools/prepare_vcmi.py --check` passed in all three
checkouts. All 26 selected official input hashes, including pointers, ledger,
checkpoint inputs, historical reports and owner-lock contents, remained exact.
The recovery receipt separately binds 1,401 deadline files and 118 files from
the excluded evaluation. Current closure checks found no matching old processes.

## Private preservation

The private repository `tkorsi/heroes-rl-training-state`, branch
`codex/checkpoint1229-handoff`, preserves a 45-file, 5,486,028-byte restore-only
snapshot at `ae985b40afec212aca0f2164016033f0e895283b`. A fresh network download
matched every file and restored the same complete state. Commit
`9f812e5d2b8ae85de42ce631e1df38840ad54d36` separately adds the recovery receipt
and readback evidence; its six new files also passed independent download and
hash comparison. Raw captures and checkpoints remain private.

This compact snapshot is not the full approximately 8.7 GiB raw run or complete
ancestry archive. Historical paths remain intact. Full lineage availability,
Linux relocation and native runtime acceptance are not established by these
copy/restoration checks. The recovery receipt is scoped to the reviewed Mac
artifact root, not a portable cloud authorization.

## Cloud finding and minimal model usage

One explicitly authorized cloud chat and one authorized inspection follow-up
completed. Setup stopped before downloading or launching the harmless two-minute
probe: the required managed environment-status tool was unavailable and the
network-policy file was absent. No PID, heartbeat or completion evidence exists.
This is **setup blocked, lifetime unmeasured**, not an observed process-survival
failure. The [cloud operations note](../operations/checkpoint1229-cloud-continuation-2026-10-29.md)
records the task identities, skill requirement and official documentation.

The training script makes no model/API calls and owns deadlines, admission,
checkpointing and cleanup. Using an agent for preparation and a final/failure
inspection avoids constant model supervision. Official documentation does not
establish detached-process survival after a cloud turn or a guaranteed ten-hour
command lifetime. Reliable unattended execution needs a host with an independently
controlled job lifetime. No cloud training or recurring monitor was started.

## Manual command

Keep the Mac awake with its lid open. The existing idle-sleep assertion does not
prevent the observed lid-close sleep. Run either duration from the main checkout:

```sh
cd /Users/orsi/Projects/Heroes
./continue-original-1229.sh --hours 1
```

For the requested ten-hour window, use `./continue-original-1229.sh --hours 10`.
The wrapper defaults to one hour and four collectors, calculates the remaining
target from verified ancestry, and rechecks state before launch. Insert `check`
after the script name for a no-game preflight. These commands are prepared for
manual invocation; validation did not launch them in training mode.
