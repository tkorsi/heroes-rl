# Four-process manual continuation integration — 28 October 2026

## Scope and hypothesis

The user requested merging the latest Codex performance work and Claude's work.
Codex branch `codex/parallel-eight` at `16e60daf5e3dea4ef83b1f8f9ee7cf3f069d7bfe`
is integrated with adaptations for the current local continuation contract.
Four independent collectors may reduce collection wall time without changing
the existing four-game PPO batch. Timing different fresh-seed cohorts cannot
establish an identical-workload speedup or a complete future campaign duration.

Claude's original5000 `573f3d1`, phase2 `74afd86`, acceleration `4d7688b`, and
embedded-base `b70442d` are already ancestors of main through selective merges.
`8b81931` retained main's engine recipe; `3e2a268` incorporated the latest
original5000 Python work. The only additional historical Claude ref contains a
Mac evidence file subsequently duplicated and deliberately removed. There is no
newer Claude implementation to import. The experimental embedded native backend
was excluded deliberately; final reviewed native code was validated on Linux,
while its older Mac evidence predates those fixes. It is not accepted here.

## Integration review findings

The old collector assumed a benchmark-only checkpoint metadata field and did
not activate process-local policy support in child interpreters. It also assumed
eight slots, while real manual plans require four. These assumptions must be
corrected before enabling concurrent manual collection. Child failure evidence
must retain finalization errors, and a failed overall batch must revoke every
receipt and report that revocation. No integrity gate is relaxed.

The new execution choice is `collection_workers`, with values 1 or 4. Historical
plans without the field retain serial execution. It is separate from the PPO
configuration. Eight-game diagnostic plans retain an explicit separate contract;
the manual runner does not expose eight-game batches. The source transition is
pinned to reviewed bytes and historical checkpoint metadata remains unchanged.

## Latest official state before diagnostics

The one-hour manual run `run-20261028T173849Z-7ee63337` closed at 18:28:30 UTC
with `stopped_budget / compute_budget_reserve`, exit 0, no supervisor error,
verified owned cleanup, and a hard deadline respected. It advanced 992 to 1,095
with 103 complete PPO updates and 412 admitted losses. Its 24 evaluation games
were losses. Full restoration reports 4,380 cumulative rollouts, 597,916
decisions and 20,134 optimizer steps. The original target has 1,864 admitted
training games, 3,136 remaining, and sequence 735 next. Process inspection found
no remaining processes owned by that run.

## Prospective validation

Use an isolated copy of checkpoint 1,095 and its complete parent descriptor,
current accepted terminal-replay runtime and the original map. Reserve fresh
seeds in a private diagnostic ledger, outside the official sequence range.
Exercise serial collection and four workers, both seats, bounded deadlines,
full native replay, policy reconstruction, failure controls and cleanup. A
separate diagnostic PPO update may exercise complete four-receipt admission and
exact restoration; it must never be published as an official descendant.
Snapshots must prove official pointers and parent checkpoint bytes unchanged.

Engine source is unchanged. The generated dependency must match the canonical
recipe, and native validation must identify the accepted runtime binaries.
No official training, cloud worker, Teacher or Claude launch is part of this work.

The following sections record the executed validation, including failed attempts.

## Offline validation before native launch

The focused integration suite ran 130 tests: 129 passed and one skipped. The
complete suite ran 1,680 tests: 1,633 passed and 47 skipped in 222.774 seconds. The native diagnostic
harness has five unit controls and independent review. The generated VCMI
checkout exactly matches the unchanged canonical recipe.

Interim development checks failed on stale source pins while review edits were
still in progress; the final reviewed bytes were pinned and the checks rerun.
Collector test fixtures needed current schemas and episode directories. Harness
controls exposed an unavailable optional NumPy conversion and a resolved-path
fixture difference; hashing now uses tensor bytes without NumPy. A potential
receipt-object-ID reuse in the four-arm harness was fixed by retaining consumed
receipts for the study lifetime, without clearing admission history. An initial
pytest invocation could not run because pytest is not installed; all reported
suite results use the project's unittest runner. None of these attempts ran
native games or modified official training state.

## Initial native preflight refusal and acceptance refresh

Attempt `run-20261028T184411Z-aed69e44` on merge candidate `497ce29` failed
before reserving seeds or launching a game. The existing runtime acceptance
correctly rejected changed `builtin_gameplay.py`, `builtin_training.py` and the
new parallel collector. Generated engine agreement passed; binary hashes and
engine source were unchanged. Cleanup passed and official files were unchanged.
This is a source/acceptance mismatch, not an engine failure or failed episode.

The strict acceptance checks remain unchanged. Refresh all 16 native garrison
controls and 28 legacy controls with fresh seed ranges, then eight frozen-1095
evaluation games through the existing candidate-runtime protocol. Reuse the
immutable binary freeze and loader evidence, and create new acceptance and
machine descriptors rather than overwrite historical bytes. This is a separate
preregistered validation phase, not an extension of the failed study's deadline.
The parallel harness now accepts an explicitly bound `--machine` descriptor; its
seven offline controls passed. The original machine file and all official
checkpoint pointers remain unchanged.

The refresh passed on `884e7c3`: all 16 garrison and 28 legacy controls passed
exact replay and cleanup. Eight fresh frozen-1095 policy games then passed with
530 exactly reconstructed choices, eight natural losses, zero PPO updates and
exact full-state preservation/restoration. The new immutable release descriptor
is `terminal-replay-parallel-v1`, SHA-256
`f1b730d6d2d05771b694db738e337e0e6abe0080da0534d62dd09a704c6c30c7`.
It refreshes Python source acceptance on the identical native binary/recipe; it
does not claim a new engine build. The old acceptance descriptor remains exact.

The new default original-map machine filename is
`machine-original-parallel-v1.json`. Its historical predecessor is retained
unchanged for completed sessions. The acceptance/source-transition and launcher
suite ran 128 tests after these changes: 127 passed and one skipped.

## Complete collection and PPO diagnostic

Candidate `06c55bf` passed all 16 fresh games in four independent arms on both
seats. All 1,073 choices passed replay and exact policy reconstruction, and all
16 games were natural losses. Each arm performed one isolated four-game PPO
update, verified its checkpoint round trip, and restored the original complete
1095 state. The four updates remain private diagnostic evidence and are not
official descendants. Collection preserved model, Adam, Python/Torch/minibatch
RNG, sampler, progress and pool exactly. Cleanup and the 15-minute deadline
passed; every snapshotted official file remained unchanged.

| Collection | Seat | Choices | Collection + verification | PPO + checkpoint round trip | Adam steps |
| --- | --- | ---: | ---: | ---: | ---: |
| Serial | Red | 212 | 8.69 s | 9.80 s | 7 |
| Four workers | Red | 549 | 13.83 s | 28.82 s | 18 |
| Serial | Blue | 208 | 8.64 s | 9.97 s | 8 |
| Four workers | Blue | 104 | 7.15 s | 5.30 s | 5 |

Fresh seeds differ across arms and the trajectories differ substantially. These
figures do not establish a paired workload speedup. The generic runner and
existing continuation wrapper retain one collector by default. Four collectors
are explicitly selected with `--collection-workers 4`. Parallel evaluation
dispatch has offline coverage only; the eight fresh native evaluation games
above used the existing serial candidate protocol. No same-seed equivalence or
long-run stability claim is made.

The actual continuation wrapper passed offline check mode from this worktree
with four collectors and a one-hour cap: checkpoint 1,095 restored exactly,
3,136 games remained and sequence 735 was next. Main-checkout synchronization
and its independent check are recorded in the compact results file.

[Compact results](../../validation/parallel-continuation-results-2026-10-28.json)
contain source identities, state component hashes, test results and limitations.
All raw captures, checkpoints, detailed reports and failed-attempt evidence
remain under the private durable artifact root.
