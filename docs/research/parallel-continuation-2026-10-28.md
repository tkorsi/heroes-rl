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

Validation results will be recorded after execution, including failed attempts.

## Offline validation before native launch

The focused integration suite passed 130 tests (one skip). The complete test
suite passed 1,680 tests (47 skips) in 222.774 seconds. The native diagnostic
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
