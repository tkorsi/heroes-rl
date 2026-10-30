# Self-play storage and verification architecture

The user requested faster data storage and checking, plus visible action progress
inside a game. This change preserves the map, model, available actions, visible
interaction guard, reward semantics, PPO, budgets and checkpoint lineage. It does
not activate the separate tiny-model experiment or start any gameplay.

## Data ownership

Previously, `neural-rollout.json` embedded both complete player histories as
pretty-printed JSON. The same histories already existed in manifest-bound player
JSONL files. Verification loaded this large JSON tree, decoded it, then expanded
the complete typed object back to JSON for an exact equality check. The training
verifier also reconstructed both transport histories directly and then repeated
the identical reconstruction inside native replay verification.

The new `heroes-context-rollout-storage-v1` envelope separates storage from the
unchanged `heroes-context-rollout-v1/v2` trajectory contract:

- Player JSONL histories remain the authoritative saved learner inputs. Their
  relative paths, byte counts, file SHA-256 and canonical-history SHA-256 bind the
  envelope to the exact same-player declaration in `manifest.json`.
- The envelope explicitly retains every sampled action, full offered support,
  interaction guard, conditional/joint probability, value estimate, recurrent
  prefix, sample boundary, bootstrap, role and model identity. None is inferred
  from a different player's observations or omitted to reduce storage.
- Loading streams and strictly decodes history records one line at a time. All
  original typed history, rollout, paired finalization and sample validations
  still run. Records must roundtrip exactly. Unknown fields, duplicate JSON keys,
  nonfinite numbers, missing or extra frames, invalid references and symlinks
  fail closed. Paths must remain inside the copied episode.
- The envelope is published last: a complete temporary file is flushed and
  synced, then linked exclusively to its final name and the directory synced.
  Existing artifacts cannot be overwritten. A failed attempt cannot acquire an
  admission capability merely because a complete artifact exists.

Raw bridge, native packet/request, live-history and finalization evidence remain
separate because they verify different boundaries. Old captures are immutable;
`load_collected()` also accepts the legacy inline format with its full roundtrip
check. Reporting and the Swift exporter continue reading unchanged manifest-bound
histories. Legacy-only consumers must use this loader to read new envelopes;
the historical optional `experiments/perf/offline_profile.py` remains legacy-only
and is not part of the runtime or this validation.

## Verification and admission

The verifier independently reloads the artifact from disk, restarts each player's
private recurrent state and RNG, resamples every action, and compares every value
and conditional/joint probability exactly. Native replay still reconstructs both
transport histories and checks the actual engine's packets and request accounting.
The training coordinator uses that freshly returned history proof instead of
calling the same history reconstructor a second time. It still requires both
player proofs, checks persisted native reports and finalization, rehashes all
bound files, and admits only four fresh, complete, single-use training receipts.
No cached report, progress file or on-disk receipt can grant admission.

This is deliberately a storage-boundary change, not a new binary event protocol,
compression dependency, lossy summary, or asynchronous verification bypass.
Native collection and policy reconstruction remain required work. Future changes
to raw capture encoding need their own version and equivalence validation.

## Progress channel

Each active game publishes a small disposable snapshot at most once per second
plus phase transitions. It contains only both policy-choice counts and the
existing ordinary/reserved budgets. The terminal-owning supervisor renders it
even when worker logs are redirected; worker completion JSON remains unchanged.
Collection, finalization, storage and verification have separate labels.

The bar measures **choice allowance consumed**, not expected time to victory or
verification completion. In a training batch the labels identify the current
game slots, which are reused in the next batch. Reserved mandatory-dialog choices
are shown separately. Display writes have no fsync, cannot affect policy RNG,
and are never read by verification, checkpoint recovery or admission. Missing,
corrupt or unwritable display files do not bypass or block scientific checks.
No CPU profiling subsystem is restored.

## Validation protocol

Use a fresh private copy of the completed frozen-1753 doubled-choice game. Keep
the original capture, checkpoint, RNG, progress, pool and seed ledgers unchanged.
Compare the old and new serializers and loaders on exactly the same saved game,
including full decoded equality. Reconstruct both policies and replay the copied
native capture under an independent 900-second deadline, with owned cleanup.
This starts no game, consumes no new gameplay seeds, and admits no samples.

Timing is one measurement per operation on one archived long cutoff game; caches
are not flushed. It cannot establish full-game or training-campaign throughput.
The copied-game attempt passed in 273.45 seconds under its independent deadline;
owned cleanup passed with exit code zero. All 158 protected original files stayed
unchanged. The actual accepted `vcmireplay` verified 102,220 transactions and
39,894 client packets, with zero unchecked records. Both policies reproduced all
11,763 actions, 11,765 values and 30,487 conditional probabilities exactly across
89,446 history frames. Neither diagnostic rollout became optimizer eligible.

| Same archived game | Previous storage | New storage |
| --- | ---: | ---: |
| Rollout artifact | 2,482,037,019 bytes | 30,279,926 bytes |
| Complete episode, replacing only that artifact | 6,533,263,254 bytes | 4,081,506,161 bytes |
| Serialize and persist rollout | 18.24 s | 1.30 s |
| Reload and strict typed roundtrip | 55.49 s | 43.67 s |
| Write plus reload/roundtrip | 73.73 s | 44.97 s |

The envelope is 98.78% smaller; the projected complete retained episode is 37.53%
smaller. The measured write/load portion is 1.64 times faster. Policy reconstruction
and native replay remain additional work; these figures are not a whole-game
speedup. The full offline test suite was started only after the storage and policy
timing phases had finished; the native replay duration is validation, not a
comparative performance measurement.

Regression controls cover legacy/external equality, natural/cutoff/failure and
terminal closures, expanded guard support, tampered inputs, duplicate/partial
records, wrong paths and atomic publication failures. Progress controls compare
complete synthetic rollouts and both RNG states with/without the display, exercise
an actual short subprocess with redirected output and a timeout, and check failed
worker/cleanup handling. Initial focused test attempts exposed stale synthetic
fixtures: missing launch-output paths, the old mocked serializer, a historical
resume test using current rather than its accepted historical sources, and an
expanded fixture missing its cutoff declaration. The fixtures were corrected;
no production checks were relaxed. The full copied-game diagnostic passed on its
first attempt.

The exact source transition pins six reviewed files to commit `3bd8100`: the
rollout decoder/collector, storage codec, progress display, self-play verifier,
training controller, and local adapter's generated source pins. All other reused
engine, model, PPO, native learner and replay sources remain pinned to the
accepted `b47c1b9` runtime base. The original checkpoint metadata remains intact.
The completed 100-game campaign's historical continuation gate stays closed;
this release does not create a new campaign or accept the tiny model for play.

The complete offline suite passed: 2,177 tests run, 2,103 passed and 74 skipped
for optional/platform-specific controls. The final progress robustness changes
received focused regression checks after that suite; malformed deeply nested
sidecars and closed output streams cannot interrupt shutdown or integrity checks.
The benchmark tool minimum-deadline parser also received a focused correction
(parent60 seconds includes a ten-second cleanup reserve); its measured default
900-second path and all measured storage/runtime sources are unchanged.

After the final source-pin, display robustness and benchmark CLI corrections,
the focused integration suite ran153 tests:152 passed and one optional control
was skipped. The original complete suite and the focused rerun are reported
separately rather than added as unique test coverage. See
[compact validation](../../validation/rollout-storage-2026-10-30.json).
