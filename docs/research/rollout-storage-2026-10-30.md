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
Results and all failed diagnostic attempts are recorded below after completion.
