# Collection and verification pipeline diagnostic

The next-launch pipeline passed 16 excluded native games using frozen checkpoint
1,392 and source `e89a50d7a7bdf0cd1a48e28b85f35f036c2b4064`. Eight baseline games
and eight candidate games used the same newly reserved diagnostic schedule,
covering four Red and four Blue games in each arm. Every paired game history
matched. Each arm reconstructed 2,183 policy choices exactly. All 16 games ended
in natural losses; this experiment does not demonstrate improved strategy.

The baseline collected four games before starting verification. The candidate
started native replay and coordinator policy checks after each collector had
exited, transferred its sealed capture and passed identity and native-cleanup
checks. Replay jobs remained bounded at four. The coordinator retained all
policy reconstruction and receipt creation. No background verifier could admit
experience, and no optimizer or admission function was called.

| Arm | Red collection plus verification | Blue collection plus verification | Total |
| --- | ---: | ---: | ---: |
| Verification after collection | 21.077 s | 12.536 s | 33.613 s |
| Verification during collection supervision | 19.555 s | 12.279 s | 31.834 s |

The candidate's observed elapsed reduction was **5.29%**, or **1.056 times** the
baseline throughput for these verified collection batches. This is one baseline
arm followed by one candidate arm, without repetitions or counterbalancing. The
user's independent training run remained active and could contend for CPU time.
The result does not establish a whole-training speedup, sustained CPU occupancy,
or performance on other checkpoints or workloads.

The existing collection and verification counters remain additive exclusive
coordinator wall time. `collection_elapsed_seconds` includes verification work
inside collection supervision and final collector cleanup.
`overlapped_verification_seconds` measures that coordinator work; it is not a
measurement of simultaneous CPU usage. The complete verified batch counter is
the appropriate comparison above.

The experiment preserved the original map, architecture, PPO configuration,
opponent, terminal-only rewards and disclosed `visible-interaction-budget-v1`.
It restored all model, Adam, global and named RNG, progress, sampler and pool
state before collection, checked that state afterwards, then restored the copied
parent independently and compared it exactly. Every fresh receipt was revoked.
Official samples admitted and optimizer updates were both zero. The copied
parent files and their immutable originals remained unchanged. The driver opened
only private diagnostic leases and wrote no official training state or locks.

The 900-second independent deadline covered the study and its guarded collector,
engine and replay descendants. The study completed in 76.45 seconds; shutdown
returned zero with owned cleanup verified before the deadline. Its log contained
no warnings. The prepared engine sources passed `tools/prepare_vcmi.py --check`;
gameplay used the existing accepted terminal-replay-repair2 engine binaries.

Earlier failures remain part of the record:

- Attempt 1 reserved eight diagnostic seeds but stopped before gameplay because
  full-parent restoration rejected a changed native learner header. Cleanup
  passed, no experience was produced, and those seeds remain consumed. A reviewed
  source-compatibility correction was independently restore-tested; the eventual
  candidate also restored the original native learner implementation after a
  separate performance study rejected its replacement.
- A read-only dependency check on the next frozen release ran before its VCMI
  checkout existed. No game or seed reservation followed that check. Preparation
  and the subsequent exact recipe check passed before attempt 2.
- Two intermediate offline test invocations failed because the new synthetic
  fixture imported a verifier while a model-identity function was mocked. The
  fixture now imports before applying the mock; the final collection/replay suite
  passed 35 tests. Six additional driver tests passed for private paths, input
  integrity, leases, collector scope and distinct durable seed reservations.

Raw captures, model files and full reports remain private. The compact result
and private-evidence hashes are in
[`pipeline-collection-results-2026-10-29.json`](../../validation/pipeline-collection-results-2026-10-29.json).
