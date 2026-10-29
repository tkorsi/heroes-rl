# Collection and verification pipeline diagnostic

The next-launch pipeline passed **32 excluded native games** across two fresh
cohorts using frozen checkpoint 1,392. Correctness passed; **repeatable net
throughput improvement has not been established**. The final cohort used source
`34e9dd7dff2cad4b7ec92c8aa328cdf491d08476`, after the acceptance dependency inventory
was completed. The earlier successful cohort at
`e89a50d7a7bdf0cd1a48e28b85f35f036c2b4064` remains in the evidence.

Each cohort contained eight baseline games and eight candidate games on the same
newly reserved paired schedule, covering four Red and four Blue games per arm.
All eight pairs matched in the first cohort; seven of eight matched in the second.
Exact reconstruction checked 4,366 choices in the first cohort and 3,308 in the
second, for **7,674 choices** overall. All 32 games ended in natural losses; this
experiment does not demonstrate improved strategy.

The baseline collected four games before starting verification. The candidate
started native replay and coordinator policy checks after each collector had
exited, transferred its sealed capture and passed identity and native-cleanup
checks. Replay jobs remained bounded at four. The coordinator retained all
policy reconstruction and receipt creation. No background verifier could admit
experience, and no optimizer or admission function was called.

| Cohort | Arm | Red collection plus verification | Blue collection plus verification | Total |
| --- | --- | ---: | ---: | ---: |
| First | Verification after collection | 21.077 s | 12.536 s | 33.613 s |
| First | Verification during collection supervision | 19.555 s | 12.279 s | 31.834 s |
| Second | Verification after collection | 13.491 s | 19.612 s | 33.103 s |
| Second | Verification during collection supervision | 15.416 s | 19.349 s | 34.765 s |

The first cohort showed a **5.29% elapsed reduction**, or **1.056 times** the
baseline throughput for that verified workload. The second candidate took
**5.02% longer**, with a nonidentical native trajectory in one pair; its
identical-workload speedup is therefore explicitly null. Neither cohort was
counterbalanced: baseline preceded candidate. The user's independent training
run remained active and could contend for CPU time. Removing the phase barrier
is verified, but these observations do not establish a repeatable net speedup,
whole-training speedup, sustained CPU occupancy, or broader performance.

The second cohort's differing pair was Blue episode 7. Exactly two intermediate
day-five visible observations differed in opponent position and nearby occupancy
or resource visibility. The native server captures also contain different
opponent input packets. All 346 learner native input packets, all 480 learner
actions, decision sequence numbers and supports match. The different visible
history changes later recurrent prefixes and some values or probabilities, which
each arm reconstructs exactly from its own history. This is evidence of native
opponent trajectory variation, not a timestamp-only digest difference. The
collector, model, settings and runtime are unchanged between arms; there is no
evidence of a pipeline contract violation. Exact workload equivalence remains
unproven for that pair and is not claimed.

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

Each 900-second independent deadline covered its study and guarded collector,
engine and replay descendants. The studies completed in 76.45 and 78.41 seconds;
both shutdowns returned zero with owned cleanup verified before their deadlines.
Both logs contained no warnings. The prepared engine sources passed
`tools/prepare_vcmi.py --check`; gameplay used the existing accepted
terminal-replay-repair2 engine binaries.

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
