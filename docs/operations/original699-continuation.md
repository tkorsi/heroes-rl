# Continue the original 5,000-game target from checkpoint 699

The user-launched run `run-20261026T181133Z-062cd75a` stopped on 27 October 2026
at 08:55:28 UTC. It admitted 280 games / 70 complete PPO updates, with 0 wins, 275 losses
and 5 verified collection cutoffs. Checkpoint699 retains the complete model,
Adam, RNG, sampler, pool and progress state. Batch 336 exceeded the inner update
clock after 466 attempted Adam steps; the transaction rolled back, admitted no
samples or update, and produced no after checkpoint. That attempt stays consumed.

The user authorized a disposable Mac acceleration comparison, a bounded timeout
fix and preparation of the remaining 4,720 games. This documentation does not
claim a Mac speedup before that comparison is completed and reviewed.

## Manual commands after the Mac validation

```sh
cd /Users/orsi/Projects/Heroes
./continue-original-4720.sh check
./continue-original-4720.sh
```

The first command checks/restores files only. The second is the explicit user
launch. No agent, LLM API, cloud trainer or automatic retry is involved. Do not
use the older `train-original-5000.sh` for this continuation: it now refuses when
an official original-map run exists, preventing a reset to 629 and a new target.

Each invocation audits the immutable checkpoint 699 anchor, closed processes, exact final
restore and the official latest pointer. Later invocations may use only complete,
verified descendants of 699 from this same continuation. The remaining target is
recomputed from four-game ancestry, so a manual restart cannot reset 4,720.
A missing/inconsistent ledger fails closed. Failed batch 336 is never retried;
first training uses 337 or a later already-reserved ledger position. Fixed paired
evaluation is excluded and its 100-update cadence counts the original 70 updates.
The current checkpoint is restored before baseline games at every fresh launch.

## Operational budget correction

The historical batch allowed 3,600 seconds, with an update offset 3,540 seconds
from worker start and 30 seconds of cleanup reserve. The continuation declares
10,800 seconds shared wall time and update offset 10,740 seconds. Only these two
configuration values change. The 600-second per-game allowance, 2,400-second
collection phase, 3,540-second evaluation phase, 30-second cleanup reserve,
640 Adam-step ceiling, PPO mathematics, rewards, architecture, map, opponents and
365-day / 5,120-choice bounds remain unchanged. Historical plans are not rewritten.

This continuation defaults to an immutable 14-day (336-hour) safety cap, with five minutes for
finalization. Every worker phase remains clamped to that session's compute cutoff.
Timeouts still roll back the whole update and stop the campaign; no failed or
partial experience is admitted. `--hours 48` selects a shorter manual window; the generic runner still refuses more than 48 hours.
Only this audited original699 continuation may select up to 336 hours;
`--batch-wall-seconds` accepts only a bounded 3,600–21,600 seconds. Neither option
extends an already started session. Ctrl-C requests closure after the bounded
current batch. The guardian retains independent shutdown enforcement. Even the 14-day cap can end
a session before all 4,720 games finish; the same wrapper then calculates the
remainder on a later explicit manual launch. It never opens a new window by itself.

The script's `summary.json` includes per-session admitted/remaining games and an
`original_campaign` total toward 5,000. Raw evidence and backup checkpoints remain
under the existing durable `Heroes-artifacts/local-script-training/original/`.
All disposable benchmark states are excluded from this official lineage.

## Why the former 48-hour cap was too short

The 70 admitted original-map batches took 13.66 hours in total, including native
collection, replay/admission checks, PPO, durable preservation and independent
restoration. Mean batch time was 702.5 seconds; median was 340.8 seconds and the
range was 10.5–2,920.9 seconds. Red batches averaged 1,319.2 seconds and Blue
85.9 seconds. Baseline evaluation and failed batch336 are excluded from these
accepted-throughput measurements and retained separately in the evidence.

At that whole-run mean, the remaining 1,180 batches project to 230 hours
(9.6 days). The last 20 batches project to 322 hours and the last 10 to 351 hours.
These are historical workload extrapolations, not confidence intervals or a
promise: policy behavior and game lengths evolve. Even a hypothetical twofold
speedup would imply roughly 115–175 hours using those observed rates. A 48-hour
cap would therefore probably stop substantially before the requested target.
The 14-day bound is a prospective operational choice for a new explicit manual
launch, not an extension of the failed session or of the current eight-hour Mac
benchmark. The benchmark supplies its own measured speedup before final launch
advice. See the compact [timing and budget record](../../validation/original699-duration-plan-2026-10-27.json).
