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
use the older `train-original-5000.sh` for this continuation: it selects 629 and
would describe 5,000 new games rather than the existing experiment's remainder.

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

A fresh session still has an immutable 48-hour safety cap with five minutes for
finalization. Every worker phase remains clamped to that session's compute cutoff.
Timeouts still roll back the whole update and stop the campaign; no failed or
partial experience is admitted. `--hours` can select a smaller manual window;
`--batch-wall-seconds` accepts only a bounded 3,600–21,600 seconds. Neither option
extends an already started session. Ctrl-C requests closure after the bounded
current batch. The guardian retains independent shutdown enforcement. The 48-hour cap can end
a session before all 4,720 games finish; the same wrapper then calculates the
remainder on a later explicit manual launch. It never opens a new window by itself.

The script's `summary.json` includes per-session admitted/remaining games and an
`original_campaign` total toward 5,000. Raw evidence and backup checkpoints remain
under the existing durable `Heroes-artifacts/local-script-training/original/`.
All disposable benchmark states are excluded from this official lineage.
