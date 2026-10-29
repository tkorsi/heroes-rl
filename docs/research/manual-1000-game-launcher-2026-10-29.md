# Manual 1,000-game continuation with elapsed time

The user requested publication for independent review and a manual script for
1,000 training games. `train-1000.sh` targets **1,000 additional admitted training
games**, or 250 complete four-game PPO updates. Evaluation games are excluded.
No official training was started while implementing or validating the launcher.

After verified cloud closure and import of the latest complete state and consumed
reservations, run from the synchronized main checkout:

```sh
cd ~/Projects/Heroes
./train-1000.sh --check
./train-1000.sh
```

The default runner safety cap is ten hours, including its cleanup reserve. Use
`--hours HOURS` to select a different cap between 0.25 and 336 hours. The launcher
selects the accepted native build and the available CPU budget automatically;
`--cpu-cap CPUS` can explicitly limit resources. This is not a claim of continuous
100% CPU utilization. The existing continuation verifies the latest complete
model, Adam, RNG, progress, sampler and pool state and consumed sequence ledger.
No checkpoint number is embedded in this launcher. A request larger than the
verified remaining original 5,000-game target is rejected before creating a
session or reserving a game seed.

The runner stops at the target or earlier on a failure, deadline or manual stop.
It retains the original campaign's admitted and remaining totals separately
from the new per-session cap. Map, architecture, PPO settings, batch size,
opponent, terminal-only rewards and visible-interaction-budget-v1 are unchanged.
No partial or failed batch becomes an admitted update.

The timer uses a monotonic clock and prints a final line such as:

```text
Total running time: 02:14:08 (8048.12 seconds); exit status: 0.
```

This illustrative format measures the whole invocation, including startup,
evaluation, checkpoint preservation and cleanup. It does not imply the target
was reached: a clean manual or budget stop can also return zero. Check
`target_games_met` and `admitted_training_games` in the runner summary.
SIGINT/SIGTERM are forwarded only to the directly owned controller; the timer
waits for its existing bounded cleanup and preserves its result. A process
killed with SIGKILL, host shutdown or power loss cannot print a final line.

## Validation and remaining readiness boundary

The [compact evidence](../../validation/manual-1000-game-launcher-2026-10-29.json)
records 144 focused tests: **143 passed, one Linux-only fork-overlap test skipped**.
Seven new cap tests cover parser boundaries, verified remaining-target rejection,
frozen-source forwarding, exact restoration in check mode, separate lineage
accounting and a mocked worker stopping at exactly 250 admitted updates. Eleven
new launcher tests exercise argument restrictions, elapsed output, child failure
status, direct-PID SIGTERM and terminal-group SIGINT with awaited cleanup.
Every child in these timer tests is a disposable mock.

Initial cap and timer test expectations each mishandled macOS `/var` to
`/private/var` canonicalization; the test fixtures were corrected. An intermediate
84-test runner group also rejected the changed runner at its intentionally stale
source gate (four failures, seven errors, one skip). After review and pin refresh,
the final 144-test group passed. Review aligned the timer's minimum duration with
the existing runner and corrected its help text about checkpoint preservation.

The native acceptance gate now binds 260 source files, including the complete
manual launcher chain. The actual accepted library loaded successfully and
reported `heroes-native-learner-v1`; its manifest and binary hashes are unchanged.
There were no native numerical edits, rebuilds or new performance claims.
Reviewed fast-path pins, shell syntax and `tools/prepare_vcmi.py --check` passed.

**Mac launch readiness remains false.** The real worktree check refused the
existing read-only cloud ownership fence with exit status 2 and printed elapsed
time. Neither a fresh latest cloud checkpoint nor its current remaining-game
count was verified here. Local checkpoint 1229 is not a substitute for a newer
cloud state. All six official pointer/lock snapshots retain their exact hashes,
modes and inode identities. This task does not authorize releasing the fence or
starting another worker. Raw evidence and full state remain private.
