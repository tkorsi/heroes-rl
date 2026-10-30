# Self-play reporting no longer holds training ownership

## Cause and authorization

The user requested that descriptive reporting stop blocking new runs, that the
existing launcher be safely unblocked, and that the one-game doubled-choice test
remain a manual launch. All changes are made in `/Users/orsi/Projects/Heroes`.

The continuation completed 92 new games and 23 updates at 19:15:13 UTC, bringing
the separate self-play experiment to 100 admitted games and checkpoint 1753.
Its successful shutdown receipt was written before optional behavior analysis.
The old supervisor then analyzed 184 player histories totaling 33,593,003,064
bytes serially while retaining both campaign locks. The analyzer repeatedly
materialized and decoded histories and performed another full history audit.
It had no reporting deadline, progress output, or incremental result. A native
process sample showed active JSON parsing and Python garbage collection.

This was a reporting and resource-ownership defect, not an engine failure,
failed PPO update, admission rejection, or checkpoint publication failure.

## Safe current-process recovery

Before stopping the old report, the recovery checked the exact supervisor PID,
start time and command, its caffeinate-only child, and absence of all 370
worker/guardian PIDs from 185 deadline records. The registered lineage and full
checkpoint state restored exactly at 1753. Both training report and shutdown
receipt declared completion with verified cleanup and unchanged protected inputs.

At 19:56:50 UTC a SIGINT was sent only to the identified report supervisor, PID
25616. It exited and released its existing lock descriptors. No lock file was
deleted or replaced and no process group was signalled. The supervisor and its
caffeinate child were confirmed absent. All 128 protected files remained
byte-identical, and a read-only acquisition/release proved both locks available.
No gameplay, seed reservation, optimizer update or restart occurred.

The interrupted optional full behavior report was not completed. Its raw
histories and the previously completed, separately validated compact analysis
remain private and unchanged. The saved training success was not rewritten as
a failure merely because optional reporting was interrupted.

## Repair and validation

Training no longer invokes descriptive analysis automatically. Once supervision
has verified completion and cleanup, the launcher releases its campaign lease,
prints the saved training result and elapsed time, and offers a separate report
command. Explicit reports require completed-run evidence, write to a fresh
private output directory, use a bounded worker, and show per-game progress.
They neither acquire the campaign lease nor reserve seeds or update weights.

The focused suite passed 105 tests, including 10 new reporting controls and a
new control for the separate controller source pin. A real report-only command
against the completed run began while the campaign lease was deliberately held
by a diagnostic checker, printed game progress, and stopped at its three-second
deadline. Its incomplete-report receipt and owned cleanup passed; all 128
protected files remained unchanged. This is an intentional timeout test, not a
completed replacement report. The canonical engine check also passed.

The first synthetic test attempt found a `/var` versus `/private/var` path-alias
problem in the output-containment control; resolving the output path fixed it.
That attempt briefly launched only a report worker against temporary fixtures.
Review also caught a possible false reporting-success receipt on user interrupt;
the corrected path records interruption, cleans its worker, and re-raises.
Final reporting-plan hashes are now rechecked by both worker and supervisor.
No failed diagnostic experience exists to admit because no game was launched.

An AST comparison confirmed that 39 existing controller functions, including
collection, verification, PPO, checkpoint publication and cleanup, are unchanged.
Only the optional analyzer and top-level dispatch changed; three report helpers
were added. The expensive reporting algorithm itself was not optimized, and no
throughput or playing-strength improvement is claimed. Progress counts are not
resumable per-game reports.

The evaluator retains the exact checkpoint ancestry and original engine, learner,
adapter and replay source baseline. Only the reporting controller is pinned to a
separately reviewed commit. Tampering with either baseline remains an error.
The map, policy, PPO, rewards, revealed movement support and collection budgets
are unchanged. The user alone launches the one-game evaluation:

```sh
cd /Users/orsi/Projects/Heroes
./play-one-selfplay-2x.sh --run
```

The [compact validation](../../validation/reporting-lock-fix-2026-10-30.json)
records final checks. No native game is run during preparation; actual capture
and replay of the doubled-choice evaluation are checked on the manual launch.
