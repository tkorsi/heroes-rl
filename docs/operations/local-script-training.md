# Manual local training

The user runs a local script, waits for its result, then asks for analysis.
The script makes **no language-model API calls**, starts no cloud agent and creates no
scheduled task. The cloud Teacher has finished and handed back the full state;
see [its final handoff](teacher-cloud-to-local-handoff-20261026.md).

## Current original-map continuation

The first original-map run stopped at checkpoint 699 after 280 admitted games.
Its remaining 4,720 games use `./continue-original-4720.sh` after the Mac 699
validation; see [the continuation instructions](original699-continuation.md).
The legacy wrapper now refuses an existing original-map run, preventing a reset
to 629. The continuation keeps its own bounded 14-day manual window; no automatic
restart is scheduled.

## Historical first original-map launch: 5,000 training games

The user invoked this after the easier8x8 run finished; do not repeat it for the continuation:

```sh
cd /Users/orsi/Projects/Heroes
./train-original-5000.sh
```

This one-shot wrapper selects the unchanged original two-Castle 20x20
hero-management-v2 map and the final full state from
`run-20261026T160706Z-0be8eb77`. It does not read a moving checkpoint pointer.
Clean shutdown, full restoration, unchanged initial inputs and original deadline
completion must be recorded before it can start. Both presets share the existing
execution lock, so the current run is never overlapped or interrupted.

`--games 5000` means **newly admitted training games**, 1,250 complete PPO batches,
2,500 games per seat. All complete valid training games contribute, including
losses; a win is not guaranteed, and one win does not establish reliable learning.
Sampling remains the current policy distribution, not an external uniform random
action generator. Model, Adam, RNG, sampler and pool continue from the final
parent. Rewards, architecture, PPO, day/choice bounds and native rules are unchanged.

Baseline/final games and paired checks every 100 updates are excluded from the
5,000: a completed run ordinarily has 112 additional evaluation games. The exact
same final checkpoint is used for both final seats. The wrapper has a **48-hour
safety cap**, with five minutes for finalization; it stops earlier at the game
target or on failure/manual stop. `target_games_met`, `remaining_training_games`
and `stop_reason` distinguish reaching 5,000 from an incomplete stop. No automatic
restart, recovery, model supervision or deadline extension is scheduled.

Original-map outputs use `Heroes-artifacts/local-script-training/original/` and
`machine-original-parallel-v1.json`. The historical `machine-original.json`
is retained unchanged for completed sessions. Engine/policy seed bases are 40,000,000/50,000,000,
separate from the current easy run. At sequence266 the first seeds are40,001,064
and50,001,064; reservations persist before launch, including failed attempts.
The fixed evaluation seeds remain710001--710004 and720001--720004 on the original
map. Their map binding distinguishes them from earlier easy-map comparisons.
The asset bundle still retains checkpoint412 as historical input provenance;
the wrapper's selected parent is the final current run, never this seed fallback.

## Run on the prepared Mac

```sh
cd /Users/orsi/Projects/Heroes
./train-local.sh check
./train-local.sh --hours 2
```

`check` restores and verifies existing files without launching a game. The second
command is the explicit launch. It starts from exact checkpoint 412 on the first
run and from the last fully verified local checkpoint on subsequent runs. The
user can choose another duration with `--hours` (0.25–48). The final five minutes
are reserved for preservation and cleanup. Preparation happens before the clock;
the baseline games, training, evaluations and preservation share one immutable
wall/monotonic budget. A new manual invocation creates a new experiment window,
never an extension of an old session.

The default easier preset keeps the accepted **easier8x8 stage3** distribution, both
seats, Nullkiller Easy, BattleAI, 365 days, 5120 choices, the existing architecture
and PPO settings, and terminal-only rewards. It does not silently introduce the
proposed harder owned-town/enemy-hero scenario. That scenario needs its own
validated preset. The recent transfer study found checkpoint 412 won 24/24 varied
easy starts but 0/32 original starts; more training on the easy preset is not a
claim that the original-map problem is solved.

Before PPO, the script runs four frozen evaluation games per seat through the
current source and pinned cold runtime. Every game needs completion, native replay,
policy reconstruction and cleanup. A failure stops the run without admitting data.
Only complete verified four-game training batches advance the checkpoint. Both
fixed evaluation seats are repeated every five updates and at the end when the
remaining budget can cover the pair. Evaluations never enter PPO. A final partial
pair must not be reported as a complete paired comparison.

The script prints progress after each admitted update. `Ctrl-C` requests a stop
after the current bounded batch; it does not start another batch or final check.
A separate process guardian enforces deadlines if the Python collector hangs.
The Mac is kept awake against idle sleep while the worker runs; keep the lid open.
Closing the terminal may interrupt the active batch; its partial experience is
not admitted. No automatic technical repair, retry, or restart occurs.

## Files to analyze afterward

Each run lives outside managed worktrees:

```
/Users/orsi/Projects/Heroes-artifacts/local-script-training/run-<UTC>-<id>/
```

- `RESULTS.md`: concise final result.
- `summary.json` and `campaign-report.json`: admitted updates, training versus
  evaluation outcomes, failures and the latest complete state.
- `session.json`: frozen source, runtime, map, checkpoint, seeds and deadlines.
- `research-log.jsonl`: append-only intentions and results.
- `batches/`: full plans, histories, native replay, policy audits and failed or
  unstarted attempts.
- `checkpoints/`: independently restored copies of completed model, Adam, RNG,
  progress and pool state.
- `shutdown.json`: supervisor cleanup result.

`latest-run.json` points to the run, while `latest-checkpoint.json` advances only
after a complete verified batch and backup restoration. `./train-local.sh status`
reads the latest report without waking an agent. Supply its `RESULTS.md` or run
folder to the analysis chat after completion. Public code/log changes are committed
separately from raw local artifacts; the script does not depend on GitHub writes
between batches. Private archival/publication can be performed during the final
analysis, with local durability and remote publication reported separately.

## Reproducibility and setup

`tools/prepare_local_training.py` prepares the machine-local descriptor at
`Heroes-artifacts/local-script-training/machine.json`. It verifies the exact
checkpoint 412 pair and update411 policy dependency, both maps, accepted engine
recipe, native binaries, libraries/resources and historical acceptance receipts.
Nothing is reinitialized or trained by preparation.

The runner freezes its committed source into an independent Git checkout under
`Heroes-artifacts/releases/`, then prepares/checks the ignored canonical VCMI
dependency. It uses the existing accepted cold runtime; no engine build or
experimental embedded backend is used. Editing the normal repository during
training cannot change the frozen release. Do not delete durable inputs/releases
while a run is active. `HEROES_PYTHON` can explicitly select the project's Python;
the prepared interpreter and Torch identities must match.

Historical seed sequences are not reused: local script training uses a distinct
20,000,000/30,000,000 engine/policy seed namespace with a persistent ledger that
consumes a sequence before launch, including failed attempts. The initial sequence
number is 266; this does not reuse the old cloud seed formula. Fixed evaluation
seeds remain 710001–710004 and 720001–720004 for descriptive before/after checks.

Offline tests and actual checkpoint restoration validate this launcher before
handoff. No gameplay campaign was launched while preparing it for the user.
The first explicit invocation still has to pass its real current-runtime gate.
