# Eight-process collection diagnostic

## Scope

The user requested parallel collection of eight games and a measured comparison,
without changing the current training run. Work is isolated on
`codex/parallel-eight` in a separate checkout. The official original-map session,
checkpoint pointer, seed ledger, release and launch scripts are not updated by
this work. At the initial process inspection no local training worker was live;
the official last completed parent was checkpoint 699.

This is a collection experiment. It launches no PPO campaign and makes no
learning-quality or complete-training-throughput claim. Eight games per PPO
update are a new batch-size choice: two successive four-game updates would use
different policies for their second four games and are not an equivalent
baseline.

## Implementation

`experiments.learner.parallel_collection.collect_batch` collects eight scheduled
games with either one or eight independently guarded Python processes. Each
child uses one Torch thread, a fixed copy of the same policy, its own recurrent
history and sampling RNG, a separate native engine, profile, bridge socket and
capture directory. Schedule order determines the result and receipt order,
irrespective of completion order.

Training capture transfer requires a coordinator-owned, single-use prelaunch
capability attached to the actual guarded child and its private result pipe.
The parent retains the original training binding, checks the child's completion
and capture digest, and independently verifies the capture before minting a
receipt. Loading an old capture or receipt file cannot admit it to PPO.
Verification runs in the coordinator after all collection processes have closed.
All slots must succeed; a failed or timed-out slot stops the batch without
replacement. The independent process guardians stop only their owned groups.

An immutable configuration explicitly selects the new path:

```json
"parallel_collection": {"batch_size": 8, "workers": 8}
```

The serial comparison selects `workers: 1` with the same eight-game schedule.
`builtin_training.worker` dispatches to this collector only when the explicit
configuration is present. Eight verified receipts may feed one bounded PPO
update; historical four-game plans retain their existing admission rules.
The unchanged 16-decision windows and two-window minibatches need an explicit
maximum of up to 1,280 Adam steps for eight full 5,120-choice games. There is no
change to policy features, rewards, PPO coefficients or the native game engine.
The official manual continuation remains pinned to its reviewed four-game
configuration; this diagnostic does not promote a new campaign release.

## Prospective comparison

Use the immutable original-map checkpoint 699, the accepted cold macOS native
runtime, hero-management-v2, Nullkiller Easy and BattleAI. Preserve the existing
365-day, 4,960 nominal-choice and 5,120 total-choice limits. Allocate a fresh
diagnostic seed namespace and durable output directory independently of the
official campaign. Copy and bind the input checkpoint before launch.

The primary cohort is eight Red games collected serially and the same eight
games collected by eight concurrent workers. Red is selected prospectively:
checkpoint 699's Blue episodes often end almost immediately, so a balanced
four-Red/four-Blue batch would keep only four workers occupied for most of the
measurement. This Red-only benchmark does not establish both-seat throughput.

One study has a 45-minute hard bound including setup, collection, verification
and cleanup, with a 600-second per-game collection bound. Both arms share the
same absolute study deadline. There are no retries, replacements or deadline
extensions. The collected data and any live receipts are discarded as training
inputs; no optimizer step or official progress update occurs.

Report collection wall time separately from the complete verified collector
wall time. Preserve every failure and unstarted slot. Each successful game must
pass exact policy reconstruction and independent native replay. Compare ordered
semantic histories, choices, endings and rewards: identical seeds alone do not
guarantee identical native-opponent execution. Raw history hashes can differ in
bookkeeping timestamps. A divergent cohort is not an exact matched-workload
speedup. One ordered serial-then-parallel measurement has no confidence interval
and cannot separate warm-cache, background-load and thermal effects.

## Results

Pending implementation checks and the bounded native comparison.

## Reproduce a separate diagnostic

From the committed experimental checkout, using the locked macOS interpreter:

```sh
/Users/orsi/Projects/Heroes/.venv/bin/python tools/benchmark_parallel_collection.py
```

The command allocates a fresh output directory under
`/Users/orsi/Projects/Heroes-bench/parallel-eight/` and reserves new diagnostic
seeds. It requires checkpoint 699 and the accepted runtime at the bound local
paths. Use `--seat blue` only for an explicitly requested separate Blue study.
The source checkout must stay unchanged until both arms have finished.

The native diagnostic exercises training-mode collection and the full fresh
receipt verifier. It explicitly revokes those capabilities with
`BatchCollection.discard()` after each arm, verifies unchanged model weights and
global RNG, and never calls the optimizer. Persisted evidence remains available.
