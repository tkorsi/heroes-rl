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

The corrected comparison used implementation
`e9c23b7f7c311e3a4de51e36afd0e84e24af07a4` on an Apple M4 Max with 16 CPU cores
and 128 GiB RAM. Both arms completed eight independently verified games with
18,945 sampled decisions: seven natural losses and one choice-limit cutoff per
arm. Native supervision records confirm a peak of one versus eight concurrent
engines. The complete diagnostic took 495.43 seconds.

| Measured stage | One collector | Eight collectors | Observed ratio |
| --- | ---: | ---: | ---: |
| Collection, including child startup and cleanup | 161.20 s | 53.69 s | 3.00x |
| Independent verification | 131.80 s | 135.29 s | 0.97x |
| Complete verified collection call | 293.08 s | 189.10 s | 1.55x |

These are **observed ratios**, not a successful exact-workload comparison.
Six full semantic history pairs match; episodes 1 and 3 contain different
observed enemy-hero positions before later learner decisions. All sampled
actions and decision counts match, but seven decisions in episode 1 and one
decision in episode 3 have different value/log-probability records. The strict
comparison remains `verified_nonidentical_games`, with
`identical_workload_speedup=null` and supervisor return code 1 retained. Equal
seeds did not produce identical native behavior. No history was normalized,
replaced or rerun to make the result pass.

All 16 games individually passed native replay and exact policy reconstruction.
Both arms preserved the frozen weights and coordinator Python/Torch RNG. All
16 fresh receipts were revoked after verification, zero samples were admitted
to PPO, and no optimizer update occurred. All 32 recorded child/native groups
closed under the independent guard before the original deadline. The bound
checkpoint, map and native-runtime inputs remained byte-identical.

The user's official continuation was active during the measurement and
continued afterwards. Its mutable checkpoint/sequence pointers changed as it
progressed; the benchmark never acquired its owner lock, wrote those files,
changed its release or signalled its processes. This concurrent workload limits
the interpretation of the timing ratio. The benchmark does not establish a
3x speedup for complete training: verification remains serial and PPO was not
timed here.

All 83 targeted tests completed: 80 passed and three Linux-specific checks were
skipped on macOS. They cover real harmless process overlap/ordering, cancellation
and descendant cleanup, deadlines, forged/replayed/cross-slot transfers,
complete eight-game admission, and a small synthetic eight-game PPO update.
The isolated generated engine also passed the exact canonical-recipe check.
See the [compact result](../../validation/parallel-eight-collection-2026-10-27.json).
Raw captures and guard records remain outside Git under
`/Users/orsi/Projects/Heroes-bench/parallel-eight/run-20261027-preflight-fixed/`.

The initial diagnostic at implementation `8af8d98` stopped after its first
serial game with `missing_source_identity`: this isolated checkout lacked the
generated `vcmi/` source checkout required by finalization. The game is an
unscored technical failure, seven serial slots and all eight parallel slots
remain unstarted, and the independent guard verified all owned processes closed.
No receipt was admitted and no optimizer ran. Evidence is retained under
`/Users/orsi/Projects/Heroes-bench/parallel-eight/run-20261027-8af8d98/`.

Preparation now checks the generated engine recipe before launching any native
process. The corrected study uses its own prepared dependency, new diagnostic
seeds and a new output directory, with only the time remaining before the
initial study's hard deadline (2026-10-27 16:47:03.662968 UTC). The failed game
is neither retried nor substituted into the corrected cohort.

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
Prepare its ignored engine dependency with `python3 tools/prepare_vcmi.py`;
the harness checks exact agreement before launch. This dependency supplies
source identity only; the native binaries still come from the frozen runtime.

The native diagnostic exercises training-mode collection and the full fresh
receipt verifier. It explicitly revokes those capabilities with
`BatchCollection.discard()` after each arm, verifies unchanged model weights and
global RNG, and never calls the optimizer. Persisted evidence remains available.
