# Native Mac learner utilization investigation — 2026-10-29

The accepted candidate reduces measured PPO update time by 35–36% versus the previous native build. It uses all 16 frame workers during parallel preparation, while average whole-update occupancy remains below full capacity. Raw captures, checkpoints and stack samples remain private.

The original 28.7% measurement was real. Replay workers existed but frequently lacked work, while the producer and replay paths incurred allocation, tensor-destruction and callback-state overhead. Candidate 1 did not establish an improvement over the accepted native implementation. Candidate 2 reduced elapsed time and CPU work substantially. Candidate 3 also reduces serial frame-encoding time. No learning settings changed.

The experiment preserves the map, architecture, opponent, four-game contract, PPO settings, recurrent boundaries, sampler order, terminal-only +1/-1/0 rewards and visible-interaction-budget-v1. Disposable updates reuse verified archived experience and never advance official state, reserve game seeds or admit diagnostic samples. Archived benchmark batch numbers do not identify the current checkpoint.

The historical main-checkout `./test-native-load.sh` result is in `validation/native-load-launcher-2026-10-29.json`. With a 16-CPU budget and 120-second comparison deadline, native PPO took 15.767479 seconds versus Python's 28.423626 seconds. Native CPU consumption averaged 4.594830 cores, or **28.717686%** of available capacity, despite a peak of 15 simultaneous replay jobs. Process lifetime peak RSS was 827,244,544 bytes. All 13 exact checks passed; official files were unchanged and cleanup was verified. The accepted runtime was `mac-native-v1-attempt6`, manifest SHA-256 `5a575c084479ac163ad152890ef58ef9e4cb4f842eca23d9f330f8733ddc477a`.

Average cores equal process CPU seconds divided by PPO elapsed seconds. This is neither total Mac activity nor GPU/full-campaign utilization. Peak RSS includes imports, restoration and audits. The original single execution order was not a counterbalanced performance study.

The external macOS sample in `baseline-sample-20261029T171108Z/` observed the actual worker for 15 seconds, using clean source `9e15a03bec7cbd6bb92525e5fb4cc4af5b8c9b2f` and archived batch 846: four histories, 584 decisions, 20 optimizer steps. Its complete resulting state matched the accepted saved state; process identities and sampler/worker/supervisor cleanup are recorded. Exclusive stack observations subtract direct-child counts; all 168,915 parsed nodes had nonnegative counts. Ancestry categories overlap and are not CPU seconds.

- Replay threads waited on their empty-queue condition variable in **65,134/85,335 observations (76.3%)**. This establishes starvation during the sampled replay-thread lifetime, not 76.3% idle time for every training phase.
- Main-thread `fold_head` accounted for 878/6,064 observations, or 21.8% of backward observations; 876 originated from eager folding in `Scope::add`. Of the 878, tensor/view destruction appeared in 502 and allocation/free in 476; only 72 ended at the vector-add kernel.
- Within `Leaf::replay`, allocation/free appeared in 7,581/20,135 observations (37.7%) and RecordFunction machinery in 4,814 (23.9%). Its 4,012 mutex/ulock waits split into 2,381 under callback lookup/rebuild and 1,631 under allocation/free. Stacks include thread-local-state restoration and callback-scope rebuilding.
- No `take_gil` frames appeared. Two `PyGILState_Check` samples and `PythonEngine` names do not establish GIL serialization.

The sampled update took 18.301662 seconds, versus the earlier unprofiled 15.767479 seconds. Sampling materially perturbed performance; these stacks support attribution, not a candidate speed comparison. Denominators and raw evidence are preserved in the sample directory.

Candidate 1 moved FIFO gradient reduction and contribution disposal to a dedicated thread, flattened contribution capture and prewarmed independent text graphs in parallel. It preserved each parameter's addition order. At budget 16, backward uses 14 replay workers, one reducer and the producer; the pending queue is bounded at 1,024. Temporary preparation threads operate while persistent replay threads sleep, so total OS thread count exceeds the active compute budget.

At source `474dcb9f63c96ded5cf7b6f9081b9bdd029b8698`, build attempt 1 passed 37 focused tests and both archived comparisons passed all 13 exact checks. Batch 821 took 0.578381 seconds native versus 0.814085 Python. Batch 846 took 16.587573 native versus 33.750590 Python, averaging 5.085816 cores. The latter is approximately 5.2% longer than the historical accepted native result; its higher Python/native ratio does not establish native improvement. Single-order measurements do not establish a reliable regression either. The large run's queue peaked at 47, with no backpressure, indicating that reducer capacity alone was not the remaining limit.

Candidate 2 adds a bounded execution schedule for retained first-order CPU text-backward graphs. It retains the same ATen numerical nodes, InputBuffer accumulation and contribution hooks, while avoiding repeated Engine scheduling and per-node thread-local callback-state restoration. Dependency/sequence ordering is preserved; every valid branch must reach an explicitly permitted alias. Buffers and captures belong to each invocation. The FIFO reducer still preserves original contribution order.

Eligibility is restricted to ordinary dense CPU float tensors, one ATen intra-op thread and an exact built-in-node allowlist. Unsupported contexts, hooks, modes or graphs fall back before any tape node executes. Execution errors propagate without Engine retry because contribution hooks may already have run. Separate tape/Engine counters establish actual acceleration-path use. Worker TLS is captured when its scope starts; a profiler introduced later only on the parent thread is not automatically propagated, an existing worker-path limitation.

Candidate 2 source is `5622486a2b74e928941fa4f70a94ec189fccb6b5`. `pre-forward-comparisons.json` records nine qualification pairs: eight combinations of old/new runtime, two archived workloads and Python/native execution orders, plus a small batch-821 candidate pair. Every pair passed all 13 exact comparisons and cleanup, with zero official updates. Checks cover initial/final complete state and components, model, gradients, Adam, RNG/generator, sampler order, reports and source/input provenance.

The following are arithmetic means of the two execution orders per runtime/workload:

| Workload | Accepted native time | Candidate-2 time | Time reduction | Mean cores, old → new | CPU seconds, old → new |
| --- | ---: | ---: | ---: | ---: | ---: |
| Batch 846 | 16.563020 s | 12.120415 s | 26.82% | 4.403835 → 2.961903 | 72.937401 → 35.895707 |
| Batch 763 | 5.473148 s | 4.136668 s | 24.42% | 4.191485 → 2.773146 | 22.940106 → 11.472597 |

Corresponding Python means were stable: 29.721496 versus 29.790113 seconds for batch 846, and 9.465810 versus 9.477726 for batch 763. Candidate 2 reduced CPU work by approximately 50% while running faster; its lower average CPU percentage is therefore not evidence of a slowdown. Both new batch-846 runs used the tape for all 88,480 replays, and both batch-763 runs for all 29,894, with zero Engine fallbacks. The small candidate batch took 0.458915 seconds. Two orders on two workloads remain a limited sample, not a full-campaign throughput estimate. Candidate-2 large-workload loss forward still averaged 2.774675 seconds.

Candidate 3 parallelizes only independent differentiable frame features. Initial review rejected direct construction against shared parameter accumulators: installed PyTorch `Node::topological_nr()` writes the non-atomic `has_parent_` flag during graph construction. Precreating those accumulators would not eliminate concurrent writes. Blind parallel graph construction was **not executed**; this was a design rejection, not a failed diagnostic run.

Instead, every frame worker owns private detached aliases and private AccumulateGrad nodes for every parameter, including text-occurrence edges. Workers read immutable parameter storage and prewarmed leaves; scope occurrence accounting is atomic. After all workers join, main validates every terminal substitution against the original parameter identity, storage, metadata and depth-zero accumulator. Only then does it reconnect equivalent-depth terminal edges and assign fresh main-thread sequence numbers to all reachable nonleaf nodes, in original per-frame creation order. Each frame is rebased immediately before its unchanged recurrent transition, value head and conditional heads. This preserves chronological global priority and gradient-addition order.

Review also identified SavedVariable weak-accumulator lifetime before building this candidate. Parameter-alias tensors and strong accumulator owners are retained through backward and released only after worker joins. Activation caches retain their original frame scope. Forward-AD tangents on parameters/initial state disable parallel preparation; unsupported callback/mode contexts use serial execution before worker dispatch. Unexpected constructed graph types fail closed before recurrence or PPO integration. No accepted experience is silently retried.

Attempt 5 passed **59/59 compiled tests in 20.092 seconds**. Coverage includes raw-byte outputs/gradients across multiple windows at 1/4/16 workers, skipped decisions and unused value-head ordering, coordinate/quantity/management fixtures, unchanged RNG/parameter versions, missing-preparation rejection, worker-failure cleanup and retry, and two complete PPO updates with exact model/gradient/Adam/RNG plus positive actual frame-parallelism and tape counters. The fresh attempt-6 binary passed seven complete paired comparisons: the small batch 821, both execution orders for batches 763 and 846, and both orders of the previously accepted native build on batch 846. Every pair passed all 13 exact-state/provenance checks and owned cleanup. Nine learning-state fields also matched across old and new builds. The five candidate executions performed **238,898 tape replays with zero Engine fallbacks** and **10,520 parallel frames**, reaching **16 concurrent frame workers**.

| Archived workload | Previous native mean | Final native mean | Time reduction | Final mean cores |
| --- | ---: | ---: | ---: | ---: |
| Batch 846 | 16.556 s | 10.535 s | 36.4% | 3.59 / 16 |
| Batch 763 | 5.473 s | 3.565 s | 34.9% | 3.28 / 16 |

Batch 846 uses a fresh old-build bracket. Batch 763 uses the earlier same-session old-build bracket; its Python controls were about 2% faster in the later measurement, so the exact percentage should not be generalized beyond these runs. Final batch-846 times were 10.450–10.621 seconds, versus 16.372–16.741 previously. Its loss-forward phase fell from about 3.166 to 1.202 seconds. Final process lifetime peak RSS was 775–786 MB, versus 810–825 MB in the compared old-build runs.

Average utilization is roughly **20–23%** on these final workloads, lower than the earlier 28.7% single-run result. This is an efficiency gain, not continuous core saturation: CPU work approximately halved while elapsed time fell. The fixed four-game collection/verification batch, chronological recurrence, ordered reductions and independent Python behavior checks still limit parallelism. This work does not claim full-campaign speed, GPU utilization, or better policy quality.

| Build / review event | Recorded outcome |
| --- | --- |
| Private tape lifetime review | Input references originally survived routing; corrected before qualification to match Engine destruction before refcount-sensitive accumulation |
| Attempt 2 | Compile failure: metadata helper called non-const `device()`/`stream()` through `const Node&`; corrected signature |
| Attempt 3 | Dynamic import failure after compilation: installed libtorch did not export `InputBuffer::variables` |
| Attempt 4 | Fresh build/load succeeded after using that helper's complete implementation: moving its public buffer member |
| Attempt-4 controls | Initial 55-test run had one incorrect test-cardinality assertion for a one-byte text; corrected fixture, then all 23 targeted rechecks passed without a numerical discrepancy |
| Attempt 5 | Candidate-3 build/load and all 59 controls passed |
| Attempt 6 | Fresh build/load after a comment-only correction; manifest SHA-256 `919951d970eefa356ed19c901ddc3441e108035da186957b5cbe107b00ace623`; seven paired comparisons passed exact state and cleanup |

Build failures are preserved in `build-attempt*.log`; neither compilation nor source agreement alone proved runtime usability. Graph ownership and lifetime concerns were corrected during design/review and must not be described as observed training failures.

These changes depend on validated PyTorch 2.9.1 internals, revision `5811a8d7da873dd699ff6687092c225caffcf1bb`; upgrades require fresh qualification. Primary references are the exact-version [Engine](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/torch/csrc/autograd/engine.cpp), [InputBuffer](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/torch/csrc/autograd/input_buffer.cpp) and installed `function.h`. Ordered reduction, recurrence, small operations and allocation remain limits; filling every CPU is not itself a throughput result.

The canonical engine check passed: generated `vcmi` exactly matches the pinned upstream, root patches and owned engine sources. No engine source or runtime was changed. The full-suite and real-main receipts are recorded in the [validation report](../../validation/native-mac-utilization-2026-10-29.json). The previous acceptance remains available in [its archived record](../../validation/native-mac-learner-initial-2026-10-29.json); the current native gate binds the new build and reviewed source hashes.

Official checkpoint/run pointers and both `0444` ownership locks retain their recorded values. The local pointer is not evidence of the latest cloud checkpoint. Cloud closure and latest-state import must be independently verified before releasing Mac ownership; this repair neither changes those fences nor starts training. Evidence root: `/Users/orsi/Projects/Heroes-artifacts/run-investigations/2026-10-29-native-utilization/`.

Run the bounded, isolated load check from the synchronized main checkout:

```sh
cd ~/Projects/Heroes
./test-native-load.sh
```

It verifies the accepted source and actual binary, compares disposable archived states, prints measured CPU/memory and parallel worker counts, and verifies cleanup. It reserves no game seeds and admits no official update. The separate real continuation check remains blocked by the ownership fence:

```sh
./continue-original-native.sh --check --hours 10 \
  --native-learner /Users/orsi/Projects/Heroes-artifacts/native-learner-builds/mac-utilization-v1-attempt6/manifest.json \
  --native-sha256 919951d970eefa356ed19c901ddc3441e108035da186957b5cbe107b00ace623
```


Full offline validation on the final source and actual attempt-6 binary ran 1,800 tests: 1,753 passed and 47 platform/optional-fixture checks were skipped. Fast-path pins passed; source/runtime hashes remained unchanged and test leaders, guardians and process groups were absent afterward. The broader suite includes the compiled learner/tape/frame controls.

The actual synchronized main-checkout load command passed at `fc70a2a`: native
PPO took **10.39 seconds** versus Python's **29.10 seconds**, with **16 peak frame
workers**, **22.4% mean CPU capacity**, 0.75 GiB process lifetime peak RSS and all
13 exact-state checks. Owned workers were cleaned up. Main's canonical-engine
check also passed. The real ten-hour continuation command in check mode returned
2 at the existing ownership fence before reaching the training runner. All six
official checkpoint/run pointers and lock identities, bytes and modes were unchanged.
These checks did not launch training or reset the local state to an older checkpoint.
