# CPU pipeline and isolated NumPy preparation, 29 October 2026

## Prospective protocol

The user requests removal of remaining CPU bottlenecks and the NumPy warning
for the next manual launch, without interrupting the active training run or
changing learning settings. The active run uses frozen source
`10d3d4f5038025a276b33a0a3d81096e5778130b`, the existing main-checkout virtual
environment and the accepted `mac-utilization-v1-attempt6` native learner.
Those artifacts, process identities, ownership locks and official state are
not modified by this work. New diagnostics have independent output directories,
deadlines and owned cleanup; no automatic training launch is authorized.

The fixed scientific contract includes the original map, four-game on-policy
batch, model architecture, PPO settings, minibatch order, chronological recurrent
transitions, gradient addition order, Adam updates, RNG and sampler progression,
opponent, terminal-only +1/-1/0 rewards and `visible-interaction-budget-v1`.
Efficiency and completed verified work per second are the performance measures;
CPU occupancy alone is not evidence of improvement.

The first two completed batches of the active run, sequences 1033 and 1034,
are the primary immutable diagnostic workloads. Their four histories contain
997 and 718 sampled choices respectively. Copied inputs retain exact historical
receipts and are independently audited; disposable updates cannot mint admission
receipts or write official checkpoints. Compare the accepted and candidate
native implementations in both execution orders, with identical copied parents,
captures and thread budgets. Compare complete model, gradients, Adam, global and
minibatch RNG, progress, sampler, pool and reported losses. Historical batch 846
is a further workload if needed to test a different trace shape.

Initial observations identify two candidates: overlap verification of a sealed
completed game with remaining game collection, and reduce empty-edge routing in
native text backward. Pipeline activation requires an explicit resource plan;
smaller CPU budgets retain sequential phases. Completed-game receipts remain
coordinator-owned and the entire batch must pass before PPO starts. Neither
proposal permits speculative next-policy games or a larger learning batch.

NumPy is added to a separate hash-locked environment. Every existing package
version, Python ABI and Torch library must remain identical. The new environment
must pass a Tensor/NumPy bridge check without warnings and exact numerical smoke
checks. Historical machine descriptors remain immutable; any future interpreter
selection must pass separately reviewed environment and runtime acceptance.

Validation will record unsuccessful designs, builds, tests and measurements as
well as accepted results. Timings collected while the user's training continues
are subject to contention and are not clean whole-Mac capacity measurements.
The normal official continuation check must continue to reject duplicate workers
while the current run owns its locks. This work will not stop that run to obtain
a benchmark or preflight result.

## Results

The isolated Python environment adds NumPy 2.3.5. All ten preexisting locked
package versions, Python 3.13.5 ABI, Torch 2.9.1 configuration and Torch library
hashes match the active environment. The bidirectional Tensor/NumPy bridge,
deterministic tensor/Adam/RNG smoke comparison and the actual selection probe
pass without warnings. Preparation did not select the environment for training
or modify the old environment. Historical machine descriptors remain immutable.

The pipeline starts verification only after a collector exits, transfers its
sealed capture and proves native cleanup. Native replay processes use fresh
exec, never post-Torch fork. Verification stays coordinator-owned, results retain
schedule order, and any failure cancels the whole batch and revokes early
receipts. Resource plan v2 permits overlapping four collectors, four replay
workers and the coordinator only with a budget of at least nine CPUs. Smaller
budgets retain sequential phases; historical resource plan v1 remains exact.

The native candidate removes duplicate undefined edges from text occurrences to
their terminal parameter accumulators during owned deferred backward. It retains
one readiness edge per parameter and computes every original numerical text
contribution, folded in the same outer-engine order. It does not fuse arithmetic,
change recurrent order, alter floating-point flags or regroup gradient sums.
Unknown graph nodes, mixed scopes and observable hooks retain the original graph
before any edge mutation. Actual workload acceptance remains mandatory.

### Diagnostic attempts and corrections

- Native build 1 passed 42 focused tests. Review added terminal Tensor-hook
  checks; build 2 passed 44 tests. A subsequent saved-tensor-hook probe showed
  that hooks retained from forward could observe changed occurrence arity.
  Neither build was accepted for training.
- Build 3 added explicit saved-variable checks for the ordinary ATen graph and
  required a hook-free compiled proof for private text leaves. Its 48 tests
  passed, including persisted hooks, mixed scopes and failed-backward cleanup.
  The first complete copied batch-1033 update matched full state, losses and
  sampler order, but reported zero compacted occurrences. The 34.985 versus
  33.220-second profiled timings are not optimization evidence: the candidate
  safely fell back to its original graph.
- A bounded first-minibatch inventory stopped before backward or Adam. It found
  two scalar `MulBackward1` nodes missing from the exact allowlist; the other
  84,699 nodes were already covered. Build 4 adds that precise ATen type after
  inspecting its ABI-bound saved fields. All 48 tests pass, including complete
  PPO tests that now require actual compaction engagement. Build flags retain
  `-fno-fast-math` and `-ffp-contract=off`.
- Pipeline native attempt 1 stopped before gameplay because the historical
  checkpoint gate lacked explicit native-header transitions. Its eight private
  seeds remain consumed; cleanup passed and no experience or update was made.
  The source-pin generator now includes canonical native learner sources and
  tests reject changed historical or candidate header bytes.
- Two intermediate pipeline test invocations exposed an import-order leak in a
  mocked test fixture. A report-integration test then used a stale simplified
  fixture without the new timing fields. The corrected targeted suites pass;
  none of these failures involved official gameplay.
- One combined environment test command named a nonexistent test module. The
  corrected environment/machine/cloud selection ran 78 tests successfully. The
  new acceptance reader was corrected during review to distinguish bound replay
  paths from path strings and reconstructed choices from consumed PPO targets.
  An unconsumed terminal choice must be reconstructed but must not become a
  training target.

Native controls use an explicit isolated diagnostic mode with fresh private
outputs, a separate serialization lease, original replay/history gates and
independent deadlines. The default harness mode still excludes official
training. Isolated mode never opens an official ownership lock. The full
pipeline diagnostic copies all parent files and revokes its ephemeral receipts;
its reports preserve the actual training-style capture phase and explicitly
record zero admitted samples and zero optimizer updates. Raw files stay private.

Final copied-workload measurements, fresh native results and next-launch
acceptance are pending. No candidate is accepted by this progress record.
