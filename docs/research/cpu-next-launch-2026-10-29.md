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

Implementation and validation are in progress. No candidate is accepted merely
by this prospective record.
