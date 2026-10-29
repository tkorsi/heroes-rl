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

The rejected native candidate removes duplicate undefined edges from text occurrences to
their terminal parameter accumulators during owned deferred backward. It retains
one readiness edge per parameter and computes every original numerical text
contribution, folded in the same outer-engine order. It does not fuse arithmetic,
change recurrent order, alter floating-point flags or regroup gradient sums.
Unknown graph nodes, mixed scopes and observable hooks retain the original graph
before any edge mutation. This implementation is not selected for the next launch.
The accepted C++ learner and PPO implementation are restored to their exact
`10d3d4f` bytes.

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
- Build 4's complete copied-1033 update engaged compaction and remained bitwise
  exact. At the production 16-thread budget, both unprofiled execution orders
  regressed: baseline 16.998 versus candidate 20.142 seconds, then candidate
  19.018 versus baseline 18.196 seconds. Mean candidate time was 11.3% higher.
  A single two-thread pair improved from 31.966 to 29.008 seconds, but that does
  not justify promoting a build slower at the normal resource budget. Remaining
  planned comparisons were cancelled before launch after this decision. The
  graph eligibility scan appears to outweigh the removed routing work; this is
  an interpretation, not a separately isolated causal measurement.
- One private benchmark coordinator redundantly applied process niceness twice;
  its worker failed before PPO with `EPERM`. A fresh attempt retained one inherited
  nice-15 setting. All completed optimizer comparisons used private copied state,
  matched model/gradients/Adam/RNG/progress/sampler/pool, losses and sampling order,
  and verified owned cleanup. Contention from the user's run remains a limitation.
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
- A source-transition test run correctly rejected a stale acceptance-helper pin
  after review hardening. Regenerating the exact pin resolved that mismatch;
  no acceptance check was disabled.

Native controls use an explicit isolated diagnostic mode with fresh private
outputs, a separate serialization lease, original replay/history gates and
independent deadlines. The default harness mode still excludes official
training. Isolated mode never opens an official ownership lock. The full
pipeline diagnostic copies all parent files and revokes its ephemeral receipts;
its reports preserve the actual training-style capture phase and explicitly
record zero admitted samples and zero optimizer updates. Raw files stay private.

### Numerical preservation and native controls

The NumPy environment also reproduced the already-admitted batch-1033 numerical
update on a private copy: model, Adam, Torch/Python/minibatch RNG, pool and every
numerical PPO report field matched exactly. Diagnostic progress and sampler
bookkeeping intentionally remained at parent 1,392. The original admitted
checkpoint-1,393 counters were independently derived and audited; substituting
only those counters in memory reproduced its complete state digest. No diagnostic
checkpoint was published and no update was applied again to official state.

The unchanged accepted engine runtime passed 16 fresh native garrison controls
and 28 legacy controls, each with original replay checks, bounded deadlines and
verified owned cleanup. The full suite ran 1,990 tests in 339.107 seconds:
1,932 passed and 58 platform or optional-evidence cases skipped. The actual
accepted native learner was supplied to the suite. Subsequent closure validation
passed 10 focused tests, including a temporary-repository regression proving
changes to each of three newly inventoried acceptance helpers reject selection.
Counts overlap and must not be added as independent experiments. A final focused
test launcher initially named `test_parallel_replay`; its prelaunch inventory
assertion refused to start. The corrected selection uses `test_parallel_verification`,
with that failed selection retained in private evidence.

Review found that three executed helpers were missing from the official reviewed
source inventory. They were added without exempting that inventory from the
adapter AST gate. The first private acceptance descriptor remains preserved and
unselected. A second fresh policy cohort validates the final dependency closure;
the final descriptor supersedes the first without changing historical evidence.
The source freeze is `34e9dd7dff2cad4b7ec92c8aa328cdf491d08476`.

See the [pipeline study](pipeline-collection-2026-10-29.md) for both completed
cohorts, every failed attempt and the limitations on its throughput estimate.
The first cohort matched all eight pairs and reduced collection plus verification
time from 33.613 to 31.834 seconds (5.29%). The second took 33.103 versus 34.765
seconds and matched seven of eight pairs. Its one differing Blue game contained
two different native opponent observations; all 480 learner actions and 346
learner-side native input packets were identical, while independent native replay
confirmed opponent input differences. Each arm exactly reconstructed its own
recurrent state, values and probabilities. That comparison has no identical-workload
speedup estimate. Both cohorts used baseline-first order during a live user run.
The phase barrier is removed, but reliable net throughput improvement is not
established. The fresh policy validation totals 32 games and 7,674 reconstructed
choices, all natural losses; none entered PPO.

### Next manual launch

Final independent review found that changing the generic shell launcher's default
interpreter broke its historical easier8x8 machine identity. Interpreter selection
now follows the explicit map profile: original-map launches use the prepared
environment, while the legacy default keeps its original interpreter. Explicit
`HEROES_PYTHON` overrides and argument forwarding remain supported; no historical
machine descriptor is rewritten. Ten isolated wrapper regressions verify both
paths, argument forwarding, overrides and child exit status. Their first fixture
run compared macOS `/var` and `/private/var` spellings of the same temporary cwd;
normalizing the expected cwd corrected that test-only failure.

Final combined selection after the wrapper correction passed 223 of 227 tests
with four documented skips in 51.686 seconds. These overlap the preceding suites.
The actual main-checkout `./continue-772.sh --check --hours 10` at `7cd022f`
passed machine/environment/native acceptance and then correctly refused the
existing official owner lock (exit 2, 7.79 seconds, no NumPy warning). It started
no gameplay. A full official continuation check cannot pass while that owner is
active; repeat it after the current run ends. No lock was replaced or released.

The final read-only audit at 22:06 UTC restored checkpoint 1,468 exactly, with
468 games remaining in the continuation goal and 1,644 in the original target.
The active supervisor and worker still used frozen `10d3d4f` and the old Python
environment. Protected immutable session, machine and native build identities
remained unchanged. No owned diagnostic process remained in the final process
sweep. These are time-bounded snapshots of a run that continues to advance.

Final runtime selection and main-checkout preflight results are recorded in
`validation/cpu-next-launch-2026-10-29.json`. The future machine descriptor is
`machine-original-native-v4.json`; old descriptors, runtime and virtual environment
are not rewritten. The normal ownership check must refuse another campaign while
the user's current run is active. Preparation does not authorize or perform an
automatic restart. After it ends, the user can check and manually continue the
same goal using `./continue-772.sh --check --hours 10` and
`./continue-772.sh --hours 10`. The launcher resolves the latest verified checkpoint
and recomputes the remainder; it never resets to 1,392 or adds another 772 games.

The remaining serial recurrent and ordered reduction dependencies are real limits
under the fixed learning contract. This work does not establish full-system CPU
saturation, whole-campaign speedup, policy improvement or long-run stability.
