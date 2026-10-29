# Native Mac learner and resource scheduling

## Scope and preserved experiment

The user requested C++ training loops and fuller use of the Mac's available
processors. The inspected machine has 16 CPU cores (12 performance and four
efficiency cores) and 128 GiB RAM. This work is isolated from the live, frozen
Linux cloud run. It authorizes implementation and disposable diagnostics, not
another official training session or removal of the Mac/cloud ownership fence.

The original map, model architecture, four-game on-policy batch, recurrent
windowing, PPO coefficients, opponent, terminal rewards, action support and
visible-interaction-budget-v1 remain fixed. Existing Python admission,
checkpoint serialization, Adam ownership, RNG/sampler state, seed ledger and
deadline supervision surround the native numerical implementation. Increasing
the batch size or collecting against stale weights would change the experiment
and is not used to fill idle cores.

## Measured bottleneck

The latest completed Mac training portion, run `20261028T192106Z-44be2ecc`,
contained 134 admitted updates and used four collectors, no forked prefix
helpers, and two text-replay threads. Summing its batch timing reports gives:

| Phase | Seconds | Share of batch time |
| --- | ---: | ---: |
| Collection | 494.82 | 18.5% |
| Verification | 628.72 | 23.6% |
| PPO | 1,241.08 | 46.5% |
| Other, restoration and checkpoint overhead | 304.72 | 11.4% |
| Total | 2,669.33 | 100% |

Within PPO, backward and clipping used 779.63 seconds, forward 270.81, prefix
reconstruction 113.33, behavior verification 65.84, history encoding 4.92, and
Adam 0.61. A C++ wrapper around the Python loop or a replacement of Adam alone
would not address this workload. Many phases really were serial, although the
whole run was not single-core.

## Native execution and provenance

The native ATen implementation operates on the existing parameter tensors. It
executes semantic tree and map encoders, chronological GRU prefixes and loss
windows, selected-action conditional likelihoods, and clipped PPO losses.
Compiled recipes use the same checked visible histories and the same action
enumeration/guard code as the reference distribution. There is no second
strategy mask, hidden-state input or alternative simulator.

Independent prefix encodings use a bounded C++ queue; chronological recurrent
transitions retain their order. Small ATen kernels retain one intra-op thread.
The resource plan uses available CPU/affinity/quota information and records
per-phase budgets instead of multiplying every process by the host core count.
Four-game collection and replay remain bounded to four independent jobs.

The revised native path also reuses text encodings within each minibatch and
replays their backward graphs on up to 15 worker threads. The main thread folds
individual parameter-edge contributions in the original autograd encounter
order. Workers never aggregate these contributions in completion order. The
scope collects every replay result and checks parameter identities, layouts and
versions before installing text gradients. It joins all workers before Adam
updates parameters.

Native builds require fresh external directories. Their manifests bind source,
compile flags, Python ABI, Torch configuration, actual Torch library bytes and
the actual loaded extension. Loading refuses a stale or changed library and
never silently compiles or falls back. Official native selection also requires
an acceptance report bound to the complete reviewed integration source.

## First candidate: exact but slower

Candidate `30faf61ac85ad4c88354ee64c10c06cbb83d932b` and build-manifest SHA256
`41134f8d5bda1807b6974c35d0230aa01d813e88715b788fa34b9249a54dd369`
passed the compiled numerical controls. These included every prefix boundary
with 1/4/16 workers, selected conditional/joint probabilities, model gradients,
and two complete PPO updates with exact Adam and RNG state. Additional controls
covered coordinates, quantities, management, dialogs and capabilities.

Three bounded historical-capture pairs then used disposable copies of the exact
same parent state. Every model, gradient, Adam, RNG, progress, sampler/pool,
source and input comparison passed. Each pair had an independent 120-second
deadline and verified owned cleanup. No game was launched, no seed reserved,
and no historical experience was admitted into official training again.

| Historical batch | Samples / Adam steps | Python seconds | Native seconds | Python/native |
| --- | ---: | ---: | ---: | ---: |
| 821 | 15 / 2 | 0.714 | 1.353 | 0.528x |
| 763 | 210 / 8 | 10.672 | 20.103 | 0.531x |
| 846 | 584 / 20 | 32.107 | 58.842 | 0.546x |

This candidate failed the performance objective and was not accepted for
official training. Prefix reconstruction improved 3.63x and 4.61x on the two
larger fixtures, but forward became 2.48–2.55x slower and backward 1.90–1.96x
slower. The native path had lost the reference's text-leaf reuse and concurrent
backward replay. This motivated a native implementation of that exact ordered
contribution algorithm. Single-pair timings are not a counterbalanced estimate
of full training throughput.

Earlier successful build attempts are retained: attempt 1 lacked the improved
prefix cache, and attempt 2 preceded worker-local ATen initialization hardening.
An early mixed test invocation also failed because reviewed source pins still
described the preceding implementation and a benchmark test module was
misnamed. Those preparation errors were corrected; integrity checks were not
relaxed.

The subsequent review found and corrected worker startup exception handling,
cleanup that could obscure the primary backward failure, and duplicate tensor
identities in the direct native constructor. The last case produced incorrect
gradients in a deliberately aliased, unsupported model; canonical PPO already
rejected that parameter inventory, and the C++ API now rejects it explicitly.
Failure controls retain diagnostic evidence privately, install no partial text
gradients, join their workers and permit a later fresh scope. One early test
incorrectly reused a model after registering a Python parameter hook; PyTorch
retained its wrapper hook, so the fixture was corrected to use a separate model.
The rejected test attempt is retained alongside the passing fresh-process runs.

## Independent replay execution

The new verification path uses fresh interpreter processes, without forking an
initialized Torch runtime. The coordinator retains policy resampling, history
checks, input bindings, ordered receipt construction and complete-batch
admission. Child completion binds a nonce and the exact bytes of its ordinary
replay report; no serialized training receipt is trusted. Failure cancels and
cleans up the complete wave before any complete-batch admission.

Four actual copied batch-821 captures passed exact-runtime native replay:
1,949 transactions, 1,660 client packets and 188 opponent requests, with none
unchecked. The replay wave took 0.911 seconds and the entire diagnostic 1.449
seconds. Original evidence and tested source/binary bytes were unchanged; all
four process groups and the outer guardian group were verified empty. These
were archived replays, not new games, policy training, or a measured end-to-end
training speedup.

## Final numerical and performance acceptance

The final implementation is `710163612323329e83c9ab85abdd5e1b243d207f`.
Fresh build attempt 6 has manifest SHA256
`5a575c084479ac163ad152890ef58ef9e4cb4f842eca23d9f330f8733ddc477a`.
The installed extension itself was loaded and exercised; matching source alone
is insufficient. All 12 compiled controls passed, including exact text gradients
with zero, three and 15 workers, two complete model/Adam/RNG update comparisons,
unsupported execution rejection and asynchronous failure cleanup.

The broad offline suite passed 1,719 of 1,766 tests, with 47 platform/optional
skips. Review then tightened the final module-order guard: parameter-free
activations cannot be removed or reordered, GRU flat weights must identify the
canonical parameters, and overridden model introspection rejects. The final
focused suite passed 139 of 140 tests, with one platform skip; it includes the
compiled controls and the native, replay, source-pin and continuation tests.
Official worker entry now checks native acceptance before changing execution
globals. Replay counters are read after joining workers. Independent final
review found no remaining issue within the canonical execution contract.
`python3 tools/prepare_vcmi.py --check` passed; no engine source changed.

Four new pairs on the final source counterbalanced execution order. PN means
Python first, NP means native first. Each PPO update used the same four recorded
games and complete parent state. The reference includes its existing optimized
text-leaf replay with two workers.

| Historical batch / order | Samples / Adam steps | Python seconds | Native seconds | Python/native | Native mean CPU cores |
| --- | ---: | ---: | ---: | ---: | ---: |
| 763 / PN | 210 / 8 | 9.247 | 5.443 | 1.699x | 4.25 |
| 763 / NP | 210 / 8 | 9.290 | 5.465 | 1.700x | 4.24 |
| 846 / PN | 584 / 20 | 29.299 | 16.228 | 1.805x | 4.39 |
| 846 / NP | 584 / 20 | 29.194 | 16.147 | 1.808x | 4.37 |

Every pair passed all 13 exact comparisons: complete starting and resulting
state, model, gradients, Adam, global and minibatch RNG, progress, sampler/pool,
report, parent, source and input audits. All owned processes closed within their
120-second bounds. The earlier revised candidate also passed batch 821 (15
samples, two Adam steps), taking 0.564 versus 0.708 seconds. Those intermediate
measurements are retained separately from the final-source pairs.

Actual concurrent text replays reached 15 in every final pair. On batch 846 PN,
prefix reconstruction fell from 3.962 to 0.883 seconds, forward from 6.114 to
3.143, and backward/clipping from 17.792 to 9.873. Native recipe compilation
added 0.899 seconds. The resource plan allows 16 prefix workers, 15 backward
workers plus the coordinating thread, and four collection/replay processes in
their respective phases. It does not oversubscribe every Torch process with 16
intra-op threads.

These are measured **PPO-update** gains, not full campaign throughput or learning
quality gains. No new gameplay, seed reservation, official admission or official
optimizer update occurred. No continuous 16-core occupancy is claimed: ordered
recurrence, ordered gradient folding and the unchanged four-game on-policy batch
limit independent work. Filling all RAM or switching this small exact CPU model
to GPU would not by itself establish acceleration or numerical equivalence.
Python still owns action enumeration, admission, the outer minibatch schedule,
exact existing Adam object, serialization and supervision; numerical hot loops
and parallel text backward run in C++. Only the canonical CPU float32 model is
accepted, without custom hooks, module changes or higher-order gradients.

The [compact acceptance record](../../validation/native-mac-learner-2026-10-29.json)
binds the actual build and complete reviewed integration source. Raw captures,
disposable checkpoint states, failed attempts, build logs and cleanup evidence
remain private in the durable artifacts directory.

## Manual continuation and cloud ownership

The new `continue-original-native.sh` is an opt-in launcher. It defaults to one
hour and all available CPU cores, accepts a smaller `--cpu-cap`, and requires the
exact accepted manifest and SHA256. It uses the existing verified original-map
ancestry and computes remaining games from admitted experience. It does not
choose checkpoint 1229 or any fixed remaining-game count.

The Mac's two ownership locks remain read-only because ownership was handed to
the cloud. The real ten-hour check correctly refused at that fence, without
reaching the runner or changing lock bytes, modes or inodes. Native backend
acceptance is therefore **not current Mac launch readiness**. The cloud's latest
complete model/Adam/RNG/progress/sampler/pool state and consumed-seed ledger must
first return through a separately verified closure and import. Do not remove the
fence or resume the older local checkpoint as part of this optimization.

After that handoff, the exact check command from the main checkout is:

```sh
cd /Users/orsi/Projects/Heroes
./continue-original-native.sh --check --hours 10 \
  --native-learner /Users/orsi/Projects/Heroes-artifacts/native-learner-builds/mac-native-v1-attempt6/manifest.json \
  --native-sha256 5a575c084479ac163ad152890ef58ef9e4cb4f842eca23d9f330f8733ddc477a
```

When the verified handoff and this check both pass, removing only `--check`
requests the manual ten-hour continuation. This work did not run that command
in start mode, restart training, alter cloud source or remove either fence.
