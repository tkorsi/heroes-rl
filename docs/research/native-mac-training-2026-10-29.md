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
scope joins all workers and checks parameter identities, layouts and versions
before installing text gradients or allowing Adam to update parameters.

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

## Acceptance status

Implementation and performance validation continue. No native manual-training
acceptance is claimed by the first candidate's numerical success. The current
compiled-build decision and limitations belong in
`validation/native-mac-learner-2026-10-29.json` once final validation is complete.
