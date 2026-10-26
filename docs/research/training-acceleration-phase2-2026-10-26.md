# Phase 2: complete verified PPO batch acceleration (scientific log)

Assignment: `docs/operations/claude-acceleration-phase2-20261026.txt` (main 83456186). Target: at least
another 2x on the complete verified four-game PPO cycle over PR30 head 4d7688b, on this Linux
allocation (4 vCPUs). Benchmark games, updates and checkpoints are disposable and separate from the
official lineage. Raw evidence stays in the private benchmark root on this machine; this file and the
compact records under `validation/` are the public part.

## Prospective protocol (recorded 2026-10-26 14:10 UTC, before any benchmark batch)

### Inputs and identities

- Official checkpoint 412 (`train-0265-seat1`, private commit 5577155): policy
  `1803823cab4a2d74f436e8425e31a1dbe1892e58844955945e2c963d176d9b88`, resume
  `09eee900a5fbff507a05c39e8ad4379e5b95177608b38cb69390a3cb56fee321`, weights
  `c310ae6b0b295149641802577fa4469c7a1662d530108c0289e3ce8f9c726d84`, progress 412/1648/95836.
  Its parent 411 (`train-0264-seat0`, commit 603be07): policy `9aeac520…`, resume `15163566…`,
  progress 411/1644/95817; both match 412's plan bindings.
- Maps (hashes equal the plans): open8x8 stage 3 `curriculum-3-0.vmap` `7d888602…`, `curriculum-3-1.vmap`
  `75303cc2…` (recovered byte-exact from the archived episode profiles); original v8 scenario
  `hero-management-v2.vmap` `ccfc8467…`.
- Configuration: 412's plan configuration unchanged (v8 `adventure-capabilities-v1`, Nullkiller Easy,
  BattleAI, 365 days, 4960 + 160 choices, PPO and Adam settings, `torch_threads` 1).
- Torch. 412 was written by `2.9.1+cpu`; this machine's PyPI wheel is `2.9.1+cu128` (same release),
  and the CPU wheel index is blocked by the network policy. The strict loader accepts neither
  relocation. `tools/acceleration2/relocate_checkpoint.py` therefore writes disposable benchmark
  copies of 411 and 412 whose payload differs only in its `torch_version` field. It proves every other
  field equal to the archived bytes' payload (every tensor bitwise, Adam, RNG, progress, sampler,
  pool, metadata) and that the strict loaders restore the archived state exactly. Relocated 412:
  policy `364b13ca…`, resume `19fdfd17…`; relocated 411: see the receipt. Baseline and candidate both
  use this one local Torch build, so the comparison is unaffected. Cross-build numeric equality
  against the archived Linux run is a separate check (below).

### Orchestration (the same for baseline and candidate)

PR30 has no campaign driver; the official overnight campaign used
`experiments/learner/overnight_training.py` and `tools/overnight_curriculum_campaign.py` on
origin/codex/teacher-overnight-20261026 with fixed `/workspace` paths. The benchmark ports both
without changing their checks: `experiments/learner/bench_training.py` (frozen-release checks, exact
parent restore, plan freezing, seed scan over all 172 official plans, per-call identity checks,
unchanged `builtin_training.worker`) and `tools/acceleration2/bench_campaign.py` (prepare, guarded
supervisor, verify the preserved copy, tar and hash-check the evidence, commit, push and confirm the
pushed blob). Differences: a private benchmark root instead of `/workspace`; a local bare Git
repository instead of the private GitHub repository (no shim); no acceptance gate or periodic
evaluations, so chains stop before an evaluation is due (at most four updates after 412); a
benchmark seed range (engine 5100001+, policy 5200001+, checked unused against every official plan);
a monotonic timeline per batch. `builtin_first_run.check_identity` gets the same three-line scope
dispatch the overnight branch added.

### Variants

- Baseline B: this harness commit (4d7688b plus harness only) with the 4d7688b native runtime.
- Candidate C: the head of `claude/project-thread-iba9jw-acceleration-phase2`, with its own runtime.
  Each measured candidate commit and runtime is frozen and recorded before its measurement set.

### Workloads

- easy: open8x8 stage 3, the training distribution of 412.
- original: `hero-management-v2` with the same configuration (longer games and larger PPO batches).
- Each campaign starts from relocated 412 and runs four sequential batches (updates 413 to 416) on
  seats 0, 1, 0, 1 with four same-seat games each. Batch 1 is the cold start; batches 2 to 4 are
  steady state.

### Measurements

- Per batch: controller prepare; supervisor lifetime (worker start, each game's collection, history
  and policy verification, native replay, admission, PPO components, checkpoint round trip,
  preservation); post-supervisor verification; archive, commit, push and acknowledgement.
- compute-only cycle: batch intent to verified preserved checkpoint (excludes publication).
  End-to-end durable cycle: batch intent to confirmed remote acknowledgement. Verified updates per
  hour = 3600 / mean steady-state end-to-end cycle.
- CPU seconds of the whole process tree, peak RSS, descriptors and threads of every owned process
  (external `/proc` sampler), per batch.
- Network/publication variation: the local push has no network. Extra campaigns emulate Teacher's
  recorded push medians (4.5 s native Git, 18.3 s API shim) with a fixed sleep before the push;
  these are labelled emulations.

### Design and analysis

- Same host, sequential, nothing else heavy running. Paired seeds: B and C campaigns of the same
  round use the same sequence numbers; rounds use disjoint sequence blocks. Round 0 is warm-up. Order
  alternates (B C C B, then C B B C). Medians and ranges are reported per workload; ratios are ratios
  of medians.
- Success requires the steady-state end-to-end durable cycle (and separately the compute-only cycle)
  to be at least 2.0x faster on both workloads, with every batch verified, and the same hardware
  budget (4 vCPUs, one campaign at a time). Any use of the idle vCPUs by C is reported as CPU seconds
  and parallel efficiency, and a single-worker comparison is reported too.
- Exactness: where a B and C batch played identical games, their checkpoints must be bitwise equal
  (weights digest, Adam state, RNG, progress). Divergent games are classified with self-repeats and
  recorded-action replay.
- Optimizer-only A/B: restore relocated 411 and the archived verified 412 batch, repeat the update,
  and compare losses, gradients, model, Adam, RNG, progress and sampler order with the reference and
  with the archived 412 result.
- Bounds: each campaign is capped at its plan budget (3600 s per batch) with owned process groups,
  the existing watchdogs and failure retention; failed attempts are kept and reported.

### Hypotheses (from Teacher's recorded costs and the code map)

- H1. Fixed per-batch work dominates small batches. In the 412 batch (59.5 s controller cycle) games
  took about 4.4 s and PPO 4.8 s; identity checks, interpreter starts, restores and publication took
  the rest. PR30's `check_identity` runs 10 times per batch and `TrainingBinding.recheck` 24 times,
  re-hashing unchanged binaries (about 1.9 GB of SHA-256 and 130 Git subprocesses per batch, an
  unmeasured estimate).
- H2. PPO dominates larger batches: over 157 official batches `_update` was 48.7% of the mean 109 s
  cycle (backward 27.8 s, loss forward 15.2 s, behaviour verification 7.5 s), at 0.29 to 0.40 s per
  sample, rising with batch size.
- H3. Removing duplicated work with equivalent coverage, exact PPO restructuring, the embedded
  engine with bounded per-process reuse, and overlapping independent verification can reach 2x on
  the complete cycle; collection alone cannot.
- H4. No single component is sufficient: on the easy workload overhead dominates, on the original
  workload PPO does, so both must improve.

## Attempts and corrections

(appended in order; nothing is removed)
