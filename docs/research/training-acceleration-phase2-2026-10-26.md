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

### A1. Baseline smoke batch (14:16 UTC, `smoke-B-easy-1`, B = 3cb6a78, sequence 9000, seat 0)

Purpose: check the harness end to end and locate the time. Not a timed comparison (single batch,
development work ran alongside). Result: passed; 340 samples from games of 26, 36, 232 and 46
decisions. Compute cycle 286.1 s, durable cycle 287.9 s (local push 1.8 s), prepare 4.0 s,
supervisor-to-first-game 6.5 s, games 3.1/1.2/3.6/1.1 s, gaps between games (verification of the
previous game plus the next game's setup and identity checks) 3.5/3.8/8.9 s. PPO 238.7 s: behaviour
verification 26.1, prefix reconstruction 12.8, loss forward 74.9, backward and clip 123.5, Adam 1.3.
Process-tree CPU 290.8 s. So H2 holds here (PPO 83% of the cycle) and H1 is secondary for a batch of
this size; on this host PPO costs 0.70 s per sample, about twice Teacher's recorded 0.29 to 0.40.

### A2. Exact behaviour verification in evaluation mode (E1) and a prefix node memo

Hypothesis: `_verify_behavior` recomputes every frame in train mode, where the exact inference
memos are disabled although every allowed module is mode-independent; running it in eval mode
with the memos, then restoring every module's mode, is bitwise identical. Result on the smoke batch
(optimizer-only, same restored state): behaviour verification 26.1 s to 1.5-1.8 s; outputs, model,
gradients, Adam and generator bitwise equal to the reference. Prefix reconstruction gains a
call-local internal-node memo with the inference memo's exact-tree rule (5.4 s against 12.8 s).
Loss forward and backward are unchanged (about 66-78 s and 124-130 s): bitwise-exact PPO cannot
change autograd's gradient summation order, so the exact PPO gain on PPO-heavy easy batches is
about 1.2-1.4x. Tests: `tests/test_ppo_phase2_acceleration.py` (mixed modes restored; poisoned
probabilities rejected by both paths and not laundered by a warm memo; a stale memo after
`.data` weight changes rejected; hooks keep the reference path; the node memo equals uncached
states; the update equals the reference).

### A3. Separate numerical experiment: shared graph encodings (opt-in, never a default)

Within one minibatch's autograd graph, a text/number leaf or exact internal tree is encoded once
and the same tensor serves every later frame. Forward values, log-probabilities, values and losses
are bit-identical; parameter gradients differ only in floating-point summation order. Selection
is bound in the plan (`experiments`) and in every produced checkpoint's metadata
(`numerical_experiments`), checked by every identity check; no environment variable can select it.
Smoke batch: PPO 238.7 s to 17.5 s (optimizer-only). The resulting update differs from the exact
update by 4.45e-4 of its L2 norm, deterministically (identical across repetitions). The user chose
to count this path toward the target as an opt-in candidate pending manual review (14:34 UTC);
exact remains the default and nothing is deployed to Teacher.

### A4. Fail-closed incident during profiling (14:41 UTC, `prof-X-easy-1`)

I created an untracked helper file in the measured candidate checkout while its campaign ran. The
next identity check failed closed ("Frozen checkout must remain clean") and the batch stopped
without an update. Correction: separate development (`p2dev`) and measured (`phase2`) worktrees;
measured checkouts are only fast-forwarded between measurement sets. `prof-X-easy-2` (py-spy
attached, so its timings are inflated and not used) then passed: PPO 6.1 s for 112 samples.

### A5. Identity-check hashing (H1)

Profile of one benchmark identity check: about 0.9 s, mostly SHA-256 of the same runtime,
implementation and parent files bound by several helpers (`check_frozen` ran twice per check).
Change: `experiments/headless/digest_scope.py`, a per-check memo keyed by resolved path, device,
inode, size, mtime and ctime observed both before and after hashing; every check opens a fresh
scope, outside a scope every call rehashes, and files of at least 1 MB are prefetched by four
threads. Microbenchmark: hashing per check 0.78-0.80 s to about 0.40 s; prefetching small files
too was slower (0.78 s, interpreter-bound), hence the 1 MB threshold. Tests:
`tests/test_digest_scope.py` (rewrites with forced-equal mtime, atomic replace, truncation and a
change during hashing are all rehashed; scopes never share; prefetch errors stay with the check).

### A6. Original-workload baseline warm-up (14:18 UTC, `r0-B-original`, B, sequence 0, seat 0)

Games of 2, 60, 93 and 2525 decisions (69.1 s for the long game). The worker then spent about
165 s verifying the long game and entered PPO for 2680 samples. PPO exceeded the update phase
budget (worker start + 3540 s): `UpdateRolledBack: TimeoutError`; model, Adam and all RNG state
were restored, no checkpoint was published, and the campaign stopped (fail closed). Caveat: this
run shared the host with my profiling (one extra busy core most of the time), so it is not a clean
baseline time; the clean baseline PPO time of the same verified batch is measured optimizer-only
(A9). A py-spy sample 35 minutes into this PPO: backward 48%, prefix reconstruction 35%, loss
forward 12%. Prefix reconstruction replays every earlier frame of an episode under the current
weights for every minibatch, so its cost grows with the square of the episode length.

### A7. Long-game verification profile (episode 4 of A6, 2525 decisions, read-only re-run)

strict JSON parse of the 112 MB rollout 5.0 s; canonical re-serialization 2.3 s (twice in the
real check); `load_builtin_episode` 9.9 s; Python history check 12.5 s; policy reconstruction 21.1 s
(warm repeat 21.9 s); `verify_episode` 62.1 s (its own history check 10.6 s, JSONL record parsing
16 s, wire cross-check 10.7 s, report writes 3 s, waiting on `vcmireplay` about 26 s); behaviour
report 14.4 s. All of it ran serially in one process.

### A8. Experiment PPO on the A6 batch (optimizer-only, candidate e33fa7c, 15:04 UTC)

820.5 s in total: prefix reconstruction 689.9 s (84%), loss forward 48.1, backward 62.7, behaviour
verification 15.9. So on long games the experiment alone leaves prefix reconstruction as the
bottleneck (H4). Note: the frames of that episode number 16044, of which 15302 have semantics
and only 4162 are distinct encodings; one full replay took 11.0 s, 1.8 s of it recomputing every
frame's memo key.

### A9. Exact prefix keys and exact prefix helpers (candidate 097f7b9)

Hypothesis: (a) the memo keys of an unchanged history can be computed once per update; (b) frame
encodings under fixed weights are independent pure functions, so forked helpers can compute
disjoint shares bit for bit while the parent keeps every check and every recurrent step in order.
Microbenchmark on the A6 episode (full replay; one other core busy): serial 11.0 s; with three
helpers 4.0-5.5 s (encodings 3.1-4.0 s, chain 1.1-1.4 s); with two 3.5-7.2 s; every state bitwise
equal to the serial result. Safety: the parent recomputes the first encoding of every helper's
share and requires identical bits; helpers keep only their socket and standard streams, never
touch the optimizer and are reaped when the update ends, whatever its outcome. Tests:
`tests/test_prefix_pool.py`.

### A10. Overlapped per-game verification (candidate f55703f)

Hypothesis: native replay (`verify_episode`, unchanged) is independent of the worker's rollout,
history and policy checks, its own history check is independent of the `vcmireplay` run, and the
behaviour report is a pure function of the finalized capture files. Change: capture files are bound
first, the native replay runs in a forked child (result or exception returned exactly) while the
parent checks rollout, history and policy; inside `verify_episode` the history check runs in a
thread while the main thread waits on `vcmireplay` (a history error is still the error reported);
the behaviour report runs in a forked child and is written only after verification passes. No
check was removed. Tests: `tests/test_parallel_verification.py` plus the existing 110 verification
tests.

### A11. Back-to-back identity checks and the canonical prepare check (candidate 373b299)

Two kinds of identity check were exact repeats with nothing in between: the benchmark worker's
plan reader and `builtin_training.worker`'s first action; and a passed attempt's final check and
the next attempt's opening check. Each pair keeps one check. `tools/prepare_vcmi.py --check` now
runs while the benchmark builds the plan; no plan is written unless it passed. Tests extended in
`tests/test_builtin_training.py` and `tests/test_builtin_first_run.py`.

### A12. First full candidate smoke fails closed on an import path, then passes (15:30 and 15:36 UTC)

`func-XH-easy-1` (candidate 373b299, experiment plus three helpers, sequence 9000, seat 0) failed
closed in the first game: `contextual_env` imports `contextual_evidence` as a top-level module, so
the new relative `digest_scope` import raised `ImportError`; no update was attempted. Fix 23c1922:
absolute import with a direct-hash fallback (same digest) in `contextual_evidence` and `scoring`,
plus a subprocess test of both load paths. The unit tests had not caught it because they import the
package form only. `func-XH-easy-2` (23c1922, CPUs 1 to 3 only while the A13 baseline ran on CPU 0)
then passed: the same 340 samples, decisions 26/36/232/46 and outcomes as A1's baseline batch on the
same seeds; compute cycle 53.0 s, durable 55.1 s, PPO 17.4 s, process-tree CPU 76.3 s. Not a timed
comparison (pinned, shared host).

### A13. Baseline PPO on the long A6 batch, optimizer-only (15:22 to 16:25 UTC)

B (3cb6a78) restored relocated 412 and updated on the verified A6 batch (2680 samples; histories of
33, 422, 637 and 16044 frames), pinned to CPU 0; CPUs 1 to 3 ran single-thread correctness jobs and
the A12 smoke part of the time. PPO took 3779 s: behaviour verification 275, prefix reconstruction
1646, loss forward 623, backward 1218, Adam 12.6. This exceeds the whole update phase budget (3540 s
from worker start, of which this batch's games and verification had already used about 240 s), so
the baseline cannot complete a batch containing a game of this length on this host, contended or
not; A6's failure was not caused by contention. (A6 refers to this measurement as A9; it is A13.)
The experiment (A8) took 820.5 s on the same batch; its update differs from the baseline update by
2.08e-4 of the update's L2 norm.

### A14. Exactness of the exact paths, optimizer-only

Same restored state, same verified batch, single repetitions unless noted; digests cover model,
gradients, Adam state, minibatch generator, the sampled window order and the PPO report.

- A1 batch (340 samples): B 261.5 s; C (candidate, no helpers) 227.5 s; CH (three helpers) 217.6 s.
  C and CH are bitwise equal to B in every digest. X 18.4 s and XH 23.5 s are bitwise equal to each
  other (XH ran with four processes on one CPU) and differ from B by 4.45e-4 of the update's L2 norm.
- 411 to 412 (relocated 411 and the archived verified 412 batch, 19 samples, two repetitions each):
  B 8.0/8.3 s, C 7.0/7.1 s, CH 7.7/7.6 s, all bitwise equal to each other and identical across
  repetitions. All three differ from the archived 412 model and Adam state (the minibatch generator
  matches): the archived run used Torch `2.9.1+cpu` on another host, this host uses `2.9.1+cu128` on
  CPU, so this is cross-build drift present in the baseline itself (measured with the same bench: L2
  difference 1.05e-3 of the update, largest element difference 7.1e-5). XH differs from B by at most
  6.6e-6 per element.
- Native runtimes: `runtime/base` and `runtime/cand-0` are byte-identical builds (same SHA-256 of
  every binary and library); phase 2 changed no native code.

### A15. Frozen candidate and measurement set (16:26 UTC)

Candidate frozen at a9b90d3 (`/home/claude/phase2`, clean); baseline 3cb6a78 (`/home/claude/p2base`,
clean). `tools/acceleration2/run_rounds.py` runs the schedule strictly one campaign at a time and
refuses a dirty or moved checkout. Easy workload, two rounds, sequences 10 to 13 then 20 to 23, seats
0, 1, 0, 1, order B XH CH X then X CH XH B. Nothing else heavy runs during the set; a read-only code
review (file reads only) runs alongside.

### A16. Pilot easy pair on a9b90d3 (16:25 to 16:36 UTC, `e1-B`, `e1-XH`, sequences 10 to 13)

The first two campaigns of the A15 set ran before I stopped the schedule for A17; they are kept as a
pilot pair, not as part of the final measurement set. All eight batches passed, and both variants
played identical games in every batch (samples 83, 80, 196, 89; the same decisions and outcomes),
including batches 2 to 4, whose parents already differ by the experiment's update difference.

| batch (seat) | samples | B compute / durable (s) | XH compute / durable (s) | B PPO | XH PPO |
| --- | --- | --- | --- | --- | --- |
| 10 (0), cold | 83 | 90.6 / 91.9 | 30.4 / 31.7 | 51.3 | 3.6 |
| 11 (1) | 80 | 92.1 / 93.3 | 30.1 / 31.3 | 55.2 | 3.5 |
| 12 (0) | 196 | 190.3 / 192.0 | 40.5 / 42.1 | 148.7 | 12.0 |
| 13 (1) | 89 | 89.6 / 91.0 | 32.6 / 34.0 | 53.8 | 4.5 |

Campaign launch wall time 476.7 s against 146.9 s; process-tree CPU 473.5 s against 172.2 s. Outside
PPO the candidate spends about 27 s per batch (B about 38 s): worker start 5 s, prepare 2 s, gaps
between games 2 s each, games 3 to 5 s, publication 1.3 s.

### A17. Independent review and hardening (15:47 to 16:50 UTC)

A read-only review workflow (five reviewers: fork safety, PPO exactness, check coverage,
verification overlap, test adequacy; then one skeptical verifier per dimension) reported 23
findings; 12 survived verification, all rated low (every one fails closed), 11 were refuted. The
measurement set was stopped and the candidate changed:

- Behaviour verification in PPO (A2) ran on the producer's persistent inference memo, so it was no
  longer independent of the collector's cached state (refuted as a defect, since the memo is exact by
  construction, but the baseline had one memo-free recomputation and A2 removed it). It now starts
  from empty memos and discards them, so it still recomputes every stored probability and value from
  the parameters. A test poisons every entry the producer left and shows the old path fails it.
- Background checks and prefix helpers defer SIGALRM, SIGINT and SIGTERM while a child is forked and
  recorded and while it is stopped; a check is started inside the caller's `try`, and a stop that is
  interrupted kills and reaps at once before the interrupt is raised. Fork failures close their
  pipes.
- The history check that overlaps the native verifier runs on a daemon thread; an interrupt during
  the native run no longer waits for it (in the serial order the alarm interrupted it directly).
- The capture is bound before the native fork only from a manifest equal to the fresh capture's;
  otherwise the persisted checks run first, so a tampered capture reports the serial order's error.
- The benchmark worker's reader requires the benchmark scope and takes the experiment and helper
  choices through the same function the identity check uses.
- Prefix helpers: the parent now recomputes the first and one unpredictable later encoding of every
  share (tests poison only later rows).
- New tests for each item, for overlapped failure combinations, for the analysis child's failure
  paths, and an overlap test that fails under the serial order.

### A18. Final measurement set (from 16:56 UTC)

Candidate refrozen at a9304eb after a passing functional smoke (`func-XH-easy-3`, the A1 batch: same
340 samples, compute 46.7 s, durable 48.6 s). Baseline 3cb6a78 unchanged. Schedule
(`schedules/v2.json` in the private root, run by `tools/acceleration2/run_rounds.py`, one campaign at
a time, nothing else heavy on the host):

- easy round a, sequences 30 to 33: B, XH, CH, X; easy round b, sequences 40 to 43: X, CH, XH, B;
- publication emulation, sequences 60 to 63: XH and B with an 18.3 s delay before each push, then B
  and XH with 4.5 s;
- original workload, sequences 0 to 3 (the A6 batch first): XH, CH, then B, with a four-hour window.

The first launch of `a-B-easy` was stopped during its setup when I moved the driver out of the tool
session (a two-hour limit there would have killed a running campaign); the partial directory is kept
and the campaign ran as `a-B-easy2`.
