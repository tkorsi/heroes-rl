# Original-map local runner acceleration (overnight scientific log)

Assignment: `docs/operations/claude-original5000-overnight-20261026.txt` (main b4fe70c), with
`docs/operations/claude-acceleration-phase2-20261026.txt` as the correctness and benchmark contract.
Hard stop 2026-10-27T11:00:00Z; heavy work stops 10:50 UTC. Host: Linux x86-64, 4 vCPUs (one thread
per core), 16 GB, Python 3.13, Torch 2.9.1+cu128 on CPU. Nothing here is a Mac measurement.
Integration guide: `docs/research/original5000-local-acceleration-integration-2026-10-26.md`.
Live status: `reports/cloud/claude-original5000-status.json`.

## Prospective protocol (recorded 2026-10-26 18:40 UTC, before any local-runner measurement)

### Deadline watchdog

`tools/acceleration3/overnight/watchdog.py` (started 18:16:32 UTC, pid 27780) recorded both
deadlines once, with their CLOCK_MONOTONIC equivalents and the boot id, and refuses any restart that
would move them. Benchmark roots are registered by pid and kernel start time; at 10:50 UTC it
terminates only those roots, their descendants and their process groups, verifies that none remain
and writes a stop receipt. It was tested on a TERM-ignoring owned tree (including a grandchild in
another session), an unowned process (untouched) and a stale registration (ignored).

### What is measured

- Path: main's manual local runner (`tools/run_local_training.py`, `local_script_training.py`), the
  real per-batch sequence: parent restore with source compatibility, plan freeze, guarded
  supervisor and worker, four games with native collection, history, native replay and policy
  reconstruction, admission, PPO/Adam, checkpoint round trip, local preservation, independent
  restore of the preserved copy, `latest-checkpoint.json`, evaluation batches, final restore. No
  per-batch publication (the runner has none).
- Baseline ("accepted local cold path"): main (a8ac411 after the final 629 synchronization; its
  runner files are byte-identical to b4fe70c) with a cold runtime built here from main's engine
  recipe (`tools/prepare_vcmi.py --check` passes; same CMake options as the earlier build).
- Candidate: the same runner with the fast path selected explicitly; exact is the default
  candidate, the reordered-gradient experiment is a separate explicit opt-in, reported separately.
- Benchmark port (identical for baseline and candidate): Linux machine description, a benchmark
  seed namespace and ledger (never 20M/30M or 40M/50M), disposable private roots, the Linux
  Torch relocation (validated: 629 and 412 restore exactly; only the recorded torch version
  differs), bounded `--max-updates`.
- Parents: update 629 (the next manual run's parent; disposable copies of the published package
  3f50750) for the primary comparison; 412 copies for continuity with phase 2.
- Workload: original hero-management-v2 (ccfc8467...), both seats, 365 days, 5120 choices,
  Nullkiller Easy, BattleAI, unchanged rewards, architecture and PPO.

### Design

- Paired seeds: baseline and candidate campaigns of one block use the same benchmark sequence
  numbers and parent, so their first batch plays the same seeds; exact candidates must produce
  bitwise-identical checkpoints whenever the games are identical, the experiment diverges after its
  first update. Order alternates between blocks (B C X, then X C B). Campaigns run one at a time.
- Per batch: intent to verified preserved checkpoint (compute) from the runner's own records;
  cold start = launch to the first training batch (baseline evaluations included); process-tree
  CPU seconds; peak RSS, descriptors and threads of every owned process (/proc sampler).
- Reported: per-batch distributions, medians and tails per variant; ratio of medians and paired
  per-batch ratios; verified updates per hour; projected wall time for 1,250 updates and 112
  evaluation games with stated assumptions; memory behaviour across a campaign.

### Hypotheses

- H5. On the original map a batch's cost is dominated by PPO when a game is long (prefix
  reconstruction grows with the square of episode length; loss and backward linearly), and by
  per-game verification and fixed per-batch work otherwise. The local runner adds per-batch fixed
  work that Teacher's driver did not have (several `restore_parent` calls, each with a source
  compatibility check that runs `git show` per learner file).
- H6. The exact fast path reduces long-batch PPO by about 1.5x (A13/A14 of the phase-2 log) and the
  whole cycle by less; the opt-in experiment reduces long-batch PPO by about 4.6x (A8). Whether
  exact reaches 2x on the whole original-map cycle depends on the share of fixed and verification
  work; we expect it does not without further reductions.
- H7. The accepted path cannot complete a batch whose PPO exceeds the update budget (3540 s from
  worker start); on this host that happens near 2,500 decisions in one game (A6/A13). A single
  such batch stops the 5,000-game run.

## Attempts and observations

### O1. Synchronization and integration branch (18:13 to 18:50 UTC)

Main moved b4fe70c → b858a72 → a8ac411 during the evening; the later commits add only documents and
validation records (the 629 handoff and the full-archive receipt), so the runner and learner files
are unchanged. The integration branch merges the PR30 Python and phase-2 exact library changes into
main and restores main's `engine/` tree byte for byte (the NativeSession opponent and Python module,
patches 0030 to 0032, and the native capture-skip change are left out), so the engine recipe equals
the accepted one and no new runtime acceptance is needed. A read-only analysis of PR30 found no
Python change on the cold path that depends on PR30's native code (no new option, variable, wire
field or native output); only the native time saving is lost. `builtin_replay` imports
`embedded_evidence` unconditionally, so that module stays.

The main finding of the same analysis: main's runner would refuse the 629 parent under any changed
learner file, because `restore_parent` requires every learner/collector file recorded in the parent
(`source_sha256`, 33 files at 3268f2f) to be byte-identical, apart from two reviewed adapter and
routing transitions. Twelve files differ in the fast path. Integration therefore needs an explicit,
pinned, reviewed source transition (O4).

### O2. Linux port of the manual runner (18:46 to 18:53 UTC)

Main's preparation pins the Mac runtime path, `.dylib` names and hashes, Mac acceptance receipts,
Torch `2.9.1` or `2.9.1+cpu`, the 412 input bytes and the official seed namespaces. The benchmark
port (`tools/acceleration3/bench_site/sitecustomize.py`, loaded only through PYTHONPATH and only when
`HEROES_LOCAL_BENCH_SPEC` names a specification) leaves every source file byte-identical and changes
in memory exactly: the Torch build string the loader sees while reading a checkpoint written by the
CPU wheel (payload and every other check unchanged); the machine description check (this host's
pinned runtime, parent bytes, map, data and prerequisites, then the same exact parent restore); and
the original profile's seed namespace (benchmark 60M/70M). `tools/acceleration3/run_local_bench.py`
writes the machine description and specification per campaign and runs `run_local_training.py
start` from a clean checkout, as `train-local.sh` does. Main's offline check (`run_local_training.py
check`) then passed on this host with an exact 629 restore (model, Adam, all RNG, progress, pool).

First launch failed closed in the first game's controller: `python -m
experiments.learner.local_script_training` runs the adapter as `__main__`, whose top-level code and
`main()` run before a post-import patch could apply. Fix: such an entry imports the module under its
own name (patched) and calls `main()`, which is all its `__main__` block does.

### O3. First baseline smoke from 629 and a failure I caused (18:53 to 19:08 UTC)

Main's accepted path played the eight fixed baseline evaluation games from 629 (seeds 710001-4 and
720001-4, both seats): 301, 1315, 896 and 536 choices for seat 0 (28, 150, 84 and 53 s per game
including verification), then 200, 238 and 4464 choices for seat 1 (19, 25 s, and 256 s of
collection alone for the long game). 629 plays long original-map games: one game in eight reached
4464 of the 5120 choices.

At 19:05 I deleted the baseline build's object folder to free disk space. The copied runtime's
binaries carry an ELF RUNPATH that names that folder, so the next native replay could not load
`libvcmi.so`: "Independent native replay failed" for the 4464-choice game; the batch and the run
stopped without admitting anything (correct fail-closed behaviour). The concurrently started
candidate smoke failed the same way in its first game. Fix: the runtime copy is self-contained
(Mods and config dereferenced; `LD_LIBRARY_PATH` names the hashed copy) and the build path is a
link to the same files. Both smokes were rerun (s4-B, s4-C). These two runs are excluded from the
measurements; their records are kept.

### O4. The opt-in fast path in the manual runner (19:00 to 19:05 UTC, commit 7c987b6)

- Default unchanged: `run_local_training.py start` now releases `ACCEPTED_IMPLEMENTATION` (a8ac411)
  unless `--fast-path` is given, so `./train-original-5000.sh` runs main's accepted code byte for
  byte even after this branch is merged. The accepted commit must be an ancestor of the checkout.
- `--fast-path` releases the current commit. Its adapter continues an accepted parent through
  `FAST_PATH_SOURCES`: for each of the twelve changed learner/collector files, the accepted digest
  it may replace and the exact digest it must have (generated and checked by
  `tools/acceleration3/fast_path_pins.py`; any other change still fails). The restore proof lists
  every such file and sets `policy_ppo_unchanged` to false (`reviewed_exact_fast_path`). The adapter
  itself crosses through the existing reviewed AST transition, extended to the a8ac411 adapter.
- Execution choices are bound in the session and every plan (`execution`): overlapped or serial
  verification, exact prefix helpers (0 to 8), and the separately reported `shared_graph_encodings`
  experiment, which is also written into every checkpoint's metadata. They take effect only after
  the worker's identity check. No environment variable selects them.
- Serial fallback: `background_check.FORK = False` forks nothing and runs each step in the worker
  when its result is requested, which is the order of main's code. It is the default on macOS
  (fork safety with Torch loaded is unverified there); `--overlap-verification` opts in.
- Each local identity check opens one digest scope (an unchanged file is hashed once per check);
  the worker's plan reader no longer repeats the identity check that `builtin_training.worker` runs
  as its first action.
- Benchmark-only phase-2 and PR30 harnesses and records (bench_training, embedded probes and
  records, tools/acceleration and acceleration2) are left out of this tree; they remain on their
  branches.

Tests: `tests/test_local_fast_path.py` (pins equal the tree and the accepted commits; accepted 3268f2f
and a8ac411 parents continue; a fast parent continues without transition; unreviewed bytes on
either side, an unlisted file and a tampered parent digest are rejected; only known execution
choices; choices take effect only from a plan; the serial fallback never forks and returns the same
value and error as the fork; default start releases the accepted commit, fast options require
`--fast-path`, macOS defaults to serial). The 674 related existing tests pass (three modules that
import sibling test modules by top-level name were rerun from `tests/`); the nine
`test_prepare_local_training` tests fail on this host on main too (they require Torch `2.9.1` or
`2.9.1+cpu`).
