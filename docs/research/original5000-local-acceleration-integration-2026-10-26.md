# Integration guide: opt-in exact fast path for the manual local runner

Branch `claude/project-thread-iba9jw-original5000`; scientific log
`docs/research/original5000-local-acceleration-2026-10-26.md`; Linux results
`validation/original5000-local-acceleration-linux-2026-10-27.json`. Review it as a normal pull request;
nothing here was merged or launched on the Mac.

## What changes for the user

Nothing, unless `--fast-path` is given.

- `./train-original-5000.sh` (and `train-local.sh` without new options) releases
  `ACCEPTED_IMPLEMENTATION = a8ac411` as its frozen checkout, so the default run executes main's
  accepted runner, adapter, learner and collector byte for byte, even after this branch is merged.
  `start` refuses if that commit is not an ancestor of the checkout. The fast-path options are
  rejected without `--fast-path`.
- `./train-original-5000.sh --fast-path` releases the current commit. Its session and every batch
  plan record `execution` (implementation `exact-fast-path-v1`, verification mode, prefix helpers,
  text-leaf replay, replay threads, experiments), and the runner prints the choice before the first
  game.

Design consequence for review: once merged, the default no longer follows main's HEAD. A later
reviewed change to the default learner has to move `ACCEPTED_IMPLEMENTATION` explicitly. This is
deliberate (the merged fast-path release refuses to run without `--fast-path`, so HEAD can never
become the default by accident), but it changes how future learner work reaches the default.

## Options

| command | meaning |
| --- | --- |
| (none) | accepted implementation a8ac411, unchanged |
| `--fast-path` | exact fast path; on macOS serial verification and no prefix helpers |
| `--fast-path --overlap-verification` | also overlap per-game native replay and behaviour report in forked children (Linux default) |
| `--fast-path --prefix-helpers N` | also compute PPO prefix frame encodings in N forked helpers (0 to 8), bit for bit, and (by default) replay text leaves on N threads during the backward |
| `--fast-path --replay-threads N` | replay text leaves on N threads (0 to 8) during the backward, bit for bit; threads only, no fork (default: as many as `--prefix-helpers`) |
| `--fast-path --no-text-replay` | compute every text leaf occurrence separately, as the accepted learner does (bit-identical; the in-lineage fallback if the replay ever refuses) |
| `--fast-path --experiment shared_graph_encodings` | NUMERICAL experiment: reordered gradient summation; reported separately; never a default |

What the exact fast path changes, all bit for bit: per-game verification without repeated work and
optionally overlapped; PPO prefix states reconstructed with a call-local node memo and optionally in
forked helpers; and the replay of repeated text leaves. Within one minibatch graph, each distinct
text leaf (`text_gru(byte(...))`) is computed once, and every occurrence replays that leaf's own
backward graph with its own incoming gradient, returning each contribution to the five leaf
parameters separately, in the reference order (log O13 to O16). With replay threads
(`--replay-threads N`, by default as many as `--prefix-helpers`), the replays of the minibatch's one
bracketed backward run on that many threads while the backward continues,
and their contributions are added in the same order, and installed as the gradients, before
gradient clipping and the optimizer step. The bracket defers only onto trainable leaf parameters
without gradients or hooks, and only the occurrences that its own backward reaches (identified by
the autograd graph task, recorded on a fresh view of the loss that nothing else reaches, and only
on the bracket's intra-op thread count, which every replay checks); any other backward, including
one nested in a hook or run on another thread, replays inside the autograd engine. The replay refuses
(raises; the update is rolled back) on anything outside its exactness argument: a leaf parameter
replaced, frozen or changed by an autograd-visible in-place operation inside the minibatch, a
contribution count that differs from its graph, with replay threads a gradient to a leaf parameter
from elsewhere (a nested `.backward()` included), a parameter frozen or hooked during the backward,
or an intra-op thread count changed during it after an occurrence was queued, a failed replay or interrupted backward (no further deferral in that minibatch), or a
higher-order or second backward. It keeps the reference path, silently and exactly, for tied or
frozen leaf parameters, autocast, anomaly mode, torch function or dispatch modes, module, global
module or saved-tensor hooks. Two changes are not detected and are outside the exactness argument
(neither occurs in the training loop): an in-place write through `.data` (it bypasses the version
counter) and a hook registered on a leaf parameter's gradient accumulator node.

The exact options produce bitwise-identical games, PPO updates, Adam state, RNG streams and
checkpoint weights to the accepted implementation (tests and paired Linux campaigns below). The
experiment does not: its updates differ from the exact update by about 2e-4 of the update norm per
batch and the lineages diverge after the first update. It is an opt-in candidate pending manual
review, not a substitute for the exact learner in the official lineage.

## Lineage rules (fail closed)

- An accepted parent (629 at 3268f2f, or any a8ac411 checkpoint) continues under `--fast-path`
  through the reviewed, pinned source transition `FAST_PATH_SOURCES` (each of twelve learner and
  collector files: the accepted digest it may replace and the exact fast-path digest it must have;
  any other byte difference refuses). The restore proof lists every replaced file.
- Every other file the run executes (all of `experiments/`, the admission verifiers in
  `validation/probes`, the learner lock, the runner, its preparation, `prepare_vcmi.py`,
  `train-local.sh`) must equal the a8ac411 blob or its `FAST_PATH_TREE` pin, checked on every
  restore; untracked Python files in those folders refuse too. `fast_path_pins.py` regenerates the
  pins after a reviewed change and otherwise reports any difference.
- A fast-path checkpoint can be continued only by the fast path, and only by a release whose learner
  digests, adapter and tree pins equal those of the release that wrote it: there is no reviewed
  fast-to-fast transition list, so a checkpoint written by one fast-path commit is refused by a later
  one that changed any pinned file ("Policy, PPO or collector implementation changed" or "Unreviewed
  historical local adapter"). `start` always releases the checkout's HEAD, so keep HEAD at the same
  commit for the whole fast-path lineage. Adopting a later fix means restarting from an accepted
  checkpoint (for example the original 629 parent) or adding a reviewed transition.
- The pinned accepted release has no reverse transition and refuses a fast-path checkpoint. So once
  the official run starts with `--fast-path`, the in-lineage fallback (same commit) is
  `--fast-path --serial-verification --prefix-helpers 0 --no-text-replay`, which executes every
  verification step serially in the worker in main's order, with no fork, and computes every text
  leaf separately. Continue the lineage with `train-local.sh`, not `train-original-5000.sh`: the
  latter always restores the finalized easy parent (`--parent-final`), so it would start a new
  lineage from update 629 instead of continuing. The continuation restores the run root's
  `latest-checkpoint.json` (or an explicit `--parent`):

  ```sh
  ./train-local.sh --profile original --fast-path --serial-verification --prefix-helpers 0 \
      --no-text-replay --games REMAINING_GAMES --hours 48 --eval-every 100
  ```

  A continuation is a new run: it plays its own eight baseline evaluation games first and counts
  `--eval-every` from its own first update.
- A run with `--experiment` writes `numerical_experiments` into every checkpoint's metadata, and a
  continuation must request the same experiment set; an exact run refuses an experimental parent.

## Validation commands

On any host with the project's Python (no game, no Torch build requirement beyond the locked one):

```sh
python tools/acceleration3/fast_path_pins.py            # "pins match": pins equal the tree and a8ac411/3268f2f
python -m unittest tests.test_local_fast_path tests.test_local_script_training tests.test_replayed_text_leaves \
    tests.test_prefix_pool tests.test_parallel_verification tests.test_ppo_phase2_acceleration
(cd tools/acceleration3 && python -m unittest test_mac_acceptance test_bench_tools)
python tools/prepare_vcmi.py --check
```

Run them from the repository root (the test modules import `experiments` and `tests` as packages).

On the Mac, bounded, from a clean checkout of this branch, while the manual run is NOT active
(the tool holds the official owner lock and refuses otherwise; it writes only into
`~/Projects/Heroes-bench/...`, uses disposable copies of the final 629 checkpoint and the benchmark
seed namespace 60M/70M, and never touches the official lineage):

```sh
"$HOME/Projects/Heroes/.venv/bin/python" tools/acceleration3/mac_acceptance.py --dry-run
"$HOME/Projects/Heroes/.venv/bin/python" tools/acceleration3/mac_acceptance.py --order AF --updates 1 --hours 4
```

`AF` (about two hours on this Linux host's numbers, 4,412 s for A and 2,366 s for F; the Mac is unmeasured) plays the eight
fixed baseline games and one training batch with the accepted implementation and then with the
macOS fast-path default, from the same parent and seeds. It exits 0 only if both campaigns ended
cleanly (every requested batch admitted, no bound reached, nothing interrupted or left running, the
expected commit released with the execution choices its letter names, the checkout unchanged) and
every exact campaign had at least one
training batch with identical games from the same parent as A, each with bitwise-identical learner
state; `summary.json` in the bench folder lists every problem and has per-batch times, game
lengths, PPO times, peak memory and process counts. Games are compared on every observation and
decision record. The engine delivers some notices asynchronously, so two runs with the same seeds
can see two of them in the other order even on the accepted implementation (on Linux, two accepted
runs differed in one of twelve games, the same game in which accepted and fast path once differed);
such a batch is reported as not comparable rather than as a failure of exactness, and the check
needs one batch with identical games (repeat with `--order AFF` if the first pair differs). If the A campaign's training
batch is stopped by the update wall limit (on this Linux host the accepted path could not finish
that batch, sequence 266 from 629, within the fixed 3,540 s worker budget), the summary reports it
as a problem: it means the accepted implementation cannot run that batch on the Mac either, which
the official run would meet as a failed batch and a stop. Optional:
`--order AOOA` tests forked verification on macOS (fork safety with Torch loaded is unverified
there), `--order AHHA` adds two prefix helpers (and two replay threads), `--order AFFA` gives a
counterbalanced timing pair, `N` checks the `--no-text-replay` fallback, `T` adds two replay
threads without any fork (the candidate macOS setting if forked helpers are not wanted), `X` adds the numerical
experiment (reported apart). Ctrl-C stops after the current batch as with
`train-local.sh`; every runner also has its own `--hours` hard stop.

## Measured on Linux (not a Mac estimate)

See the log, sections O5 onward, and the validation record
(`validation/original5000-local-acceleration-linux-2026-10-27.json`). Summary for the review, all on
the original map from the final 629 parent, every campaign alone on the same three CPUs:

- The accepted implementation (a8ac411) cannot complete training batch 266: its update stops at the
  3,540 s budget after 83 of 125 optimizer steps (batch 3,523 s). Its complete length is estimated at
  about 4,600 to 4,900 s; every ratio against it is a lower bound.
- The exact fast path with prefix helpers and deferred replay (480aa1a, which computes the same values
  in the same order as the final candidate 5635c7c) completed batch 266 in 1,248 s, at least 2.8x the
  censored accepted batch (about 3.7 to 3.9x against its estimated length). Over six consecutive
  training batches (266 to 271) its mean was 902 s (median 761 s, slowest 1,810 s against the 3,540 s
  budget), about 4.0 verified updates per hour; cold start 361 s, evaluation batches 115 and 222 s.
  Projection for 1,250 updates and 26 evaluation batches: about 314 hours on this host. Peak over
  the six batches: 8.5 GB RSS (during the largest batch, 5,831 decisions; memory follows the batch
  size and falls back after it), 45 descriptors, 21 threads, 10 processes, with no descriptor or
  thread growth.
- Against PR30 (4d7688b): its optimizer has no multi-CPU option, so the like-for-like single-CPU
  optimizer-only comparison is 1.76 to 1.85x on batch 266 and 1.67x on batch 267; on the same three
  CPUs (PR30 leaves two idle during PPO) the fast path's batch 266 is about 3x shorter.
- Exactness: identical games give update 630 `fe1378ef...` and 631 `46439a9b...` with the accepted
  path, PR30, the fast path, every replay candidate and the `--no-text-replay` fallback.
- The numerical experiment `shared_graph_encodings` (reordered gradient summation, never a default)
  ran batch 266 in 477 s; it is reported apart and is not part of the exact figures.

These are Linux numbers; the Mac has other CPUs, another BLAS and no measured fork safety for the
helpers, so validate there with the bounded command above before choosing any default.

## Rollback

Do not pass `--fast-path`. To remove the code path entirely, revert the merge; checkpoints written
by the accepted path are unaffected either way.
