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
  experiments), and the runner prints the choice before the first game.

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
| `--fast-path --prefix-helpers N` | also compute PPO prefix frame encodings in N forked helpers (0 to 8), bit for bit |
| `--fast-path --experiment shared_graph_encodings` | NUMERICAL experiment: reordered gradient summation; reported separately; never a default |

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
- A fast-path checkpoint can be continued only by the fast path. The pinned accepted release has no
  reverse transition and refuses it. So once the official run starts with `--fast-path`, the in-lineage
  fallback is `--fast-path --serial-verification --prefix-helpers 0`, which executes every
  verification step serially in the worker in main's order, with no fork. Returning to the accepted
  path means restarting from an accepted checkpoint (for example the original 629 parent).
- A run with `--experiment` writes `numerical_experiments` into every checkpoint's metadata, and a
  continuation must request the same experiment set; an exact run refuses an experimental parent.

## Validation commands

On any host with the project's Python (no game, no Torch build requirement beyond the locked one):

```sh
python tools/acceleration3/fast_path_pins.py            # "pins match": pins equal the tree and a8ac411/3268f2f
python -m unittest tests.test_local_fast_path tests.test_local_script_training
(cd tests && python -m unittest test_prefix_pool test_parallel_verification test_ppo_phase2_acceleration)
python tools/prepare_vcmi.py --check
```

On the Mac, bounded, from a clean checkout of this branch, while the manual run is NOT active
(the tool holds the official owner lock and refuses otherwise; it writes only into
`~/Projects/Heroes-bench/...`, uses disposable copies of the final 629 checkpoint and the benchmark
seed namespace 60M/70M, and never touches the official lineage):

```sh
"$HOME/Projects/Heroes/.venv/bin/python" tools/acceleration3/mac_acceptance.py --dry-run
"$HOME/Projects/Heroes/.venv/bin/python" tools/acceleration3/mac_acceptance.py --order AF --updates 1 --hours 4
```

`AF` (about MAC_AF_ESTIMATE on this Linux host's numbers; the Mac is unmeasured) plays the eight
fixed baseline games and one training batch with the accepted implementation and then with the
macOS fast-path default, from the same parent and seeds. It exits 0 only if both campaigns ended
cleanly (every requested batch admitted, no bound reached, nothing interrupted or left running, the
expected commit released, the checkout unchanged) and every exact campaign had at least one
training batch with identical games from the same parent as A, each with bitwise-identical learner
state; `summary.json` in the bench folder lists every problem and has per-batch times, game
lengths, PPO times, peak memory and process counts. Games are compared on every observation and
decision record. The engine delivers some notices asynchronously, so a faster learner can see two
of them in the other order (once in 13 paired games on Linux, at the end of a game); such a pair is
reported as not comparable rather than as a failure of exactness, and the check needs a pair with
identical games (repeat with `--order AFF` if the first pair differs). If the A campaign's training
batch is stopped by the update wall limit (on this Linux host the accepted path could not finish
that batch, sequence 266 from 629, within the fixed 3,540 s worker budget), the summary reports it
as a problem: it means the accepted implementation cannot run that batch on the Mac either, which
the official run would meet as a failed batch and a stop. Optional:
`--order AOOA` tests forked verification on macOS (fork safety with Torch loaded is unverified
there), `--order AHHA` adds two prefix helpers, `--order AFFA` gives a counterbalanced timing pair,
`X` adds the numerical experiment (reported apart). Ctrl-C stops after the current batch as with
`train-local.sh`; every runner also has its own `--hours` hard stop.

## Measured on Linux (not a Mac estimate)

See the log, sections O5 onward, and the validation record. Summary for the review: RESULTS_SUMMARY.

## Rollback

Do not pass `--fast-path`. To remove the code path entirely, revert the merge; checkpoints written
by the accepted path are unaffected either way.
