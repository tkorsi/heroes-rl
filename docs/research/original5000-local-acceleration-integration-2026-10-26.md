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
  text-leaf replay, experiments), and the runner prints the choice before the first game.

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
| `--fast-path --prefix-helpers N` | also compute PPO prefix frame encodings in N forked helpers (0 to 8), bit for bit, and replay text leaves on N threads during the backward |
| `--fast-path --no-text-replay` | compute every text leaf occurrence separately, as the accepted learner does (bit-identical; the in-lineage fallback if the replay ever refuses) |
| `--fast-path --experiment shared_graph_encodings` | NUMERICAL experiment: reordered gradient summation; reported separately; never a default |

What the exact fast path changes, all bit for bit: per-game verification without repeated work and
optionally overlapped; PPO prefix states reconstructed with a call-local node memo and optionally in
forked helpers; and the replay of repeated text leaves. Within one minibatch graph, each distinct
text leaf (`text_gru(byte(...))`) is computed once, and every occurrence replays that leaf's own
backward graph with its own incoming gradient, returning each contribution to the five leaf
parameters separately, in the reference order (log O13 to O15). With helpers declared, the replays
of the minibatch's one bracketed backward run on that many threads while the backward continues,
and their contributions are added in the same order, and installed as the gradients, before
gradient clipping and the optimizer step. The bracket defers only onto trainable leaf parameters
without gradients or hooks; any other backward replays inside the autograd engine. The replay
refuses (raises; the update is rolled back) on anything outside its exactness argument: a leaf
parameter replaced, frozen or changed inside the minibatch (in place or through `.data`), a
contribution count that differs from its graph, with replay threads a gradient to a leaf parameter
from elsewhere or a parameter frozen or hooked during the backward, a failed replay or interrupted
backward (no further deferral in that minibatch), or a higher-order or second backward. It keeps
the reference path, silently and exactly, for tied or frozen leaf parameters, autocast, anomaly
mode, module, global module or saved-tensor hooks.

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

`AF` (about MAC_AF_ESTIMATE on this Linux host's numbers; the Mac is unmeasured) plays the eight
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
counterbalanced timing pair, `N` checks the `--no-text-replay` fallback, `X` adds the numerical
experiment (reported apart). Ctrl-C stops after the current batch as with
`train-local.sh`; every runner also has its own `--hours` hard stop.

## Measured on Linux (not a Mac estimate)

See the log, sections O5 onward, and the validation record. Summary for the review: RESULTS_SUMMARY.

## Rollback

Do not pass `--fast-path`. To remove the code path entirely, revert the merge; checkpoints written
by the accepted path are unaffected either way.
