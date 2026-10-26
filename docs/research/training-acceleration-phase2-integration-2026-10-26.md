# Phase 2 acceleration: integration guide

Branch `claude/project-thread-iba9jw-acceleration-phase2`, based on PR30 head 4d7688b. Manual review
and merge only. Measured candidate: a9304eb (later commits touch only documentation, records and
benchmark tools). The scientific log with every attempt, failure and correction is
`docs/research/training-acceleration-phase2-2026-10-26.md`; compact results are in
`validation/training-acceleration-phase2-2026-10-26.json`.

## What changes by default (exact; no switch)

These keep every check and are bitwise identical to 4d7688b on the PPO outputs measured (model,
gradients, Adam state, minibatch generator, sampled window order and report; A14).

| Change | Files | Exactness and coverage argument | Tests |
| --- | --- | --- | --- |
| Behaviour verification in evaluation mode with fresh exact memos (E1) | `ppo.py` | The admitted modules behave identically in both modes; every module's mode is restored; memos start empty and are discarded, so nothing cached by the collector or the policy reconstruction is reused | `test_ppo_phase2_acceleration.py` |
| Prefix reconstruction: call-local exact node memo; memo keys computed once per update | `contextual_model.py`, `ppo.py` | The node memo follows the inference memo's exact-tree rule; a frame's key is a pure function of the unchanged history | `test_ppo_phase2_acceleration.py`, `test_prefix_pool.py` |
| Per-check digest memo | `digest_scope.py` and every `check_identity` | One hash per unchanged file inside one identity check (path, device, inode, size, mtime and ctime observed before and after hashing); every check opens a fresh scope; outside a scope every call rehashes | `test_digest_scope.py` |
| Overlapped per-game verification | `builtin_gameplay.py`, `builtin_replay.py`, `builtin_first_run.py`, `background_check.py` | The native replay and the behaviour report run in forked children on files bound first; the parent applies every original check to their results; the serial order's error is the error raised | `test_parallel_verification.py`, `test_builtin_gameplay.py`, `test_builtin_first_run.py` |
| Back-to-back identity checks | `builtin_first_run.py`, `builtin_training.py` | Only exact repeats with nothing in between are skipped (a passed attempt's final check and the next attempt's opening check); the final check always runs | `test_builtin_first_run.py`, `test_builtin_training.py` |
| Top-level module hashing fallback | `contextual_evidence.py`, `scoring.py` | Same digest when the repository root is not importable | `test_digest_scope.py` |

## Opt-in execution choices (bound in the plan)

Neither has an environment variable. A caller that wants one binds it into its immutable plan, and
every identity check compares the plan with the campaign context.

- `prefix_helpers: N` (exact). `ppo.PREFIX_HELPERS = N` forks N helpers per update that compute
  disjoint shares of the prefix-reconstruction frame encodings under the parameter bits the parent
  sends with every request; the parent recomputes two encodings of every share, applies every
  recurrent step itself, and reaps the helpers when the update ends, whatever its outcome. Bitwise
  equal to `N = 0` (A14). Uses up to N more CPUs during prefix reconstruction only; report CPU
  seconds when comparing.
- `experiments: ["shared_graph_encodings"]` (numerical experiment, never a default). Within one
  minibatch graph an exact tree or leaf is encoded once; forward values and losses are
  bit-identical, gradients differ in summation order only (update difference 2.1e-4 to 4.5e-4 of
  the update's L2 norm, A13 and A14). It is recorded in `checkpoint_metadata.numerical_experiments`
  of every checkpoint it produces. The user chose to count it toward the target as an opt-in
  candidate pending manual review; it must not be deployed to Teacher.

## Porting to an official driver

The benchmark driver `experiments/learner/bench_training.py` mirrors
`origin/codex/teacher-overnight-20261026:experiments/learner/overnight_training.py`. I did not
change any Teacher branch or checkout. To use the changes there:

1. Merge this branch's library changes (everything outside `bench_training.py`, `tools/` and
   `docs/`). The exact defaults then apply without further changes.
2. The overnight driver's `read_plan` runs `check_identity`, and `builtin_training.worker` runs it
   again as its first action. To remove that exact repeat, give the worker a reader that checks
   the plan's scope and binding but not the identity, as `bench_training.read_plan(check=False)`
   does; keep the check in every other reader.
3. Its `check_identity` should open one `digest_scope.scope()` per call, as `bench_training` does.
4. To opt into helpers or the experiment, add the plan fields and the identity requirements
   exactly as `bench_training.execution_options` does, and set `ppo.EXPERIMENTS` and
   `ppo.PREFIX_HELPERS` only from a plan read by that function.
5. `prepare` can run `tools/prepare_vcmi.py --check` while it builds the plan, but must not write the
   plan unless the check passed (see `bench_training.prepare`).

## Overlap with other lanes

- Codex's recurrent-prefix lane: cea6376 (integrated) is in the base; 0d38977 is not. My changes to
  `ContextualActorCritic.prefix_states` (optional `vectors` and `keys` arguments, the call-local
  node memo) and `frame_key`/`frame_keys_until` sit on top of cea6376's cache. A merge with later
  work in that function needs care: the helpers rely on `_can_cache_prefix()` and on
  `_frame_features` under `no_grad` with a fresh memo being a pure function of the frame.
- The native runtime is unchanged (phase 2 changed no C++); `runtime/base` and `runtime/cand-0`
  were byte-identical builds.

## Operational notes

- Forks: background checks and helpers fork only from a single-threaded point of the worker
  (after the game's reader threads end), close every inherited descriptor except their own pipe or
  socket and the standard streams, stay in the worker's process group (its deadline guardian kills
  them with it), and defer SIGALRM, SIGINT and SIGTERM while they are forked, recorded and stopped.
- Memory: a forked child shares the parent's pages copy-on-write; the summed RSS in the resource
  samples double-counts those shared pages. Helpers live only for one update.
- Torch: this host runs `2.9.1+cu128` on CPU; the archived checkpoints were written by
  `2.9.1+cpu`. Benchmark copies were relocated (`tools/acceleration2/relocate_checkpoint.py`); the
  same 411 to 412 update differs across the two builds by 1.05e-3 of the update in the baseline
  itself, so cross-host bitwise claims need the same Torch build.

## Results

See the scientific log (A18 onwards) and the compact record.
