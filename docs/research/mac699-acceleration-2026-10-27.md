# Mac checkpoint 699 acceleration validation

## Prospective protocol — 27 October 2026, 12:16 UTC

The user authorized an actual Mac benchmark on disposable copies of final
checkpoint 699, correction of the operational update timeout, and preparation
of a manual continuation for the remaining 4,720 original-map games. The
benchmark does not launch that official continuation or promote its output.

The official original-map run ended at 699 after 280 admitted games and 70 PPO
updates. Batch 336 consumed its attempt and timed out during PPO; its update
rolled back completely. Its four games and attempted optimizer work remain
excluded from official training. This study neither reruns that official
attempt nor uses its seed namespace. Official continuation must start from 699
with fresh attempt bookkeeping (next unused official sequence 337).

### Hypotheses and comparisons

1. Exact acceleration will reduce complete verified original-map batch time on
   this Mac, including native collection, replay, policy reconstruction, PPO,
   checkpoint round-trip, local durable preservation and independent restore.
2. For identical captured games from the same full parent, accepted and exact
   accelerated implementations will produce bitwise-identical model, Adam,
   RNG, progress, sampler and opponent-pool state.
3. Raising the declared operational batch allowance from 3,600 to 10,800 seconds
   will permit longer valid PPO updates without weakening rollout/admission
   checks or changing learning settings. A finite watchdog remains mandatory.

The ordered comparison is A, F, T, with two complete four-game updates per path,
starting independently from identical full 699 copies. A is accepted code; F is
exact text-leaf replay with serial verification and no prefix helpers; T adds
two replay threads. A and F use one Torch intra-op thread and no helpers.
T uses additional CPU parallelism on the same Mac; its wall-time result is not
called a single-core speedup. Each path first performs the ordinary two-seat
eight-game baseline; those evaluations have zero training samples/Adam steps.
All resulting updates 700 and later are disposable diagnostic outputs.

The host is the user's Apple M4 Max Mac (16 CPU cores, 128 GiB), Python 3.13.5 and
Torch 2.9.1 CPU. The actual machine, source, runtime, map, parent and script hashes
are bound in the runtime preregistration. No Linux ratio is substituted for a
Mac measurement. One ordered comparison gives no statistical confidence
interval or counterbalanced thermal/cache estimate.

The original hero-management-v2 map, cold accepted VCMI runtime, Nullkiller Easy,
BattleAI, 365-day/5,120-choice limits, original architecture/PPO/Adam settings,
terminal-only rewards, complete four-game batches and strict verification remain
unchanged. No demonstrations, strategic masks, reward shaping or experimental
reordered-gradient path is used. The only operational configuration changes are
`shared_wall_seconds=10800` and `phase_deadlines_seconds.update=10740`.
Training collection remains 2,400 seconds, per-attempt 600 seconds, evaluation
3,540 seconds and cleanup reserve 30 seconds. Both A and candidate receive the
same declared allowance through a benchmark-only shim whose hash is recorded.
Historical checkpoint bytes and metadata are not rewritten.

### Isolation, deadlines and acceptance

A durable benchmark ledger allocates a fresh diagnostic namespace before every
invocation, including failed preparation/dry runs. Namespaces start at 80,000,000
(engine) and 80,050,000 (policy), advancing 100,000 per study. All paths within one
paired study intentionally share those diagnostic seeds, starting at sequence 266
(Red), then 267 (Blue); official 40M/50M and older 60M/70M namespaces are untouched.
The official owner lock remains held throughout and inherited by benchmark
runners. All writes are outside the official artifact root.

The whole study has one eight-hour wall/monotonic deadline, including setup,
checks and preservation. An independent process guardian is armed before the
harness executes. Subsequent paths inherit only remaining time; no path resets
the study clock. The runner's conservative admission reserve is unchanged.
A path that cannot start or finish safely is reported incomplete; no additional
benchmark window is silently granted.

A complete comparable full-state match is required to claim exactness. VCMI's
asynchronous behavior can make equal-seed games differ; such pairs are marked
not comparable, never silently normalized or called equivalent. If necessary,
a separately recorded optimizer-only comparison may reconstruct the immutable
failed 336 captures from the same full 699 copies. That is a diagnostic replay,
not an official retry or sample admission, and it must finish within the same
study budget. Its optimizer-only timing is reported separately from full-cycle
timing.

Timeouts are censored observations, not completed baseline timings. Report
all attempts, failures, unstarted paths, raw-evidence bindings, parent hashes
before/after, cleanup receipts and practical ratio limitations. Do not infer
a reliable 4,720-game duration from one short batch.

## Results

Pending: no gameplay or PPO had been launched when this protocol was committed.

## Prospective long-update supplement — 2026-10-27T12:30:41.592814+00:00

Before the first native benchmark PPO update completed, the timeout regression
check was made explicit: if the native comparison does not itself complete an
update beyond the old effective wall bound, or if native histories are not
comparable, reconstruct the four immutable failed336 captures from disposable
full699 copies. Compare the accepted optimizer with the fastest exact candidate
that passed the Mac correctness gates. This includes all16058 recorded samples
and is diagnostic only: no native game is replayed or official sample admitted.
Compare full model, gradients, Adam, every RNG, unchanged progress/sampler/pool
and before/after input hashes. Its independent guardian uses the original
20:21:52.592870UTC study hard deadline and a per-update maximum10800seconds;
no supplementary window is added. Optimizer-only timing is reported separately.
