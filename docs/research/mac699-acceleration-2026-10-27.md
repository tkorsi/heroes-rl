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

## Status at preregistration

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

## Prospective missing-Blue supplement — 2026-10-27T14:30:37.455011+00:00

The accepted path completed Red266 in5434.37seconds and exactly restored700,
but its unchanged conservative admission rule requires six times the slowest
batch and therefore declined the second requested training batch. Its ordinary
final evaluations remain separate. The planned two-update full-cycle study is
incomplete; no deadline or admission rule is changed to force completion.

Before candidate Blue267 has completed, register a small conditional correctness
supplement: if a candidate266 history/full-state pair matches A and candidate267
passes its checks, compare accepted and the fastest validated exact candidate
optimizer on those existing Blue267 captures from identical disposable full700
parents. Use the independent guarded capture driver, compare model, gradients,
Adam, all RNG and unchanged progress/sampler/pool plus input bindings, and retain
all results. There are no new native games, official samples or promoted weights.
This optimizer-only supplement must fit the same20:21:52.592870UTC study deadline
and cannot be reported as completion of the missing accepted Blue full-cycle
measurement. A supplemental exactness failure prevents candidate adoption.

## Completed observations and engineering decision — 27 October 2026

The native study closed at 15:26 UTC and its outer guardian verified cleanup.
The additional captured Blue comparison also closed and passed its independent
guards. The official parent remains 699; its owner lock is free and its next
unused sequence remains 337. No official game or update was launched.

| Implementation | Matched Red full cycle | PPO portion | Full-cycle ratio versus accepted |
| --- | ---: | ---: | ---: |
| Accepted A, one Torch thread | 90.57 min | 82.81 min | 1.00x |
| Exact F, no replay threads | 43.34 min | 38.72 min | 2.09x |
| Exact T, two replay threads | 35.53 min | 31.04 min | 2.55x |

These are intent-to-durable-checkpoint-and-restore measurements, including native
collection and verification. There is no GitHub publication latency in this local
cycle. T uses additional CPU parallelism and is 1.22x faster than F on this case.
The primary prefix-reconstruction component fell from 50.71 to 18.66 minutes
(A to T); backward/clipping fell from 18.61 to 9.08 minutes. All three processed
the same 15,510 samples with 485 Adam steps. Their semantic native histories and
complete model, Adam, RNG, progress, sampler and pool state matched exactly.
The full-state digest is `d360418850669629534f59f2a6a5cc311c195536c540c537df584e9dcca6f71f`.
F and T also completed Blue267 (37 samples, two Adam steps), with identical
histories and full state. The separately preregistered captured Blue supplement
restored identical full700 parents and compared accepted A with selected T:
model, gradients, Adam and all RNG state matched exactly, as did the native
reference model/Adam/minibatch generator. Diagnostic progress stayed unchanged;
this was not new native gameplay or an official update.

The original two-update full-cycle protocol remains **incomplete**, with its
original `passed=false` and outer return code1 preserved. A completed only its
first update because the unchanged conservative admission rule required more
remaining time than the original eight-hour budget allowed. Its ordinary final
pair then ran. The actual count is52 native games:24 baseline evaluations,
20 disposable training games (five updates), and8 A final evaluations, rather
than the nominal48. The diagnostic training outcomes were0 wins,14 losses and6
collection cutoffs; baseline was24 losses and final evaluation7 losses/1 cutoff.
All evaluation data had zero PPO samples and Adam steps. A Blue267 is explicitly
unstarted; no deadline was extended or admission rule weakened.

One A/F Red baseline history comparison differs in three late view frames after
the learner has no heroes, without a subsequent differing learner decision.
The nonidentical evaluation-history finding is retained. A/T baseline histories
match. Raw file hashes also differ in incidental timing records; only the
specified semantic training histories are claimed identical.

The selected manual default is T: exact text-leaf replay with two threads,
serial verification, no prefix helpers and no numerical experiment. The measured
F mode remains a manual fallback using `--replay-threads 0`. This engineering
choice rests on the matched long update, the Blue supplement, exactness tests,
full restoration and independent review; it does not turn the incomplete
original protocol into a pass. The beyond-old-limit A update also demonstrates
the timeout correction:5,432.96 seconds of worker execution completed under the
new allowance, compared with the old3,510-second effective bound.

One ordered workload per implementation supplies no confidence interval or
counterbalanced estimate. The result is not a forecast for all4,720 games,
proof of strategic learning, an original-map win, or a Mac measurement against
the separate older PR30 optimized baseline. Linux measurements are kept separate.
All benchmark checkpoints remain disposable. Approximately16GiB of full captures
remain in durable local storage; the separate private publication contains full
copied checkpoints and compact bound verification evidence, not a full new
remote capture archive. See the [compact result](../../validation/mac699-acceleration-results-2026-10-27.json)
and [manual continuation](../operations/original699-continuation.md).

The [private preservation receipt](../../validation/mac699-preservation-2026-10-27.json)
binds package commit `c83d311fc27f6396011e28582fdcd4c692527c7b` and
receipt commit `f697c31bdd45174f5b74a1bab1b53879d558409b` on private
`tkorsi/heroes-rl-training-state`, branch `codex/mac699-acceleration-20261027`,
path `benchmarks/mac699-20261027`. All298 package blobs were
acknowledged remotely; eight checkpoint files were independently downloaded and
full700/701 states restored exactly. These states are explicitly ineligible for
the official continuation.
