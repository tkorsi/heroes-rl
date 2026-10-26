# Local curriculum and overnight continuation: 26 October 2026

This is a retrospective synthesis of the completed experiments, not a new
preregistration. No games or optimizer updates were launched for this audit.
The original prospective records and failures remain unchanged on their
immutable experiment commits.

## Results and accounting

| Stage | Model updates | Admitted games | Wins | Losses | New decisions | New Adam steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Local easier-start campaign | 146 → 254 (+108) | 432 | 64 | 368 | 32,704 | 1,151 |
| Cloud continuation | 254 → 412 (+158) | 632 | 563 | 69 | 22,491 | 919 |
| Combined new curriculum training | +266 | 1,064 | 627 | 437 | 55,195 | 2,070 |

Both campaigns report zero horizon cutoffs and zero failed/partial gameplay
admitted to training. Final cumulative state is **412 PPO updates, 1,648
training rollouts, 95,836 decisions and 3,545 Adam steps**. These cumulative
figures include earlier stages; the table contains only the new curriculum
training. An optimizer step is not a PPO update.

Local held-out checks comprised 184 games (29 wins, 155 losses); cloud checks
comprised 272 games (225 wins, 47 losses). They are excluded from PPO. Repeated
checks of the same four starts per seat are not independent new test cohorts.
The 64-game local acceptance and 64-game Linux acceptance were also separate:
7/57 and 25/39 wins/losses, respectively, plus separate controls.

## What improved, and what remains untested

Local fixed evaluation changed from Red 0/4 and Blue 0/4 to Red 2/4 and Blue
1/4. Linux established its own baseline after exact restoration and native
acceptance: **1/4 per seat at update 254**. Its final update 412 won **4/4 per
seat**, with both final evaluations bound to the same checkpoint. Every
held-out batch contributed zero training samples and zero Adam steps.

The cloud trajectory was not monotonic. Paired fixed-check win totals were:

| Checkpoints | Wins out of eight |
| --- | --- |
| 254, 256, 261 | 2 |
| 266, 271 | 8 |
| 276, 281 | 6 |
| 286, 291, 296 | 8 |
| 301 | 7 |
| 306 | 4 |
| 311 | 2 |
| 316 | 4 |
| 321, 326, 331, 336, 341 | 6 |
| 346 through 411 at five-update intervals, then final 412 | 8 |

The matched Linux comparison retains engine seeds 710001–710004 and policy
seeds 720001–720004 in both seats, with identical source, runtime, map and
configuration bindings. Native opponent scheduling is not fully determined by
those seeds. This is positive descriptive evidence on a small, repeatedly
tested **easier open8x8 distribution**, not an independent estimate of broad
win rate, original-map generalization or universal mastery. The original map
was not evaluated after this curriculum, and no new evaluation was launched
after the cutoff.

## Observed behavior

The cloud report aggregates verified training **and held-out** histories:
3,706 movement attempts, 1,347 changed endpoints, 919 moves toward the visible
opponent town, 788 town-capture evidence entries and 79 attacker-role events.
It records **zero completed recruitment actions and zero army transfers**.
These are action/evidence totals, not counts of unique towns, unique games or
units. The report does not provide a matched before/after behavior comparison;
the attacker role alone does not distinguish neutral from enemy targets.
Therefore the results do not demonstrate a learned “buy an army, then attack”
strategy. Changed weights and aggregate movement counts cannot establish that
specific explanation.

## Timing, failures and preservation

Local controller stopped at **03:03:55 UTC**, and its finalization watcher
exited at 03:04:43, before the original 03:10:53 hard deadline. It published
the final handoff only after owned shutdown and restoration, disabled local
resume, and transferred ownership to cloud.

The cloud clock began at **03:26:51.744627 UTC**. Its controller stopped at
**09:49:22**, the transport watcher at 09:49:41, final held-out gameplay ended
at **09:52:25**, and autonomous preservation finished at **09:53:26**. These
precede the fixed **09:55 compute cutoff / 10:00 hard stop**. The separate
post-deadline process audit is recorded in the
[root audit receipt](../../validation/overnight-root-final-audit-2026-10-26.json).
The cloud's [10:11 UTC process scan](../../validation/overnight-cloud-shutdown-2026-10-26.json)
found no live campaign-owned or native processes and all five ownership locks
unheld. Some unreaped zombie entries remain; they are exited processes, not
running workers. At 10:00:00.075838 UTC the registry guard killed the original
watchdog, pre-empting its own final status write; the registry's final receipt
and later process scan confirm shutdown. Root's 10:12 UTC Mac scan found no
commands matching the local campaign/finalization paths, and local autoresume
remains paused. The post-deadline audit ran no gameplay, PPO or repair.

Retained technical failures include an offline capture-filename mismatch
(zero games), HTTP503 publication failure after valid update 265, a recovery
prelaunch guard-scope failure (zero games), and plain-Git authentication failure
while publishing completed Blue evaluation 286. Exact checkpoint publication
was subsequently acknowledged. Blue evaluation 286 was reconciled append-only
without repeating its games or adding optimizer steps. The existing connected
GitHub API publication route repaired transport; frozen learner/runtime source
and the original clock stayed unchanged.

The 158 cloud training sequences **108–265** are unique and contiguous. Their
original journal contains 157 passed records and one publication-failed record
(118), whose complete four-game update was later durably acknowledged. Thus
158 complete batches were admitted, but not every original operational record
was marked passed. No failed or partial gameplay was retrospectively admitted.
Final campaign analysis reports zero failed gameplay attempts and zero
unstarted campaign games; these counters do not erase failed preparation.

Final checkpoint **412** is remotely preserved in private training-state
commit `5577155c0b9e9d859d795edc0831d9d8e800d9d6`. Autonomous final evidence
is at `2f8d587c7a4a82f6b08bb752a1322ea31fd100df`. Independent-copy restoration
receipts match model, Adam, RNG, progress and pool. Final model file SHA256 is
`1803823cab4a2d74f436e8425e31a1dbe1892e58844955945e2c963d176d9b88`;
resume SHA256 is
`09eee900a5fbff507a05c39e8ad4379e5b95177608b38cb69390a3cb56fee321`.
Raw captures, game data and checkpoint bytes remain private.

Reported training-batch wall time totals 12,546.61 seconds (3h29m), including
collection, audit, native replay, PPO and checkpoint supervision, but excluding
private publication and acceptance. It is not the full overnight elapsed time
or a pure simulator speed measurement. Claude's experimental acceleration was
not used in this campaign.

## Evidence and reproducibility

- [Imported local final analysis](../../validation/curriculum-v3-final-2026-10-26.json)
  and [local final status](../../reports/local/teacher-curriculum-status.json).
- [Imported cloud final analysis](../../validation/overnight-cloud-final-2026-10-26.json)
  and [cloud finalizer status](../../reports/cloud/teacher-overnight-status.json).
- [Post-deadline shutdown audit](../../validation/overnight-cloud-shutdown-2026-10-26.json),
  originally published at `81480c1d9e996c3052f0ef49c4139983d368c8b0`.
- [Local prospective protocol and corrections](https://github.com/tkorsi/heroes-rl/blob/6dd709821fb10a377aa856d80113074b3056f1c3/docs/research/curriculum-v1.md)
  and [append-only local journal](https://github.com/tkorsi/heroes-rl/blob/6dd709821fb10a377aa856d80113074b3056f1c3/validation/curriculum-v1-research-log.jsonl).
- [Cloud prospective protocol and corrections](https://github.com/tkorsi/heroes-rl/blob/d75de47355426c2f0a7165b13db6c52f75671ee3/docs/research/overnight-cloud-2026-10-26.md)
  and [append-only cloud journal](https://github.com/tkorsi/heroes-rl/blob/d75de47355426c2f0a7165b13db6c52f75671ee3/validation/overnight-cloud-2026-10-26.jsonl).
- Executable local source: `cb3d324cb09919383dac7e97a97bc0696dcd588d`;
  cloud source: `30629a84bc3048ee64ae4f91186fbc0832e4ca25`.
  Restoration, runtime, dependency and seat-map hashes remain in the bound
  reports. The public report import does not merge operational source changes.

All runs retained the original model/PPO configuration, terminal-only rewards,
365 days, 5,120 total choices, Nullkiller Easy and BattleAI. Curriculum changes
initial conditions; it does not replace VCMI rules. Demonstrations, scripted
learner actions, shaped rewards, hidden state and strategic action masks were
not introduced. Historical preparation failures and scripted rush probes remain
outside the admitted training lineage.
