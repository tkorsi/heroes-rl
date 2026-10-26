# Frozen checkpoint 412 acceleration validation

## Prospective protocol

Recorded on 2026-10-26 before any game in this study. The user requested a
check of Claude's acceleration with our trained network and configuration,
followed by a new assignment to Teacher. This study is diagnostic only and
does not reopen the expired overnight training window.

Hypothesis: the proposed exact optimizations preserve checkpoint 412's choices
and observed game outcomes while reducing collection and verification time.
The earlier seed-21 benchmark is supporting evidence, not this result.

- Frozen policy: update 412, policy SHA256
  `1803823cab4a2d74f436e8425e31a1dbe1892e58844955945e2c963d176d9b88`.
- Optimizer checkpoint SHA256:
  `09eee900a5fbff507a05c39e8ad4379e5b95177608b38cb69390a3cb56fee321`.
  Restore and retain it; do not take an optimizer step or overwrite either file.
- Base runtime/learner: `b70442d2466fb2b9f309e56d52aeaaab25dcc445`.
- Accelerated runtime/learner: PR30
  `4d7688b98fc1e20ab349676cc0a9e719bcfd6321`; its executable changes match
  measured commit `6c120c9591efc7549650c1fca310c0c03f27d8f2`.
- Both builds are independent canonical VCMI checkouts on the same Mac,
  Release configuration, using the same Python/Torch installation. These
  measurements will not establish Linux throughput or cross-platform bitwise
  equality. Existing same-public-version CPU checkpoint relocation must pass.
- Use both immutable final easier8x8 seat-specific maps, v8
  `adventure-capabilities-v1`, Nullkiller Easy, BattleAI, 365 days, 4960 normal
  choices plus 160 dialog-reserve choices (5120 total).
- Compare base cold, accelerated cold and accelerated embedded. Each round
  runs four frozen-policy games per seat/configuration, using engine seeds
  710001--710004 and policy seeds 720001--720004. Three rounds give at most
  72 games. Round one is warm-up; rounds two and three supply timing summaries.
  Rotate configuration order by round to reduce order bias. Run sequentially
  after builds/tests finish, with no concurrent benchmark workers.
- Turn on `HEROES_VERIFY_NATIVE_CAPTURE=1` for both accelerated configurations
  during warm-up; require observed verification counts, then turn it off for
  timed rounds. Keep every native replay and frozen-policy reconstruction.
- Maximum 300 seconds per game, 60-second delivery watchdog; the root runner
  enforces an additional bounded process-group deadline and preserves failures.
  Stop on a failed source/checkpoint/replay/reconstruction/cleanup check.
- Compare paired player-visible histories, decisions, outcomes, and replay
  acceptance. Distinguish known native opponent nondeterminism from any new
  discrepancy; do not claim exact equality when it is not observed.
- Report collection, replay, policy-audit and complete probe-game timings
  separately. These are not full PPO/admission/publication timings.
- No training samples, PPO/Adam updates, demonstrations, reward shaping,
  strategic action masks, altered model parameters or changed game rules.
  Repeated seeds are intentional paired diagnostic replicates, never training.

Raw captures, model files and restore receipts live in the private artifacts
directory and private training-state repository. Public results will bind them
by immutable hashes. Any correction to this protocol will be appended with its
time, reason, and affected attempt before the corrected attempt runs.

## Next Teacher assignment

After this gate, give the existing cloud Teacher an English instruction to
prepare and perform a bounded frozen-policy transfer evaluation on varied
starting coordinates and the original scenario, comparing checkpoints 146,
254 and 412. Retain an independently evaluated cold fallback if the embedded
gate fails. This next diagnostic must have a prospective manifest, both seats,
separate outcome/behavior metrics, zero PPO admission, private preservation and
an English research report. It does not authorize a new unbounded PPO campaign.

## Preparation record

Before gameplay, both independent native Release builds completed and canonical
dependency checks passed. Exact checkpoint restoration passed on the existing
same-public-version CPU Torch loader. The first 100-test focused run found a
pre-existing report sanitizer defect for quoted paths containing spaces (99
passed, one failed). The narrow reporting correction passed all 100 tests on
repeat. Six native contract tests also passed. The dedicated loader adds
strict input/update/interface/budget/RNG checks; it does not relax the original
probe's historical v7 checkpoint gate.

Independent review found that a cold game starts its own process session.
The diagnostic supervisor therefore verifies PID/start time/argv and ancestry,
plus each native binary's unique output argument, before shutdown. An offline
dummy child in a separate session was terminated with its parent and left no
live owned process. The whole study is capped at 2700 seconds, without any
extension on failure. The dedicated probe is fail-fast after a failed row;
unstarted rows remain recorded. These corrections precede all study gameplay.

### Preserved startup correction

Study1 stopped at 12:36:21 UTC before any game: resolving the selected virtual
environment's Python symlink launched Anaconda's other Torch installation.
The strict checkpoint loader rejected it. Zero games, samples and updates
resulted; all 72 scheduled games remained unstarted. The fresh study2 keeps the
original 13:21:20.318085 UTC diagnostic deadline and the same input bytes/seeds.
The runner now preserves the virtual-environment interpreter path and anchors
wrapper ownership to its original start time and command before any signalling.
No game-rule, learner or checkpoint check was weakened.
