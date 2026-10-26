# Exact acceleration of real VCMI games

This branch makes real VCMI games, their player-local capture, the frozen policy
in the loop and the per-episode verification faster without changing any
observable output. The game is the pinned VCMI 1.7.5 C++ state and rule code
with Nullkiller Easy as the opponent and BattleAI for automatic battles. Rules,
fog, RNG, legal actions, opponent logic, routing and event and query ordering
are not changed. Against the base code's run of the same scenario, every
outcome is equal and 71 of 74 learner histories are byte-identical; the other 3
diverge after 34 identical frames on maps where the base code's own runs
diverge the same way (see [Exactness evidence](#exactness-evidence)).
Replaying the base code's recorded v7 and v8 policy games through the
accelerated collector reproduces every choice, log-probability, value and
recurrent prefix exactly. Every ended game's replay verifies under
`vcmireplay`. Replay files themselves were not compared with the base: they
embed the wall-clock start time, and the contextual replay tape records the
learner's sends and the opponent's requests in an order that varies between
runs of the same code, so no two runs' files are byte-equal, base runs
included. Every
correctness check still runs, including the four frozen-weights checks per
decision. No training, PPO update, Teacher work or Mac time was used.

Evidence: [acceleration record](../validation/linux-acceleration-2026-10-26.json)
(A/B runs on two hosts, repeated-reset soak, scaling, engine-free controller
and audit replays, native capture verification counts, review-fix costs and the
review's findings) and
[revalidation record](../validation/linux-acceleration-revalidation-2026-10-26-compact.json)
(24 real-game runs of the final code with native capture verification switched
on, the 4 runs that resubmit base-code games, and every comparison, including
the 24 with the base code's runs of the same scenarios). The base runs they are
compared with are in
[the embedded engine record](../validation/linux-embedded-2026-10-25-compact.json).
`tools/acceleration/` holds the scripts behind the runs, the comparisons and
both records (see [Reproducing](#reproducing)). The run scripts are cleaned-up
copies of the ones that made the runs, which differed only in local paths and
in deleting episode directories as they went; `measure.sh` puts the host B
measurements in the order they ran. `record.py` regenerates the acceleration
record byte for byte from the run outputs and the committed notes file
`record-notes-2026-10-26.json`, which supplies the commits, the platform text,
where each input lies and the summary of the independent review.

## Results

Unless a line names another revision, numbers are for the final code
(`6c120c9`) against the base, the embedded engine branch (`b70442d`, code
identical to `d74e1f5`). Linux x86-64 cloud VM, 4 vCPUs, 15 GiB, Python 3.13,
Torch 2.9.1 on CPU, VCMI Release build. The container was restarted onto
different hardware during the final measurements, so every timing section names
its host and base and accelerated are only compared on the same host. Host B
(Xeon @ 2.80 GHz) ran the A/B tables, the soak, the controller and audit
replays, the review-fix costs and the scaling table. Host A (Xeon @ 2.10 GHz,
the machine of all earlier work) ran the same embedded A/B, an earlier scaling
run, and the revalidation, base-game resubmissions and capture verification of
`6c120c9` (see [Exactness evidence](#exactness-evidence)) before the restart.
Host B is slower overall (about 1.5× on most parts) and about twice as slow at
the base code's inference, so its policy-row speed-ups are larger; recorded-row
speed-ups, which involve no inference, are about the same on both hosts.

Each A/B round runs every workload once per configuration, base then
accelerated, on an otherwise idle machine: 4 games per run (seats red, blue,
red, blue), natural endings, every game replayed by `vcmireplay` and every
policy game audited. The A/B tables (embedded, cold, and where a decision's time
goes) show medians of rounds 2 and 3; round 1 is warm-up and listed in the
record. All 72 A/B runs passed.

### Embedded backend

| Workload | Decisions/s | Collection games/s | Full game incl. replay and audit | Host A: decisions/s, full game |
| --- | --- | --- | --- | --- |
| v7 policy | 17.1 → 42.6 (×2.50) | 0.113 → 0.283 | 16.3 → 7.8 s (×2.07) | ×2.03, ×1.86 |
| v8 policy | 17.1 → 42.3 (×2.48) | 0.199 → 0.494 | 9.8 → 4.7 s (×2.09) | ×2.06, ×1.83 |
| v7 recorded actions | 78.5 → 101.2 (×1.29) | 0.52 → 0.67 | 5.1 → 3.8 s (×1.35) | ×1.27, ×1.37 |
| v8 recorded actions | 66.9 → 90.7 (×1.36) | 0.78 → 1.06 | 3.4 → 2.6 s (×1.30) | ×1.32, ×1.31 |

Policy rows run the labelled freshly initialized v7/v8 model (seed 21) on
`hero-management-v1`; the games average 151 (v7) and 86 (v8) decisions.
Recorded rows resubmit the delivered actions of a base-code policy run, so they
measure the engine, capture and adapter without inference. On host A the same
rows ran at 31.7 → 64.4 decisions/s (v7 policy) and 118 → 150 (v7 recorded).

### Cold executable backend (the current training default)

The adapter, inference and verification changes are shared, so the cold
backend gains too (host B):

| Workload | Decisions/s | Full game |
| --- | --- | --- |
| v7 policy | 16.4 → 34.4 (×2.09) | 16.7 → 8.6 s (×1.95) |
| v8 policy | 14.7 → 29.3 (×1.99) | 10.6 → 5.8 s (×1.83) |
| v7 recorded actions | 57.0 → 67.3 (×1.18) | 6.2 → 4.7 s (×1.32) |
| v8 recorded actions | 44.2 → 52.3 (×1.18) | 4.2 → 3.3 s (×1.25) |

Against the cold base, the accelerated embedded backend runs v7 policy games at
×2.59 decisions/s (×2.13 per full game) and v8 at ×2.87 (×2.26).

### Where a decision's time goes now

v7 policy, embedded, host B, milliseconds per decision:

| Part | Base | Accelerated |
| --- | --- | --- |
| Frozen policy (controller) | 42.4 | 11.9 |
| Engine wait (own actions, Nullkiller turns, battles) | 5.5 | 4.2 |
| Player-local ingest (decode, offers, history checks) | 4.9 | 2.7 |
| Evidence ledger | 0.60 | 0.55 |
| Other collection work | 5.2 | 4.2 |
| Collection total | 58.5 | 23.5 |

Per v7 policy game, the replay check (evidence binding, wire and history
verification in Python, then `vcmireplay`) takes 2.1 s (base 3.1) and the
policy audit 2.0 s (base 4.1), so verification is now a little over half of a
full game. The records do not time `vcmireplay` on its own. Inside the engine
(v7 recorded, 4 games, 603 steps), advancing the game takes 2.0 s (base 2.7),
of which Nullkiller turns take 0.65 s for 22 turns (base 0.75) and automatic
battles 0.13 s for 4 battles (base 0.28).

Step latency is measured over the whole environment step, from submitting a
response to the next request: response validation, the engine advance,
player-local ingest and the ledger write, so it is not engine-only. The step
that ends a game also includes that game's finalization (terminal checks,
ledger close, manifest and history writes): 0.29 to 0.34 s per 4-game v7
recorded run in rounds 2 and 3, in both configurations, so that step falls in
each game's tail and can be its slowest. In the three long v7 recorded games of
each run (323, 87 and 187 steps), the per-game p50 is 5.0 to 5.9 ms (base 6.6
to 8.2) and the per-game p99 31.5 to 91 ms (base 52 to 140), over rounds 2 and
3. The fourth game has 6 steps, so its p99 is just its slowest step,
finalization possibly included: 85 and 156 ms accelerated, 326 and 317 ms base.

Engine-free replay of the base code's recorded policy games through the
collector, every choice and value required identical (see
[Exactness evidence](#exactness-evidence)), 3 interleaved repeats on host B:
10.3 ms per decision against 38.8 for the base code for v7 (×3.8, 603
decisions) and 11.3 against 41.9 for v8 (×3.7, 342 decisions).

When the same model works through the same games twice, the second pass finds
what the first left in its inference memos. In the policy audit
(`reconstruct_behavior`, as training's admission check runs it), the
accelerated code's second pass takes 52 to 56% of its first (medians of 3
repeats: v7 8.9 then 5.0 ms per decision, v8 9.0 then 4.7), the base code's 90
to 94% (v7 25.8 then 23.1, v8 23.9 then 22.4). Through the collector the
accelerated second pass takes 67% of its first (10.6 then 7.1 ms) and the base
code's 96% (41.8 then 40.2).

### Resets, CPU, system calls and memory

Embedded runs use one engine per run with one reused environment per seat
(two environments taking turns: 12 games each in the 24-game end-turn runs, 6
each in the 12-game policy runs); the cold rows start a new game process per
game. No replay or audit, host B. `perf stat` counts
the whole probe process: interpreter start, imports, model construction, engine
load, the games and shutdown. The per-game and per-decision rows divide those
whole-run totals, so they include one-time work and overstate steady-state
cost; compare base against accelerated rather than reading them as absolute
costs. The end-turn rows cover 3 interleaved rounds per configuration, round 1
included. The policy rows run 12 games of the v7 policy workload (seed-21
model, 1024 choices, 365 days), 88 decisions per game on average.

| Measure | Base | Accelerated |
| --- | --- | --- |
| Embedded reset p50, one reused environment per seat (24 games per run, 12 per seat, 3 runs) | 0.109–0.113 s | 0.085–0.100 s |
| Embedded engine load, once per worker (warm, 3 runs) | 0.56–0.59 s | 0.60–0.79 s |
| Cold reset p50 (new process per game, 8 games) | 0.80 s | 0.76 s |
| CPU per run of 24 end-turn games, incl. startup and load | 12.6–13.2 s | 10.5–10.9 s |
| System calls per end-turn game, incl. startup and load | 21,000–21,500 | 20,300–20,400 |
| CPU per decision, 12 v7 policy games, incl. startup and load | 62.3 ms | 28.8 ms |
| System calls per decision, same games | 514 | 474 |
| Context switches per decision, same games | 16.6 | 14.4 |
| Resident memory after the same 12 games | 701 MiB | 693 MiB |
| Threads, descriptors after every game | 4, 7 | 4, 7 |

Threads and descriptors stay the same after every game in both configurations.
Resident memory is read after engine load and after each game, and grows mostly
in the first game. Over 24 end-turn games, which run no inference, it grows by
46 to 51 MiB from about 538 MiB after load in both configurations, 36 to 39 MiB
of it in the first game and 7 to 12 MiB over the other 23; at the end it is
still growing by about 0.1 to 0.25 MiB per game on average over the last 12
games, mostly in steps under 0.1 MiB with occasional jumps of up to 2.3 MiB.
Over the 12 policy games, from about 542 MiB after load, it grows by 159 MiB for
the base and 151 MiB for the accelerated code. The first game adds 136 and 108
MiB; games 2 to 12 add 23 MiB for the base (678 → 701 MiB) and 42 MiB for the
accelerated code (651 → 693 MiB), 5 MiB of that in its last game. The
accelerated code's faster later growth fits the memos it adds (up to 64 MiB of
observations and 16,384 internal-node encodings; the leaf memo of 65,536
entries per kind was already in the base code and is used the same way there),
but after 12 games it still ends 8 MiB below the base; 12 games are not enough
to show where it levels off. The engine-load code was not changed; the warm-load
difference above is not attributable to the branch. The first load of each
configuration took 1.7 s (base) and 1.3 s (accelerated), probably a cold file
cache (not recorded; the native data load inside it was 0.30 s and 0.25 s, as in
later runs), and is excluded from the ranges.

### Workers on one machine

K probe processes at once, each with its own embedded engine: v7 policy, 4
games per worker with identical engine and policy seeds (seats red, blue, red,
blue), full rows with `vcmireplay` and the policy audit. Wall time includes
process start, imports and engine load. One run per worker count and
configuration, host B:

| Workers | Base games/s | Accelerated games/s | Accelerated decisions/s |
| --- | --- | --- | --- |
| 1 | 0.064 | 0.131 | 19.7 |
| 2 | 0.120 | 0.223 | 33.6 |
| 4 | 0.238 | 0.434 | 65.5 |

From 1 to 4 workers throughput grows ×3.3 accelerated and ×3.7 base on host B,
so the gain falls from ×2.05 at one worker to ×1.82 with the machine full
(single runs; host B's single-worker speed-up varied from ×2.00 to ×2.16 across
the A/B rounds). An earlier run of the same workload on host A, on code
`89f377e` before the review fixes (in the record as `scaling_host_a`), scaled
×3.9 accelerated and ×3.8 base (0.087 → 0.325 base, 0.156 → 0.605 accelerated
games/s). These games average 151 decisions; at 4 workers host B finishes about
1,560 verified policy games of that length per hour, host A about 2,200 (on
`89f377e`).

## What changed

Native engine (C++, `engine/ExplorerRL`, `engine/NativeSession`, patch 0032):

- Unchanged-view capture check. Many player-local captures in the map context
  see the same view as the last one; the capture verification confirmed at
  least 1,000 such skips in each 4- or 6-game embedded run, but the share of all
  captures was not recorded. The capture compares the current visible
  state with the stored observation field by field, with the same engine
  queries and handle allocations in the same order as building it, and returns
  early only on an exact match on the same day. Any difference, window or other
  context builds the observation as before.
- Replay digest. The recorder serializes the game state once for both the
  combined state digest and its game-state part, instead of twice (patch 0032).
  The eight smaller handler parts (queries, randomizer, battles and so on) are
  still serialized inside the combined digest and again for their own part
  digests. `vcmireplay` still recomputes the combined digest with the
  unchanged `stateDigest` and each part with its own fresh serialization; it
  compares the parts only when the combined digest differs.
- Ledger and transport. Ledger records are written as one compact string
  without an intermediate copy; in-process policy messages are moved, not
  copied.
- `HEROES_VERIFY_NATIVE_CAPTURE=1` builds every skipped observation as well,
  fails the game on any disagreement, and reports confirmed skips on stderr.
  The first version read `EXPLORER_RL_VERIFY_CAPTURE`, which both launchers
  strip from the engine environment, so that check never ran in probe games.
  This branch fixes it, a source-level test now guards the switch, and the
  revalidation below ran with it on.

Adapter (`experiments/headless`, `experiments/learner/contextual.py`):

- A decision's observation is decoded once, not twice.
- Tiles that pass the exact type checks are shared through a bounded cache keyed
  by their typed values; a tile list byte-equal to the previous one reuses its
  decoded tuple. Equal values of other types take the reference path and
  failures are never cached.
- The detached frame encoder returns exact JSON leaves directly.
- Environments hand their own validated frames to the collector, which uses them
  only when each delivered record has exact JSON types and exactly the
  canonical text of its frame; otherwise it decodes the records as before.

Frozen policy inference (`experiments/learner/contextual_model.py`,
`contextual_rollout.py`, `builtin_rollout.py`):

- Internal tree nodes and whole observations are memoized under the existing
  inference memo's validity rules: eval mode, no autograd or autocast, no hooks,
  same configuration and thread count, and bit-identical parameters of the
  modules involved. Hits add exactly the cache entries the reference adds,
  except at the last root of a frame whose cache nothing reads afterwards (every
  non-decision frame in `forward_step`), where a hit returns without visiting
  its subtrees. The
  new memos are bounded (64 MiB of observations, charged for everything an entry
  keeps; 16,384 internal nodes) like the existing leaf memo (65,536 leaves per
  kind), and cleared whole when full, so a miss only recomputes.
- One policy walk yields the sampled action and its evaluation.
- The weights check keeps a private copy of the digested bytes and compares
  every bit against it as integer words. When all tensors share one dtype the
  words are 64-bit, or integers of the element width when the bytes do not fill
  whole 64-bit words; when the dtypes differ, the bytes are zero-padded to a
  multiple of 8 and viewed as 64-bit words. It reads the state dict by a plain
  module traversal when `state_dict()` is plain module code, and calls
  `state_dict()` otherwise (hooks, overrides). Non-CPU or strided tensors, a
  layout difference or any mismatch recompute the original digest, so every
  decision equals the previous digest comparison.

Per-episode verification (`contextual_replay_history.py`,
`contextual_evidence.py`, `contextual_protocol.py`, `builtin_gameplay.py`):

- History verification encodes each reconstructed frame once instead of four
  times; anything the canonical encoder rejects takes the original encoding.
- Strict JSON hooks build objects with `dict()` and fall back to the original
  loop and error only when a key repeats.

## Exactness evidence

The revalidation, the base-game resubmissions and the capture verification of
the final code ran on host A.

- Real games, revalidation of the final code with capture verification on
  (`HEROES_VERIFY_NATIVE_CAPTURE=1`): 24 runs, embedded and cold, both seats, v7
  and v8, policy, end-turn, visit, level-up, poison and reuse scenarios. All
  passed, with every ended game replayed by `vcmireplay` and every policy game
  audited. Every engine process (embedded host, cold game and replay processes)
  confirmed at least 100 unchanged-view skips against a full build (counts are
  logged at powers of ten). An embedded host process keeps one count across
  every game of its run: in the 4- and 6-game embedded runs (9 of 16) the host
  confirmed at least 1,000, in the 2-game visit runs and the 1-game
  after-poison run at least 100. The processes that played or replayed the
  1,684-frame v8 policy game also confirmed at least 1,000. The exception is the
  stalled poison games, which confirmed at least one before they were failed by
  design. A disagreement fails the game; none occurred.
- Against the base code, scenario by scenario: each final run was compared with
  the base code's run of the same scenario, seeds and backend
  (`compare_base.sh`). 71 of 74 games have byte-identical learner histories and
  every outcome is equal, including the policy games, whose actions the
  accelerated code sampled itself (v7 and v8, 4/4 on each backend), end turn,
  chest red, mine red, level-up red, poison, recovery after poisoning,
  environment reuse and resubmitted actions. The three others are blue-seat
  games (chest blue on both backends, level-up blue embedded) that diverge after
  34 identical frames with equal outcomes.
- The base code itself varies there: in its own record, chest blue and level-up
  blue each matched 1/2 between backends, and in its pre-merge baseline mine red
  diverged after 64 frames across backends and between repeated runs of one
  backend. Nullkiller seeds its hero-chain search from `std::random_device`, so
  a match on any map only means both runs took the same Nullkiller variant.
- The accelerated engine resubmitting the base code's 8 policy games (v7 and v8,
  both seats, played on the base embedded backend) on both accelerated backends
  produced byte-identical learner histories and equal natural outcomes in all 16
  replays.
- Within the final code: embedded against cold matched in policy 4/4 and 4/4,
  end turn 6/6 and 4/4, chest red, mine red and both level-up maps 2/2 each, and
  chest blue 1/2 (the divergence above); one reused environment against a new
  environment per game on the same loaded engine 6/6; policy actions resubmitted
  to a new embedded run 4/4 and 4/4.
- Engine-free replay (host B) of the base code's 4 recorded v7 policy games (603
  decisions, 4,023 frames) and 4 recorded v8 policy games (342 decisions, 2,317
  frames) through the accelerated collector requires every choice with its
  log-probability and conditionals, every value and estimate, every recurrent
  prefix and the history to equal the persisted base rollout; all matched in
  every repeat. The audit replays (`audit_passes.py`) reconstruct every choice
  and value of the same games exactly in both passes.
- Differential unit tests compare the Python fast paths with the code they
  replaced or an equivalent reference: encoder subtrees (key order and leaf
  identity), decoders, strict JSON outcomes and error messages, weights-check
  decisions against the original byte-list digest under hooks, overrides and NaN
  payloads, and memo hits against cold recomputation. One-walk sampling has no
  direct unit test; the behaviour audit checks it by resampling and
  re-evaluating every saved choice, in the collector tests and in every real
  policy game. The native changes have no unit tests. They rest on the capture
  verification, `vcmireplay` and the base comparisons of learner histories
  above. `vcmireplay` checks every combined state digest the patched recorder
  writes; it reads the recorded part digests only after a combined mismatch, to
  name the diverging part. So the part digests that patch 0032 now derives from
  the combined serialization are correct by construction, but no run checked
  them.
- Independent adversarial review (summarized in the record): reviewers for the
  native engine, decoding, inference, per-episode verification and tests read
  the diff at `89f377e`, and two skeptics tried to refute each finding. Native,
  decoding and verification reported nothing. Five findings came back and one
  survived both skeptics. Four are fixed on this branch; the fourth, a coverage
  gap, is narrowed but not closed:
  - The weights check raised for a one-tensor module whose storage starts
    inside an 8-byte word (the multi-tensor real model was not affected). Such a
    view is now copied before it is read as words. The new test fails on the
    reviewed code.
  - The observation memo's 64 MiB bound counted only part of what an entry
    keeps. It now charges the key, the cache dict, every tree object and every
    tensor once; on the four recorded v7 games it charges 32.7 MB for 29.0 MB an
    independent deep measurement finds (×1.13). The charge costs about 0.25 ms
    per stored entry, that is per memo miss (host B, `review_fix_costs.py`). The
    new tests fail on the reviewed code.
  - Environment frames replaced their records after a text comparison that does
    not tell a tuple or a subclass from the exact JSON type, which the strict
    decoder does. They are now used only when every record also has exact JSON
    types; anything else takes the original strict-decoding path. The check
    costs about 0.44 ms per decision (host B, `review_fix_costs.py`). Records
    from the environments always have exact types, so no real game was
    affected. The new tests fail on the reviewed code.
  - No automated test covers the native unchanged-view capture skip (both
    skeptics called it a coverage gap, not a defect). A regression test now
    pins the `HEROES_VERIFY_NATIVE_CAPTURE` switch: it checks that the switch
    still guards the skip and the disagreement failure, and that it passes the
    environment filters of the embedded environment and every cold and replay
    launcher. It fails on the first version of the switch and passes on the
    reviewed code. The test reads source text and does not run the skip; the
    skip itself is covered only by the opt-in capture verification, which was
    on for every final revalidation run.
  - One differential test class did not run when its file was executed as a
    script. The main guard moved to the end of the file; no new test covers it.

## What was measured and not changed

These come from informal profiling on host A during the work (`strace` and
`perf` traces were not kept), except where noted:

- Nullkiller's TBB worker threads issued most `sched_yield` calls (about 28 per
  decision) while idle. Limiting its threads could change the opponent's search,
  so it is left alone.
- Engine load spent about 25 ms seeking through the LOD archive indexes (about
  90,000 `lseek` calls through `tellg`). It happens once per embedded worker and
  once per cold game; the cold reset (0.76 to 0.80 s on host B in the soak
  above) is mostly process start and library initialization.
- Ledger and history writes flush per record (about 45 writes per decision)
  because the files are crash evidence; batching them would weaken that.
- Training's admission check still runs its own history verification next to
  the one inside the replay check, and every weights check still runs; removing
  either would be a lost check, not a speedup.

## What this means for training

On the cold backend that training uses today, collection plus verification per
policy game in the probe's rows falls from 16.7 s to 8.6 s for v7 (×1.95) and
from 10.6 s to 5.8 s for v8 (×1.83); the accelerated embedded backend, which
also avoids the 0.8 s cold reset, brings the v7 game to 7.8 s (host B; on host A
the embedded v7 game fell from 9.5 s to 5.1 s). These are the probe's games:
freshly initialized seed-21 models on `hero-management-v1`, about 151 (v7) and
86 (v8) decisions per game, on Linux. The models, maps and machines that
training uses were not measured, and per-game seconds scale with game length
and host, so plan with the ratios or decisions per second.

The probe's rows are not training's admission path, and they differ in both
directions. Each probe row verifies the history once (inside the replay check)
and audits with a separate model object rebuilt from seed 21 inside the audit
timer (or loaded from `--checkpoint`), so its audit starts from empty memos and
includes the rebuild. Training's `verify_training_capture` also runs its own
history verification, hashes the capture files and writes reconstruction and
receipt files, which the probe does not time; and it audits with the collecting
model object, so its reconstruction finds what collection left in the model's
memos. The audit replays above show the size of that kind of saving: a second
audit pass over games the model has already worked through takes 52 to 56% of
the first for the accelerated code (about 4 ms less per decision), against 90
to 94% for the base. There the first pass, not collection, filled the memos; a
collection pass encodes the same observations and trees, so it should leave the
same encodings, but that path was not timed.

That sharing holds only while a game's encodings fit in the memo bounds, which
last as long as the weights are unchanged (a PPO update starts them anew). A
memo that overflows is cleared whole, so when the distinct encodings between a
game's collection and its audit exceed the bounds (for observations about 940
at the 72 KB per observation that the measured games charge; 16,384
internal-node encodings), the audit gets little of it. Where it applies, the
reconstruction still recomputes the recurrent core, the head layers, sampling,
values and every check. A memo hit requires the same observation, or the
identical exact tree, and bit-identical encoder parameters, so its result is
the recomputed one. What it no longer does is encode a second time what
collection already encoded: whole observations (map and observation tree), and
the internal-node encodings of each frame's semantics tree and of every
action-candidate tree. These are the inputs of the recurrent core and of the
policy heads, so the heads recompute only their own layers on inputs taken from
the memos. The base code already shared its text and number leaf encodings this
way, but no internal-node or observation encodings. Clearing the model's memos
before the audit restores the full recomputation.

The PPO update and anything else outside game capture are not measured here;
the total speed-up of a training batch is
`(capture + verification + update) / (capture' + verification' + update)`, so
it approaches these numbers only when the update is a small share. On the
embedded backend (host B), frozen-policy inference is about half of the
remaining collection time for v7 policy games (45% for v8), the engine (own
actions, Nullkiller and battles) 18% (22% for v8) and player-local ingest 11%
(11% for v8). On the cold backend the same parts are about 41%, 17% and 11% for
v7 and 34%, 18% and 8% for v8, and most of the rest is other collection work.

## Reproducing

Each configuration is a worktree whose VCMI source is prepared with
`tools/prepare_vcmi.py` and then built in Release with the embedded engine on
(`-DENABLE_EMBEDDED_ENGINE=ON`, the `heroes_embedded_engine` target included),
as in [the embedded engine build](embedded-engine.md#build). One probe run:

```
cd ROOT && .venv/bin/python -m validation.probes.validate_embedded_engine run \
  --binary vcmi/build/bin/vcmiclient --replay-binary vcmi/build/bin/vcmireplay \
  --data GAME_DATA --output OUT --backend embedded --controller policy \
  --random-policy-seed 21 --interface adventure-dialogs-v1 \
  --map experiments/explorer/maps/hero-management-v1.vmap \
  --episodes 4 --choices 1024 --max-days 365
```

Use `--controller recorded --recorded BASE_POLICY_RUN` for recorded rows,
`--backend cold` for the cold backend, `--interface adventure-capabilities-v1`
for v8, and `HEROES_VERIFY_NATIVE_CAPTURE=1` for the native capture check.
`validate_embedded_engine compare --left A --right B --output COMPARISON.json`
compares histories, and `collect` builds a path-free compact record.

`tools/acceleration/` holds the scripts behind the records:

- Revalidation record: `revalidate.sh` (the 24 runs and the within-code
  comparisons; run with `HEROES_VERIFY_NATIVE_CAPTURE=1`), `base_replay.sh`
  (resubmitting the base code's policy games), `compare_base.sh` (each final run
  against the base code's run of the same name) and `capture_counts.py` (the
  path-free capture verification summary). The compact record was built with
  `validate_embedded_engine collect --runs FINAL/*/ RESUBMITTED/*/ --attach
  FINAL/compare-*.json FINAL/codec.json RESUBMITTED/compare-*.json
  BASE_COMPARISONS/compare-base-*.json capture-verification.json --output
  RECORD` (the trailing slashes select the run directories; the same folders
  also hold the scripts' `.out`, `.err` and comparison files).
- Acceleration record: `measure.sh` runs the host B measurements in the order
  they ran there (cold A/B, soak, scaling, v7 controller replay, embedded A/B,
  then, in later sessions, the two-pass, audit, v8 replay and review-fix
  measurements) and writes the host B directories and line files that
  `record-notes-2026-10-26.json` names: `ab.sh` (interleaved A/B, embedded and
  cold), `soak.sh` (one reused environment per seat under `perf stat`),
  `scale.sh` (concurrent workers), `controller_replay.py` (engine-free
  controller replay, `--repeat 2` for the second pass), `audit_passes.py` (the
  policy audit twice with one model) and `review_fix_costs.py` (the costs of two
  review fixes). The v7 controller lines came from an earlier copy of
  `controller_replay.py` that did not yet print per-pass times; the record uses
  only their per-decision times. Host A's embedded A/B (`ab.sh`) and scaling
  lines (`scale.sh`, on `89f377e`) were made the same way before the restart.
  `record.py NOTES RUNS_ROOT OUT` then builds the record; RUNS_ROOT must hold
  the host B outputs, host A's `ab-final/` and `scale-host-a/results.txt`, and
  the `capture_counts.py` summary at
  `accel-evidence-final/capture-verification.json`.
