# Map garrison adapter repair, 28 October 2026

## Prospective scope

The user authorized repairing the owned-mine adapter failure and related normal
game interactions before the next manually launched training session. This work
does not launch official training or extend any previous campaign deadline.

The last manual run started on 27 October at 20:05:05 UTC and stopped at
20:20:39 UTC. It admitted 212 natural losses in 53 complete four-game PPO
batches, advancing checkpoint 717 to 770. Sixteen evaluation games were separate
losses. The original 5,000-game target has 4,436 games remaining. Failed sequence
408 is consumed; the next unused sequence is 409. Its first game reached an
unsupported interface after 110 choices; three games were not started. That
batch supplied no samples or optimizer updates. Final restoration and shutdown
receipts passed; checkpoint and backup hashes were independently rechecked.

The failing action was a legal `move_to` to (5, 16) on day 3. It revisited the
player's sawmill. Native VCMI opened a garrison query, but ExplorerRL's handler
accepted only neutral-monster joining. The generic error incorrectly called this
"joining neutral army". This is missing adapter coverage, not a learner action
error, time limit, or PPO numerical failure.

## Hypotheses and acceptance gates

1. Supporting native map-object garrison queries resolves owned-mine revisits.
2. Normal and anti-magic garrisons require the same query lifecycle, with native
   ownership and troop-removal restrictions represented in legal action support.
3. Explicit close, inspection and supported transfers must remain policy choices;
   no automatic close, strategic movement mask, demonstrations or shaped reward
   are introduced. Terminal-only rewards and model/PPO dimensions remain fixed.
4. Existing hero exchange and neutral joining histories remain compatible. New
   map-garrison records must pass strict visible-history and native replay checks.

Validation is isolated and supplies zero training samples or optimizer updates.
Bounded scripted interface probes will cover both seats, capture/revisit of a
mine, empty and populated garrisons, ordinary and anti-magic variants, removable
and locked troops, full armies, explicit close and neutral-joining regressions.
Scripts are test controls, never learner demonstrations. Invalid withdrawal,
hero-last-stack removal and hidden-state disclosure must remain rejected.
Any discovered failures and repairs are appended below rather than rewriting
these hypotheses. No benchmark checkpoint replaces official checkpoint 770.

## Results

Implementation and validation are in progress. No successful native acceptance
or next manual launch is claimed by this prospective entry.

### First native attempt retained

The first locked-garrison control used a new frozen macOS candidate and failed
before stack inspection: the move delivered the native garrison query but the
adapter timed out waiting for engine completion after 30 seconds. It admitted
no training data and owned cleanup passed. This does not yet reproduce the
suspected occupied-slot transfer issue. Two later harness errors (the stack
inspection/return action names) and a fixture instance-name warning were found
by independent review and corrected before a fresh attempt. Old captures,
source hashes and the failed report remain in the private diagnostic directory.

### Mixed object-file build diagnosed

The second retained control opened the native query but failed strict history
validation because the pending query was followed by a map decision. Inspection
of the preserved candidate2 dylib established inconsistent compiled struct
layouts: `receiveGarrison` wrote the optional-engagement flag at offset `0x4c4`,
but the stale `installOwnHeroExchange` object read `0x4c0`. A copied source file
had an older modification time than an object built from the previous contents.
Canonical source agreement alone did not establish object-file freshness.

Both candidates were rejected. All owned C++ source/header dependencies were
forced to rebuild. This exposed an optional-player-ID compilation error that
had also been hidden by a stale object; it was corrected before rebuilding.
No successful native acceptance or change to official training is inferred from
these builds. The original two captures and binary/disassembly evidence remain
preserved. Final acceptance requires the rebuilt runtime's real replay tests.

### Native locked-destination defect reproduced

The fully rebuilt third candidate opened the garrison and passed explicit
stack inspection/back. Its valid one-unit deposit (hero slot0:25 to24;
occupied guard slot1:2 to3) received a native rejected acknowledgement.
`CGameHandler::arrangeStacks` compared the unrelated destination slot0 against
the source remainder. Patch0030 instead rejects a negative transfer into a
locked destination, preserving the withdrawal restriction and allowing deposits.
The predicate no longer emits an ERROR/system complaint merely for checking a
locked army. This is a native valid-action correction, not a performance change.
The failed attempt supplies no training and remains separate from acceptance.

### Rebuilt native acceptance

Frozen runtime `local-v8-garrison-repair1`, source84fb492, passed all16 new
map-garrison controls and all28 existing native-capability regressions for both
seats. Every control passed strict history audit, independent native replay and
owned shutdown within its independent deadline. These are scripted interface
controls only, with zero PPO samples or optimizer steps. Allied-garrison rights
have offline coverage only; the accepted two-player terminal protocol does not
provide a nonterminal allied native fixture.

The complete offline suite ran1,614 tests,33 skipped, with no failures. Historical
batch407 was independently reconstructed under the new parser:4 games,876frames,
130actions/values and362 conditional probabilities, all exact. The official770
model/Adam/RNG/progress/pool state also restored exactly.

The first frozen770 policy diagnostic failed during preparation, before any
worker/game: its launcher used the default Linux prerequisite path because the
Mac artifact-root environment had not been initialized before imports. Its
parent copy and eight reserved seed pairs remain preserved and are not reused.
A corrected launcher and a fresh diagnostic attempt are required; this failed
preparation is not native acceptance or learning.

### Final frozen-policy acceptance and manual continuation

The corrected launcher used fresh seeds in `policy770-diagnostic2`. All eight
actual frozen770 policy games completed naturally (eight losses,580 choices).
Native replay, strict policy reconstruction, source/runtime identity, guarded
choice support and owned cleanup passed for every game. Full model/Adam/RNG/
progress/pool state restored exactly and remained unchanged; zero diagnostic
samples or optimizer steps were admitted. The preparation failure above and
its consumed diagnostic seed range remain retained.

A separate offline audit applied the actual770 model twice to16 completed
scripted native histories (126 decisions per pass). Both inference passes matched
exactly, including new garrison history tokens and available close/back actions.
Those fixture actions were scripted controls, not choices sampled by770 and not
training demonstrations. Two failed audit setup attempts remain referenced.

Release `garrison-dialogs-v1` now requires the frozen runtime manifest/load
preflight, canonical source agreement,16 new native controls,28 legacy native
regressions and the eight policy reconstruction proofs. Its accepted descriptor
SHA256 is `772444aa025f27208226af18796262cf807d4646c047f4e9f5504cf046ac3e98`.
The final targeted offline checks ran67 tests with one skip and no failures.

The read-only continuation check passed from source0df9db5: exact770 restore,
4,436 remaining games, fresh sequence409 and a10-hour session cap. The existing
visible interaction budget, architecture, PPO settings and terminal-only rewards
are unchanged. No official training was launched. Main-checkout readiness is
recorded separately after configuration activation.

Private commit `000897d9ec410173fc315aa00f98a0844635116e` on
`tkorsi/heroes-rl-training-state`, branch `codex/local-script-final770-checkpoint`,
preserves all1,816 diagnostic evidence/receipt files in a16.5MB archive. An
independent remote read-back verified every member hash. External game-data
symlinks and runtime binaries are excluded. Checkpoint770 has its separate
full-state restoration receipt. See the compact results for immutable hashes.
