# Temporary troop-slot restriction and passive owned-army information

The user requested a second temporary simplification alongside automatic combat:
the policy buys troops but cannot rearrange individual slots. The restriction and
its removal checklist are recorded in [the permanent register](../temporary-restrictions.md).
The user explicitly retained whole-hero entry/exit from the town garrison and
requested current own-army composition and combat statistics without clicks.

## Source findings

Native town recruitment inserts creatures into the upper army: the garrison hero,
or the town itself. Banning whole-hero garrison entry as well as individual transfers
would prevent a visiting hero from collecting purchases through the existing
interface. `swap_garrison` therefore stays available. Native entry into an empty
hero garrison combines the town army with that hero subject to normal capacity
rules; exit preserves the hero's army. No automatic scripted transfer was added.

The previous observation did not contain full army information. The map roster
contained hero location and movement; troop type/count appeared only in opened
hero/town/exchange windows. Opening an individual stack added upgrade choices,
not its attack/defense or other current combat statistics. The user's assumption
was corrected before implementing the extension.

## Versioned behavior

`visible-interaction-budget-recruit-only-v1` adds seven explicit exclusions to the
existing revealed-movement/visible-budget support: split, merge, swap and transfer
stacks; open a stack; select an upgrade; apply an upgrade. All historical support
versions retain their prior behavior. Collection, stored probabilities, independent
policy reconstruction and native recipe compilation use the same selected version.
No penalty, reward shaping or strategic movement rule was added.

`own-army-details-v1` is an explicitly negotiated, additive v8 observation extension.
Its fields are absent from historical observations and their neural semantic trees.
Every current owned hero and town has one row, including empty town armies; a
garrison hero is not duplicated as the town's upper army. Rows contain occupied
slots, public creature ID/name, count, current attack, defense, damage bounds,
health, speed, shots, casts, morale, luck and shooter/flying/undead flags. Native
values use the same current stack/hero modifiers and ranged-mode convention as
the adventure creature popup. These are not base-creature constants or a prediction
of a particular battle. Conditional abilities are not exhaustively enumerated.

Only owner-filtered callback heroes and towns feed the new view. Raw observations,
negotiation, launch identity, finalization and native replay header bind the extension.
Native replay regenerates the complete payload; field omission or tampering is not
normalized away. The encoder binds each army to its owned subject without treating
opaque object handles as learned features. Model tensor shapes and the preserved
random tiny initialization remain unchanged, but inputs/action probabilities change:
this is a new experiment, not an exact speed optimization or checkpoint update.

## Validation

A new build directory compiled all 782 native objects, including client, server,
replay and all five AI libraries. Configuration took 2.014 seconds and compilation
57.441 seconds. The canonical dependency check passed. The frozen runtime is
`Heroes-artifacts/runtime/local-v8-passive-armies-v1`; it contains nine freshly
signed binaries, 592 resources and 87 hash-bound external dependencies. Actual
client loading resolves its frozen `libvcmi`, without the donor build directory.
The existing accepted runtime remains unchanged. This clean build avoids reusing
objects after the adapter class-layout change.

The first bounded native attempt passed all six technical prefixes in 9.638
seconds under a 600-second supervisor deadline: legacy mode, original-map Red/Blue
recruitment and garrison controls, authored artifact Red/Blue controls, and the
frozen tiny policy. Its 24 mechanics checks passed. The tiny seats made 64 and 25
choices; every choice and value reconstructed exactly, without weight changes.
All six captures passed native replay and independent history reconstruction.
After the historical-trainer isolation fix, a second attempt used fresh seeds and
passed all six prefixes and 24 controls in 10.232 seconds. Its tiny seats made
62 and 64 exactly reconstructed choices. Both attempts are retained; the final
acceptance binds the final 214-file implementation inventory.
Adding one to a captured own-stack attack was rejected by native regenerated-record
comparison, not merely by a file-hash gate. The artifact fixture tests a declared
Sword of Judgement stat effect; it is not evidence that the original map starts
with that artifact.

No technical prefix was a completed scored game. These checks do not establish
playing strength, useful travel, a higher win rate or faster complete games.
The original map, budgets, rewards, small model tensors and original initialization
are preserved. All raw evidence stays in the private durable artifact directory.

The first full offline pass ran 2,239 tests and reported 10 failures, 13 errors and
33 skips. The failures exposed historical source pins that needed a narrow opt-in
compatibility review and a naive reference encoder that included absent optional
fields. Earlier focused test development also exposed fixtures using explicit
null fields, incomplete owned-army rows, invalid creature name identifiers and
an incorrect expectation that a raw history reader would enforce policy support.
Fixtures and contract checks were corrected; no failed observation was normalized
away and no replay or admission gate was removed. The review also caught a real
isolation gap: the historical local trainer inherited the shared registry of
experimental support versions. It now accepts only its two previously released
versions and rejects passive-army declarations at session, plan, configuration,
checkpoint-metadata and lineage boundaries. Thus a new tiny experiment cannot
silently change a historical training continuation. An additional focused test attempt started before the pin writer finished and
loaded stale constants; a tuple/set fixture typo also failed. The corrected,
stable focused pass ran 49 tests successfully (one skipped). Final results are
recorded in the compact validation receipts below. The second full run executed
2,252 tests with one failure and 33 skips: one additional legacy test still
expected all globally registered experiments to be admitted by the old trainer.
Its expectation now explicitly names the two historical versions. Only that test
fixture changed after the second full run; production behavior was not loosened.
The affected module and launcher/native focused regressions were then rerun.

## Manual use and limits

```sh
cd /Users/orsi/Projects/Heroes
./play-one-tiny-selfplay.sh --check
./play-one-tiny-selfplay.sh --run
```

Only the user launches the full game. Both seats still get 9,920 ordinary and
320 reserved dialog choices; the hard limit is 10,240 per seat and day 365.
The default capture deadline is 1,200 seconds, with a 3,600-second total deadline.
The progress bars and total runtime output remain. This script plays one frozen
untrained tiny-policy diagnostic, with no PPO update or training admission.

The earlier user-launched tiny game remains historical evidence under the old
revealed-support mode: Red 3,613 choices, Blue 6,907 choices, day-limit cutoff,
no winner, exact replay/reconstruction, zero updates. Its consumed seeds must not
be reused. The new mode uses the same persistent tiny ledger and its next unused
reservation. The trained 1753 self-play state and original 1728 campaign remain
protected and are not relabeled with this intervention.

The [restriction register](../temporary-restrictions.md) contains the required
removal checklist. The legacy 1753 evaluator retains its existing runtime and old
observation/support; acceptance of the new tiny runtime is not a claim that the
historical evaluator has been rebuilt.

Native release evidence: [compact acceptance](../../validation/passive-armies-native-2026-10-30.json).
