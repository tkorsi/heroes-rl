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

## Validation status

Implementation validation is in progress. A clean native build and frozen runtime
are required because the adapter class layout changed. Full training and ordinary
evaluation games remain manual. Any technical native controls are isolated,
bounded, freshly seeded, excluded from PPO, and reported separately with failures.
