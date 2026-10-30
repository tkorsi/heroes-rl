# Temporary experiment restrictions

These restrictions simplify early adventure-policy learning. They are not final
rules of Heroes III and must not be forgotten when interpreting results.

| Restriction | Current scope | Removal work |
| --- | --- | --- |
| Automatic combat | Adventure experiments delegate tactical combat to BattleAI. | Add and validate a separate battle policy before claiming end-to-end learned play. |
| No manual troop-slot management | The next tiny-policy experiment uses `visible-interaction-budget-recruit-only-v1`. | Restore stack rearrangement and upgrades in a separately versioned experiment after the adventure baseline is evaluated. |

The second restriction was explicitly requested on 30 October 2026. It excludes
`split_stack`, `merge_stacks`, `swap_stacks`, `transfer_stack`, `open_stack`,
`select_upgrade` and `upgrade_stack`. Buying creatures remains available, including
already upgraded creatures offered by a dwelling. Native recruitment chooses its
usual insertion/merge slot; the policy does not choose a troop-slot layout.

The user explicitly retained `swap_garrison`: moving a whole hero into or out of
the town garrison is allowed. This preserves the existing way to collect purchased
town troops without selecting individual troop slots. Capacity restrictions and
the existing visible interaction budget still apply. Artifact management is not
part of the new prohibition. Mandatory queries keep their explicit policy-selected
answers or close action; no dialog is silently answered or closed.

Consequences must be reported: manual transfers between heroes or garrisons are
unavailable, existing troops cannot be upgraded through the stack popup, and an
overflow query may have to be closed without receiving its offered troops. These
are deliberate experimental limitations, not claims that those actions are illegal.

The accompanying `own-army-details-v1` observation extension exposes each owned
hero and town's current composition and numeric adventure-map stack statistics
without opening a window. It does not reveal enemy armies. It does not enumerate
every special ability or predict damage against a particular enemy. The information
extension can remain after slot management is restored; it is not a reward bonus.

## Required removal checklist

- [ ] Review this restriction after evaluating adventure exploration and recruitment.
- [ ] Register a new support version restoring troop-slot operations and upgrades.
- [ ] Validate town/hero transfer, garrisons, neutral overflow and mandatory exits.
- [ ] Verify exact policy probabilities, historical replay and checkpoint provenance.
- [ ] Compare wins, natural completions, useful travel and repeated operations.
- [ ] Update this register and the research log when the restriction is lifted.

Historical captures retain their original support. The trained 1753 evaluation
and the completed training campaigns are not relabeled as restricted experiments.
This change does not authorize launching a training campaign.
