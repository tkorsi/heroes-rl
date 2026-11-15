# Temporary experiment restrictions

These restrictions simplify early adventure-policy learning. They are not final
rules of Heroes III and must not be forgotten when interpreting results.
The current cycle uses `hero-management-v4`; historical maps and captures keep
the rules under which they were created.

| Restriction | Current scope | Removal work |
| --- | --- | --- |
| Prebuilt starting taverns | `hero-management-v4` retains the v3 starting Tavern in both towns at no resource cost. Hiring still costs native gold and requires a free visitor slot. | Remove each starting Tavern in a new map version; compare recovery after hero loss. |
| Calendar neutral spawning restored | Since v4, native special-month conditions can create wandering neutral armies. This does not mean new armies appear every month. Initial neutrals, existing growth and town growth retain their native rules. | This restriction was removed on November 12, 2026. Keep v3's disabled-spawning map for historical provenance. |
| Automatic combat | Adventure experiments delegate tactical combat to BattleAI. | Add and validate a separate battle policy before claiming end-to-end learned play. |
| Manual troop-slot management restored | New cycles from checkpoint 1,768 use `visible-interaction-budget-hero-relative-stack-management-v1`, restoring split, merge, swap, transfer, stack inspection and upgrades. Historical recruitment-only captures retain their original support. | Prepared for the next user-run cycle; not yet observed in new games. |
| No artifacts anywhere | `no-artifacts-v1` removes all hero equipment and disables every artifact, including spellbooks and war machines, throughout the game. Starts, hires, purchases, loot and rewards are covered. | Restore normal artifact acquisition in a separately versioned experiment; report that removing catapults and spellbooks changes siege and spell access. |
| One-action town cooldown | `visible-interaction-budget-recruit-only-town-cooldown-v1` excludes `open_town` after a player closes a town, until that player completes a different action. It also applies after an empty visit. | Review repeat openings and menu alternation after the frozen game. Remove the cooldown by selecting a different support version and retain the original version for replay. |
| Consecutive identical commands | `visible-interaction-budget-hero-relative-cooldown-v1` retains the town rule and excludes the previous command for each player. Another hero, tile or quantity is a different command. Mandatory choices and window exits stay available, and a repeated end turn is allowed when it is the only available action. | Measure repeated commands, travel and menu alternation. Select a separate support version to remove the restriction. |
| Revisited swap layouts | New cycles use `visible-interaction-budget-hero-relative-stack-cycle-v1`: optional `swap_stacks` cannot recreate a complete owned-army layout seen since meaningful visible progress. Window reopening and reversed slot arguments cannot bypass this rule. Split/merge/transfer/upgrade actions retain their existing support. | Introduced for the user-run 1,000-game continuation from 1,769; evaluate useful consolidation and action counts. Historical versions keep their original masks. |

The historical troop-slot restriction was explicitly requested on 30 October 2026. It excludes
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

The user stopped the tiny-network experiment and retained this restriction for
the next trained-1753 manual game. Historical captures retain their original
support. The prior trained-1753 evaluation and completed training campaigns are
not relabeled as restricted experiments.
This change does not authorize launching a training campaign.

On 31 October 2026 the user first chose to remove all opening equipment, then
explicitly expanded the rule to every artifact for every hero throughout the
game. This includes ordinary artifacts, spellbooks, war machines and all later
acquisitions. Normal initialization and starting-bonus RNG draws still occur;
gold/resource bonuses are retained. The map and army composition are unchanged.
Native rules reject prohibited acquisitions, and the adapter reflects those
rules. No policy dialog is silently closed. Empty artifact windows may still be
opened; this rule is not a claim that repeated empty-window visits are useful.
Review and remove the entire `no-artifacts-v1` restriction explicitly before
restoring ordinary artifact behavior. Siege and spell access intentionally differ.

On 7 November 2026 the user added the tavern and calendar-spawn restrictions and
chose gradual removal of temporary restrictions in later experiments. Each
removal must declare its rule/version and parent checkpoint, preserve the prior
variant, and distinguish a rule effect from policy learning. The v3 variant
passed native validation and was used by training cycles through checkpoint
1,763; historical one-game modes retain `hero-management-v2`.

On November 12, 2026 the user restored native calendar spawning in v4 while
retaining starting taverns and the other restrictions. The selector checks that
v4 differs from v3 only by the spawn override and descriptive text. It reuses the
pinned, replay-verified native-default spawning control from v3 acceptance;
this is not a claim of a new v4 mechanics capture. Subsequent training has used
v4, and the latest complete published checkpoint is 1,765. See the
[research journal](research-log.md) for each run's map, parent and outcome.


On November 15, 2026 the user requested restoration of troop-slot management
for 100 fresh games continuing complete checkpoint 1,768, with CPU inference and
MLX PPO. The new support version retains hero-relative coordinates, consecutive
command exclusion, town reopening cooldown, revealed movement and visible
interaction budgets. Only the seven troop-slot action exclusions are removed.
Native legality, capacity and mandatory-dialog rules remain. Starting taverns,
no-artifacts-v1 (including spellbooks/war machines), automatic combat, neutral
spawning and scenario limits are unchanged. Historical rules remain available.
No new game or training run has been performed for this rule change yet.
