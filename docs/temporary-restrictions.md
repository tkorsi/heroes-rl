# Temporary experiment restrictions

These restrictions simplify early adventure-policy learning. They are not final
rules of Heroes III and must not be forgotten when interpreting results.

| Restriction | Current scope | Removal work |
| --- | --- | --- |
| Prebuilt starting taverns | The scenario `hero-management-v4` gives both starting towns a Tavern at no cost. Hiring still costs gold and needs a free visitor slot. | Remove each starting Tavern in a new map version; compare recovery after hero loss. |
| Automatic combat | Tactical combat is delegated to VCMI's BattleAI. | Add and validate a separate battle policy before claiming end-to-end learned play. |
| No artifacts anywhere | The engine setting `no-artifacts-v1` removes all hero equipment and disables every artifact, including spellbooks and war machines, for the whole game. | Restore normal artifacts in a separately versioned experiment; report that removing catapults and spellbooks changes siege and spell access. |
| Interaction guard | `visible-interaction-budget-hero-relative-stack-cycle-v1` ([heroes/game/guard.py](../heroes/game/guard.py)) keeps the policy from sampling these legal actions: a navigation step already taken twice since the last visible progress, further stack management by a hero or town after 32 such moves in a day, a swap that recreates an army layout already seen since the last progress, a move to an unrevealed tile, to the hero's own tile, to its previous target or to a target that failed since the last progress, the previous command repeated exactly, and reopening a town right after closing it. Mandatory choices and window exits always stay available. | Measure repeated commands, travel and menu alternation, then select a new guard version. |
| Game horizon | A game is stopped at day 365 or after 163,840 answered decisions per seat and then trained on without a win or loss. | Review once games end naturally well before the limit. |

Not restricted: calendar neutral spawning (restored in v4 on November 12, 2026)
and manual troop-slot management (split, merge, swap, transfer, stack inspection
and upgrades, restored from checkpoint 1,768).

The policy also receives `own-army-details-v1`: each owned hero and town's
composition and stack statistics without opening a window. It reveals nothing
about enemy armies and is not a reward.

Earlier guard versions (recruit-only, town cooldown, hero-relative cooldown,
stack management) were removed from the code in the November 2026 refactor. The
current version contains their rules except the recruit-only exclusion. Captures
made under the earlier versions can no longer be re-evaluated by this code; the
versions remain in Git history up to commit `5efe811`.

## How these decisions were made

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
