# Artifact-free experiment and latest-game audit

## User-authorized scope

The user first requested artifact-free starts, explicitly including spellbooks
and war machines, then clarified that **all artifacts must be disabled for all
heroes throughout the game**. The final policy is `no-artifacts-v1`, recorded in
`server.experimentEquipmentPolicy` and preserved by replay. It covers starting
heroes, the tavern pool and later hires, ordinary artifacts, spellbooks and war
machines, purchases, loot and rewards. Native initialization still consumes its
original RNG draws; starting gold/resource bonuses, original map bytes and troop
composition remain intact. Equipped bonuses and cached morale/movement/mana are
removed or recomputed. Siege and spell access intentionally change.

The default remains native behavior. The opt-in changes native game rules and
corresponding available actions; it is not an optimization. Unknown options and
unsupported or mispackaged runtimes fail before gameplay. No policy-controlled
dialog is silently answered. Empty artifact windows can still be opened. The
restriction is temporary and must be explicitly reviewed and removed later.

The one-game manual launcher retains trained checkpoint 1753, hidden size 64,
feature size 32 and 74,344 parameters; both seats are frozen. It retains the
recruit-only troop-slot restriction, passive own-army information, revealed move
targets, automatic combat, terminal-only rewards and doubled choice limits.
The historical model/Adam/RNG/progress/sampler/pool state and prior captures are
preserved. No full policy game or training campaign is authorized by preparation.

## What the latest completed game actually showed

The source is the verified private run `run-20261030T214036Z-d1d3a205`, not the
older collar/necklace incident. Initial Red and Blue artifact panels contain only
a locked catapult (artifact type 3, slot 16), automatically supplied by native hero
initialization. No treasure artifact appears in either player's recorded panels
throughout this game. Red's later spellbook is first recorded on day 212 on Ingham,
hired on day 211; the first-aid tent appears on Rion, hired on day 292. These came
with recruited heroes rather than map treasure. Exchange panels can list both
heroes' equipment; repeated catapult entries are distinguished by their subject.

Blue did not defeat Red's hero. On day 18, Blue's Edric attacked Red's town with
25 pikemen and lost to its 42-pikeman garrison. Orrin was alive at (0,18).
Red later hired the defeated Edric on day 225, making the same named hero reappear
on the other side. Red hired 12 times (11 distinct hires); Blue hired none.

The original weak neutral camps were defeated on days 2 (Red) and 14 (Blue).
Later neutral objects truly spawned at month boundaries, not merely through fog:

| Day | Creature | Position | Displayed replay event |
| ---: | --- | --- | ---: |
| 141 | Troglodytes | (10,10) | 2570 |
| 197 | Lizardmen | (5,18) | 3641 |
| 225 | Wolf Riders | (4,11) | 4714 |
| 281 | Gremlins | (7,9) | 7873 |

Both players received the corresponding new-month notices. The native global
trace proves each object was absent before creation. The ordinary new-month
double-growth/deity-of-fire path spawns wandering monsters; no experiment patch
overrides that mechanic.

Red lost 12 heroes in encounters with the new camps: five to Lizardmen, four to Wolf
Riders and three to Troglodytes. When Orrin died on day 197 he still had 25 pikemen,
while 381 pikemen remained in the home town. At day 356, the town had 675 pikemen
and the surviving hero had 14 pikemen and five archers. The policy never selected
the retained whole-hero garrison entry/exit action to collect those recruits.

Red targeted Blue's castle once on day 76, from (0,10), but the move ended at
(12,8) with no movement points. It did not complete the attack. Neither player
captured a town or mine. The game ended on day 356 at Red's choice limit, with
no winner and no weight update. These facts show behavior, not neural motives.

## Validation protocol

Build into a fresh directory and freeze a separate runtime; do not modify the
accepted previous runtime. Validate the actual library loading and canonical
engine recipe. Reserve an independent diagnostic namespace before each attempt.
Compare eight explicitly paired original-map seeds under native and empty-start
settings, inspecting both players. Require all random bonus branches and unchanged
resources/army composition. An authored equipment fixture exercises combination
artifacts, backpack contents and spellbooks. Additional controls must establish
that later hires have empty inventories, forbidden purchases are unavailable,
and later ordinary artifact acquisition cannot bypass the restriction.
Every technical capture requires native replay. A wrong-setting replay must fail.
A short frozen-1753 prefix must reconstruct both policies exactly. These controls
are unscored, excluded from PPO and not evidence of improved strategy or speed.

The first focused launcher suite ran 52 tests and found one outdated synthetic
source-inventory fixture. It omitted the newly required equipment source pins.
The fixture was extended; production source rejection remained intact. Final
native results and manual readiness will be recorded after acceptance.

Independent review found the cached mana issue; the follow-up audit also covered
movement and mixed-army morale. Two compilation attempts rejected direct access
to the private movement field. Both failed build logs are retained; the code now
uses the native public setter. The final rebuild passed. No C++ header or object
layout changed; the initial fresh build compiled all 782 objects, and the reviewed
source edit rebuilt its translation unit and linked dependents before freezing.

The first native attempt (`attempt-20261030T221059Z-bfc2a0`) failed after
3.44 seconds. The default case passed seven controls; the empty-start case
still exposed the native catapult. This was a real configuration defect, not
a permissive test mismatch: the new server setting was present in requested
settings but missing from the schema's required list. VCMI normalizes settings
by discarding optional nodes, so the engine never received the intervention.
The failed capture, frozen runtime and consumed diagnostic seed schedule remain
private and immutable. Cleanup and protected-input checks passed. The correction
adds the setting to the required schema with its native default, and preflight
will inspect the packaged schema as well as the compiled feature marker.

Before native attempt two, the user broadened the scope from opening equipment
only to a permanent temporary ban on spellbooks and war machines for every hero,
including purchases. The initial-only candidate was never accepted or launched
as a full game. The intermediate proposal used `empty-start-no-books-machines-v1`;
the failed first attempt remains separate.
An intermediate offline preflight review also caught that upstream schema files
contain JSON comments. The parser now handles comments while preserving quoted
URLs and escapes, and was checked against the real generated schema.

The final scope clarification also prohibited ordinary artifacts, including later
acquisitions. The intermediate book/machine-only proposal was not frozen or run.
The accepted candidate must use `no-artifacts-v1`, not either narrower proposal.

Independent review of the expanded acquisition paths found that a new raw
"unavailable" notice would not match the adapter's strict notice contract. The
implementation must recognize one exact, versioned equipment-restriction notice
while retaining rejection of every unrelated unknown message. The native factory
and ground-artifact controls will exercise that path; no arbitrary notice is
automatically accepted or closed.

Validation is scoped to the original map and the explicit diagnostic fixtures.
Known limitation: arbitrary authored mixed artifact/nonartifact rewards can still
produce notice shapes that the existing adapter does not support (including empty
reward text or XP components). The native item prohibition remains enforced, but
such a map can stop with an adapter rejection. The original map has no such reward
objects. This work does not claim general modded-map support.

The final-scope launcher test initially found an outdated expected label
(`Starting gear`); the test now checks the truthful whole-game `Artifacts OFF`
summary. One pin-generator invocation used system Python without PyTorch and
stopped before edits; it was rerun with the locked project interpreter.

## 4 November follow-up: behavior and second diagnostic failure

Both seats used identical frozen checkpoint-1753 weights, with different policy
sampling seeds and histories. Red built its Tavern on day 165 and first hired on
day 187. Blue never built a Tavern or any other building; its hall displayed
Tavern construction as allowed in 16 post-defeat decisions, with ample gold.
Hiring itself therefore never became available. At the day-18 battle Red had
only the original Fort, no Citadel/Castle upgrade. The 42-pikeman garrison beat
Blue's 25 pikemen and retained 31. Blue's pre-attack observation identified the
enemy town but did not expose its exact garrison or fortification level.

Red targeted x=0 in 1,655 of 1,906 movement commands (86.8%). It knew Blue's
town by day 50. After approaching it on day 76, it returned left on day 77.
The model scores absolute coordinates without color-relative mirroring; a
positional policy bias is plausible, but this trace cannot identify its learned
cause. Ten of twelve fatal neutral encounters came from adjacent guarded-tile
moves, not direct camp targets. Examples include one skeleton on day 271 and
one pikeman on day 310. The observations expose camp type/location but not
stack quantity or an explicit guarding-radius field. These limits matter when
interpreting threat assessment.

Native attempt `attempt-20261030T222257Z-52f098` passed eight paired controls
(112 checks, every random starting-bonus branch), then failed on the default
authored ground-artifact control. The move reached the Centaur Axe and spent
100 movement; the adapter rejected the ordinary artifact event notice before
returning a result. This was an adapter contract failure, not a route failure.
The narrow repair binds the exact catalog artifact event to its matching
displayed artifact component, AUTO mode and v8 only. Unknown or mismatched
notices still fail. The failed capture and consumed schedule remain preserved.

Following the current publication instructions, local tests, fixtures and the
diagnostic harness are preserved privately outside the publication diff. Earlier
unpublished implementation commits remain on a local evidence branch. No test
or diagnostic-tool changes will be pushed. The runtime acceptance receipt also
remains local; the production launcher binds its exact digest.
