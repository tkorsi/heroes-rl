# Artifact inventory audit, 28 October 2026

## Scope

The user questioned whether Orrin's cancelled equipment request referred to a
nonexistent artifact. The earlier repair verified cancellation provenance and
exact replay, but did not inspect the source inventory. A slot address alone
does not prove that an item exists.

This audit reconstructs failed batch 555, episode 3, from recorded commands.
It starts no new game, reruns no Nullkiller decisions, reschedules no consumed
seed, admits no experience and changes no training runtime or setting.

## Confirmed origin and request

The exact map SHA256 is
`ccfc8467847067a75091d68626e5b241839036ab729b5520ce4445a75a45b7b3`.
Both authored hero inventories are empty; no artifact objects are placed.
Separately, native lobby settings request `RANDOM` starting bonuses for both
players. `CGameState::initStartingBonus` resolves these to artifact, gold or
resources. Both resolved to artifacts in this recorded game:

| Stage | Orrin, Red, hero 1 | Edric, Blue, hero 3 |
| --- | --- | --- |
| Initialized game | Collar of Conjuring, type 76, instance 241, neck 2 | Necklace of Swiftness, type 97, instance 242, neck 2 |
| Packet 523, `BattleResultsApplied` | Collar remains equipped; looted necklace 242 enters backpack 19 | Necklace removed through battle loot |
| Packet 524 | Inventory unchanged | Defeated hero removed |
| Request 91 after packet 526 | Attempts necklace 242 from backpack 19 to occupied neck 2 | Already removed |

The same necklace instance 242 is followed throughout. The original learner
artifact-panel history independently shows it equipped before the battle.
The policy performed no artifact move in its 30 choices.

At the recorded send point, the necklace exists, both slots are unlocked, and
native slot compatibility passes with replacement allowed. Orrin attempted to
replace Collar of Conjuring with the looted Necklace of Swiftness. That equipment
request remains verified terminal-unapplied. The preceding battle loot transfer
executed; the requested equipment replacement did not.

## Transparent diagnostic code

`tools/audit_artifact_inventory.py` creates separately instrumented copies of
the production verifier using checked unique anchors. It hashes all originals,
generated sources and its helper. `tools/native/ArtifactInventoryAudit.h` reads
initial inventories and resolved bonuses, records inventory changes after
regenerated client packets, and inspects artifact requests at their recorded
send point. Presence, instance/type/name, source/destination contents, locks,
compatibility and execution status are separate fields. Unsupported locations
are unknown, not reported as empty.

Its distinct `heroes-artifact-inventory-diagnostic-v1` report schema and
`training_admissible=false` prevent ordinary training admission. Full inventories
remain private forensic evidence and never enter policy observations. Neither
the accepted engine recipe nor the frozen training runtime was modified.

All 21 verifier translation units were compiled afresh without precompiled
headers; the two instrumented units replaced only diagnostic copies. Unchanged
native archives and original nonverifier runtime bytes were retained. Loader
tracing verified the intended copied library. Final replay checked all 90
authoritative inputs, 532 client packets, 91 requests, 30 responses and 777
transactions, with zero divergences or unchecked records. Native read-only
empty-slot and missing-holder controls reported absence. Four focused tooling
tests and canonical `prepare_vcmi.py --check` passed.

The first diagnostic invocation incorrectly combined legacy `--observations`
with `--contextual`. The verifier refused it before library loading or replay;
its report is retained. The corrected command uses the contextual ledger's
existing observation/message checks. An intermediate successful audit and the
final rebuild after correcting unsupported-location metadata are retained too.

All 64 retained audit evidence files were published privately and independently
downloaded and rehashed at commit `03c5f00206c734b5e03396f6e8e2f05463a46ca7`.
The first raw-content download failed with `transform: short source buffer`;
base64 API read-back succeeded. This transport failure is separate from replay.

## Limits

Compatibility is not full server authorization. Production terminal accounting
proves cancellation without execution, not gameplay legality. This additional
offline audit proves the actual source item and origin for this request. It
supports ordinary hero slots and labels other location kinds unsupported.

For the concrete neck destination, native `moveArtifact` rejects an empty source
with `No artifact to move!`. Source review also noticed that the separate
`FIRST_AVAILABLE` destination accesses the source before that null check. That
branch was not requested here or exercised by this audit; it is not claimed as
validated or repaired.

The [compact evidence](../../validation/artifact-inventory-audit-2026-10-28.json)
binds private raw receipts. Historical checkpoint 916 and failed evidence were
not modified. The follow-up `check --hours 10` refused because a separate manual
run already held the training lock. At 16:06 UTC, that run used frozen release
`36e0b90`, a one-hour cap, and reported checkpoint 953. This is a timestamped
observation, not a final campaign audit. No locks were bypassed or workers
stopped, and this audit launched no official training. The earlier remaining
count and next sequence describe the prior handoff, not this active run.
