# Update92 to v8 preparation

This document defines the reviewed preparation and launch gates. Actual activity
is recorded in the dated launch/campaign evidence. The old local365 session is
consumed and must remain untouched. The new authorization permits
at most two hours locally, starting at actual launch, only after all requested
v8 features pass native validation and the coordinator explicitly releases it.

The [Python preparation checks](../validation/builtin-v8-preparation-2026-10-25.json)
record local v7 regression results and an isolated, explicitly uncommitted v8
contract snapshot. They are not native game validation.

## Exact parent

Use `/Users/orsi/Projects/Heroes-artifacts/runs/local365-continuation-092/plan.json`
and its `execution/` directory. The immutable identities are:

| Artifact | SHA-256 |
| --- | --- |
| `plan.json` | `aa65d37019dc62252fee9d6269719c31d48b9df011fc9886abf59c314835517e` |
| `execution/report.json` | `a76c8e1ec41e7ce4196a1f54c4bb343ee1a1a92a25e2913bf6b94402e05c353a` |
| `execution/after-policy.pt` | `1d6f9a68b61fd60d6c6e49155b2c8b479c0324b2d3c5bd89ddda3e97fbaab344` |
| `execution/after-resume.pt` | `1c21b70deaf35de410501cf1b6e1b37768dfc054cbcd23bfe86df6c05f483ce4` |

The parent has 92 updates, 368 rollouts, 21,062 decisions and 783 Adam steps.
The read-only [audit](../validation/update92-v8-resume-audit-2026-10-25.json)
restores canonical, backup and private copies and inventories every named tensor,
Adam slot, RNG, progress and archived-pool identity. It also binds the consumed
old session and controller result. Refresh into a **new** output path:

```sh
HEROES_ARTIFACT_ROOT=/Users/orsi/Projects/Heroes-artifacts \
/Users/orsi/Projects/Heroes/.venv/bin/python tools/audit_update92_v8_resume.py \
  --artifact-root /Users/orsi/Projects/Heroes-artifacts \
  --output /tmp/update92-v8-parent-refresh.json
```

## Explicit contract transition

The target is `adventure-capabilities-v1`, history/actions v8, protocol 10,
model `heroes-context-model-v6`, distribution `heroes-context-policy-v6`.
The coordinator owns its definition and native implementation. The built-in
adapter resolves only exact v7 or v8 profiles and rejects mixed declarations.
Existing v7 filenames, serialized envelopes and default collectors remain v7.
Replay, finalization and training admission retain all native proof requirements.
Ordinary resume explicitly rejects an interface transition, even if a runtime
migration object approves source changes.

The current target design adds semantic candidates without adding parameter
modules. Before migration, pin the final committed target and verify this again:

1. Restore both exact parent files through their original v7 contract. Match all
   39 parameter names, order, shapes and dtypes against the target model.
2. Copy each named tensor unchanged. Restore Adam groups, per-parameter slots,
   step counters, Python/Torch/minibatch RNG, progress and sampler state exactly.
   Refuse unmatched names or shape changes; do not initialize new tensors silently.
3. Produce distinct v8 checkpoint identities and explicit transition metadata
   binding both source commits, old file hashes and target contract. The weight
   digest includes model configuration, so it changes even when tensors do not.
4. Preserve the original archived pool as historical evidence. If v8 checkpoint
   loading requires compatible pool copies, derive new identities and record
   each old-to-new digest mapping while proving tensor equality. Never execute
   archived neural opponents or relabel old pool hashes.
5. Save to a fresh migration directory, restore the resulting policy/resume pair
   independently, and compare every named tensor, Adam slot, RNG and progress
   field. Preserve originals and publish the derived pair privately. No PPO step
   is needed or permitted as a migration check.
6. Establish a new both-seat v8 frozen baseline after release. Equal tensors do
   **not** imply equal probabilities: additional legal actions change support.
   Keep v7 and v8 behavior cohorts separate.

The implemented converter and fresh-session controller below must be reviewed
against the final contract before use; their launch command requires release. The
legacy local365 controller remains bound to its consumed session and v7 scope.

## Release gates and fresh campaign

All of the following are required before the coordinator release:

- Native marketplace exchange; spellbook/war-machine purchases; artifact
  inventory/equipment/exchange/assembly; adventure spells; level-up skill choices;
  Eagle Eye/Stables notices; full-army neutral joining. Validate legal and invalid
  paths, visibility boundaries, resource costs and exact capture/replay.
- Final source and runtime identities, clean checkout and
  `python3 tools/prepare_vcmi.py --check`; compatible native verifier and map hash.
  Teacher does not modify Claude's embedded engine or Linux performance work.
- Asset-free v7 regression tests and v8 integration tests. Tests using mocked
  native proof are labelled synthetic and cannot satisfy native feature gates.
- Reviewed explicit checkpoint migration and independent restore proof.
- Fresh campaign/session/lock/controller names, never old local365 paths. Bind
  configuration, migration receipt and release evidence before a single launch.

At launch create the new deadline and watchdog from the actual start time, with
an at-most-7,200-second wall budget covering the whole controller and a reserved
cleanup/preservation period. Use 365 days, 5,120 total choices, Nullkiller Easy
and BattleAI, terminal-only rewards and complete four-game PPO batches. Require
both-seat held-out evaluation, exact capture/history/policy reconstruction and
native replay before admission. Retain all failures and unstarted games.

The failed sequence93 and its seeds are burned. Begin at sequence94 or the next
unused sequence after scanning all existing plans; do not retry93 as new
experience. Stop on technical failure or deadline, preserve the last fully
admitted update in independent local and private copies, verify cleanup, and
report all outcomes. No cloud launch is authorized by this document.

## Implemented preparation and launch tooling

The Teacher tooling now implements the converter and separately gated session
controller. It requires an explicit coordinator release; this document is not
that release. Native suite completion and a final committed source/runtime are
prerequisites. Temporary conversion of the real
update92 pair passed exact named-tensor, Adam, RNG, progress and archived-pool
checks; those temporary files were removed. See
[the exact-parent test](../validation/v8-converter-exact-parent-test-2026-10-25.json).

The ordinary target map is **hero-management-v2**, SHA-256
`ccfc8467847067a75091d68626e5b241839036ab729b5520ce4445a75a45b7b3`.
`builtin_v8_migration.map_identity` verifies its authored scenario using
`create_capability_scenarios.verify_scenario(fixture=None)` plus exact bytes.
The sole baseline map change is the native hero level cap from1 to0. v1's
verifier remains unchanged. The migration records source, interface, runtime
and map transitions explicitly.

After the final source/runtime is committed, these preparation commands create
no session and launch no gameplay (replace the explicit path placeholders):

```sh
export HEROES_ARTIFACT_ROOT=/Users/orsi/Projects/Heroes-artifacts
PYTHON=/Users/orsi/Projects/Heroes/.venv/bin/python
$PYTHON tools/v8_teacher_campaign.py prepare-migration \
  --parent "$HEROES_ARTIFACT_ROOT/runs/local365-continuation-092/plan.json" \
  --runtime-bin /absolute/final-v8-runtime/bin \
  --data "$HEROES_ARTIFACT_ROOT/game-data" \
  --game-map /absolute/hero-management-v2.vmap \
  --output "$HEROES_ARTIFACT_ROOT/migrations/update92-v8-final"
$PYTHON tools/v8_teacher_campaign.py verify-migration \
  --plan "$HEROES_ARTIFACT_ROOT/migrations/update92-v8-final/plan.json"
$PYTHON tools/v8_teacher_campaign.py preserve-migration \
  --plan "$HEROES_ARTIFACT_ROOT/migrations/update92-v8-final/plan.json" \
  --private-repo "$HEROES_ARTIFACT_ROOT/private-local365-state"
```

Private preservation copies only the derived baseline and its manifests into
`checkpoints/local-v8-update92/migration-<plan-digest>/` on the existing authorized
private result branch. It verifies the copy, commits/pushes explicit paths and
then reconstructs the pair from immutable Git blobs, verifying bytes and exact
restore again. The public report contains no raw checkpoints or game assets.
A private preservation receipt is mandatory in the release and checked before
the one-use session is burned.

The coordinator-authored release JSON must use schema
`heroes-local-v8-update92-v1-release`, `coordinator_released: true`, a nonempty
`release_message`, exact `authorization` and `configuration` from
`builtin_v8_session`, current clean `source`, migrated `runtime` and `map`, and
bindings for `migration_receipt` and `private_migration_receipt`. A required
`native_acceptance` binding points to the final 28-case compact summary. The gate
rederives its bound raw reports, both-seat Dimension Door visibility comparisons,
current native/codec implementation fingerprint and frozen runtime/client bytes.
Documentation and controller-only changes do not invalidate mechanics evidence.
Each of the 26
`feature_cases` contains `feature`, integer `player`0/1, `passed: true`,
`replay_verified: true`, and bound native diagnostic `plan` and `evidence`
(report). Supported feature names are the module's `FEATURES` tuple. Validation
matches actual passed case/seat rows and the released runtime; Boolean labels
alone cannot satisfy a feature gate. Optional `additional_evidence` binds other
review/validation reports. No preparation command manufactures this release.

Only after explicit release:

```sh
$PYTHON tools/v8_teacher_campaign.py launch \
  --release "$HEROES_ARTIFACT_ROOT/local-v8-update92/release.json" \
  --private-repo "$HEROES_ARTIFACT_ROOT/private-local365-state"
```

The fixed new session is `local-v8-update92/campaign-session.json`. Exclusive
session and worker locks prohibit a second launch. Its supervised worker must
have the original supervisor as its parent. The initial both-seat v8 baseline
runs **inside** the new two-hour budget before any update; later fixed held-out
pairs run each fifth update. Batch plans retain the existing strict admission
and independent preservation gates. Legacy cloud/local365 controllers and their
consumed session paths are unchanged.

An intended batch path and full schedule are journaled atomically before any
prepare/execute/preservation subprocess. Failures retain partial raw capture,
existing reports, unstarted rows and a separate failure/preservation record;
private publication failure is explicit and leaves the last admitted parent
unchanged. No failed batch is retried. Cleanup uses the single absolute wall
and monotonic hard deadline across all operations, scans only journaled
session-owned run directories and reports any incomplete cleanup as a failure.

## Mandatory query boundaries

Built-in cutoff, replay reconstruction and collection use
`PlayerHistory.requires_choice`. Deferred choices remain genuine policy samples:
`choose_dialog`, `choose_level_up`, `choose_spell_target` and
`cancel_spell_target`. The evidence checker reconstructs the exact delivered
panel at each reply, rejects optional actions in the drain, and requires actual
native issuance/acknowledgement through the unchanged history contract. No
cleanup reply or acknowledgement is synthesized.

Neutral joining uses the explicit exchange context and native `close` query
reply. As with the existing own-hero exchange, its ordinary pending decision can
be a cutoff: the one-player runtime stops with the captured unacted decision,
without issuing a synthetic dismissal. It must not report that query as closed
or the game as naturally completed. Native neutral-joining/replay evidence is
still a release gate.
