# Compact cloud-to-local handoff:26 October2026

The transfer study and overnight PPO campaign are closed. This handoff starts no
model, game, recovery, trainer, watchdog or recurring supervision. Root owns
review/synchronization and any separately authorized, user-invoked local script.

## Public results and changed files

Finished study: `82e022e6145814ef2b96c9b21ec463f94e26ffc7`, branch
`codex/teacher-transfer-eval-20261026`, based on
`deddbf934675a768bd49d677762ae0622d51a3f7`.

Primary paths: `validation/transfer-eval-2026-10-26.json`,
`validation/transfer-eval-2026-10-26.jsonl`,
`docs/research/transfer-eval-2026-10-26.md`,
`reports/cloud/teacher-transfer-status.json`, and
`configs/teacher-transfer-eval-20261026.json`.
The machine handoff lists every changed file. Its own final publication commit
is the Git commit containing this document; it does not change the frozen study.
Operational cloud scripts have absolute cloud paths and expired one-shot
sessions: do not reuse them as a local training runner. No competing runner was
added for the new operating model. PR29/30 remain experimental source, not an
implicit merge or authorization for PPO.

## Exact checkpoint and dependencies

Private repository: `tkorsi/heroes-rl-training-state`. Full checkpoint412 is at
`5577155c0b9e9d859d795edc0831d9d8e800d9d6`, prefix
`checkpoints/overnight-20261026/campaign-v1/http-transport-003/train-0265-seat1`.

| File | Bytes | SHA256 |
| --- | ---: | --- |
|after-policy.pt|638195|`1803823cab4a2d74f436e8425e31a1dbe1892e58844955945e2c963d176d9b88`|
|after-resume.pt|1266293|`09eee900a5fbff507a05c39e8ad4379e5b95177608b38cb69390a3cb56fee321`|
|plan.json|58426|`4ceb55e2f477c8406e90eaa9a2b79c4a13514bd37b984b626e2ea79f24f95e23`|
|report.json|11258|`20c070ce2390315176073db880498ad1a7c86ddca2f4451bd82ad2c979327110`|

Both `plan.checkpoint` and `checkpoint_metadata.initial_checkpoint` require the
same update411 policy:638195bytes, SHA256
`9aeac52025a9bd0cd307303e8b06f732aefa640795ccf9e4fd9d7ba1f0ba65bd`.
It exists at the same private commit under
`checkpoints/overnight-20261026/campaign-v1/http-transport-003/train-0264-seat0/after-policy.pt`.
Do not edit the original plan/report or serialized metadata to relocate paths.

Private handoff `eb37149e06e751b6215fd4aad370badca94275cd` contains compact
`handoffs/2026-10-26-cloud-to-local-frozen412/manifest.json` and unchanged maps
under that prefix's `maps/` directory:

- `curriculum-3-0.vmap`:1644bytes, SHA256
  `7d88860234664580c564fc2c71554fd813519ace9904ff368dc30eef3c352019`.
- `curriculum-3-1.vmap`:1645bytes, SHA256
  `75303cc2b19c1d0ddc5fbac35315a951dfc9bb6e780b4245508f8a2a78e691f1`.
- `hero-management-v2.vmap`:2354bytes, SHA256
  `ccfc8467847067a75091d68626e5b241839036ab729b5520ce4445a75a45b7b3`.

These are authored maps, not redistributed game assets. Original licensed Data
stays private; use the existing local installation. Historical private transfer
commit `c7257a047373a55901e04bd9fa024dbb6df964d3` remains available if needed.

## Local macOS restore-only instructions

Use the exact Git blobs above (`git show COMMIT:PATH > LOCAL_FILE`) and verify
all sizes/SHA256s with `shasum -a256` before loading. Arrange a new local bundle:

```
LOCAL_BUNDLE/checkpoint/{after-policy.pt,after-resume.pt,plan.json,report.json}
LOCAL_BUNDLE/dependencies/update411-after-policy.pt
LOCAL_BUNDLE/maps/{curriculum-3-0.vmap,curriculum-3-1.vmap,hero-management-v2.vmap}
```

Root's already validated Mac harness is
`941f60b91b34e601d44e01a7659120df3fa6e2e1` (accelerated) or
`278322fbbf48b68dbc2d1d9e6c77712e87f61192` (base). Preserve the virtualenv
interpreter path; do not resolve its symlink into a different Python installation.
Use CPU Torch public release2.9.1, with the existing strict relocation option
`historical_torch_version="2.9.1+cpu"`; do not fake the historical Python/Torch
metadata or weaken any checkpoint checks.

A root-invoked verification command, **not executed by this handoff**, is:

```bash
"$VENV_PYTHON" "$FROZEN_SOURCE/tools/validate_checkpoint_acceleration.py"   --input "$LOCAL_BUNDLE" --prepare-only --backend cold --seat 0 --episodes 1   --binary "$ACCEPTED_LOCAL_BIN/vcmiclient"   --replay-binary "$ACCEPTED_LOCAL_BIN/vcmireplay"   --data "$LOCAL_GAME_DATA" --output "$FRESH_RESTORE_RECEIPT_DIR"
```

`--prepare-only` verifies full state and map identity without gameplay; it is not
native-runtime acceptance or a trainer. `verify_pair` restores the caller's
ambient RNG on return. A future authorized local training script must therefore
restore the actual working model+Adam, named minibatch generator, Torch/Python
RNG, progress, sampler and opponent pool from `after-resume.pt` after setup and
before its first rollout. Use the original plan PPO/Adam config, metadata from
`_metadata(plan, report['plan_sha256'])`, `at_rollout_boundary=True` and
`generators={'minibatch': generator}` in the existing checkpoint loader.
Do not resume from policy-only weights or reseed/reinitialize learned state.
Expected412 state:1648rollouts,95836decisions,3545Adamsteps; tensor digest
`c310ae6b0b295149641802577fa4469c7a1662d530108c0289e3ce8f9c726d84`.

Root's private Mac validation is `be0313b756675acf8d55723c010653f7ff5e87a8`.
Same-host native replay/reconstruction passed there. Strict archived Linux-to-Mac
numeric reconstruction failed identically in both Mac variants; no tolerance
was relaxed. Exact checkpoint restoration is distinct from cross-platform
trajectory equality. A future authorized local run needs its own same-Mac gates;
historical Linux captures must not be silently admitted as Mac training data.

## Findings, seeds and closure

Valid transfer results (Red/Blue wins):146=`1/12,1/12`,254=`9/12,5/12`,
412=`12/12,12/12` on varied easy positions. All three=`0/16,0/16` on the
original scenario.168valid games,40wins,128losses,0cutoffs;2unscored native
startup failures and22unstarted invalid-fixture entries. The8Linux gate games
are separate. Hero **visitable** position is anchor-(1,0,0):our original
anchor-only preflight missed off-map x=-1 for anchors x=0. Both runtimes failed
on that invalid fixture, so this is not an acceleration-specific regression.
Four layouts were excluded structurally across all checkpoints/seats; no frozen
start was moved or failed attempt retried.412 targeted all four new visible
town coordinates, but original-map winning strategy was not demonstrated.

Exact-continuation provenance, if separately authorized: next training sequence
266, Red, engine seeds911065--911068 and policy seeds921065--921068. Next fixed
periodic update416 (origin146, every5updates); the diagnostic did not advance
training progress/pool/sampler. Closed transfer reservations are engine810001--810032
and830001--830032, policy820001--820032 and840001--840032. Unstarted invalid
entries remain quarantined. Seeds outside those reservations begin810033/820033
and830033/840033; this is bookkeeping, not permission to run them.

All owned controllers, native workers and watchdogs are stopped; ownership locks
are unheld. No queued automatic restart or recurring model supervision was
created. Original checkpoint and evidence bytes are preserved. Complete transfer
captures:private `d00675305595e62078f8c9bab85b22d67b990504`; independent audit
seal:`24ae5b2ecb9ef035484887eb1ac9df47682ce347`. Root may propose an owned-town,
active-enemy-hero short-board bridge under a new explicit local budget; this
handoff authorizes no training, cloud recovery or new provisioned resource.
