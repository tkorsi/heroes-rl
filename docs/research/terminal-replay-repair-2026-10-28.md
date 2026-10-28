# Terminal native-request replay repair, 28 October 2026

## Prospective scope

The user requested investigation, repair, isolated validation and a safe manual
continuation after the latest local failure. This authorizes no official
training, automatic restart, cloud work or additional strategic intervention.

The actual run `run-20261028T131340Z-684287b9` advanced checkpoint 770 to 916
in 146 complete four-game updates: 584 admitted natural losses. Sixteen separate
evaluation games were also losses. The original target now has 1,148 admitted
games and 3,852 remaining. Failed sequence 555 is consumed; sequence 556 is next.
Its first two games completed, the third ended naturally but failed replay
validation, and the fourth never started. The whole batch supplied zero samples
and zero optimizer updates. Checkpoint 916 must not be replaced by 770.

The third game's terminal server request was `MakeAction` (type 198), request 90.
Nullkiller then queued `ExchangeArtifacts` (type 190), request 91, during final
callbacks. The terminal protocol cancelled this request before server execution.
Python replay accounting rejected its type before invoking native replay; its
outer admission error misleadingly mentioned supplemental compatibility. This
is a validation coverage failure, not an unsupported adapter query, native
crash, time limit, PPO failure or publication failure. The audit also found
`TradeOnMarketplace` (193) in the C++ terminal-tail allowlist but absent in Python.

## Acceptance gates

1. Preserve and independently restore the complete latest model, Adam, RNG,
   sampler, progress and opponent pool, plus original failure evidence and seeds.
2. Keep terminal cancellation tightly constrained: one unmatched native request,
   exact serialized packet identity, native controller, no authoritative
   execution and placement inside the final barrier's packet interval.
3. Reproduce the accounting rejection using copied recorded evidence, without
   launching the consumed game seed. Replaying captured commands is an excluded
   diagnostic and cannot retroactively admit the failed batch.
4. Build a new runtime from a fresh build directory; verify actual loaded
   binaries, source recipe, focused native accounting controls, all 16 garrison
   controls and all 28 legacy capability controls. Use fresh diagnostic seeds,
   bounded deadlines and owned cleanup.
5. Run eight frozen checkpoint-916 policy games in a separate namespace, with
   native replay, strict policy reconstruction and exact unchanged parent state.
   Diagnostic data never enters PPO. Record failed and unstarted attempts too.
6. Keep the original map, architecture, PPO settings, Nullkiller Easy/BattleAI,
   terminal-only rewards +1/-1/0 and `visible-interaction-budget-v1` unchanged.
7. Publish reviewed source and compact evidence; preserve raw evidence privately.
   Activate only an accepted runtime and verify the actual main-checkout
   `./continue-original-4720.sh check --hours 10`. The user launches continuation.

## Investigation notes

The main checkout's generated `vcmi/` still had the pre-garrison recipe and no
build directory. Its entire contents were preserved before fresh preparation;
the failed run itself used the correctly pinned frozen garrison runtime.
The removed implementation worktree is not needed to recover official state.

The saved session budget is two hours, although the user supplied a ten-hour
launch command in this investigation. The recorded failure occurred after about
one hour, before that deadline. No inference about the original shell invocation
is made; the final readiness check must explicitly verify ten hours.

## Repair and validation results

Implementation commit `c6d2cb21f391b595d35b25f5b9378b2b971c30b8` adds only the
concrete `ExchangeArtifacts` terminal tail to native accounting and aligns Python
with that type and the existing native `TradeOnMarketplace` type. Unknown adjacent
request types, query replies, bulk artifact exchange, duplicate tails, altered
serialized identities and requests outside the verified terminal interval remain
rejected. Admission now reports the actual replay failure before refusing a
receipt. No engine gameplay rule or ExplorerRL action changed; canonical verifier
source is tracked in `engine/ExperimentReplay/RequestAccounting.h`.

A byte-identical diagnostic copy reproduces the original Python rejection.
Supplemental native replay uses the original client, libvcmi and AI binary bytes
with only the rebuilt verifier replaced. It checks all 90 authoritative inputs,
532 outgoing/client packets, 91 requests and 777 transactions with no unchecked
operations or divergences, and reconstructs all 30 learner choices. The cancelled
request remains separately accounted and unexecuted. Original evidence and its
failed admission remain unchanged; this diagnostic cannot admit the old batch.

The runtime was built from an empty build directory (782 build steps), frozen
with nine signed artifacts and 592 resource files, checked for actual local
library loading, and verified against the canonical dependency recipe. Native
packet-codec tests passed 594 checks. The stable full offline suite ran 1,628
tests: 1,595 passed and 33 platform/optional cases were skipped. After acceptance
pinning, 97 targeted tests ran: 96 passed and one was skipped.

All 16 fresh garrison controls and 28 fresh legacy controls passed strict history,
independent native replay and bounded owned cleanup. Eight actual frozen-916
original-map policy games then passed exact replay and policy reconstruction:
591 choices, eight natural losses, zero training samples and zero optimizer steps.
Model, Adam, Python/Torch/minibatch RNG, sampler, progress and pool remained exact.
Seven games exercised terminal cancellation (five MoveHero and two EndTurn);
ExchangeArtifacts and marketplace cancellation did not recur in these eight.
Their coverage is the preserved actual ExchangeArtifacts replay and native
packet/accounting fixtures. Allied-garrison native coverage remains unproven;
its existing permission tests are offline only. No strategic improvement is
claimed.

Release `terminal-replay-v1` binds runtime
`/Users/orsi/Projects/Heroes-artifacts/runtime/local-v8-terminal-replay-repair1`.
Accepted descriptor SHA256:
`fffce8b1ee74c010f235b97a88547adc6ada3dce65251ac959a4b6dbe6e032b8`.
The acceptance gate requires the preserved-failure diagnostic, full native proof,
all 44 interface controls and all eight frozen-policy reconstruction proofs.

## Retained unsuccessful preparation and publication attempts

- Initial fresh dependency preparation overlapped a canonical header edit and
  correctly rejected the expected-tree mismatch. The preserved old dependency
  was copied back, synchronized from canonical overlays/patch0030, and verified
  before the empty-directory build. No game used the rejected preparation.
- The first full offline suite overlapped the final source-pin edits: nine
  failures and five errors were source-integrity rejections. Its log remains;
  the stable committed-tree rerun passed all 1,628 tests with the same 33 skips.
- An initial offline CLI probe had an import-path setup error. The corrected
  intercepted shell/parser/re-execution probe verified 36,000 seconds on both
  clocks without launching a campaign.
- Initial acceptance aggregation used `source.commit` instead of
  `source.git_commit`; it failed before writing output. Correct aggregation
  reused the already passed evidence, with no repeated game or seed.
- The first bulk private archive publication failed at transport, separate from
  the training failure. Incremental publication and independent read-back are
  recorded in the checkpoint preservation receipt.

The historical migration proof uses generic policy-support transition labels
for adapter source-pin updates. In this repair they do not represent an additional
policy-support intervention: the source review establishes unchanged policy
support, rewards, architecture and PPO.

See [compact repair evidence](../../validation/terminal-replay-results-2026-10-28.json)
and [checkpoint-916 preservation](../../validation/local-script-final916-sync-2026-10-28.json).
The user alone starts any continuation; preparing and checking it starts no games.


## Actual main-checkout readiness

At source `7997ca10ef5c46b831571b087a1ae61927eaa830`, the clean main checkout
passed the default `./continue-original-4720.sh check --hours 10` after activation
under the global and original-profile owner locks. It restored checkpoint 916
exactly, calculated 3,852 remaining games and unused sequence 556, and confirmed
a ten-hour cap. Owner locks were available afterward and no native processes
remained. The old machine descriptor is preserved privately. No check launched
a game, sampled training experience or applied an update.

The exact user command is:

```sh
cd ~/Projects/Heroes
./continue-original-4720.sh --hours 10
```

The historical filename does not determine the remaining target. Sequence 555
and all four reserved seed pairs remain consumed, including its unstarted fourth
game. Unknown terminal request types still fail closed; this repair admits only
the five reviewed concrete types under the unchanged terminal constraints.


## Complete private preservation

Private branch `codex/local-script-final916-checkpoint` preserves the complete
checkpoint and run package at `78d793355b17a099eb906870a3368e7972bceb7c`, the
excluded diagnostics at `f481857133b49b48e6bd83215d7bfaaf9e015554`, and final
read-back receipts at `0082b8317f386e443af55d84d48534ddc47e405b`.
Independent remote bytes restored the exact 916 model, Adam, Python/Torch/minibatch
RNG, sampler, progress and pool: 3,664 cumulative rollouts, 526,188 decisions and
17,676 Adam steps. All 146 successful batches and the unique consumed-seed
lineage were audited. The official pointers and original evidence were unchanged.

All 15 run-archive parts were independently downloaded and all 19,536 files
plus 603 symlinks verified. The separate diagnostic archive passed remote
read-back for all 1,707 files and 53 symlinks. Runtime binaries and external
game-data targets are excluded; symlinks are retained without dereferencing.
Transient DNS/TCP/HTTP failures, the bounded shallow-clone alternative and the
initial missing-manifest read-back aggregation are retained in the preservation
receipt. Final remote ref and receipt read-back passed; no transfer processes
remain.
