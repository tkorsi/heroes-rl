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

Results are pending. Preparation alone is not successful native validation or
permission to launch official training.
