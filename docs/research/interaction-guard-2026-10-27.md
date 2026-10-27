# Visible interaction budget: prospective protocol

Registered on 2026-10-27 at 19:21 UTC, before guarded native diagnostics.
The user authorized fixing repetitive choices while retaining rewards only for
the game outcome. Official training remains a manual launch.

## Motivation and hypothesis

The first 68 admitted games of the manual continuation from update699 contained
90,117 choices: 54,709 capability-panel opens/closes and 19,116 movement-limit
responses. Ten games reached the 4,960 ordinary-choice limit on days1–2.
This descriptive historical cohort is not a prospective paired control.

Hypothesis: a generic, visible-history interaction budget reduces these loops
without choosing a strategy for the learner. It is a disclosed change to policy
support, not an exact performance optimization or a claim that all excluded
actions are illegal. Rewards, PPO settings and tensor architecture remain fixed.

## Frozen rules: visible-interaction-budget-v1

- Permit two optional opens/selections per semantic target per visible-progress
  epoch. UI revision changes and reversible rearrangements do not renew it.
- Permit32 reversible army/artifact operations per actor per game day. Reset
  this quota only on a new day. This explicitly limits potentially useful
  management, so it is an interface intervention with a tradeoff.
- Exclude native moves with zero movement points or to the current tile; the
  accepted native implementation returns before route execution for both.
- Suppress a repeated failed, no-effect movement target until relevant visible
  world/hero progress. Do not exclude partially successful moves.
- Keep closing/back, ending the turn, mandatory dialogue and mandatory choices.
- Use only the player's delivered history. Never rank tactical targets, force
  recruitment/attack, substitute a sampled action, add penalties or shape rewards.

The guard snapshot must be immutable, versioned and reconstructed from the full
native history. Collection, replayed policy probabilities and PPO use that same
support. Native v8 history and offers remain intact. Historical unguarded
artifacts must retain their original support even when the new guard is enabled.

## Acceptance before manual release

1. Unit tests cover budget/reset/escape semantics, support serialization/tampering,
   hierarchical probabilities, historical compatibility and synthetic PPO.
2. Review source and exact full-checkpoint restore. Use the latest fully admitted
   checkpoint of the now closing manual run, never disposable benchmark weights.
3. Run at most eight frozen-policy native diagnostic games (four per seat) on the
   original map, with fresh engine seeds starting at82010000 and policy seeds at
   82050000. No PPO admission or optimizer steps. Launch only after offline gates;
   a separate45-minute wall budget starts at diagnostic launch and includes cleanup.
4. Require complete identity, native replay, strict policy reconstruction and
   cleanup for each claimed valid game; retain failed/unstarted entries. Check
   actual choices against all guard invariants and report natural outcomes and
   cutoffs separately. Compare action counts descriptively with the historical
   cohort, without claiming a paired causal speedup or improved strategy.
5. Preserve full model/Adam/RNG/progress/pool state and the official continuation
   ledger. Publish all results and failures before enabling the manual wrapper.

No official training or automatic continuation is authorized by this protocol.
If the gate fails, repair offline under a new recorded diagnostic attempt; never
admit failed/partial experience or silently rewrite these hypotheses.
