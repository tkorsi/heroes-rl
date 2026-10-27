# Visible interaction budget: prospective protocol

Registered on 2026-10-27 at 19:21 UTC, before guarded native diagnostics.
The user authorized fixing repetitive choices while retaining rewards only for
the game outcome. Official training remains a manual launch.

## Motivation and hypothesis

The first 68 admitted games of the manual continuation from update 699 contained
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
- Permit 32 reversible army/artifact operations per actor per game day. Reset
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
   a separate 45-minute wall budget starts at diagnostic launch and includes cleanup.
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

## Results and retained validation deviations

The prior manual run closed at 19:15:13 UTC with full checkpoint 717: 72 new games,
18 updates, 62 natural losses and 10 collection cutoffs. The original 5,000-game
target now has 352 admitted games and 4,648 remaining; next training sequence 355.
The full 717 model/Adam/RNG/progress/sampler/pool and 716 dependency were privately
published and restored from an independent network clone. See the
[preservation receipt](../../validation/local-script-final717-sync-2026-10-27.json).

Guarded diagnostic source `decec45` ran eight fresh, frozen checkpoint 717 native games at
19:30 UTC. Each game passed native replay, strict policy reconstruction, guard
invariants and cleanup. All were natural losses, with no cutoffs: 34, 79, 47, 117
choices for Red and 21, 42, 10, 2 for Blue (352 total; mean 44). All data remain
diagnostic: zero PPO samples and optimizer steps.

The independent audit found zero moves with zero movement points or to the
current tile. Optional navigation never exceeded two openings per semantic key
within a progress epoch; management peaked at 17 operations per actor/day.
There were 42 capability opens/closes in total, 152 movement attempts with 31
endpoint changes, 114 no-route responses and 16 end-turn choices. Recruitment,
hiring, upgrades and town capture were not observed. Red's final visible days
were 2, 5, 2, 3; Blue's were all 2. These results show the targeted loops are bounded
in the tested games, not that the policy learned a winning strategy or that all
possible no-effect actions are eliminated.

Two harness failures are retained rather than relabeled successful:

1. Attempt 1 stopped before gameplay because the invocation pointed the artifact
   root at the wrong directory for prerequisite receipts. Attempt 2 used the
   actual root and a separate diagnostic output directory.
2. After all eight games individually passed, attempt2's final state comparator
   rejected `OrderedDict` versus `dict` before comparing tensors. Its supervisor
   then failed to serialize the shutdown receipt because `finished_utc` was
   supplied twice, after process cleanup. The original overall report remains
   `failed`; its final in-memory Adam/RNG/pool assertion chain was not reached.

The harness now normalizes model mappings and emits one cleanup timestamp; three
regression tests cover identical state, tensor/Adam/RNG/sampler changes and the
shutdown receipt. A separate read-only reconciliation verified every persisted
game proof, all copied/official 717 hashes, exact complete restoration and actual
owned shutdown. It replayed no games and performed no updates. This postcollection
audit replaces the missing final bookkeeping evidence, not the retained original
failure verdict. The accepted diagnostic result is explicitly
`passed_with_postcollection_harness_correction`.

Strict reconstruction matched all 352 actions and 945 conditional probabilities
exactly. Native evidence checked 6,923 transactions, 758 requests and 4,500 client
packets, with none unchecked. Full raw evidence and all failed attempts are
preserved privately; compact immutable references are in the
[native preservation receipt](../../validation/interaction-guard-native-preservation-2026-10-27.json).

The historical first 68 unguarded games averaged 1,325 choices, versus 44 in these
eight guarded games. Seeds, checkpoints and cohort sizes differ, so this is a
descriptive comparison, not a measured paired speedup or a 5,000-game ETA.
No native PPO update was tested in this diagnostic; synthetic PPO and prefix
reconstruction tests passed with the versioned support. Future official training
remains an explicit manual launch from 717 and preserves the terminal-only reward.
