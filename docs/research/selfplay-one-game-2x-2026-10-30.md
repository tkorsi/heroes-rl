# One frozen self-play game with doubled choice limits

## Authorization and purpose

On 30 October 2026 the user requested a script to play one game with twice as
many choices, to launch manually. This is a separate frozen-policy evaluation
from the latest verified completed self-play checkpoint, currently 1753. It does
not change the four-game PPO update contract or apply a one-game update.
Both players use separate model instances with the same checkpoint weights and
independent private memory and policy RNG streams. One game cannot establish a
learning trend or balance the previously observed Red/Blue asymmetry.

The completed 100-game experiment ended at 1753 with 25 full updates from 1728.
Its learner outcomes were 3 wins, 8 losses and 89 collection cutoffs; all 11 natural
victories belonged to Blue. The learner captured 3 towns. Across the continuation,
the first 20 versus final 20 games revealed 189 versus 223.8 new tiles per game,
but stack swaps rose from 68.9% to 70.2% of choices and collection cutoffs were
17/20 versus 18/20. Longer games and changing seeds/opponents prevent attributing
these descriptive changes to learning.

## Declared intervention

| Setting | Completed training | Single-game evaluation |
| --- | ---: | ---: |
| Normal policy choices per player | 4,960 | 9,920 |
| Reserved dialog allowance per player | 160 | 320 |
| Hard choice ceiling per player | 5,120 | 10,240 |
| Maximum game day | 365 | 365 |
| Physical games per manual invocation | Four-game training batch | One evaluation game |
| Optimizer updates | One per complete training batch | Zero |

The cutoff starts when either player reaches the normal allowance. The reserve
only permits mandatory policy-selected dialog answers before a safe cutoff; it
does not permit more ordinary movement. The independent final dialog-chain
limit remains eight answers, so the 320-choice reserve is an outer ceiling.

The existing map, network, engine, native learner library, revealed-target support,
visible interaction budgets and terminal scoring stay unchanged. The existing
mandatory-dialog drain and technical continuation limits also remain unchanged;
the larger reserve does not authorize scripted policy choices. No exploration
reward, strategic mask, demonstration or hidden observation is added.

The technical capture deadline is 1200 seconds, doubled from 600 to accommodate the
larger choice budget. The default total budget is 3600 seconds for collection and
verification, with bounded cleanup. A technical failure is unscored and never
restarted automatically. The runner retains its failed-attempt evidence.

## Isolation and verification

A separate durable evaluation directory and seed ledger consume exactly one fresh
engine seed and two policy seeds before launch. Reusing an output or consumed
reservation is rejected. Check mode reserves nothing and performs no gameplay.
The official training lineage, original Nullkiller state and model/Adam/RNG/
progress/sampler/pool files remain hash-protected. The normal resource lease
prevents starting alongside an existing campaign; it never interrupts the owner.

The driver uses the existing collector and native replay APIs. The saved paired
rollout must roundtrip exactly; both player histories and both frozen policies
must reconstruct; the actual engine replay must pass. Diagnostic trajectories
cannot become training admission capabilities. No training receipt or optimizer
update is created. All core learner, collector, replay, engine and dependency
sources are required to match the accepted base release byte for byte. The
campaign controller's reporting-only repair is pinned separately to its reviewed
commit; the original checkpoint ancestry and all runtime checks remain exact.

## Manual command

```sh
cd /Users/orsi/Projects/Heroes
./play-one-selfplay-2x.sh --check
./play-one-selfplay-2x.sh --run
```

With no arguments the script performs only the check. Explicit launch plays one
game, reports its outcome and choice counts, retains private replay evidence and
prints elapsed time. The user launches it; preparation does not run a native game.

## Validation boundary

Preparation uses offline controls and a real full-state checkpoint/runtime
preflight. No new native game or policy rollout is run during preparation.
The accepted native runtime remains unchanged; actual long-game execution and
its replay are verified when the user invokes the script.

The focused offline suite passed 90 tests, including 13 new evaluator controls.
The real shell launcher check from the main-checkout working directory passed:
checkpoint 1753 restored exactly, two separately allocated frozen model copies
matched the published weights, and 126 protected inputs remained unchanged. Zero
seeds were reserved and no gameplay or optimizer update ran. The canonical engine
check passed in the existing accepted runtime checkout; corresponding sources in
this new checkout are identical. The [compact validation](../../validation/selfplay-one-game-2x-2026-10-30.json)
records the source and private receipt hashes.

Review found and corrected nested deadline timers, final plan-hash rechecking and
shutdown evidence on protected-input failure before any native launch. The
original supervisor was still computing its post-run report during preparation.
The subsequent [reporting repair](reporting-lock-fix-2026-10-30.md) safely stopped
that report-only process after checking the completed checkpoint and worker
shutdown. Reporting is now separate from training ownership. An actual active
campaign still blocks evaluation before any new seeds are reserved.
