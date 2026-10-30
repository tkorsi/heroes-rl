# Independent tiny policy candidate

The user requested a much smaller neural network for a separate experiment,
explicitly distinguishing model simplification from reducing the action set.
This candidate uses the smallest practical width already supported by the
existing contextual and C++ learner implementation: hidden size8, feature size4,
with the same max_map_side256 and v8 interaction contract.

| Quantity | Current self-play model | Tiny candidate |
| --- | ---: | ---: |
| Recurrent hidden state | 64 | 8 |
| Encoded feature width | 32 | 4 |
| Trainable parameters | 74,344 | 15,460 |
| Training progress | Checkpoint1753 | No games or updates |

This is a width reduction, not a replacement of the architecture with a linear
or feedforward model. The spatial CNN, text/tree encoders, ordered reducer,
recurrent memory and typed policy/value heads remain. Fixed-width map embeddings
and convolutions limit further parameter reductions by changing these two
dimensions alone. Less capacity can also hurt learning; no strength improvement
is established. Changing weights/dimensions cannot retain the learned1753
policy or its Adam history exactly, so no trained weights were resized or
distilled. Checkpoint1753 and all other official state remain unchanged.

The prospective contract retains the original map, player-visible inputs, full
action set, existing visible-interaction-budget-revealed-v1, four-game PPO
batches, Adam lr0.0003, unchanged PPO configuration, frozen batch-start self-play
opponents and terminal-only rewards +1/-1/0. No action reduction, strategic mask,
shaping, demonstration or hidden information is introduced. No game-seed
reservation or automatic training launch is part of this preparation.

`configs/selfplay-tiny-v1.json` records the independent random initialization.
`tools/prepare_tiny_policy.py` checks that contract by default; explicit
`--prepare --output` creates a fresh private candidate only under the durable
tiny-policy-experiments directory and refuses overwrites. Initial policy,
Torch/Python/minibatch RNG and frozen initial opponent pool restore exactly.
The candidate has no optimizer checkpoint and no update history; a future
training initializer must create fresh Adam state rather than load a diagnostic
optimizer artifact. It is not silently selected by existing launchers.

## Offline evidence and limits

The actual accepted C++ library matched Python prefix state bit for bit on128
recorded observation frames using the saved tiny policy. Recorded observations
were used only for inference diagnostics, without action imitation, loss,
gradient or training sample creation. Sixteen unique controls passed, including
real native prefix/gradient parity, synthetic complete PPO updates with exact
Adam/RNG agreement, expanded coordinate/quantity/management/dialog/capability
actions, checkpoint restoration and independent preparation guards.

A seven-repeat warm-prefix microbenchmark on those same128 frames measured
12.61ms median for the tiny model versus14.93ms for the trained1753 baseline
(about1.18x). This excludes feature compilation, policy action sampling, engine
communication, recording, PPO and replay. It is not an end-to-end throughput
claim. The two policies also have different weights and capacity, so no playing
strength or learning-efficiency conclusion follows.

The first test selection named a nonexistent unittest class; it was corrected.
An expanded native test initially skipped under the other suite's manifest
environment name and then passed with its explicit manifest. A baseline
microbenchmark load correctly rejected missing expected plan/opponent metadata;
the corrected attempt supplied exact published metadata. Logs remain private.

The saved candidate and all126 protected official files were rechecked.
No native game, full training campaign, or candidate optimizer update ran.
Actual tiny-policy gameplay/reconstruction and a separately reviewed manual
training entry point remain required before declaring training launch readiness.
See [compact validation](../../validation/tiny-policy-2026-10-30.json).
