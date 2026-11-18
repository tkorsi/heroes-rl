# How it works

The [README](../README.md) says how to run training. This guide follows one
game from the engine to a weight update and names the code at each step.

```text
heroes/train.py          restore → [ collect 100 games → update → save ] …
   └ Pool                ten actor processes (heroes/train.py: worker)
       └ actor.play      one game, both seats, the learner's trajectory
           ├ engine/     vcmiclient process, two sockets, typed frames
           ├ model/      frames → GRU state → action distribution, value
           └ game/       what may be offered, done and sampled
   └ learner/            targets, loss windows, MLX update
   └ checkpoint.py       one file per update
```

## The game as the policy sees it

VCMI runs headless with both players controlled by the **ExplorerRL** plugin
([engine/ExplorerRL](../engine/ExplorerRL/)). Whenever a player may act on the
adventure map, the plugin sends what that player can see and waits for an
answer. Each player has its own TCP connection to the Python side; messages are
one JSON object per line.

[`heroes/engine/protocol.py`](../heroes/engine/protocol.py) turns the messages
of one seat into **frames**, appended to that seat's `PlayerHistory`
([`heroes/game/contract.py`](../heroes/game/contract.py)):

| Frame | Meaning |
| --- | --- |
| `view`, `event` | something became visible: the map, a battle result, a notice |
| `decision` | the player must act; carries the observation and the offers |
| `submitted` | the action the policy answered with |
| `consumed`, `issued`, `acknowledged` | the engine took the action and ran its commands |
| `context`, `context_opened`, `context_invalidated` | a window opened or closed |
| `result` | how the action ended |
| `superseded` | the engine withdrew a decision or closed an action another way |
| `terminal` | the game ended for this player: win or loss |

`PlayerHistory.append` checks every frame against the order the engine must
follow and refuses anything else, so a protocol bug fails the game instead of
reaching training. The contract also carries the profiles of earlier, smaller
interfaces (history v2 to v7); the engine and the checkpoints use the newest,
`adventure-capabilities-v1` with history v8.

An **observation** is the visible map (terrain, roads, objects), the player's
heroes and towns, resources and the open window. **Offers** say which actions
are legal now. An **action** is typed: `move_to` a tile, `open_town`,
`recruit` a count, `choose_dialog`, `end_turn` and so on.

## One game

[`heroes/engine/process.py`](../heroes/engine/process.py) starts a fresh
`vcmiclient` per game in its own folder, with its own profile directory, a copy
of the map and a random engine seed.

[`heroes/engine/env.py`](../heroes/engine/env.py) is the two-seat environment:

```python
state = env.reset(folder, game_number, seed)      # the first Request
while isinstance(state, Request):
    state = env.step(action)                      # the next Request, or the End
```

A `Request` belongs to one seat and carries that seat's new frames, ending with
the decision to answer. The game ends in one of three ways:

| End | When | Trained on |
| --- | --- | --- |
| `completed` | The engine finished. Both seats got complementary win and loss terminals, the engine log agrees with the sockets and the process exited cleanly. | yes, with reward +1 or −1 |
| `truncated` | A decision arrived after day 365, or a seat had answered 163,840 decisions. The game is stopped there. | yes, without reward |
| `failed` | Anything else: a crash, a protocol error, an engine that reports its own failure. | no |

[`heroes/actor.py`](../heroes/actor.py) plays the game. Both seats use the same
weights but each has its own recurrent state and random generator (`Seat`).
One seat is the **learner**. When the game ends, `build_trajectory` joins what
the learner's policy recorded while playing (action, log-probability, value)
with the frames:

- A decision is trained on once its action was resolved, by a `result` or, for
  the very last action of a finished game, by the terminal.
- Only the last action of a completed game has a reward. The game is
  undiscounted, so every earlier advantage is the final outcome minus the value.
- A truncated game ends at the last action whose following decision the policy
  saw. That decision's value stands in for the rest of the game.

## The policy

[`heroes/model/policy.py`](../heroes/model/policy.py). `ContextualEncoder`
turns each frame into a semantic tree; `ContextualActorCritic` encodes it:

- text and numbers become leaf vectors (a byte-level GRU, a small MLP);
- a node pools its children and combines them with its tag; ordered
  sequences use a second GRU;
- an observation's map goes through a small CNN, its heroes, towns and window
  through the tree encoder;
- one `GRUCell` (64 units) carries everything the policy remembers.

At a decision, `ContextualDistribution` samples an action step by step: the
kind first, then what it applies to, then a tile or a quantity. The
log-probability of an action is the sum over those steps. Engine handles and
revision numbers are never features; they only select which visible entity a
step refers to.

The **interaction guard** ([`heroes/game/guard.py`](../heroes/game/guard.py))
removes actions that are legal but lead nowhere, such as reopening the same
window or swapping stacks back and forth. It restricts what is sampled; it is
not a reward and not a game rule. See the
[restriction register](temporary-restrictions.md).

While acting, the model reuses encodings of trees and observations it has
already computed. That memo is dropped as soon as any weight changes.

## The update

[`heroes/learner/ppo.py`](../heroes/learner/ppo.py) prepares a batch:
undiscounted GAE with λ = 1, advantages normalized over the batch, and
**loss windows** of 16 consecutive trained decisions. A minibatch is two
windows. For each window the update recomputes the recurrent state from the
first frame of the game up to the window with the current weights and without
gradients, then differentiates through the window. Hidden states from play are
never reused.

| Setting | Value |
| --- | --- |
| clip ratio, value clip | 0.2, 0.2 |
| value coefficient | 0.5 |
| gradient norm | 0.5 |
| epochs | 1 |
| optimizer | Adam, 3e-4 |
| entropy bonus | none |

`ppo.update` is the plain autograd reference, correct and slow. Production uses
[`heroes/learner/mlx/`](../heroes/learner/mlx/): each actor compiles its game
into a weight-independent **recipe** (`learner/recipes.py`: which trees every
frame reads, which choices each trained action took), the learner turns recipes
into batched plans (`learner/plan.py`, `mlx/plans.py`) and runs them with
custom Metal kernels for the GRUs, the tree encoder and the sparse map CNN.
Weights and Adam state are copied into MLX for the update and copied back only
when every step succeeded.

## Checkpoints

[`heroes/checkpoint.py`](../heroes/checkpoint.py). A checkpoint is one
`torch.save` file: model configuration, weights and their SHA-256, Adam state,
the generator that shuffles minibatches, and the counters. Files are written to
a temporary name and renamed; `latest.json` is switched last.

The previous pipeline wrote a `policy.pt` and a `resume.pt` per update.
`find_legacy` locates the newest pair and `import_legacy` reads it. This
happens once, when the state folder is empty.

## Engine

| Path | Responsibility |
| --- | --- |
| [engine/ExplorerRL](../engine/ExplorerRL/) | visible-state adapter, typed actions, socket transport |
| [engine/patches](../engine/patches/) | changes to pinned upstream VCMI 1.7.5 |
| [engine/ExperimentReplay](../engine/ExperimentReplay/) | engine-side replay logging and the replay worker; the Python loop no longer uses them |
| [engine/NativeSession](../engine/NativeSession/) | an earlier reusable-host experiment; not used |
| [maps/](../maps/) | the scenario both seats play, `hero-management-v4` |

`vcmi/` is a generated checkout, ignored by Git. The canonical engine changes
are the patches and the folders above; never leave work only in the checkout.

```sh
python3 tools/prepare_vcmi.py           # fetch VCMI and apply patches and overlays
python3 tools/prepare_vcmi.py --check   # verify the checkout matches them exactly

cmake -S vcmi -B vcmi/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/homebrew -DENABLE_GOLDMASTER=ON \
  -DENABLE_STRICT_COMPILATION=OFF -DENABLE_EXPLORER_RL=ON \
  -DENABLE_LAUNCHER=OFF -DENABLE_EDITOR=OFF -DENABLE_DISCORD=OFF \
  -DENABLE_MMAI=OFF -DENABLE_NULLKILLER2_AI=OFF -DENABLE_TRANSLATIONS=OFF
cmake --build vcmi/build --parallel 2 \
  --target vcmiclient vcmiserver ExplorerRL Nullkiller BattleAI vcmireplay
```

The Python side expects the built client at `HEROES_VCMI`, by default
`~/Projects/Heroes-artifacts/runtime/local-v8-sleep-resume-v1/bin/vcmiclient`,
and the original game's `Data` folder under `~/Projects/Heroes-artifacts/game-data`.
Python dependencies are pinned in [dependencies/](../dependencies/):
Torch 2.9.1, NumPy 2.3.5 and MLX 0.31.1 on CPython 3.13.

## Earlier work

Before November 2026 the loop was a much larger system of launch plans,
receipts, replay verification and several learner backends. It is described in
[project history](project-history.md), the [research notes](research/) and the
two engine notes ([shutdown](native-headless-shutdown.md),
[reusable session](native-engine-session.md)). Paths those documents cite under
`experiments/`, `tools/`, `validation/`, `reports/` and `apps/` exist in Git
history up to commit `5efe811`.
