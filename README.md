# Heroes RL

An experimental environment for learning Heroes III adventure-map strategy with
reinforcement learning, using the real [VCMI engine](https://github.com/vcmi/vcmi).
It connects an external Python agent to the game and records synchronized
omniscient and player-fog videos.

**Current milestone:** the contextual environment and randomly initialized
recurrent actor–critic have passed bounded local validation. The opt-in v4
interface adds explicit town-garrison-to-visiting-hero troop transfer to map,
hero and town navigation, construction and exact-count recruitment. Policy and
value share a recurrent network but use only their player's ordered visible
history, with separate memory for each player. No game-trained policy exists.

Independent native replay, episode boundaries and natural-outcome controls pass
within the recorded scope. A predeclared sixteen-game contextual cohort produced
sixteen valid cutoffs and no natural endings; behavior audits separate that
sparse feedback from engine correctness. A concrete first-update plan is
prepared, **not authorized or executed**. Rules, fog and terminal-only rewards
are unchanged. The earlier uniform-random/video and fixed Easy-opponent
baselines remain available. Draft PRs remain unmerged; other platforms are not
validated.

Repository: [tkorsi/heroes-rl](https://github.com/tkorsi/heroes-rl).
Original game assets are required locally and are not included or downloaded.

This is the canonical project document. Last agreement: **25 October 2026**.

## Why this project exists

Build an agent that learns adventure-map strategy in Heroes III through
reinforcement learning, and make the development of its behaviour observable.
The original long-term idea was self-play from random initialization. After a
first baseline against a fixed built-in opponent on Easy, the next experiment
uses **two copies of our random agent**, with equal action capabilities and
separate observations. Battles resolve automatically. This first self-play
experiment establishes the environment; learning updates come later.

There are two separate tracks:

- **Experiment track — this workspace and chat:** implement the environment,
  record episodes, run controlled experiments, inspect evidence, and iterate.
- **Learning track — outside this repository:** explanations and unrelated
  learning exercises. Lessons do not need to block experiment engineering here.

## The experience we want

After a training run, the user can watch how the hero's behaviour changes:
leaving the castle, exploring, encountering neutral armies, capturing mines,
and eventually dealing with the opposing computer. Early policies may wander,
do nothing useful, or lose when the computer arrives. Keep failures visible.

Provide a short overview of a large run (for example, the first 1,000 episodes
in about one minute), plus access to individual recorded episodes. A useful
overview shows evenly spaced episodes/checkpoints with episode numbers and
outcomes. It must say which episodes were selected; it is not a claim that a
minute contains every action of all 1,000 games. An all-episodes timelapse can
also be produced, but each episode is only a glimpse at that compression.

This is the desired observation experience, not a promise of monotonically
improving play. Terminal-only rewards may produce no learning for a long time.

## Agreed constraints

- Use the actual VCMI engine locally on the Mac. No paid cloud compute.
- The learner chooses adventure actions. Combat is a fixed automatic subsystem;
  tactical battle skill is outside the first experiments.
- Start without demonstrations, scripted strategies, coordinate heuristics, or
  handcrafted exploration/mining rewards. Initial reward: natural victory +1,
  natural defeat -1. Log technical timeouts/errors separately, never as losses.
- Learn both policy and value from experience. We define the observation/action
  interface and learning algorithm, not strategic priorities or a handcrafted
  position-value formula. "Terminal-only reward" means +1/-1 at a verified
  natural result and 0 on ordinary transitions; the reward convention is unchanged.
- Only player-visible information may enter the learner. A spectator recording
  can show more for human inspection, but must never feed hidden state to it.
- Use structured game observations and typed contextual interactions, not
  screenshots, screen coordinates or externally selected random clicks. For the
  expanded interface, only the adventure map, hero window and town window are
  in scope. Window-only information and operations require an explicit open
  action; closing a window stops refreshes of its private panel contents.
  Additional interfaces, including obelisks and special dialogs, are deferred.
- In the new contextual experiment, the policy chooses a destination `(x, y)`
  and the engine computes and executes a route within the hero's available
  movement points. Do not supply engine path costs, a reachable-this-turn target
  mask or a movement-range overlay. Learn movement relationships from permitted
  observations and actual results. No auxiliary path-cost target or handcrafted
  movement reward is authorized. Retain the adjacent-step baseline separately.
- Learned recurrent state is separate for each player and resets each episode.
  Discuss any engineered last-seen input/cache before adding it to the learner;
  existing diagnostic caches are not authorization to make them model inputs.
- Freeze opponent settings during comparisons. The earlier built-in baseline
  uses active Nullkiller on Easy (difficulty index 0). Random self-play uses
  ExplorerRL on both sides with independent Python RNGs and the same action
  interface. Keep these cohorts separate. Easy is the lowest game difficulty,
  not evidence that an AI module is empirically the weakest available.
- Preserve seeds, map hashes, engine/code versions, policy/checkpoint identity,
  outcomes, and termination reasons. A seed alone does not establish deterministic
  replay; the current built-in AI has additional randomness.
- Compare fixed checkpoints on held-out seeds and both player sides. Videos
  illustrate behaviour; evaluation statistics establish whether it improves.

## Stage 1 — record real experiments

Build the recording path before training:

1. Capture native game frames from the VCMI renderer during gameplay, without
   capturing the desktop or other applications.
2. Record two synchronized views of the **same episode**: **omniscient** (open
   map for inspection) and **player** (the selected AI player's actual team fog).
   Both use the same camera following that player's heroes. The open-map pass
   must never change player fog, AI callbacks, or the learner's observations.
3. Save a timestamped frame ledger per view, isolated engine logs/settings, and
   a run manifest. Set finite frame and wall-time limits.
4. Encode each view of an individual episode to MP4 and produce a labelled, roughly
   one-minute montage across a set of episodes.
5. Verify the path on a real, bounded built-in-AI game with automatic battles.
   Label this **recorder smoke test**, with no learner and no training claim.

For future training, save episode records and checkpoints for the whole run;
record scheduled evaluation episodes from frozen checkpoints. This avoids
making every training step wait for rendering. A video of a later evaluation
is labelled as such, not presented as the original training episode. Exact
action replay is a separate capability. It is absent from this original rendered
recorder but implemented and locally verified in the separate headless PR #6
workstream described below; do not relabel old recordings as verified replays.

The player view means the map information available to a **side/team**, shared
by its heroes and towns; it is not a single hero's line of sight or a literal
visualization of the AI's internal thoughts. Preserve VCMI's normal fog rules.
Only the map perspective changes; spectator sidebars are still spectator UI.

The recorder disables engine cheats before game-state creation. This also
turns off Nullkiller's automatic map reveal and hidden-object graph. Record
actual initial fog counts and effective AI flags, then the chosen side's visible
tile count per sample. Treat these checks as evidence of configured fog, not a
complete audit of every built-in AI information access. ExplorerRL has a separate
player-visible observation interface: own heroes/towns/resources and tiles/objects
checked against the player's visibility callback. The random policy reads only
the attempt mask when choosing movement.

## Experiment 1 — uniform random movement baseline

**H1:** On a small fixed two-player map, an untrained policy selecting among
permitted primitive adventure attempts can leave its starting area and encounter
meaningful game events before an Easy opponent ends the game. First measure
the random-policy baseline and the rate of natural endings. This tests the
environment and whether terminal feedback is reachable; it is not training.

An initial learning candidate is a small recurrent policy with PPO and terminal
win/loss rewards, after validating random self-play below. Compare against the
random baseline and frozen checkpoints within the same opponent, combat, map
and action configuration. Log exploration, neutral battles, mine ownership,
survival, and wins as **metrics**, not added rewards. If feedback is too sparse,
discuss the smallest next experiment rather than silently introducing reward
shaping.

The first implementation uses `experiments/explorer/random_policy.py` over the
ExplorerRL bridge. Red is our agent; Blue is Nullkiller on Easy. A seeded Python
RNG selects uniformly among the player-visible mask's permitted attempts: eight
adjacent directions plus end-turn. The mask is not a guarantee of legal movement:
the server validates every attempt, including any unknown neighboring tile.
Actions, results and player-visible observations are saved in `policy.jsonl`.

The authored `random-baseline-v1.vmap` is a fixed 20×20 map with two Castle towns,
one hero and 25 pikemen per side, two sawmills, two neutral stacks of six pikemen,
and two gold piles. It uses normal fog and elimination rules. The initial stats
are equal; the town footprints and map edges do not give identical initial
visibility. Use both sides in a later evaluation before drawing strength claims.

This is a deliberately limited movement experiment: the agent cannot recruit,
build in towns, transfer troops or cast spells. Tavern construction is forbidden
for both sides to keep one hero per side, and heroes are capped at level 1.
Nullkiller can otherwise use its normal town economy. Visits, pickups, captures
and battles are real engine interactions. Dialog choices supported by the bridge
are also selected by the external policy; unsupported dialogs fail explicitly.
After losing its hero, the agent can end turns while its town remains, allowing
the engine to establish a natural result. The old `explorer-*` bare-duel maps
remain unvalidated drafts, separate from this authored scenario.

The baseline has no neural network, optimizer or learning updates. Record actions,
actual displacement, visited positions, fog expansion, visits, battles and owned
mines as metrics. A successful environment check requires the agent to actually
act and a natural ending to agree between its socket message and the engine log,
with a clean process exit. Wall/day/action/frame limits and technical errors are
unscored. The top-level manifest `reward` is authoritative; socket terminal messages
are evidence pending process and recording checks. Engine and policy seeds are
separate; the engine logs its actual initial seed, while deterministic replay of
the whole opponent/engine is still unverified.

## Experiment 2 — random self-play

**H2:** Two independent copies of our movement policy can play the same authored
scenario with their own fog, equal available actions, real automatic battles,
and evidence from both players. This is a no-learning environment experiment,
not a claim that self-play has already improved either policy.

`run_self_play.py` starts one `RandomPolicy` instance per side, with separate
localhost sockets, RNG seeds and action ledgers. Each native ExplorerRL instance
uses its own player callback. The policies receive visible tiles/objects, their
own state and the same nine-action attempt interface; no spectator observation
is sent to either policy. Neither side recruits, builds or receives scripted
strategic choices. Normal game rules and automatic BattleAI combat remain fixed.

The renderer captures four streams under one client-state lock:
`omniscient-red`, `player-red`, `omniscient-blue`, and `player-blue`. Each pair
shares its own hero/town camera; all four share sample IDs, timestamps and game
days. Open-map rendering does not reveal the map to either player. The player
videos show that side's map knowledge; spectator sidebars remain spectator UI.

Schema 3 manifests preserve per-player policies, actions, terminal messages and
outcomes. A scored result requires both agents to have acted, agreeing engine
and socket terminals for both sides, complementary win/loss, a clean exit and
valid recordings. `rewards_by_player` is authoritative only after these checks.
Limits or errors leave **both rewards and outcomes null**. Losing a hero while
still owning a town is not a defeat: that agent can end turns until the engine
ends the game or a declared limit is reached. Random wandering may never produce
a winner within the budget, which is itself useful evidence about sparse
terminal feedback.

## Next workstreams — simulation and learning

**Claude owns headless simulation, performance and offline replay**, tracked in
[issue #4](https://github.com/tkorsi/heroes-rl/issues/4). On 24 October the user
explicitly authorized Codex to finish the unfinished contextual adapter while
Claude was paused; Codex now owns that native/Python integration and its local
validation. Claude's other simulation/replay work remains separately scoped.
Codex also owns learning design and policy/value implementation.
Deliver engine work in staged draft PRs starting from the merged `main` branch.
Keep this README as the canonical agreement rather than creating another charter.

The training path must advance the real VCMI rules without a window, renderer,
animation waits, audio, frame encoding or presentation delays. Video production
is a separate consumer of selected episode records. A replay must reproduce the
original logical states and each player's historical fog, checked with hashes;
re-running a checkpoint on a seed is not sufficient evidence of exact replay.
Capture all relevant randomness or authoritative events, including automatic
battles. Keep omniscient replay data inaccessible to the learner.

First prove a complete headless episode, correct terminal/truncation handling
and replay fidelity. Then measure actions/second and naturally completed
games/second separately, with hardware, worker count, setup/reset time, game
lengths and limits. Thousands of complete games per second is an aspiration,
not a measured result or an acceptance promise. Use isolated processes first;
the engine has global state. Episode validity must be independent of whether
an optional video was successfully rendered.

### Initial learner specification and isolated prototype

The established baseline policy is uniform random over permitted attempts. The
legacy isolated component remains unchanged. A separate contextual neural
adapter, collector and synthetic-only PPO implementation now extend
`experiments/learner/`; see the integration section below. The first candidate is a small
recurrent actor-critic trained with [PPO](https://arxiv.org/abs/1707.06347).
This is a starting hypothesis to test, not an established best algorithm for
Heroes III.

| Component | Initial specification |
| --- | --- |
| Reward | 0 on ordinary transitions; +1 for a verified natural win and -1 for a verified natural loss, from that player's perspective. No exploration, mine, gold, survival or invalid-move rewards/penalties. |
| Observation | Only the player's structured visible map, own heroes/army/towns/resources, attempt mask, and history of its observations, actions and results. No spectator state or opponent memory. |
| Policy `pi(a \| h)` | A shared candidate scorer handles a variable set of typed, parameterized choices. Its current movement adapter exposes nine masked outputs for compatibility; movement is the first subset, not a permanent neural-output restriction. Record the exact choice support and behavior probability. The server validates attempted actions. |
| Value `V(h)` | A scalar prediction of expected future terminal return under the current policy and frozen opponent distribution, using the same permitted information and memory. No separate Q-network is required initially. |
| Self-play | Share model weights across colors, with separate recurrent state and trajectories. Start each episode with a learner seat and a frozen opponent sampled from random play or saved policy checkpoints; alternate learner colors. |

Here `h` is the player's observation/action history, summarized by recurrent
memory. Use an undiscounted terminal objective (`gamma = 1`) initially, so
winning sooner is not a separate reward. For games that eventually end in a
win or loss, the ideal value is `P(win | h) - P(loss | h)`; it is a prediction
under the training opponent distribution, not an objective score for the map.
Both policy and value learn from experience. Mines and exploration are logged
as metrics; their strategic usefulness must be learned through eventual returns.
The existing movement baseline has a uniform dialog-choice policy and automatic
BattleAI, with their versions recorded. The proposed expanded interface retains
automatic battles but does not inherit automatic or random answers to unsupported
dialogs; its bounded handling rule is specified below.

Freeze the behavior checkpoint and opponent mixture during rollout collection,
and the chosen opponent for each episode. PPO updates use learner actions with
their recorded behavior probabilities; historical opponents' actions are not
additional on-policy data.
Keep a player's transition open until its next decision or own terminal, even
when the opponent takes a full turn in between. Compute returns/advantages
within that player's trajectory, never using the other side's value or memory.
Train recurrent sequences with preserved history rather than shuffled isolated
observations. Reset memory at episode boundaries, not between player turns or
rollout chunks, and keep checkpoint identities.

A true terminal has no value bootstrap. A valid rollout/horizon cutoff may
bootstrap only from a verified next decision observation and matching recurrent
state for that player. A timeout is not a draw or loss; a killed process with no
valid boundary cannot supply a bootstrap. Exclude technical failures from
training. The current rendered, wall-limited recordings do not yet provide this
learner boundary contract. An entropy term, if used for exploration in PPO, is
an optimization term logged separately from game reward.

The three original rendered self-play runs contain no verified terminal outcomes.
Bootstrapping zero-reward trajectories alone does not supply missing win/loss
information. Before a learning campaign, establish reachable natural endings
and measure their frequency under the headless baseline. PR #6's bounded native
checks now establish reachable natural endings; they are not a learned-policy
evaluation or complete learner integration. Evaluate frozen
checkpoints on held-out seeds, both colors and a fixed opponent set; do not use
win rate against the current copy of itself as the sole measure of progress.

### Learning from scratch, discovering actions and retaining experience

**User requirement, 24 October:** start with zero prior game or strategy
knowledge. Initialize the learned model randomly. Do not supply demonstrations,
pretrained game/language embeddings, walkthroughs, utility tables, tactical
rules, object-effect explanations or preloaded spawn locations. Learn mechanics,
consequences, strategic preferences, policy and value through interaction and
terminal outcomes. No discovery, exploration or intermediate-event reward is added.

Zero prior knowledge does not mean zero input or no action interface. We must
provide observations, control primitives, observable feedback and the terminal
reward. The engine enforces game rules. Information or choices genuinely shown
by the player's game interface may be observed, but the adapter must not add
expert explanations or disclose unavailable information. Preserve the scoped
automatic BattleAI subsystem: this is initially adventure-strategy learning,
not learning every combat tactic from scratch.

The action space must be extensible and context-dependent. In later stages, as
the player explores and interacts with towns, shops and other objects, it can discover
additional choices such as construction, recruitment, purchases and upgrades.
Choices may disappear again when resources, access or prerequisites change.
Expose them through legitimate player observations/interactions, not a global
catalog of undiscovered mechanics, hidden objects or all future options. The
agent learns which choices are useful; the adapter must not choose or rank
them by strategic value.

Represent available choices with a versioned typed/parameterized contract.
Use variable candidate scoring and/or conditional choices for action type,
target, item and quantity, rather than one permanent neural output for every
concrete purchase. Preserve the exact candidate set, parameter constraints and
behavior probabilities used during collection so PPO can reproduce its action
distribution. The current nine-action mask is a bounded movement prototype,
not the final game interface. Construction, economy and shop actions are not
implemented by declaring this requirement.

The current expanded-interface agreement narrows the next implementation to
the map, hero window and town window. Shops, taverns, obelisks, spell books and
other special interfaces are not part of that implementation stage. Future
extensibility does not authorize implementing their choices now.

Plan for continued training on HotA with versioned observation/action extensions
and explicit checkpoint compatibility or migration. More objects with familiar
mechanics can reuse the representation; genuinely new mechanics may require
new interface support and learning. Do not claim current HotA support or automatic
understanding of unseen rules. Proprietary assets remain local dependencies.

Experience must persist across training games through updated policy/value
weights and resumable checkpoints. When a training loop exists, preserve its
optimizer state, training progress, RNG state and environment/schema versions
needed to resume it. Experience of object types, observable effects and spawn
patterns may change decisions in later games; their utility must be learned
from outcomes, not encoded as priorities. Merely running inference does not
update weights, and frozen evaluation episodes must not train the model.

Keep this persistent learning separate from episode-local observation history
and recurrent state, which reset at each new game and remain isolated per player.
An old map's observed locations are not current facts on a new map. Learned
expectations about likely spawns are uncertain predictions, not permission to
reveal hidden contents. An external persistent memory database is not required
for the initial learner. These are architecture requirements, not a claim that
cross-game training or checkpoint resumption has already been implemented.

### Player information and memory

An observation is information available to that player now. Memory is the
history of its permitted observations, actions and results, or a learned
representation of that history. Both policy and value may use it. The GRU is
an initial implementation candidate; the model must learn what to retain and
how to use it. It is not a scripted strategy or access to the engine's full state.

For example, when an enemy becomes hidden, the agent may remember where and
when it last saw that enemy and the attributes visible then. Historical facts
must remain distinguishable from current observations. Never refresh a remembered
position from hidden engine state or interpret disappearance as confirmed death.
Any belief about later movement is an inference learned from experience.

Preserve VCMI's normal team-fog rules. Its revealed territory is generally
persistent; visibility is not a moving line-of-sight circle around one hero.
The original bridge-v2 recorder sends player-filtered decision snapshots, which
can miss an enemy appearing and disappearing between decisions. PR #6's bridge-v3
captures sanitized visible changes between decisions and projects supported
fields into history-v1. The local review at `4fe0d7c` observed transient sightings
absent from adjacent decision snapshots. This is bounded coverage evidence, not
proof of every callback or complete support for all dialogs and window contents.
A recurrent network cannot recover an event the adapter never supplied.

Do not pass raw engine callbacks or numeric global object IDs into the learned
representation. Movement callbacks can include a hidden endpoint, and some
removal callbacks disclose hidden events. Filter each field against information
available to that player at that time. Keep engine IDs internal; if identity is
needed, assign opaque per-player episode-local handles only on legitimate first
observation. Do not enumerate hidden objects to populate memory or handles.

These remain requirements for the complete engine-to-learner adapter. The current
random agents use only the attempt mask for movement; their diagnostic logs are
not learned memory. The new isolated component can process an ordered synthetic
history through trainable recurrent memory; its weights have not learned from
game experience. It explicitly rejects raw bridge-v2 messages.

PR #6's `PlayerHistory.memory()` is a deterministic last-seen diagnostic helper:
permitted observations refresh its facts/time, and hidden entries freeze. It is
not a learned memory module and is not automatically an input to the GRU. The
default learner proposal remains chronological observations plus learned recurrent
state. An explicit last-seen feature table could make entity tracking easier,
but would change preprocessing and the memory experiment; discuss its fields,
freshness rules and comparison cohort before adding it. No such learner-input
change is agreed or implemented by this document.

### Contextual map, hero and town interface — agreement and proposal

**Status:** destination selection, engine routing and the minimal structured
map/hero/town loop are authorized for implementation and bounded local validation.
`experiments/learner/contextual.py` implements the v2 record and trajectory contract.
The opt-in native producer and Python adapter implement the minimal loop.
Local VCMI checks cover navigation, movement, fog isolation, changed armies after
closed-window battles, failure boundaries and a native terminal fixture. Compact
evidence and remaining coverage limits are in
`validation/contextual-native-2026-10-24.json`. The details below define the
execution contract; bounded checks do not establish complete interface coverage.
The first knowledge-planner stage (`b4cc7b2`) builds locally on macOS and its
generated dependency matches the root recipe. Its grid/prefix regression covers
five groups and 200 hidden-world variants; it does not run the full pathfinder
in a game. The 88 contextual contract, auditor and probe tests also pass. The
planner stage preceded the live contextual adapter. Its compact stage evidence is in
`validation/contextual-integration-2026-10-24.json`.
**Town stage, 24 October:** construction and creature recruitment are implemented
in [draft PR #8](https://github.com/tkorsi/heroes-rl/pull/8) on
`codex/town-economy`, based on draft PR #7 at `48d8e1a`. Construction passed
real VCMI validation before recruitment was added. The separate v3 interface
preserves v2 and movement-v1 behavior; compact evidence, retained failed
diagnostics and coverage limits are in `validation/town-native-2026-10-24.json`.
The policy selects the building,
creature, count and supported destination; the adapter supplies no strategy,
automatic purchase/quantity or new reward. Options, costs, prerequisites and
stock appear only where the normal town interface displays them. Keep the
authored map, tavern restriction and hero count unchanged; isolated validation
fixtures must declare every difference and never become training experience.
Broader army editing and special interfaces remain deferred. The bounded
diagnostics establish the listed mechanics, not complete gameplay coverage or
learning readiness. No training is authorized. Keep the adjacent-step `movement-v1` baseline
as a separately identified comparison. All contexts are structured headless
interfaces; opening one does not create a rendered window or use image recognition.

The intended loop, in plain English:

1. **Observe:** receive structured facts available to this player in the current
   context, plus every intervening permitted event in order. Current facts are
   distinct from earlier sightings; absence never means a hidden enemy died.
2. **Remember:** pass those observations, the player's actions and their results
   through that player's GRU. Opening/closing windows and ending turns preserve
   its state; a new episode resets it. The opponent has a separate history/state.
3. **Choose:** the environment offers supported interactions and typed parameter
   domains for this context. These describe what can be attempted, not what is
   useful. The policy supplies the distribution and chooses all parameters.
   Initial choices can be random; learning changes their probabilities.
4. **Transition:** resolve the selected operation through the adapter/engine,
   report consumption and observable results, and append intervening events.
   The next boundary is this player's next decision, or its verified terminal.
   Window navigation is an explicit transition even when game time does not move.
5. **Learn later:** use eligible experience to update both policy and value
   weights. Recurrent state is memory within a game; weight updates carry learning
   across games. Inference alone does not change weights. Ordinary rewards stay
   zero; only verified natural win/loss yields +1/-1. No strategic priority,
   intermediate reward or scripted target chooser is inserted into this loop.

#### What exists and what needs implementation

| Capability | Implemented behavior | Expanded-interface change still needed |
| --- | --- | --- |
| Live action interface | Adjacent-step baseline plus opt-in destination routing, map/hero/town navigation, construction, recruitment and explicit town-to-visitor troop transfer; automatic battles. | Wider native interruption/dialog coverage; additional operations remain deferred. |
| History and fog | Versioned v2/v3/v4 player-local histories record permitted current windows, ordered notices, command receipts and paired finalization. Real hidden-world comparisons, boundary audits and fresh contextual replay pass within their recorded scope. | Wider event coverage; historical recordings without native capture remain unsupported by contextual replay. |
| Memory and model | Randomly initialized contextual shared GRU actor–critic with separate player sessions, chronological visible history and bounded live inference checks; legacy prototype retained. | No learned game behavior; broader field/event coverage remains. |
| Candidate scoring | Contextual typed conditional policy covers navigation, construction, coordinates, recruitment counts and transfer source/destination/count; routing handles are not features. | Semantically indistinguishable dwelling controls remain equally scored; additional action interfaces are deferred. |
| Hero/town operations | Explicit owned hero/town windows, native economy panels and v4 town-to-visiting-hero transfer with bounded local evidence. | Reverse transfer, swaps, broader army editing, hero hiring, special panels and live garrisoned-hero edge cases remain deferred. |
| Learning | Contextual neural inference, collection, recurrent PPO and frozen-checkpoint plumbing passed separate bounded validation. | No optimization on gameplay; player-experience correctness and explicit launch approval govern readiness. |

The action-consumption and stream-finalization findings are **resolved in the
reviewed draft head `4fe0d7c`**, not still-open design blockers. Integration must
use that fix or an equivalent revalidated descendant, not an older build. The
remaining history-v1 limitation is separate: final visible changes after a
superseded decision invalidate the projection. A natural game result does not
make that incomplete player trajectory eligible for learning.

#### Observation and history boundary

The implemented contract identifies `adventure-windows-v1`,
`heroes-player-history-v2` and `heroes-context-actions-v2`. Store versions,
supported operations, map/engine hashes and checkpoint
compatibility in manifests. Never pool these results with `movement-v1` without
identifying the interface and observation differences. Record the destination
execution rules, pathfinder configuration and tie-breaking behavior as part of
the new interface version; the existing nine-action adapter is not compatible.

Each decision carries `{context, revision, observation, controls, offers}`. Context is
`map`, `hero(subject)` or `town(subject)`; subjects use player-local routing
handles. A revision identifies this player's observed context, not a global
engine event counter. Offers and responses bind to the decision and context
revision. Revalidate stale commands without exposing hidden reasons or executing
them in a different window. Routing handles/keys are not numeric model features;
the scorer sees only the corresponding permitted observations and parameters.
`controls` may include disabled controls and their displayed status when the
native panel actually shows them; `offers` is the currently supported attempt
set. Neither is populated from an unseen global interaction catalog.

Separate **map/global fields** from **active-window fields**:

| Context | Proposed current observation | Freshness/access rule |
| --- | --- | --- |
| Map | Public day, the player's own resources, hero position and remaining movement points, known terrain/roads, permitted objects and other owned hero/town map fields established by the interface audit. | Preserve normal player/team fog and legitimate map-visible changes. No route costs, range overlays, reachability labels or blanket inclusion of all owned-object attributes. |
| Hero | The explicitly opened, accessible hero's supported statistics, army slots and other audited hero-panel fields; supported controls. | Only after successful `open_hero`; do not expose another hero's panel. Unsupported fields are declared as coverage limits, not invented zeros. |
| Town | The explicitly opened, accessible town's supported overview, buildings and armies; controls visible in that panel. | Recruitment stock, construction options and other deeper-panel details require the appropriate explicit internal-panel navigation if the normal interface does. |

The first access scope is owned heroes/towns reachable through the player's
normal map interface. Opening an enemy object must not grant its private window.
Ownership is necessary, not sufficient: engine getters are not automatically
safe discovery APIs. Audit every field/offer against the corresponding native
interface before calling getters that can return broader information.

Freshness is per field. On close, window-only details cease to be current and
receive no refresh while closed. Their prior observations remain in chronology
and may be retained by the GRU. Reopening produces a fresh structured observation.
Map-visible facts such as the player's own resource totals must still update through their
legitimate map/global channel; closing a window must not hide those facts or
silently refresh its entire old snapshot. The present bridge exposes army and
other own-state fields without any window context, so v2 needs a field-by-field
audit rather than copying that payload wholesale or deleting all own-state data.

History-v2 must record window open/close/panel changes and their outcomes in the
same player-local order as map observations, including transient sightings.
Add explicit consumption, decision-supersession and terminal events so final
views remain representable without inventing an action or mutating a delivered
prefix. Distinguish an adapter-consumed navigation action from a server-issued
game command. Both require an observable result; neither acknowledgement alone
proves a strategic effect. Keep raw engine IDs and omniscient replay data outside
this contract. Context changes never reset recurrent memory.

The first implemented wire format is deliberately narrower than the future
panel design above. A decision frame contains the current observation, offers
and revision; its context is carried by the optional active window. Map views
contain visible terrain/roads/objects, own resources and hero/town map entries.
Hero windows add army slots, four primary skills, mana and level. Town windows
add army slots and faction only; building details and deeper controls are
deferred. The only window action is `close`. Unknown fields are rejected at every
nested record boundary, including raw engine IDs and path-cost/reachability data.

The authorized town extension uses a separate opt-in `adventure-town-v1`,
protocol 5, `heroes-player-history-v3` and `heroes-context-actions-v3`. The v2
defaults, serialized records and prefix hashes remain unchanged. Mixing versions
within either player's history or a paired finalization is invalid. The v3
environment uses a separate history API/file/manifest declaration; it does not
make legacy or v2 records interchangeable with town-operation experience.
Native acceptance evidence and its limits are reported below.

In v3, a town context also identifies its active panel and optional selected
entry. Each town action names its owned town handle and binds to the held
decision and context revision. `back` follows the actual parent panel;
`close` is available only from the town overview. Map and hero contexts retain
their previous behavior. None of these navigation operations ends the turn or
resets memory.

| Active town panel | Permitted current content | Supported selection |
| --- | --- | --- |
| Overview | The existing upper army and faction; audited hall/dwelling navigation controls. No hall catalog, construction prices, prerequisites or recruitment stock. | Explicit `open_panel`; `close` returns to map. |
| Hall | One actual displayed building per native hall slot, its opaque reference, static name identifier and displayed status. No hidden upgrade chain or detailed prices. | `open_building`, including a disabled entry to inspect its detail; `back` returns to overview. |
| Selected building details | Its static name/description identifiers, status, seven-resource price, and only the minimized requirement expression the UI displays for the applicable status. | `construct` only when the native Buy control is enabled; `back` returns to hall. |
| Selected dwelling recruitment | Actual creature cards, selected creature, shared available stock, selected unit price and native quantity-slider maximum; supported upper-army destination. | Explicit creature selection, integer count and destination in `recruit`; `back` returns to overview. |

Overlay panels do not refresh the underlying town army or faction. Building,
dwelling, offer and destination references are player-local routing handles,
not numeric neural features. Static content identifiers describe displayed
identity; they do not grant access to unshown object data or resolved game text.
Requirements preserve the displayed logical expression, not the engine's full
dependency graph or an auxiliary plan for satisfying it. The normal Grail prompt
on hall entry is an unsupported boundary, never bypassed or answered implicitly.

An activated native construction returns to the overview, matching Buy's
closure of both detail and hall. It issues exactly one building request; an
acknowledgement and verified native effect are distinct evidence. Exact price
deduction and the requested building/daily counter change must agree. No retry,
automatic prerequisite purchase or strategic ranking is part of the adapter.

Recruitment must not rely on the server's quantity clamping. The explicit count
must fit the currently displayed slider range before submission, and accepted
effects must match that exact count, cost, stock and army delta. The normal town
UI targets its upper army (town garrison or garrisoned hero); server support for
other destinations does not make those UI choices available. Capacity does not
reduce the native slider maximum. A valid count can produce the native no-room
response with no issued purchase and unchanged stock/resources. Duplicate-stack
automatic merging under a different engine setting is outside this interface;
do not silently alter the setting or perform extra army edits.

Malformed, foreign, stale or unoffered selections must not be converted into a
different valid operation. API rejection preserves the held request and history.
A valid consumed operation that the native UI rejects is recorded distinctly
from a server-issued request. All ordinary rewards remain zero. New panels obey
the same immutable-history, intervening-event and paired-finalization rules;
cutoff continuation explicitly unwinds their parent panels outside the frozen
learner prefixes. Historical contextual recordings remain unsupported by replay;
fresh v2/v3 capture uses the separate native verifier described below.

**Town validation:** at `1beba77`, real construction checks passed before
recruitment implementation. At `ffe377e`, the expanded cohort passed construction
and recruitment on both colors, selected base/upgraded prices and shared stock,
same-creature merging, full-army no-room feedback, the seven-day natural terminal
fixture and a process crash after an acknowledged build. These histories were
independently re-audited under the stricter `8053e66` contract. A final native
affordability probe at `8053e66` reached nine available archers priced at 100 gold
with only 35 gold available; its positive purchase was rejected unchanged.
The full suite passed 401 tests with 14 optional PyTorch skips. V2 navigation and
day-cutoff regressions retain valid histories; those navigation cases do not
establish additional hidden-world or closed-army-change coverage. A fresh legacy
movement game ended naturally and its offline replay verified.

All diagnostic attempts are retained. The initial construction probe failed
because of a reporting-field collision, then passed after that probe was fixed.
One recruitment-budget probe reached its declared day limit without creating
the desired shortage; it remains a failed diagnostic over an unscored valid
cutoff. The successful follow-up declares its different inspected seed; neither
is random-policy performance evidence. Capacity fixtures change only one town's
initial army, with exact field/byte isolation checked. The natural fixture is the
separately declared townless map. Neither the authored baseline, its tavern
restriction nor its hero count was changed.

The town upper-army route to a garrisoned hero is source-audited but not validated
in a real game: the pinned JSON format cannot initialize that state and this
interface has no garrison-swap action. An actual server-rejected purchase,
Grail/conditional auto-merge boundaries and exhaustive event coverage also remain
unverified. The ordinary no-room branch is a validated UI rejection without a
server request. This town-mechanics evidence does not establish a random-policy
natural-outcome rate or contextual replay, and involved no neural training. Raw records
remain local; compact provenance and coverage are committed in
`validation/town-native-2026-10-24.json`.

Reproduce the bounded probes with the prepared native build and local game data:

```sh
python3 validation/probes/validate_town_native.py --case construction \
  --binary vcmi/build/bin/vcmiclient --data "$HOME/Library/Application Support/vcmi" \
  --output runs/my-town-construction
python3 validation/probes/validate_town_native.py --case recruitment-upgrade \
  --binary vcmi/build/bin/vcmiclient --data "$HOME/Library/Application Support/vcmi" \
  --output runs/my-town-recruitment
python3 validation/probes/create_town_fixtures.py --kind capacity-full \
  --output runs/my-capacity-fixture
python3 validation/probes/validate_town_native.py --case capacity \
  --fixture runs/my-capacity-fixture/fixture.json \
  --binary vcmi/build/bin/vcmiclient --data "$HOME/Library/Application Support/vcmi" \
  --output runs/my-capacity-check
```

Informational notices are separate ordered events. VCMI's `InfoWindow` carries
no query or server reply, including in its `MODAL` display mode. The audited
baseline can emit resource/mine, experience-cap, new-week, battle-trophy and result notices;
stopping on every modal notice would prevent supported battles and natural
endings. The v2 notice contract carries its display mode, ordered text operations
over audited static identifiers and displayed numbers, and supported resource,
relative-player flag or displayed trophy components. The exact winner-only
battle-trophy template carries artifact or scroll-spell static `core:` content
identifiers from its displayed components, without instance IDs, descriptions,
inventory queries or new artifact operations. The authored map's four literal result
messages have exact stable identifiers. Arbitrary text, object IDs, unrelated
pack fields and unused substitutions are not learner inputs. The native
producer must validate each supported template and operation sequence before
normalization; this schema alone does not establish native notice coverage.

Notices enter only the recipient's history (new-week notices follow their
native broadcast routing), including between that player's decisions. They do
not create a policy action, acknowledgement, window transition, private-panel
refresh or reward. A notice stops an active route at the interaction boundary;
verified terminal finalization still takes precedence. The structured native
hook must run before text translation and cover the separate new-week path.
Unrecognized notice shapes remain explicit unsupported stops.

Legacy server replay and visible-state digests do not verify these new fields.
Fresh contextual replay independently checks window operations and payloads,
roads, ordered recipient-specific notices, and policy-action/native-command
associations. It also reconstructs both delivered histories from raw bridge
messages. Historical recordings without the native transaction tape remain
unsupported. File hashes alone do not establish semantic correctness, and no
contextual record may inherit a verified label from the legacy replay path.

Records are immutable and append-only. Navigation consumption is distinguished
from native command issuance and acknowledgement. An unacted decision must be
superseded before a new visible view or event enters its history. A consumed
action without a trustworthy result never acquires an ordinary result;
trajectory-v2 may derive only the verified terminal closure described below.
Natural finalization binds both complete recorded player prefixes to
complementary engine/socket outcomes and a clean exit. A collection cutoff is
provisional until both players have real held decisions, readers finish without
failure and bounded continuation/shutdown evidence is verified. Technical and
unsupported failures provide neither terminal rewards nor bootstrap targets.
The technical cutoff finalizer may explicitly close a window and end a turn to
obtain the other player's real boundary, at most eight actions per player. These
actions are recorded in separate live evidence outside each frozen learner
prefix; they are never policy experience or automatic route continuation.

`validation/probes/audit_contextual_histories.py` independently reconstructs both
streams and checks their immutable prefixes, paired proofs and manifest hashes.
Without a manifest it reports structural consistency only, with no effective
learner eligibility. A valid record with zero completed samples remains distinct
from a usable learner trajectory. Its output explicitly does not prove native
visibility or closed-window freshness; those require the real-game probes.

```sh
python3 -m unittest discover -s tests -p 'test_contextual_*.py' -v
python3 validation/probes/audit_contextual_histories.py \
  --red <red-history-v2.jsonl> --blue <blue-history-v2.jsonl> \
  --manifest <manifest.json> --output <new-audit.json>
```

The independent native acceptance driver is
`validation/probes/validate_contextual_native.py`. It refuses an environment
without the opt-in contextual API before launching a game. Its bounded cases
cover navigation, exploration, partial movement, closed-window battles,
cutoffs and owned-process failures; an optional same-seed pair compares hidden
opponent movement variants. A separate seeded uniform-random cohort supplies
outcome coverage. Every attempt is preserved; absent conditions are unverified,
and failed audits cannot pass an individual acceptance gate. These are evaluation
probes, not agent strategies or training examples.

`validation/probes/create_contextual_fixtures.py` prepares optional paired maps
from an actual initial v2 player frame. The base is byte-identical to the authored
map; variants change only unknown terrain/roads or append a fully hidden neutral
guard. Existing objects and victory rules remain unchanged. Its archive tests
do not prove that VCMI loads a variant or that its initial player view is equal:
both must be checked in the real engine before interpreting a paired comparison.
Terrain/road variants provide bounded coverage, not an exhaustive isolation proof.

`validation/probes/create_casualty_fixture.py` strengthens one neutral stack to
check changed army contents after a closed-window battle.
`validation/probes/create_boundary_fixture.py` prepares isolated townless and
Den-of-Thieves maps; `validate_contextual_boundaries.py` verifies their native
natural-terminal or unsupported-interface behavior. The generators preserve
all undeclared map fields and leave the authored baseline unchanged. These
mechanical evaluators are excluded from training and random-policy outcome
estimates. `validate_contextual_random.py` runs a separately seeded uniform cohort,
counts every failure/cutoff and exits unsuccessfully on native failures or probe
errors. None of these commands performs learning updates.

At `d0786d2`, all 379 discovered tests complete without failures: 365 pass and
14 optional PyTorch tests skip. The native build and exact root/dependency recipe
check pass. Real VCMI runs validate both players' map/window navigation, full
coordinate support, current-turn movement, separate fog, hidden-opponent route
invariance, paired cutoffs and unscored process failures. A strengthened neutral
stack produces five actual casualties: the old window snapshot stays immutable,
closed observations omit the army, and reopening shows the changed count.
A townless fixture reaches the engine's unchanged seven-day rule with Red −1
and Blue +1; Red's permitted postterminal hero-removal view is retained. A real
Den visit stops as unsupported with no outcome, reward or bootstrap.

Keep the random cohort separate: eight predetermined contextual attempts at
`c75d015` produced seven day cutoffs and one unsupported failure, with no natural
ending. That failure exposed an overly broad artifact-assembly check and then a
missing battle-trophy notice template. Both are fixed. The selected case's final
rerun at `d0786d2` delivers the trophy only to the winner and reaches its day
cutoff without failure after 1,047 choices. All attempts remain in the compact
evidence; none is a draw or a training example. The fixture does not establish
the contextual random-policy natural-outcome rate. Rejected native-command
coverage and contextual replay remain unverified, so the broader training gate
is still closed. No training, paid compute or merge occurred.

#### Proposed minimum interactions and parameters

Start with map controls and open/read/close windows, then add the smallest audited
internal operations below. Names describe mechanics only; no choice is ranked
or labelled as beneficial. The agent selects the operation, target and quantity.

| Context | Proposed interaction | Typed parameters and mechanics |
| --- | --- | --- |
| Map | `move_to` | Owned `hero_ref`, integer destination `(x, y)` on the current map level. The policy chooses the target; the engine plans and executes the permitted current-turn route prefix under the proposed rules below. |
| Map | `end_turn` | No parameters; remains possible without a hero while the player is still in the game. |
| Map | `open_hero`, `open_town` | An accessible owned `subject_ref` from permitted map/list observations; returns a structured window observation. |
| Hero | `close` | Return to map; no world command or automatic end-turn. |
| Hero | `transfer_stack` | `source_slot`, `destination_slot`, integer `count`; split/move/merge within this hero's visible army, with an empty or same-creature destination. Swapping unlike stacks is a separate deferred operation. |
| Town | `close` | Return to map; no automatic purchases or turn completion. |
| Town | `open_panel`, `back` | A currently displayed construction/recruitment control reference, or return to its enclosing supported town panel. Proposed explicit town-internal subcontexts, not access to unrelated dialogs. |
| Town, construction panel | `construct` | An offered `building_ref`; attempt one construction with the displayed cost/prerequisites. |
| Town, recruitment panel | `recruit` | An offered `creature_offer_ref`, a currently supported visible `destination_army_ref`, and integer `count`; attempt exactly that purchase/placement. No automatic quantity or destination selection. |

Construction/recruitment panels were explicitly authorized on 24 October as the
next incremental implementation. Their audited native navigation and information
scope define the boundary; this does not authorize unrelated windows. If the operation
requires an additional unsupported modal interface, leave it unsupported under
the rule below. In the native construction interface, town → hall → building
details reveals information progressively; do not collapse that into revealing
every building's detailed cost/prerequisites when the town first opens. Owned
towns/heroes may be opened through their normal lists without physically moving
a hero to the town. Tavern hiring, shops, spell books, artifact operations, inter-hero
exchange, obelisks and other special dialogs stay deferred. The authored baseline
also forbids tavern construction; do not quietly change the map to enable hiring.

The implemented movement baseline still sends one adjacent destination to
`moveHero`, wrapped as a one-element path. It does not implement destination
routing. The new experiment replaces the proposed `move_step` action with
`move_to`; it does not change the old baseline or its evidence.

#### Destination movement: agreed choice and proposed execution

**Agreed:** the policy selects `(x, y)`; engine routing executes movement within
the hero's available points. The learner receives legitimate hero position,
remaining movement points and known terrain/roads, then the actual resulting
position and points. It receives no planned route, predicted path cost, arrival
turn, reachable-target mask, range overlay or auxiliary cost-learning target.
The difference between observed before/after movement points is experience,
not an engine-provided prediction. Ordinary movement transitions receive 0;
verified natural win/loss uses +1/-1, while technical failures remain unscored.
Routing is an approved mechanical operation; choosing or replacing the
policy's strategic destination remains outside the adapter's authority.

**Implemented execution choices — bounded native evidence, with remaining cases listed in the validation report:**

- Use the hero's visitable map tile as `(x, y)`, on the authored single level.
  The adapter handles native coordinate conversion. Initially support ordinary
  land/road movement only, with boats, teleportation, flight and spell use
  explicitly outside the route configuration. These are public capability
  restrictions, not tests of undiscovered objects.
- For each accessible owned hero, offer every in-bounds coordinate, including
  unknown terrain, obstacles, the current position and targets beyond this
  turn's range. Keep that coordinate support even at zero movement points.
  Do not precompute, filter, order or annotate targets by route existence,
  cost, distance, usefulness or current-turn reachability. Bounds and current
  level are public; coordinate inputs are not handcrafted target priorities.
- Compute a route only after selection, using audited player/team knowledge
  and native mechanical routing with fixed configuration. It may extend beyond
  this turn; execute only its legal current-turn prefix. An all-map or omniscient
  planner is forbidden even if its route is never returned: the hero's chosen
  detour or a failure could disclose hidden state. Do not expose planner timings,
  internal search counters, future nodes or rejected hidden-object details.
- This first interface does not speculate through unexplored terrain. The native
  pathfinder blocks unexplored tiles; such destinations remain selectable but
  return `no_route_known` if no route exists in the permitted knowledge and
  supported travel modes. This does not assert that the physical destination is
  unreachable. Do not substitute a frontier, nearest reachable tile or another
  target. Movement to known frontier coordinates can reveal more terrain through
  ordinary fog rules. Speculative routing through unknown terrain would need
  a separate decision and validation.
- Once planned, follow that route without strategic detours or automatic
  replanning. Validate each native step and capture every permitted intermediate
  event. New observations alone do not trigger a handcrafted threat/utility
  rule; normal engine interruptions and the encounter stop rule below apply.
  A policy destination is one decision, while its native steps and actual
  elapsed game time are logged separately. No intermediate learned decisions
  are invented. Bound route execution and planning; a watchdog or inconsistent
  engine response is a technical failure, not a gameplay rejection.

History must link the destination decision, each issued/consumed native command,
intermediate observations and final result without duplicating the policy action.
There is one behavior probability for the policy's hero/coordinate selection,
not a policy probability for each engine-chosen step. Keep step count and elapsed
game time as trajectory metadata so the later PPO design explicitly chooses its
discount convention for this interface rather than inheriting adjacent-step
timing. This adds neither intermediate rewards nor path-cost supervision.

The adapter returns a structured result tied to the original decision/hero/target,
with a stop status, actual position and remaining points when observable, and
`destination_reached` determined from the actual visitable position. It does not
return a remaining distance, predicted cost or route feasibility for other
targets. The selected stopping semantics are:

| Situation | Intended execution and next boundary |
| --- | --- |
| Target equals the current tile | Return `reached` without movement; consume one policy decision and no movement points. |
| No movement points remain | For any different target, return `movement_limit` immediately without planning. Do not end the turn. |
| No route in permitted knowledge/supported modes | Return `no_route_known`, with no movement or automatic alternate target. This is an ordinary zero-reward result, not an episode failure or a claim about hidden terrain. |
| Target is farther than this turn permits | Follow the current-turn prefix and stop before the next unaffordable step. Return `movement_limit` and the actual endpoint/points, even if points remain. Do not cross a turn boundary or automatically end the turn. |
| Native interruption, blocked/rejected next step or route invalidation | Keep completed movement, stop the remaining route and return `interrupted` with only a legitimately observable reason. Do not refund completed steps, replan or retry automatically. |
| Visit, pickup, capture or automatic battle encountered en route | Resolve the supported native effect, then stop the route at that first interaction even if points remain. Record every permitted event and return `interaction` plus actual position/points; an interaction can occur without occupying the target tile. Do not automatically open a hero/town window or expose its contents. |
| Encounter requires an unsupported dialog/choice | Apply the bounded `unsupported_interaction` stop rule below; do not answer it or continue the route. If even a dismiss/acknowledgement is required and unsupported, it follows the same rule. |
| Hero becomes unavailable, while its player remains alive | Stop without selecting another hero. Report only the permitted own-hero event, then offer that player's next valid map decision. Losing a hero is not by itself terminal defeat. |
| Verified natural episode ending | Stop all pending movement; preserve final visible events and apply the verified terminal contract. No synthetic next decision. |
| Engine crash, timeout or incomplete event capture | Fail unscored with no bootstrap or training targets; a partial route does not turn a technical failure into an ordinary cutoff. |

These conditions are evaluated at native step/event boundaries. Technical failure
or incomplete capture prevents scoring even if a natural ending is subsequently
reported; otherwise verified terminal finalization takes precedence. An encountered
interaction or interruption stops further steps, and position/points determine normal
completion or the movement limit. At entry, an already-reached target takes
precedence over zero points. An observed interaction at the destination can
report `interaction` with `destination_reached=true`; it need not be relabelled.
If the target was not reached, discard the unexecuted route. The requested target
remains historical input only; resuming later requires a fresh policy action,
including after an explicit end-turn. No persistent auto-travel queue is implied.

The engine can use costs internally to route and enforce movement limits. It
must not supply those costs as policy/value features or supervision. Native
pathfinding needs a source and real-game fog audit, including guard relations,
object access, caches and unsupported travel modes. Reusing the graphical
movement controller wholesale is unsuitable: it includes presentation work and
automatic replies. Implement a structured headless adapter with the declared
stop rules, shared with replay and coordinated with Claude's engine workstream.

The contextual policy selects quantity through a learned binary interval tree
over every integer from 1 to the displayed slider maximum. This handles the
full native int32 domain without enumerating billions of output entries. Record
the exact support and every conditional probability so PPO recovers the joint
behavior probability. Do not let the environment fill parameters
with an external RNG. Extend the current candidate representation with typed
target/control references and permitted target features; integer handle
embeddings would violate the existing information contract.

**Availability is not utility or a guarantee of success.** Filter offers using
only current context, declared adapter capabilities and player-visible facts:
ownership/access, displayed costs/stock/prerequisites and visible army slots.
Destination coordinates use the unfiltered public domain above; do not consult
pathfinding, hidden obstacles, enemies or undiscovered objects to prune it.
The old baseline retains its separate adjacent-attempt mask. The server still
checks actual reachability, movement,
collisions, access, costs, prerequisites, stock and army invariants atomically.
A rejection reports only information the player is entitled to receive. For two
worlds with identical permitted history/context but different hidden state, the
observations, offered support and parameter domains must be identical until a
legitimate observation distinguishes them.

#### Unsupported interactions and bounded execution

An unsupported query dialog, special window or malformed/unrepresentable transition
stops collection with a distinct `unsupported_interaction`/technical status,
preserves diagnostic evidence and performs bounded engine shutdown. It is not
a defeat, draw or eligible horizon: no terminal reward, value bootstrap or
training targets. Do not answer it randomly, select a default, or wait forever.
Keep this reason separate from crashes in reporting even though both are
ineligible. A later natural result cannot repair missing interface history.
The supported information-only notices above require no dismissal command or
invented reply. Query dialogs, including a garrison dialog when revisiting an
owned mine, remain unsupported in this minimal interface.

For a visible object/control, a publicly defined support restriction may prevent
offering an operation that is known to require an unsupported interface. Never
use a hidden object's type or hidden dialog availability to remove a movement
attempt. If an unknown encounter triggers the dialog, stop only when it actually
occurs. Automatic battle resolution remains the agreed external subsystem; any
unsupported surrounding choice still follows the stop rule. The old baseline's
uniform supported-dialog policy remains available only in its labelled cohort.

Count every policy choice, including destination selections, open/close/back and
rejected attempts, toward a separate total-decision budget. The current headless action cap counts only
adventure commands and is insufficient for window loops. Navigation can consume
zero game time but receives no reward/penalty. A deliberate budget cutoff is
bootstrap-eligible only at that player's verified next decision, with matching
context, offers and recurrent state. Never invent a boundary for the opponent;
exclude unresolved data if both required player boundaries cannot be obtained
within the declared continuation bound. A wall-clock watchdog is a technical
failure, not a substitute for such a cutoff.

#### Proposed implementation and real-game validation stages

1. **Freeze versions and audit visibility.** Keep the adjacent-step baseline
   intact. Destination selection/engine routing is agreed; resolve the proposed
   unknown-target, travel-mode, interruption and encounter semantics before
   implementation. Agree the field/context matrix, internal operations and
   town-panel scope. Document permitted observations and mechanical effects;
   engine path cost remains excluded even if a normal GUI would display it.
   Use the reviewed boundary fixes. Codex owns learner contracts; coordinate
   engine transport, pathfinding and capture changes with Claude's workstream.
2. **Implement history/trajectory v2 and navigation.** Add explicit context,
   consumed-action and terminal/supersession semantics, then map → hero → map →
   town → map using structured observations. Add the audited visible-road field
   to the schema and encoder, and update candidate target bindings, keeping
   separate player recurrent states. No PPO yet.
3. **Implement destination routing without previews.** Add typed hero/coordinate
   support, the player-knowledge route adapter, bounded current-turn execution
   and structured stop results. Capture every intermediate visible change and
   distinguish the policy decision from its native command sequence. Record
   routing configuration and replay both selected targets and actual commands.
   Audit all planner inputs and remove cost/range/arrival features. No PPO yet.
4. **Validate movement and navigation on real VCMI.** Check access denial,
   stale-context commands, closing/reopening, closed-panel non-refresh and map-visible
   exceptions. Verify intervening sightings and final views, isolated players,
   zero-time loops, cutoff/crash handling and replay of recorded context actions
   alongside native game commands. Exercise destination equality, zero points,
   known obstacles/disconnected targets, unknown targets, targets beyond the
   current turn, terrain/road costs and insufficient points for the next step.
   Check en-route encounters, native interruption, hero loss, unsupported dialogs
   and no automatic next-turn continuation. Change hidden terrain/objects while
   holding player history fixed: candidate domains, planner decisions and
   pre-discovery outputs must not change. Verify that no cost/reachability
   features or labels enter policy/value inputs or targets. Validate native
   coordinate conversion and route replay, including all intervening sightings.
   Synthetic tests alone do not pass this gate.
5. **Add internal operations incrementally.** The authorized order is town
   construction first, then recruitment and quantity selection; broader army
   editing remains deferred. Verify exact costs,
   stock, slot/count changes, rejections and no hidden-mask leakage. Each step
   needs real game evidence, an explicit coverage list and replay verification.
   Exercise unsupported-dialog termination without answering it. Retain failed
   attempts rather than selecting only successful recordings.
6. **Collect bounded random-interface validation, then review integration.**
   Use predetermined seeds, both colors and matching interface versions. Recheck
   reachable natural outcomes, action consumption, complete histories, rewards
   and player-local bootstrap boundaries. Report natural endings, deliberate
   cutoffs, technical failures and unsupported interactions separately. Only
   after review authorize a learning experiment. The subsequent explicit user
   authorization permits implementing the collector/PPO now, with synthetic
   optimizer tests and bounded fixed-weight native inference only. It does not
   authorize gameplay optimization, a training campaign or paid compute.

### Player-history v1 contract

`experiments/learner/contract.py` defines the strict typed boundary and
`decode_frame` accepts its JSON representation. PR #6 now emits a projection of
this stream and local checks decode real games; that is not a complete window or
dialog adapter. Incomplete projections remain ineligible. All records have
`schema="heroes-player-history-v1"`, `episode`, `player` (0 or 1), a contiguous
per-player `seq` starting at zero, public game `day`, and a discriminated `kind`.
The sequence counts only that player's records, never hidden/global events.
Unknown fields, unsupported vocabularies, missing sequence entries and mixed
episode/player identities fail closed. Routing metadata never becomes a neural
feature.

| Kind | Payload and meaning |
| --- | --- |
| `view` | A complete current player-filtered `View` at a relevant observable change, including between the player's decisions. |
| `decision` | A complete current `View` and nine-boolean attempt `mask`; end-turn is always enabled. Its sequence is the decision identifier. |
| `action` | Own sampled action and `decision_seq`, after consuming the matching decision. |
| `result` | Own action, matching `decision_seq`, and boolean `accepted`. Acceptance means request acknowledgement, not displacement or strategic success. |

A `View` contains currently permitted tiles (position, categorical terrain,
blocked and visitable flags), currently observed objects (position, categorical
type/subtype, owner relative to the player, opaque local handle), and own state
(hero position or absence, movement, seven resource amounts, army slots/types/
counts, town positions and mine count). Unknown terrain has no tile entry;
an absent object has no current object entry. There is no hidden destination,
hidden removal/death field, omniscient replay input or handcrafted value.
The initial component supports only the authored 20x20x1 scenario and bounded
unmodded categories: 16 terrain types and 256 object types, subtypes and creature
types each. Unsupported data is rejected, never clipped or silently discarded.
Known own state remains available under normal player rules even when a map tile
is not currently visible. Exact enemy armies and attributes not explicitly in
this schema are not requested or inferred by the adapter.

Complete views are a correctness-first reference encoding, not a throughput
target. A later version may use lossless deltas. The environment must capture a
view whenever information in these fields becomes available or changes, including
a transient enemy sighting and subsequent disappearance before the next decision.
It must not emit a record solely because of a hidden change. View capture and
the decision's preceding history must be atomic and complete: no dropped,
coalesced or truncated visible transitions. A visible-to-hidden move contains
the last legitimate sighting followed by absence, never the hidden endpoint or
a claim of death. Merely receiving an engine callback does not authorize its
fields. VCMI's existing fog semantics remain unchanged.

Handles are assigned only on legitimate first observation, scoped to one player
and episode. They are used for validation/bookkeeping, but their spelling or
ordinal is deliberately omitted from this first neural encoder. This makes it
invariant to handle renaming; persistent entity-specific memory is not yet
implemented. The full chronological views retain the evidence of where and when
an enemy was seen. Only the GRU carries earlier information forward; a previous
object is never copied into a new current-view grid as if still visible.

The first encoder covers the structured fields above. In the original movement
baseline, uniform dialog selection and automatic BattleAI remain fixed external
subsystems; their commands, versions and seeds must be retained in episode
provenance. The expanded interface retains automatic battles but follows its
explicit unsupported-dialog stop rule instead of inheriting that chooser. Rich dialog text,
battle details and additional player-visible attributes require a versioned
schema extension before being admitted as learner features. The adapter must
report its observation coverage; this prototype is not an audit of VCMI callbacks.

### Typed action candidates and extension boundary

`experiments/learner/actions.py` defines `heroes-action-candidates-v1` as an
ordered nonempty set of available candidates. Each has an opaque routing key,
a categorical action type, and typed parameters with categorical slots and
category, scalar or position values. No catalog of shops, items or undiscovered
mechanics is supplied. Categorical embedding capacity is not an enumeration of
available choices. Only the environment may offer candidates, through the
player's legitimate interface; their order must not encode strategic priority.
Parameter values must be observed facts or the player's proposed control inputs,
never hidden state, handcrafted utility or expert explanations.
Raw engine IDs and arbitrary local entity-handle ordinals must not be placed
inside category/scalar arguments. Future entity targets need routing-only
identity plus permitted observed features, not numeric identifiers as features.

The implemented scorer combines randomly initialized type/parameter embeddings
with the recurrent context and returns one score per supplied candidate, using
the same weights for every choice. There is no permanent output neuron per item.
Opaque keys do not enter features. Reordering candidates only reorders scores;
renaming keys does not change scores. Typed parameters retain their binding to
their action and slot. Unsupported schema/category ranges fail explicitly.

The only game action adapter implemented here is `movement_candidates(mask)`:
available moves use type 0 with a direction parameter; end-turn uses type 1.
It exposes exactly the original attempt support and maps scores back to the
nine-action bridge interface. Each decision record stores this exact versioned
candidate set and selected behavior log-probability. The low-level scorer is
tested on variable synthetic typed choices, but history/trajectory v1 still
accepts movement only. Construction, shops, recruitment, upgrades and quantity
selection need a player-filtered engine adapter and history/trajectory v2 that
records candidate keys/parameters instead of nine-action indices. Freeze that
version and support during a PPO rollout; never reconstruct old support from
the current game state. Conditional quantity choices must retain each sampled
conditional probability and support to reconstruct their joint log-probability.

Before another scenario or HotA is enabled, version the observation/action
contracts and checkpoint configuration together. Retain compatible weights only
through an explicit tested migration, initialize new embeddings randomly, and
restart collection after migration. State-dictionary roundtrip tests cover
weight persistence now; a resumable trainer must later also save optimizer,
RNG, update counters, opponent-pool identity and sampler state. This prototype
does not implement that trainer or claim compatibility with HotA.

### Per-player trajectories and recurrent computation

`experiments/learner/trajectory.py` separates a player's history from its
decision samples and episode finalization. A sample records the selected action,
its exact mask and behavior log-probability/value, behavior and frozen opponent
checkpoint identities, and the provenance of the recurrent prefix. Opponent
actions are not learner samples. Neither an accepted end-turn nor an action
acknowledgement closes the transition: the next boundary is that player's next
decision, after all intervening visible history, or its verified natural result.
A terminal may close an outstanding action without an `action_result` record.
It may also supersede a final unacted decision: retain the observation, invent
no action/sample, and attach the terminal reward to the last actual action.

Natural finalization needs matching complementary engine/socket results for
both players, complete player histories and clean process shutdown. The Python
contract checks the supplied attestation's consistency; the external validator
must establish its truth and retain an evidence reference. Video success is not
a condition of learner episode validity. Only this natural finalization yields
+1/-1, with no value bootstrap. Ordinary transitions have reward zero.
A deliberate rollout/day/action horizon is eligible only at a verified next
decision of that same player, with the same behavior checkpoint and recurrent
prefix. It supplies a value bootstrap, never a draw label. A timeout, crash,
protocol failure or incomplete history supplies no terminal reward, bootstrap
or training targets. A history hash binds records; it does not prove visibility
or that a recurrent tensor was correctly computed.
The included reference targets use undiscounted (`gamma=1`, `lambda=1`) returns;
they specify reward/boundary semantics and are not a PPO or GAE implementation.

The model uses learned categorical embeddings, a small spatial encoder and own
state encoder, then a GRU shared by policy and value. Scalar counts use `log1p`
for numerical scale, not strategic weighting. The spatial grid represents
coordinates as observations; it adds no directions, destinations, exploration
bonuses or distance-based strategy. Every history frame updates memory. Only
`decision` frames produce a masked categorical policy and scalar value prediction.
Both heads and their shared encoder/recurrent weights remain trainable from
experience; synthetic gradient checks do not train a game policy.

Each inference session owns one player's recurrent state. Sessions can share
weights but cannot share history or hidden state. Chunks and end-turns preserve
memory; an explicit episode boundary finishes a session before reset. The core
forward operation is differentiable for eventual sequence training. The safe
inference wrapper disables gradients. Future PPO must preserve/recompute the
per-player prefix under the relevant weights, use sequence minibatches with
explicit padding/loss masks, and keep collection weights/opponent distribution
frozen. Stored behavior-state tensors must not silently become current-weight
states during repeated PPO updates. Batching, burn-in/truncated BPTT and PPO
losses are intentionally not implemented in this component.

Run the optional CPU component tests with Python 3.13 on macOS arm64 (the tested
lock target); the existing stdlib recording tooling needs no PyTorch install:

```sh
uv venv --python 3.13 .venv
uv pip sync --python .venv/bin/python --require-hashes dependencies/learner.lock
.venv/bin/python -m unittest discover -s tests -p 'test_learner_*.py' -v
```

The hash-pinned lock is generated from `dependencies/learner.in`. It is a
platform-specific prototype environment, not a verified Linux/GPU dependency
matrix. Tests without PyTorch explicitly skip the neural component. These are
synthetic contract/gradient checks, not real headless episodes, a training run,
replay validation, a natural-outcome estimate or a performance benchmark.

### Contextual neural integration and PPO

**Authorized implementation stage, 24 October:** connect PyTorch to the actual
map/hero/town interface, including construction and exact-count recruitment;
implement recurrent PPO, collection, frozen opponents and checkpoint resumption.
Only synthetic optimizer tests and bounded real games with fixed, untrained
weights are authorized. No optimizer update on gameplay is permitted. This work
is published in draft [PR #9](https://github.com/tkorsi/heroes-rl/pull/9), stacked
on validated town PR #8 at `8f6c715`, on `codex/contextual-ppo`, keeping
history/action v1 and contextual v2/v3 compatibility separate. No native source
change is required for the neural adapter.

`contextual_model.py` uses a current-view spatial encoder, learned typed field
encoder and one shared GRU actor–critic. It processes permitted views and events
in chronological order, plus the player's own actions/results. Each player has
its own zero-initialized recurrent state at each episode. Only the GRU retains
learned history; the diagnostic last-seen cache is not an input. Temporary
routing bindings refer to their explicitly current or prior observed snapshot;
they do not query the engine or accumulate a hidden-world state. Opaque handles,
sequence/revision/command counters and verification metadata are not features.
Categorical content IDs and text IDs enter randomly initialized learned layers,
without pretrained language/game knowledge or numeric utility scores. Resource and skill slots preserve their
positions; collections of objects and controls have no strategic ordering.

The policy chooses action kind, then any required visible subject/control/card,
then a full-map coordinate or exact positive count. Every coordinate in the
public rectangle is supported, including unknown and unreachable destinations.
The count tree has at most 31 learned binary choices; its interval-size base
measure gives uniform integer probability when learned logits are zero. Every
conditional and their sum are recorded with the immutable offered support.
The destination of recruitment is explicitly selected from the currently
supported native singleton destination. No adapter RNG supplies parameters.
Overview dwelling controls currently expose only opaque references and panel
kind. Semantically indistinguishable controls receive identical logits: no
hidden dwelling attributes or reference suffixes are invented to distinguish
them. The opened popup exposes the actual creature choices.

`contextual_rollout.py` maintains separate player sessions and freezes behavior
weight identities. A sampled intention becomes a transition only when the same
history records submission, consumption and a resolved result, or the explicit
trajectory-v2 natural terminal closure described below. Superseded unacted
decisions are retained as history, not samples. Intervening opponent turns remain
inside the acting player's transition. Both complete recorded histories and
matching final proofs/manifest hashes are required. Other unresolved consumed
actions make that player's whole trajectory ineligible; the collector never
attaches their terminal reward to an earlier action. Failures provide no
reward or bootstrap. A cutoff uses only the actual unacted next decision and
value from that player's matching recurrent prefix. Frozen-opponent and
diagnostic records are never on-policy training samples.

`ppo.py` implements clipped joint action-ratio loss and value regression with
`gamma=1`, default `lambda=1`, optional advantage normalization and gradient
clipping. Batches contain contiguous decision windows with all intervening
history frames. Before each minibatch it reconstructs the prefix from episode
start under the **current** parameters without gradients, detaches at the window
start, then differentiates through the full window. This is explicit truncated
BPTT, not reuse of stored behavior hidden states. Stored behavior probabilities,
values and verified bootstrap values remain fixed across updates. Initial
entropy coefficient is zero: exact entropy over the large conditional count
space is not approximated by a misleading sampled-path entropy. The optimizer
entry point currently accepts synthetic test rollouts only; enabling gameplay
optimization requires a separately reviewed authorization change.

`checkpoints.py` preserves model/config/schema identity, weight hashes,
CPU Adam optimizer/progress state when applicable, Torch/Python and named sampler RNGs,
source metadata and frozen opponent snapshots. Loading validates fields/tensors
and uses `weights_only=True`. Resume is at a boundary between rollouts; engine
processes, live sockets and episode-local GRU states are not checkpointed.
Resume validates named parameter ordering and finite moment shapes before
changing live state. Fixed policy checkpoints need no optimizer. A frozen opponent pool copies and
freezes weights, samples from an explicit fixed set, and checks for mutation.

`python -m experiments.learner.inference` is a bounded fixed-weight runner, with
no optimizer entry point. It uses one frozen opponent for each pair of engine
seed/color trials, records policy initialization/sampling seeds and source/hash
identity, saves untrained checkpoints and both complete rollout histories, and
runs the independent history auditor. It retains every attempt, verifies
unchanged weights, and reports natural endings, valid cutoffs, technical
failures and collection errors separately. Its collector records remain
diagnostics, excluded from optimization. The default diagnostic purpose makes
no outcome-rate claim; the explicit ordinary-evaluation purpose requires the
unchanged authored baseline map and reports the observed frequency within its
declared finite cohort. Collection never grants a replay-verification label;
the separate contextual replay command checks fresh recordings afterward.

Validation at `c3edda8` (core/runner), `0b90703` (natural fixture) and `8a24a4a`
(corrected crash probe) passed on local macOS arm64, Python 3.13.5 and locked
PyTorch 2.9.1:

- **456 tests passed, zero skipped.** The new tests include exact action/count
  probabilities, field/handle invariance, recurrent gradients, paired boundaries,
  synthetic PPO updates, and bit-exact next-update checkpoint/RNG resumption.
  The existing optional neural tests ran with PyTorch installed.
- Four fixed untrained-policy games on the unchanged authored map, engine seeds
  `95011`/`95012` in both learner colors, each reached its declared 192-choice
  collection cutoff. All histories and paired bootstrap boundaries audit
  correctly. They contain all eleven action kinds, five actual constructions,
  fourteen exact-count recruitments and twelve creature selections. No technical
  failure or natural completion occurred in this four-game diagnostic cohort.
- Independent prefix recomputation of 5,383 frames reproduces all 776 decision
  values, 768 behavior joint probabilities and 1,295 conditional probabilities
  with **zero difference**. Eight cutoff decisions were valued without sampling.
  All eight native diagnostic player rollouts reject PPO admission.
- Two isolated townless-fixture games load the saved untrained policy checkpoint
  and reach verified Red loss/Blue win. Both loser histories have a final consumed
  action without a resolved result. The collector excludes the entire losing
  rollout, preserving the loss in episode evidence without assigning it to an
  earlier action. **This unresolved terminal-action case remains a training
  gate**, not a completed source of negative policy/value targets.
- An intentional real-process crash after four resolved neural choices produces
  no rewards, outcomes, bootstrap or usable samples. The first crash attempt's
  probe report failed on relative-path formatting; its raw failed-game history
  is retained. The corrected reporting probe passes on a second declared attempt.
- Weights stayed unchanged in every native run; native optimizer updates were
  zero. Exact generated-engine/root-source verification passes; no engine source
  was changed for this stage. Contextual replay is still unsupported.

All eight native attempts, including the reporting failure, source/checkpoint
identities, raw-file hashes and limits are retained in
`validation/contextual-neural-2026-10-24.json`. Raw histories, checkpoints and
reports remain under ignored `runs/`. These are mechanics/collector diagnostics,
not demonstrations, gameplay optimization or an outcome-rate estimate.

With the documented locked Python environment and prepared VCMI build:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m experiments.learner.inference \
  --output runs/neural-fixed-check --engine-seed 95011 --engine-seed 95012 \
  --max-requests 192 --max-days 40
python3 validation/probes/create_boundary_fixture.py \
  --kind natural-townless --output runs/neural-terminal-fixture
.venv/bin/python -m experiments.learner.inference \
  --output runs/neural-terminal-check \
  --checkpoint runs/neural-fixed-check/untrained-policy.pt \
  --map runs/neural-terminal-fixture/natural-townless.vmap \
  --engine-seed 95101 --max-requests 256 --max-days 40
.venv/bin/python validation/probes/validate_neural_failure.py \
  --output runs/neural-crash-check \
  --checkpoint runs/neural-fixed-check/untrained-policy.pt --engine-seed 95201
```

#### Terminal closure and pre-training validation follow-up

The user authorized a follow-up in draft
[PR #10](https://github.com/tkorsi/heroes-rl/pull/10) on
`codex/terminal-validation`, based on PR #9 at `daa1e98`, to repair losing-player
terminal credit and check the remaining gates with fixed weights. No gameplay
optimizer update or training is authorized.

Trajectory-v2 and rollout-v2 derive a separate `TerminalClosure` from the final
action's recorded consumption, explicit `superseded(reason="terminal")`, natural
terminal and matching paired finalization. Its `result_seq` is null: it claims
neither command acceptance nor an ordinary result. Only that final consumed
decision receives the terminal +1/-1, with no successor or bootstrap. Earlier
transitions retain zero reward. Full recorded post-terminal views/events remain
in the history and recurrent pass. Submitted-only intentions, ambiguous earlier
gaps, failures and cutoffs never acquire this closure. `trajectory_eligible`
reports structural target continuity separately from optimizer admission;
native diagnostics and frozen opponents still cannot enter PPO. Existing
history-v2/v3 bytes and explicit trajectory-v1/rollout-v1 interpretation remain
unchanged. The native command-result protocol is unchanged.

The mirrored `natural-townless-blue` fixture removes only Blue's town and
main-town header, using the same native seven-day rule as the original Red
fixture. These isolated fixtures test terminal mechanics and are never included
in ordinary-game outcome frequency. The ordinary cohort uses the unchanged
authored map and predeclares every seed, seat, fixed checkpoint, sampler seed and
budget before launching. Every attempt is retained, including collection errors.
Cutoffs, unsupported interactions and technical failures remain unscored.
`validation/probes/audit_neural_rollout.py` independently replays the checkpoint
over every saved player frame and checks all decision values, joint/conditional
probabilities, terminal targets, actual cutoff bootstraps and rejected optimizer
admission. This is inference reconstruction, not native engine replay.

Native replay/drain audit found concrete remaining blockers. Contextual capture
bypasses the legacy observation ledger, leaving it empty. Contextual records
lack authoritative packet/receipt anchors and persisted native-request-to-action
associations; the legacy verifier omits roads, windows, town panels and ordered
notices. Historical contextual records therefore cannot be upgraded to verified
replay by removing the guard. Fresh evidence and a contextual verifier are
required. Separately, the second `PlayerEndsGame` immediately requests headless
shutdown; `NetworkHandler::stop()` stops the I/O context without an authoritative
final-packet barrier. Clean exit and bridge EOF prove the recorded stream closed,
but do not prove every late native event arrived. Observed final hero-removal
views are useful fixture evidence, not a general drainage guarantee. These
native changes remain separately scoped from Claude's legacy replay/performance
work; the ownership update is on issue #4.

Validation at `f448939` on local macOS arm64 with locked PyTorch 2.9.1:

- **480 tests passed, zero skipped**, including both colors, acknowledgement
  arrival variants, invalid/ambiguous closures, version compatibility, separate
  memory, synthetic PPO gradients and exact checkpoint/RNG resumption.
- Four fixed-weight fixture games, two with Red townless and two with Blue
  townless, reached natural complementary outcomes. All four losing final
  consumed actions now receive exactly -1 with null `result_seq`; winners receive
  +1. All eight player trajectories are structurally valid. Each loser retains
  its post-terminal `hero_lost` event and empty-own-hero view before finalization.
- The predeclared ordinary cohort used engine seeds `96111` and `96112` in both
  learner colors, with a 2,048-choice/100-day budget per game. All four reached
  verified day cutoffs: **0 natural outcomes / 4 attempts**, zero unsupported
  interactions, technical failures or collection errors. Completed-only win
  fraction is undefined (zero completed games); cutoffs are not draws. Its 6,274
  choices include all eleven action kinds, 15 constructions and 148 recruitments.
  This finite result does not establish that ordinary natural outcomes are
  reachable under the current untrained policies, or that they are impossible.
- An intentional owned-process kill after four resolved choices left every
  reward and bootstrap absent and every sample ineligible. No technical failure
  was assigned a loss or draw.
- Recomputing all 49,560 saved frames reproduces 6,536 decision values, 6,528
  sampled action probabilities and 11,017 conditional probabilities with zero
  difference. All eighteen native player rollouts reject optimizer admission;
  all eight actual unsampled cutoff decisions supply their own bootstrap values.
- Saved rollout-v1 cutoff, natural and failure artifacts still round-trip with
  identical canonical bytes. Four tampered copies of a new real rollout reject
  swapped policy roles, a forged closure reference, an invented result and a
  missing loss target. These are artifact checks, not native replay tests.
- Exact prepared-engine verification passes; native sources and binaries were
  unchanged. The replay CLI rejects the new contextual recording before replay,
  and both legacy observation ledgers are empty, as the source audit predicted.

All nine native attempts are retained. Compact source/checkpoint/artifact hashes,
the declared roster, inference reconstruction results and gate statuses are in
`validation/contextual-terminal-2026-10-24.json`; raw files remain under ignored
`runs/`. Previous native fog/freshness, construction-price and recruitment-stock
checks remain bounded evidence at their original commits, not fresh exhaustive
verification by this cohort. No gameplay optimizer update, paid compute or merge
occurred.

Reproduce the ordinary cohort with the prepared engine and saved random policy:

```sh
.venv/bin/python -m experiments.learner.inference \
  --purpose ordinary-evaluation --output runs/my-ordinary-cohort \
  --checkpoint runs/neural-fixed-check/untrained-policy.pt \
  --engine-seed 96111 --engine-seed 96112 --policy-seed 96103 \
  --max-requests 2048 --max-days 100
python3 validation/probes/create_boundary_fixture.py \
  --kind natural-townless-blue --output runs/my-blue-terminal-fixture
.venv/bin/python validation/probes/audit_neural_rollout.py \
  --checkpoint runs/my-ordinary-cohort/untrained-policy.pt \
  --episode runs/my-ordinary-cohort/episode-000001 \
  --output runs/my-ordinary-cohort/inference-audit-1.json
```

#### Native terminal barrier and contextual replay implementation

On 25 October, the user authorized Codex to finish terminal drainage and
contextual replay, investigate ordinary-game completion, and publish coherent
commits with bounded local validation. Gameplay optimizer updates, training,
paid compute and merges remain excluded. Claude's duplicate contextual adapter
is preserved on `claude/contextual-engine-parallel`; its shared branch was
restored to the earlier base. Ownership of these contextual changes was
coordinated on issue #4. Claude's legacy replay/performance evidence remains
separate.

The new contextual capture requires an explicit server terminal barrier after
the terminal-producing request, its effects and acknowledgement. The client
validates the request, complementary outcomes and packet count after all
callbacks, flushes its receipt and closes the diagnostic capture before
shutdown. Python requires the matching server/client receipts and final capture
close in addition to clean exit, complete bridge streams and no failure. Missing
or mismatched evidence and late failures remain unscored. This version supports
request-caused natural endings, including authenticated automatic neutral battle
actions; initial and timer-origin endings fail explicitly
until they have a replayable origin. The learner wire and terminal-only reward
contract are unchanged.

Contextual replay uses a fresh diagnostic transaction tape. It first reruns the
recorded server inputs through the actual game handler, then applies the
regenerated packets through the actual client callbacks and contextual adapter.
Recorded policy responses and transaction ordering drive that second pass;
observations, notices, window contents and native adventure requests must be
regenerated exactly. Automatic battle inputs remain separately identified and
bound to the authoritative server input stream. The Python verifier binds every
delivered message and action, reconstructs both persisted player histories and
proofs, and checks the map, source recipe, executable/library identities and
file hashes. A cutoff verifies only its exact captured prefix. Historical
contextual recordings without this tape remain unsupported, and legacy replay
does not acquire a contextual verification label.

Implementation is on `codex/contextual-replay` in
[draft PR #11](https://github.com/tkorsi/heroes-rl/pull/11), stacked on PR #10. The gameplay
sources and native build were frozen at `609083a` for the final validation.
The initial `29139c4` navigation capture replayed correctly, but two terminal
fixtures exposed recursive recorder destruction at process exit. Both crashed
attempts remained unscored, with all 150 samples ineligible. The fix removes
registry access from its owned object's destructor; a real-library regression
checks explicit release and both forms of process teardown. All three development
attempts are retained separately from the final cohort.

Local macOS arm64 validation on 25 October passed:

- **514 tests, zero skips**, including native barrier/lifetime checks and the
  synthetic PPO/checkpoint tests. Six focused inference tests also pass after a
  later report-label-only correction. Exact prepared-engine verification passes.
- **28 final native attempts:** four natural diagnostic endings, twenty declared
  cutoffs and four intentional process failures. These are separate probe and
  evaluation cohorts, not a pooled outcome-rate estimate.
- **23 contextual replays** check all 22,941 captured client packets, 64,995
  transactions, 16,537 messages, 8,232 responses and 4,117 native requests with
  zero unchecked records. A separate legacy cutoff replay also passes. The
  natural recording passes its positive control and rejects all **15 rehashed
  semantic mutations**, including destination, construction, exact recruitment,
  notices, window payloads, recipient/order changes and missing final close.
- Four townless fixtures cover both losing colors and learner seats. Native
  replay reaches the exact regenerated terminal receipt, and losing histories
  retain the final hero-removal event and view. All nine fresh neural attempts
  reproduce 56,430 frames, 7,564 values, 7,556 action probabilities and 12,816
  conditional probabilities with zero difference. All eighteen player rollouts
  reject optimizer admission; the eight ordinary cutoff values come from each
  player's actual next decision. The injected crash has no reward/bootstrap.
- Current replay-verified ordinary histories contain two complete enemy sightings
  between Red decisions that snapshots would miss. A closed Blue town window
  stays absent during a defender battle and shows the changed garrison only
  after explicit reopening. Historical panel values remain earlier observations.
- Construction and exact recruitment pass on both colors, alongside shared
  upgraded stock, full-army rejection/merging, construction affordability and a
  crash after acknowledged construction. The recruitment-affordability probe
  did **not** reach a resource-limited slider within either its ten-day limit or
  a separate extended 400-choice/120-second bound. Both attempts remain valid,
  replay-verified, unscored cutoffs; the missing condition is not claimed passed.

The predeclared ordinary cohort used the unchanged map and saved untrained
checkpoints, engine seeds `97111` and `97112` in both learner colors, policy seed
`97103`, and at most 2,048 choices or 100 days per game. All four reached the day
cutoff: **0/4 natural, 4/4 valid cutoffs**, with zero unsupported-interface,
technical or collection failures. The completed-only win fraction is undefined;
cutoffs are not draws. No seed, policy, map, reward or capability was changed to
obtain an ending.

The permitted histories explain limits of these trials: no town changed
ownership, 5,561 of 7,191 choices (77.3%) were navigation, and all 1,998 recruited
creatures went to town garrisons. Only one move selected a currently observed
opponent town. Both players lost their heroes in the first game while retaining
towns; this interface cannot hire replacement heroes or transfer garrison troops
into a field army. Other trials retained at least one hero. These observations
identify capability and behavior limitations; four cutoffs do not prove that
ordinary natural outcomes are impossible or that every engine defect is absent.

Compact source/checkpoint/artifact hashes, every attempt, audits and remaining
gates are in `validation/contextual-replay-2026-10-25.json`. Raw evidence remains
under ignored `runs/`. Captured collector reports preserve their original
historical `unsupported` string; separate verification reports establish replay
success. Future collector reports say `not_verified_by_collector`. No original
manifest is rewritten to grant itself a verification label.

Reproduce independent verification on a fresh captured episode:

```sh
.venv/bin/python -m experiments.headless.contextual_replay \
  runs/my-contextual-run/episode-000001 --data <game-data-directory> \
  --output runs/my-contextual-replay
.venv/bin/python -m experiments.headless.contextual_replay \
  runs/my-contextual-run/episode-000001 --data <game-data-directory> \
  --output runs/my-contextual-tamper --tamper
```

Remaining limits include initial/timer-origin completion, fresh hero-window
casualty and broader encounter/interruption coverage, the unmet recruitment
resource-limit condition, and ordinary natural outcomes. The authenticated
neutral battle-request origin has source/native-unit coverage, not a new natural
terminal fixture in this cohort. No throughput or cross-platform claim is made.

#### Follow-up: validation gaps and ordinary completion

The 25 October follow-up keeps gameplay sources, native binaries, rewards,
action support, the ordinary map and model weights unchanged. It fixes
diagnostic setup and acceptance checks, adds reproducible behavior auditing,
and separates actual interface defects from valid untrained behavior. Its
ordinary roster was committed before launch in
`validation/contextual-gap-plan-2026-10-25.json`: eight new engine seeds in both
learner seats, the same saved untrained learner and frozen opponent, 100 days
and 2,048 choices per attempt. Two runner batches preserve the predeclared RNG
sequence by offsetting the second batch's policy-seed base. Every attempt is
retained; diagnostic controls never enter this cohort or the learner.

The following bounded real-game checks now pass, with independent native
replay and zero unchecked records:

- **Recruitment affordability:** the former setup exhausted wood/ore before
  spending enough gold; waiting then increased gold. Explicit displayed
  Marketplace and Resource Silo purchases reach stock 14, gold 475, price 60
  and slider maximum 7. Count 8 is rejected unchanged; buying exactly 7 costs
  420 and leaves stock 7, gold 55 and maximum 0. Another positive purchase is
  rejected unchanged. No initial resources or prices are modified.
- **Closed hero windows, both colors:** a declared single-neutral-stack-count
  diagnostic produces a real automatic battle. The same hero survives with
  20 creatures instead of 25. Closed-window fields remain absent, the old
  snapshot remains immutable, and reopening supplies the changed army.
- **Resources, neutral battles and mines, both colors:** each selected object
  has its actual visit, issued/accepted command, visible native effect and
  cessation of further route steps after the event. The initial mine probes
  exposed an invalid test assumption: visibility of the object does not imply
  knowledge of its entrance approach. Those failed diagnostics remain valid,
  unscored, replayed cutoffs. The corrected evaluator makes a separate explicit
  move to an observed frontier before retrying; the product planner and policy
  are unchanged. Fixture comparison also rejects JSON Boolean/numeric aliases.
- **Neutral terminal requests, both colors:** isolated townless/stronger-guard
  inputs preserve ordinary defeat and fog rules. Actual day-two last-hero
  defeats end through authenticated neutral `MakeAction` requests, with exact
  server/client receipts, complete histories and final capture close. Three
  completed recordings replay. The first attempt ended before the other player
  acted and remains failed, unscored and ineligible; its evidence is retained.
- **Unchanged-map capture and victory, both colors:** controlled interface
  tests relocate the defender through its visible controls, capture its town,
  then explicitly end turns. Both games end naturally on day eight while the
  defender retained its hero until the seven-day loss. These are mechanics
  checks, not random-policy results or demonstrations. They establish that
  existing routing, capture and natural victory are reachable without army
  transfer, a new action, or a map/rule change.

`validation/probes/audit_ordinary_behavior.py` reproduces the preceding
four-game behavior counts from manifest-bound player histories alone. It treats
closed hero-window armies as timestamped historical observations. That cohort
had only one town-targeting move among 426 moves made while an opponent town
was visible. Navigation consumed 77.3% of all choices; all 1,998 recruited
creatures entered town garrisons. In one game both heroes died while both towns
survived. Standard defeat requires losing both heroes and towns (or seven days
without a town); the unchanged map forbids taverns. With no hero sources or
moving armies, this state cannot become conquering through the supported
actions. It remains nonterminal, never a synthetic loss or draw. Other games
retain heroes and still make few capture attempts. Sparse terminal feedback is
therefore an observed behavior/capability limitation, not evidence that the
engine cannot end a normal game. Army transfer is outside the current interface
and is not required for the demonstrated capture route.

The new 16-attempt ordinary cohort also finished with **0/16 natural endings
and 16/16 valid day cutoffs**, with no unsupported-interface, technical or
collection failures. Its 26,874 choices include 20,241 navigation actions
(75.3%), 65 constructions and 668 recruitments. All 7,769 recruited creatures
entered town garrisons; no town changed ownership. Only five of 1,949 moves made
while an opponent town was visible selected such a town. There were twelve
hero-loss events and one both-heroes-dead episode. These observed actions and
limits explain the measured sparsity without inventing rewards or concluding
that ordinary victories are impossible. The eight engine seeds are paired
across seats, not sixteen independent seed samples.

`validation/contextual-gaps-2026-10-25.json` records the complete follow-up:
all sixteen ordinary attempts and thirteen mechanics attempts, exact sources,
all retained failed probes, native replay and neural audits. The dated
`validation/assemble_contextual_gaps_2026_10_25.py` is a byte-identical tracked
copy of the evidence assembler. Raw recordings remain ignored. Final local
validation passes 546 tests with zero skips and exact prepared-engine agreement.
Across the separate cohorts, the 29 attempts comprise five diagnostic natural
endings, 23 valid cutoffs and one failed early neutral-terminal attempt. All 28
valid recordings independently replay with zero unchecked records; all sixteen
ordinary neural recordings reproduce their action probabilities and values
exactly. These counts do not pool diagnostics into ordinary outcome frequency.

Initial/timer-origin terminals remain explicitly outside the supported profile.
The actual recorded starts have no human players and timers disabled; the
pinned timer implementation acts only for human players. Source/config tests
and receipt rejection tests cover those exclusions, not hypothetical timer
games. An actual server-rejected movement acknowledgement remains unit-covered
but was not induced by the serialized, prevalidated ordinary action interface;
no live rejection claim is made. Broader dialogs, army editing, other maps,
platforms and throughput remain outside this bounded validation.

The user's later readiness clarification supersedes the earlier proposal to
require four ordinary natural endings before a first update. A zero-outcome
untrained cohort is sparse-feedback evidence, not by itself a correctness
failure or a requirement for competent behavior before learning. The relevant
conditions are correct player-local experience, native action effects and
replay, validated natural/cutoff/failure boundaries, and an explicit reviewed
launch budget. Without terminal events, zero-reward cutoffs can supply only
bootstrap-based targets; they provide no observed victory signal and cannot
establish strategic improvement.

The user subsequently authorized explicit town-garrison-to-visiting-hero
transfer and preparation of a small first training proposal, while excluding
hero hiring, tavern/map/rule/reward changes and automatic loading. That work is
complete within the scope below. Preparation does not authorize execution;
launch requires separate approval of the concrete plan. No gameplay optimizer
update, training campaign, paid compute or GitHub PR merge has been performed.

#### Explicit town-to-visitor troop transfer and first-run preparation

The current implementation on `codex/troop-transfer` adds the missing way to
move recruited troops from a town into its visiting hero's army. It does not
assign a strategic priority to recruitment, transfer, movement or capture.
The policy chooses `transfer_stack`, the source slot, destination slot and exact
positive count. Native rules determine validity; no automatic loading or count
clamping occurs. Forward movement into an empty slot, partial splits and merging
with the same creature are supported. Unlike-creature swaps, reverse transfers,
general army rearrangement and hero hiring remain outside this interface.

This is the opt-in `adventure-armies-v1` profile: protocol 6,
`heroes-player-history-v4` and `heroes-context-actions-v4`. The current town
overview exposes the actual upper army and actual owned visiting hero, with
seven explicit slots each. A nearby or selected hero is not substituted for the
visitor. A garrisoned hero uses the engine's minimum-army rule; a town garrison
may be emptied. The native adapter revalidates ownership, slots, counts,
capacity, overflow and current context under the state lock before issuing
one command. It checks the acknowledged result against both complete armies,
resources and context. All ordinary transfer rewards are zero.

Army rows are absent from closed town windows and other panels. Earlier views
remain timestamped history, never refreshed from hidden current state. The
actor and critic share a random recurrent network, ingest only their player's
ordered visible history, and retain separate per-player recurrent state.
Transfer source, destination and count use typed conditional policy heads with
reconstructible joint probability. Opaque routing references are not numeric
features. Version-two model/policy schemas are explicit; old checkpoints are
not silently promoted. Existing v2/v3 profile behavior remains available.

Local validation with implementation frozen at `d5ef9b0` passed **585 tests,
zero skipped**, and exact prepared-engine agreement. Evidence is recorded in
`validation/troop-transfer-2026-10-25.json`; its byte-identical evidence
assembler is tracked at `validation/assemble_troop_transfer_2026_10_25.py`:

- All ten native attempts are retained: seven valid cutoffs, two controlled
  natural endings and one intentional process failure. Nine independent native
  replays pass with 1,915 client packets, 5,294 transactions, 315 requests and
  zero unchecked records. All 32 rehashed semantic tamper cases are rejected,
  including changed transfer count/source/destination; positive controls pass.
- Both colors execute full moves, exact partial splits, same-creature merges at
  seven-slot capacity, and complete emptying of a town garrison. Counts,
  untouched slots and resources are conserved. Visitor departure/re-entry and
  closed/reopened windows supply fresh permitted views. Invalid choices are
  rejected before sending; no actual server-rejected acknowledgement is claimed.
- Two unchanged-map town-capture controls reach natural day-eight outcomes,
  with verified terminal receipts. The intentional crash after acknowledged
  transfer remains unscored and cannot provide reward, bootstrap or PPO samples.
- The final fixed-weight v4 diagnostic uses both learner colors. Independent
  inference reconstructs all 256 actions, 260 values and 441 conditional
  probabilities exactly. The Blue learner itself samples two successful
  transfers (counts two and one), with matching native effects; these choices
  were not imposed by the mechanics probe. Weights remain unchanged and both
  attempts end at valid limits. The earlier two diagnostic attempts and their
  source-snapshot limitation are retained separately from this clean rerun.
- An unchanged v3 native control replays. The saved old v3 neural checkpoint
  reproduces 61 actions, 61 values and 104 conditional probabilities exactly on
  its archived natural fixture. Every diagnostic rollout remains excluded from
  optimizer admission.

Actual server rejection and live garrisoned-hero/minimum-last-stack/overflow
cases remain outside this local evidence; those paths have source/native-unit
coverage. Initial/timer-origin completion, broader dialogs, other maps and
platforms remain unsupported or unvalidated as stated above. The transfer fix
removes a real action-interface limitation. It does not establish that a random
policy will choose useful transfers or finish ordinary games more frequently.
A future both-heroes-dead/towns-retained state still remains nonterminal under
the unchanged rules and available actions.

The user's separately requested preparation is complete. The tracked
`configs/first-training-v1.json` declares the following small proposed run;
**execution is not authorized and has not occurred**:

| Item | Fixed proposal |
| --- | --- |
| Initialization | Fresh random actor/value seed 99001; one independent frozen opponent seed 99002. No diagnostic weights or demonstrations. |
| Collection | Engine seed 99101 in both learner colors: exactly two training attempts, no replacement of failures. |
| Held-out evaluation | Seeds 99201 and 99202 in both colors before and after the update: four attempts per phase, with paired sampling seeds. |
| Total limit | Ten attempts, at most 256 total choices and 30 days each; one CPU Torch thread and 60 minutes for the entire operation. |
| Optimization | At most one PPO update, one epoch and sixteen optimizer steps; Adam 0.0003, gamma/lambda 1, clip/value clip 0.2, value coefficient 0.5, entropy coefficient 0. |
| Stop conditions | Any unsupported/native/history/replay/probability/identity failure stops the run. Invalid attempts are retained and never replaced or trained on. |
| Checkpointing | Save model, Adam, frozen pool, progress and RNG state; reload into new objects before after-evaluation. Updates interrupted by time/step limits roll back. No automatic process restart or resume CLI. |

`experiments.learner.first_run` only prepares fresh checkpoints and an immutable
plan; it launches neither a game nor an optimizer. The dormant
`experiments.learner.gameplay` entry point requires separate human approval of
that exact plan hash. It binds all implementation files, native runtime, map,
validation artifacts and policy identities. Only its fresh, schedule-bound
training captures can pass its admission gate; it repeats independent native
replay and behavior-probability checks before updating. Ordinary diagnostics,
fixtures, held-out evaluation and frozen-opponent trajectories remain excluded.
The default PPO entry point remains synthetic-only. Executor admission,
transaction rollback and checkpoint continuation have synthetic tests and code
review; an actual gameplay update is deliberately not claimed validated yet.

The local preparation is in `runs/first-training-prepared-2026-10-25/`.
`validation/first-training-preparation-2026-10-25.json` records its exact proposed
command, random checkpoint identity, budget, source commit and SHA-256 plan
`ff22a542eb881b913e34772dd124b34802a36fe55f625e248984d331c23512e6`.
The checkpoint and raw validation evidence remain ignored. Preparation's
read-only identity check passes and authorization remains `not_granted`.
Zero natural outcomes alone does not invalidate correct cutoff experience, but
an update without terminal feedback has no observed victory signal and cannot
be described as demonstrated strategy improvement. No gameplay training, paid
compute or GitHub PR merge occurred.

### Continue in a new chat

Use English and read `AGENTS.md` plus this README first. The public `main`
contains merged self-play PR #3 and offline analysis PR #2, including evidence
validation fixes. Claude's headless implementation is now in draft
[PR #6](https://github.com/tkorsi/heroes-rl/pull/6), initially reviewed locally at
`40e34a3`, then at `3a25545` after its neutral-player fix, and rechecked at
`bbfc955`. Claude addressed the four findings across `c6fb53d` and `4fe0d7c`.
The latter passed independent local review and bounded native validation,
including the additional action/finalization cases found in the first candidate.
PR #6 remains draft and unmerged; this is not complete learner integration.
Check its current status rather than treating this snapshot as live.

Codex has now specified the learner's observation/history and per-player
trajectory interfaces and implemented an isolated recurrent policy/value
component against them. Local review of PR #6 reached three natural endings in
four random self-play attempts. Their original replay failure was fixed in
`5b7c9e9`; Claude's subsequent local validation verifies 20 selected natural
replays. At `4fe0d7c`, consumption is committed at the game-state-locked issuance
boundary, pre-existing failures cannot become eligible cutoffs, last-seen times
follow permitted observations, and rendering requires the verified input bytes.
An incomplete terminal history-v1 projection is explicitly invalid and must be
excluded; retaining those final views for learning still needs a versioned
terminal/supersession contract. The full raw player history is preserved.
Current review evidence is in
`validation/headless-fix-candidates-2026-10-24.json`; earlier reviews remain in
`validation/headless-review-2026-10-24.json` and
`validation/headless-neutral-fix-2026-10-24.json`.
The current handoff is `codex/troop-transfer`, stacked on
`codex/contextual-replay` (draft PR #11). Native contextual replay, the terminal
barrier, v4 explicit troop transfer, the recurrent actor–critic and bounded
preparation are implemented and locally validated as described above. Read
`validation/contextual-gaps-2026-10-25.json`,
`validation/troop-transfer-2026-10-25.json` and
`validation/first-training-preparation-2026-10-25.json` for current evidence and
the exact unexecuted proposal. The user authorized Codex to finish the paused
contextual work; Claude's separately scoped simulation/performance work remains
coordinated on issue #4. Neither diagnostic outcomes nor preparation authorizes
training. The next execution decision requires separate approval of the saved
plan. Do not launch it or start a campaign without that approval.

Earlier v2/v3 validation and the original three truncated rendered self-play
runs remain separate historical evidence. Historical contextual recordings
without the native tape remain unsupported. Verified legacy video binding passes
at `4fe0d7c`; no clean throughput benchmark has been accepted.

## Headless self-play environment (issue #4, in progress)

Status: the legacy `movement-v1` environment has passed bounded real-game,
natural-ending, boundary and replay checks on the local Mac at `4fe0d7c`, with
later regression evidence in `validation/contextual-native-2026-10-24.json`.
The opt-in contextual interface has its own fresh-capture replay evidence above;
historical contextual recordings remain unsupported. Neither interface has an
accepted clean throughput benchmark.
PR #6 remains draft and unmerged; this is environment validation, not training.

`vcmiclient --headless` plays the real VCMI rules with no window, renderer,
audio, frame capture, animation waits or policy delays. Patch
`engine/patches/0012-headless-simulation.patch` makes the upstream headless path
playable. Without it, the client crashed on its first packet. The patch:

- uses a fallback interface mutex when no GUI engine exists;
- always creates the map-change relay, which has no views in headless mode,
  so map notifications and animation waits return at once while every
  authoritative state update still runs;
- turns the main loop into a wait for one shutdown request with an exit status;
- exits 0 only after both ExplorerRL terminals were delivered, and nonzero after
  a packet-handling exception, a lost server connection, a refused server start,
  missing game data, a silent game end or a failure that ExplorerRL reports;
  a failure is never replaced by a later success;
- starts the game from the lobby by waiting on lobby updates with a deadline,
  never by sleeping, and starts no LAN discovery for a client-hosted server;
- calls `packApplied()` on every interface once each server pack is applied;
- makes the server's `PackageApplied` carry the real result, so a rejected move
  or end of turn is reported as rejected (upstream always reported success);
- calls every AI's `finish()` before destruction;
- skips the stdin console;
- accepts `VCMI_EXPERIMENT_DIR` for experiment evidence.

ExplorerRL protocol v3 adds ordered, sanitized player-visible events between
decisions (`visible_update-v2`). After every applied server pack, each
side captures its own team-fog view and emits the difference from its previous
capture as a `visible_update`. A capture holds the visible tiles with their
terrain and blocked/visitable flags (derived only from objects the player can
see there), the objects on them that its callback reports visible, and the
player's own state: hero position, movement, seven resources, hero army, town
positions and mine count. Raw callback arguments are never forwarded. A
disappearance carries no cause or later position, a hidden tile carries no
facts, and a hidden removal or hidden tile change produces nothing. The first
capture is reported against an empty view, so the event stream alone
reconstructs what the player saw at every point. The Python adapter checks
this at every decision snapshot, including tile facts and own state. The
visible-tile list in decision observations only repeats what the events
established, so `EXPLORER_RL_SNAPSHOT_EVERY=N` (`--snapshot-every`) may send it
every N decisions (0: never) for speed; the handshake declares the period, the
own state is still checked at every decision, and events are never thinned.
Event objects stay hidden even on visible tiles, as in the game's own UI, and
within one update handles follow visible facts, never engine-ID order.
`battle_start`/`battle_complete` mark battle boundaries for the side involved,
and `level_up` reports a level-up the server applied without a choice.

Each player's stream is also projected onto the learner's
`heroes-player-history-v1` frames (`experiments/headless/history_v1.py`): a
complete `view` at each visible change, a `decision` with its view and attempt
mask, the player's own `action` and its `result`. Sequence numbers count only
that player's frames, owners are relative to the player and object identities
are the player's opaque handles. Visible events that v1 cannot represent
(battle details, visits, dialogs and their answers) stay in the full record
stream and are counted per player in the manifest; a frame order that v1
forbids marks only the projection invalid for that episode. Each terminal
carries the engine's consumption watermark, `last_action_id`: the last action
ExplorerRL issued to the server while that player's game was in progress. It
is committed at issuance under the game-state lock, where no pack can be
applied, so an action is issued either before the pack that ends the game or
not at all. It records the client's issuance, not the server's verdict: an
action issued while the game-ending pack was still in flight is kept, although
the server then rejects it, because at the terminal the client cannot tell it
from an action that itself ended the game (the server sends the end before
acknowledging that action). A delivered action without a result that the
watermark names stays the final transition, with no invented result
(`actions_without_result`). One it does not name is withdrawn from the frames
(`actions_unconsumed`, `retracted_actions`), and a response the closing socket
refused is never folded (`responses_undelivered`); the validator then rejects
a result or watermark for it. The decision left unacted stays in history as
the last frame, `superseded_decision`, but is never a sample: the terminal
reward belongs to the last action actually taken. v1 has no frame after it, so
if the terminal brings visible changes there, that player's projection is
marked invalid instead of silently losing them; the natural outcome is
classified separately. The frames
decode with the learner's `decode_frame` and `HistoryCursor`; run the check
with `HEROES_LEARNER_PATH=<folder containing learner/> python3 -m unittest
tests.test_headless` while the learner contract is not on `main`.

`experiments/headless/env.py` is the agent-environment cycle:

- `reset(episode, seed)` launches an isolated engine process with its own
  profile directory, ports and seed. It returns the first request.
- `step(value)` answers the current player's request and returns the next
  request or the episode end. A request carries only that player's sanitized
  records and history-v1 frames since its previous request, and no global
  step counter, which would reveal how much the opponent did.
- Engine object IDs stay in the adapter's private routing table. Learner
  records use opaque per-player handles, assigned on the player's own first
  legitimate observation.
- `memory(color)` keeps last-seen facts that never refresh from hidden state.
  A visible object is seen again at every capture and decision, so its
  last-seen day follows them and its capture sequence follows the latest
  visible update; a hidden one keeps those of its last observation.
- Rewards are 0 on ordinary transitions. +1/-1 exists only after both
  terminals, a clean exit and validated engine evidence.
- Day, action and collection horizons are `truncated`. Both players pass a
  day horizon within one round, so play continues until each player reaches
  its own first decision past it; that decision is the player's bootstrap
  boundary (`EpisodeEnd.bootstrap` and `bootstrap_v1_seq`). Requests at and
  after a player's boundary carry `past_horizon=True`: they are answered so
  that the other player can reach its boundary, but belong to no training
  trajectory. A natural ending during that continuation is reported in the
  manifest (`natural_terminal_after_horizon`) and stays an unscored
  truncation. A boundary decision that the terminal superseded before the
  engine took any answer to it is no boundary: no action followed the
  horizon, so that player has no cutoff and, if no other player acted past
  its own, the natural ending is scored. The action horizon is a runaway
  guard and stops at once.
- Only a stop the environment declares can be a truncation. Watchdog, crash,
  protocol violation, engine error, unclean shutdown and invalid engine
  evidence are `failed`, also during a horizon continuation. Events that
  arrived before a declared stop takes effect are folded first: a failure
  among them fails the episode, an engine that had already exited is judged by
  its exit status and terminals, and a game that had already ended finishes
  naturally, also when the engine is still shutting down and when the stop
  was a horizon reached at that very decision. Only what follows the stop is
  attributed to it. After an exit of
  the engine's own, every bridge stream is read to its end first: a stream that
  ends inside a message, fails to decode or is not finished within 5 s is a
  protocol violation even after both terminals. Neither
  truncations nor failures are scored, and a dead hero with a surviving town
  is never a loss.

The workload is the authored map with two independent seeded uniform-random
agents. Run a predetermined series and record its frozen seed schedule:

```bash
python3 tools/prepare_vcmi.py --check
cmake --build vcmi/build --target vcmiclient ExplorerRL BattleAI
python3 experiments/headless/run_headless.py --output runs/headless-smoke --episodes 3
```

`summary.json` records the host, build type, source identity, seed schedule,
per-episode status and termination reason, and per-player counts: accepted,
rejected and moving actions, end turns, town-only turns, dialogs, battles and
days. Totals over both players call the battle count
`battle_participations_red_plus_blue`, because a battle between the two
players counts once for each, and `dialog_choices_sent` counts dialog answers
(such as level-up choices), not decision cycles. The agents sample only from
the mask, so a rejected attempt is one the engine refused despite the mask. It also records step-latency percentiles,
startup and shutdown time, log volume, peak memory and end-to-end resolved
actions per second, which include relaunch time. Steady-state rates divide by
the stepping time between both bridges being ready and both terminals; they
are reported separately and never replace the end-to-end figure.

### Verified replay

Patch `engine/patches/0013-experiment-replay.patch` and the `vcmireplay` tool
(`engine/ExperimentReplay`) make an episode replayable and checkable. With
`VCMI_EXPERIMENT_REPLAY=1`, which the environment sets by default (`--replay
none` turns it off), the engine writes into `<episode>/engine`:

- `replay.jsonl`, schema `heroes-replay-v1`: the VCMI and serialization
  versions, the compiler, standard library, architecture and OS, the seed,
  the no-cheats option and the StartInfo; then every pack the server's game
  handler received, in order, each serialized on its own; after each one,
  whether it was blocked, its result, the number of packs the server sent and
  a running FNV-1a chain over every sent pack (target, type, bytes). After an
  accepted end of turn, a battle end and the game end it adds a digest of the
  server's logical state: the game state, queries, randomizer, battles, hero
  pool, player messages, turn order and turn timers. Statistics are left out
  because they hold wall-clock time, and so are their timestamps inside sent
  packs. Every record is flushed as it is written.
- `observations-p<N>.jsonl`: for each player, the client pack position and
  digest of every capture that produced a visible update and of the capture
  each policy message ended at. The ledger is flushed before each policy
  message is sent.

The server confirms an input to the client before it logs the input as
applied, so before the environment stops an engine it waits (at most 5 s)
until `replay.jsonl` covers the largest client pack position named in the
players' ledgers. The manifest's `replay` entry hashes the evidence files,
names any that are missing, and reports that wait under `tail`.

`vcmireplay` loads the recorded StartInfo and seed into the real game handler,
feeds it the recorded packs and stops at the first difference in an outcome,
the sent-pack count or chain, a boundary, a terminal, a state digest or a
player capture, which it recomputes from the replayed state at the recorded
pack position. `experiments/headless/replay.py` adds the policy side: it folds
the visible updates each player actually received (from the bridge ledger) and
checks every capture and message digest with a Python mirror of the native
digest (`tests/test_native_visible_state.py` pins the two together). A
natural ending must also replay its game-end boundary and both terminals,
whether it was scored or reported after a horizon. The manifest's labels must
be ones the environment writes: a completed episode is a natural terminal
with one winner, one loser and the rewards they earn, a truncated one names a
declared horizon, and only a completed episode names outcomes or rewards. Its
episode number must be the one its directory names, and its seed must be the
integer the replay log recorded. Why a cut-short episode stopped, beyond its
status and termination reason being such a pair, is the environment's own
label: a replayed prefix cannot confirm it.

Every evidence file must still match the hash and size the manifest recorded.
`vcmireplay` accounts for every observation record: a position must be at
least 1 and may not decrease within a player's ledger, and each player's
checked records plus those past the replayed range must equal the records
loaded. An episode is verified only when nothing is past that range and the
checked counts equal the records in each prepared ledger. The received
visible updates and the ledger captures must be numbered without gaps. An
episode the environment stopped may end its log with one input whose applied
record was cut; any other break in the record pairs is an error, and its
received evidence may lack only the engine's captures after the last update it
delivered and at most one message the engine sent as it was stopped.

Randomness audit. The server's game randomizer is seeded from the recorded
seed and is part of the state digest. BattleAI and ExplorerRL run in the
client: their choices reach the server as recorded packs, so the replay does
not depend on the client's random generators. Turn timers must be off. The
standard library's distributions differ between libc++ and libstdc++, so a
replay on another platform is refused unless `--allow-platform-mismatch` is
given. Deferred coverage: spells, multi-target attacks and other code paths
whose order depends on pointer values cannot occur on the pikemen-only map and
are not covered by any verified replay yet.

```bash
cmake --build vcmi/build --target vcmiclient ExplorerRL BattleAI vcmireplay
python3 experiments/headless/replay.py verify runs/headless-smoke/episodes/episode-000001 --data <game data> --trace
python3 experiments/headless/replay.py tamper runs/headless-smoke/episodes/episode-000001 --data <game data>
python3 experiments/headless/replay.py series runs/headless-smoke --data <game data> --count 20 --prefer-natural --tamper 1 --trace 3
```

`verify` writes `<episode>/verification/replay-verification.json`, and with
`--trace` a per-input trace for rendering, which records each battle's tile and
its two players. `tamper` changes the seed, an input byte, a boundary digest
and a player capture digest, removes an input (and renumbers the rest), swaps
two inputs and alters a received visible update; each must be rejected. A
verifier crash on a tampered recording counts as a rejection but is listed
apart from detected divergences, and an episode with fewer than four inputs is
skipped with the reason recorded. `series` writes `replay-series.json` even
when a tamper run fails. All of this material names engine IDs and hidden
state and is never a policy input.

### Offline video

`experiments/headless/render_replay.py` draws a verified trace after the
episode, never during simulation: an omniscient view labelled as
verification-only, and each player's fog view, where tiles seen now are in
full color, tiles seen before are dimmed with that player's last-seen objects,
and never-seen tiles are black. Memory changes only through the player's own
sightings. Battles and boundaries are labelled; a fog view marks only battles
that player fought. Long episodes are thinned to the frame budget, but every
frame with a new sighting or a battle, the game end and the final frame are
kept. It uses only the standard library and ffmpeg (H.264). The verification
report records the SHA-256 of the manifest and trace it verified, and the
renderer refuses inputs whose bytes differ and a report from before the
manifest label checks, so a video never shows a trace, episode number, seed
or winner that the replay did not check.

```bash
python3 experiments/headless/render_replay.py runs/headless-smoke/episodes/episode-000001 --output videos/episode-000001.mp4
```

### Series, benchmark and costs

```bash
# 1,000 predetermined attempts over 8 workers; every attempt is preserved.
python3 experiments/headless/run_series.py --output runs/attempts-1000 --attempts 1000 --workers 8
# 1, 2, 4 and 8 workers, 3 repetitions of 180 s each; end-to-end rates first.
python3 experiments/headless/benchmark.py --output runs/benchmark
# Where the time goes, from a profiled series with the full ledger.
python3 experiments/headless/run_series.py --output runs/profiled --attempts 16 --workers 1 --profile
python3 experiments/headless/cost_breakdown.py runs/profiled/worker-0
```

`tools/validate_headless.sh` runs every real-game check in order (build, tests,
smoke, profile, 1,000-attempt series, replay verification with tamper tests,
videos, benchmark, compact evidence) and resumes after an interruption:
`DATA="$HOME/Library/Application Support/vcmi" tools/validate_headless.sh`.
It stops at once unless `PYTHON` (default `python3`) is Python 3.11 or newer;
macOS's own `/usr/bin/python3` is older. Without `OUT`, its first run pins the output directory in
`runs/validation-current` (remove that file to start a new validation). A stage
that did not finish is moved aside to `<name>.interrupted-<UTC time>` and run
again, and a smoke stage in which any of the four attempts fails or does not
verify, tamper tests included, stops the run before the long stages. The build
stage configures `vcmi/build` when it has no CMake cache, leaving out the
components this project never builds (launcher, map editor, MMAI, Discord,
Nullkiller2 and translations), so it needs neither Qt nor onnxruntime. A
`vcmi/` prepared from an older recipe is never changed, and the stage prints
the one-time commands that keep it aside and prepare again. Each build and test run keeps
its own timestamped log, and `$OUT/build.log` and `$OUT/tests.log` collect
every run, so a resumed run adds to the evidence instead of replacing it; both
are hashed into the evidence. Replay
verification puts natural endings with a battle first and traces episodes until
three verified ones have a trace; a video selection with fewer videos, or
without a natural ending that shows a battle, is recorded as a note in the
evidence.

`series.json` reports natural endings, truncations and failures separately,
wins by color, episode lengths and any missing attempt; its steady-state rates
are given per worker and for all workers. `benchmark.json` is marked complete
only after its last configuration. The benchmark defaults
to the evidence the scoring needs (engine info log and metadata); add
`--ledger full --replay record` to measure with replay evidence too. The
stretch target's timed path includes replay-log writes, so only such a run
judges it; otherwise `stretch_target_1000_per_second_met` is null.

Deferred: persistent workers (each episode launches a new engine process), and
a typed action transport for economy choices.

## Git and local artifacts

This is one Git project. Open its root in SourceTree to see all Python code,
ExplorerRL C++ source, engine patches, dependency locks and documentation.
The public default branch is `main`; development branches use `codex/` names.
The root repository has no submodules.

| Path | Contents |
| --- | --- |
| `experiments/` | Agent policies, authored maps, recording tools and the headless environment |
| `engine/` | Our VCMI integration and reviewable engine patches |
| `dependencies/` | Pinned external dependency metadata |
| `tools/` | Reproducible dependency preparation |
| `tests/` | Protocol, recording, map and shutdown checks |
| `validation/` | Compact experiment evidence without game assets |

Breakout, HotA, installed applications, installation reports and unrelated
learning material are outside this repository's scope. Existing local copies
are ignored and remain available on disk. Only the clean `main` snapshot starts
the public history; earlier local development and editor checkpoint refs are
not published. Publish named branches explicitly, never a mirror of all local
refs.

VCMI is an external dependency pinned in `dependencies/vcmi.json` to upstream
1.7.5 / `6901839ae9760dfcd1d5ff2c4aa4ef6968fee752`. Its generated local checkout is
`vcmi/`, ignored by the root Git repository. Our bridge lives in
`engine/ExplorerRL/`; modifications to upstream files live in
`engine/patches/`. Those patches are the complete reviewable source changes,
not opaque references to commits in a second project. Never keep work only in
the generated checkout. The preparation script verifies it agrees with the
tracked source and patches.

VCMI and its own dependencies retain their upstream licenses (VCMI:
GPL-2.0-or-later). Their source trees are obtained as dependencies, not copied
into our Git history. Original Heroes III data and HotA installations are
user-provided local assets; installers, game data and applications are excluded.
The generated `.vmap` duel maps in this repository contain only our map JSON,
with references to external game assets. See [third-party notices](THIRD_PARTY_NOTICES.md)
for the pinned upstream license and attribution. No repository-wide license for
independent project code has been selected yet.

Raw recordings live under `runs/`; exported videos under `videos/`; generated
analysis reports under `reports/`. All are ignored by Git, along with build
products and proprietary Heroes III game data.
Compact validation manifests belong in `validation/`. Keep this document as
the single source of purpose and decisions; do not maintain competing charters.

## Current status

- VCMI 1.7.5 is installed locally and its source/build are available.
- Experiment 1 now runs our external random agent against Nullkiller Easy in
  the real engine. Three final runs completed naturally on the same native
  build, with matching engine/socket losses and clean exits. Normal fog and
  paired open-map/player recordings were verified for every run.
- The agent left its castle, expanded explored territory, and picked up 500
  gold in policy seed 1. Each run included one automatic battle against the
  opponent. No mine capture or neutral battle was observed in this series.
- Final previews are `videos/experiment-001-final-player-60s.mp4` and
  `videos/experiment-001-final-omniscient-60s.mp4`. Each includes all three final
  episodes (6, 7, 8), in recording order, at 20 seconds per episode. They use
  the same paired samples; whole clips are rescaled in time, not action replays.
  Compact evidence: `validation/random-baseline-2026-10-24.json`.
- Eight total integration attempts are preserved locally. Attempts 1, 2 and 4
  exposed a startup acknowledgement error, a shutdown exception and a shutdown
  queue race, respectively; they remain unscored. Earlier successful attempts
  3 and 5 are retained. The final series uses attempts 6–8 after the queue fix.
- Client/server/ExplorerRL/Nullkiller builds and video visual checks passed for
  the recorded series. After merging self-play and offline analysis, all 84 tests
  and exact dependency-source verification pass. The native concurrency regression
  fails on the old queue and passes after the fix. Self-play now exercises the
  external bridge on both colors, automatic neutral battles and mine capture.
  Nested game dialogs remain unverified live paths. The older bare-duel map
  drafts also remain unvalidated.
- Stage 1's earlier built-in-only smoke checks covered paired capture,
  player-only Blue and omniscient-only capture. Those frame-limited runs are
  unscored; evidence remains in `validation/recording-dual-view-2026-10-24.json`.
- Historical `videos/recorder-demo-60s.mp4` and
  `validation/recording-2026-10-24.json` used Nullkiller's automatic open-map
  cheat (runs 003/004). They validate recording only, **not** play under fog.
- No Heroes learner training has been performed here. The later 1,000-attempt
  headless series below uses uniform-random agents for environment validation.
- At the original prototype milestone, an isolated, randomly initialized shared
  recurrent policy/value component and strict player-history/trajectory validation
  existed without a live adapter, collector, PPO update or checkpoint league.
  The later contextual neural integration above supersedes that implementation status.
  Existing bridge-v2 recordings are not eligible learner histories.
  All 129 tests pass, including 45 synthetic learner checks; exact prepared-engine
  source verification also passes. Compact evidence:
  `validation/learner-prototype-2026-10-24.json`. These checks do not establish
  live event coverage, headless boundaries or reachable self-play endings.
- Claude's offline baseline analyzer is integrated with additional evidence
  validation. All eight local baseline attempts classify as five natural
  endings, two technical failures and one truncation across six provenance
  cohorts; input files remain unchanged. Schema 3 self-play analysis is not yet
  supported by this CLI.
- Draft headless PR #6 was reviewed at `40e34a3` on the local Mac, using its
  existing Release build after exact dependency-source verification and a
  no-op rebuild of all five targets. All 163 PR tests passed with the actual
  learner contract enabled. Four fresh real self-play attempts produced three
  complementary natural endings with clean exits and one unscored day cutoff;
  a separate day-1 cutoff and forced watchdog failure also remained unscored.
  All 6,793 reconstructed player-history frames pass the learner decoder and
  ordering checks. Eleven intermediate enemy-position samples across seven
  decision intervals are absent from both adjacent own decisions. These
  demonstrate intermediate capture without establishing exhaustive coverage.
- At the original `40e34a3`, native replay verified 0/4 episodes, failing on
  neutral-player identity. Review also reproduced a submitted action
  being recorded despite failed delivery, a trailing reader error being ignored
  during finalization, stale last-seen observation times, and altered render
  inputs retaining a verified label. The terminal races were injected checks,
  not failures observed in the four-game smoke. No verified video, full series,
  throughput benchmark or training was run by this review. Evidence and commands:
  `validation/headless-review-2026-10-24.json`. Engine, replay and performance
  fixes remain Claude's workstream in issue #4.
- Claude then fixed neutral-player recording in `5b7c9e9`; `3a25545` adds only
  an ignore rule. Independently inspected local reports now verify 4/4 new smoke
  replays and 20/20 selected natural replays, with all 24 tamper cases detected
  across those sets. His 1,000-attempt series reports 133 natural endings, 867
  day-limit truncations and zero failures. Three offline videos were generated;
  their trace/manifest binding issue was unresolved at that head. The benchmark was still
  incomplete at inspection. The current source-only review checkout ran 164 tests
  with two dependency-related skips; the original build-backed check passed all
  163 tests. `validation/headless-neutral-fix-2026-10-24.json` records this later
  evidence separately. At that head, training was blocked by action consumption
  and stream finalization. Last-seen memory and video evidence binding also
  required fixes; video success is not a condition of learner episode validity.
- Follow-up review at `bbfc955` confirms that none of those four fixes has
  landed. Its only change since `3a25545` correctly labels map-monster battles
  as neutral; the relevant Python implementations are unchanged. Fresh probes
  reproduce all four defects, including the different scoring results when exit
  and reader-error events arrive in opposite orders. The native visibility
  regression (10 scenarios, 200 randomized folds) and eight renderer tests pass;
  those passing checks do not cover the four defects. This recheck used injected
  failures, synthetic histories and real FFmpeg encoding, with no new game series
  or training. Evidence and commands: `validation/headless-four-findings-recheck-2026-10-24.json`.
  Portable reproducers are tracked under `validation/probes/`; their reports
  distinguish a reproduced bug from correct behavior. Exit zero alone is not an
  acceptance verdict.
- Claude's first correction, `c6fb53d`, passes last-seen and render-binding
  checks but exposed three remaining cases within the action/finalization work:
  the consumption watermark preceded native issuance; terminal views after a
  superseded decision were omitted while marked valid; and a pre-existing crash
  or reader error could become an eligible collection cutoff. The last case was
  reproduced with real VCMI after an acknowledged random action, as well as with
  controlled events. These were sent directly to Claude and recorded on PR #6.
- The corrected `4fe0d7c` passes all 183 tests, the native build and exact recipe
  check. A fresh four-game random-agent smoke produced three natural endings,
  one day cutoff and zero failures. All four replays verify, including 11 battle
  boundaries, and all eight tamper cases are detected. The eight player streams
  decode 6,793 frames; 11 intermediate enemy-position samples across seven
  decision intervals are absent from both adjacent decision snapshots. Three
  additional native cases pass: a day cutoff with both players' own bootstrap
  decisions, a post-action process crash, and a crash already completed before
  requesting a collection cutoff. Both crash cases are failed, unscored and have
  no bootstrap. Extracted production-code tests cover terminal-first,
  issuance-first and already-ended player status; reader-error ordering and
  last-seen/terminal-history probes also pass. A verified offline video renders
  from the new trace. The four original findings are addressed within this
  reviewed scope; incomplete terminal projections are excluded, not repaired
  into complete learner histories. No training or merge occurred. The existing
  old-head benchmark overlapped local review work, so this review accepts no
  throughput claim. Evidence: `validation/headless-fix-candidates-2026-10-24.json`.
  Desired-behavior probes are `validation/probes/validate_*.py`; the real-ledger
  auditor is `validation/probes/audit_headless_histories.py`. Their `--help`
  documents the reviewed checkout, learner source and output arguments. Raw
  runs, game data and videos remain ignored and local.

| Policy seed | Engine seed | Final attempt | Actions sent | Distinct hero positions | Explored tiles, of 400 | Result |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 12345 | 7 | 27 | 15 | 92 → 139 | Loss, day 2 |
| 2 | 12346 | 6 | 14 | 11 | 92 → 189 | Loss, day 2 |
| 3 | 12347 | 8 | 10 | 8 | 92 → 121 | Loss, day 2 |

These three runs establish reachable terminal feedback, not a win-rate estimate
or a learning result. The agent's current movement-only action space leaves it
unable to use the town economy as its opponent does. A training comparison must
retain and report that limitation or explicitly change the action interface.

Random self-play gives both sides the same movement-only capabilities. Its
results are recorded separately in `validation/self-play-2026-10-24.json`.
All three runs reached the 180-second wall limit with no policy error and four
valid recordings each; outcomes and rewards remain null. They reached days
19, 23 and 22. Episode 1 included two neutral battles, a town capture and a
hero-versus-hero battle: Blue held both towns after losing its hero, while Red
kept its hero but no town. Episode 2 included a neutral battle for each side.
Episode 3's Red agent captured a mine. No natural self-play ending was observed
in this original rendered series, which did not validate the full two-terminal
shutdown path. The later bounded headless validation is separate evidence.

| Episode | Engine seed | Red / Blue policy seeds | Red / Blue actions sent | Red / Blue explored tiles, of 400 | Result |
| --- | --- | --- | --- | --- | --- |
| 1 | 22345 | 1 / 101 | 104 / 112 | 189 / 332 | Wall limit, unscored |
| 2 | 22346 | 2 / 102 | 97 / 150 | 257 / 298 | Wall limit, unscored |
| 3 | 22347 | 3 / 103 | 119 / 100 | 188 / 155 | Wall limit, unscored |

Each one-minute overview includes all three predetermined episodes, 20 seconds
per episode, without selecting by outcome: `videos/self-play-player-red-60s.mp4`,
`videos/self-play-player-blue-60s.mp4`, and matching
`videos/self-play-omniscient-red-60s.mp4` / `videos/self-play-omniscient-blue-60s.mp4`.
These are uniformly accelerated recordings, not exact action replays.

Policy threads are stopped before snapshots and ledger hashes are frozen.
Native terminal routing is regression-tested in both arrival orders; scoring
requires actual delivery to both Python policies. A pre-existing, unobserved
race can omit final event-list details when an action result overlaps terminal
delivery; final state counters and terminal outcomes are checked separately.
Native worker errors or missing terminal evidence never produce a scored result.

## Run the first agent experiment

Use Python 3.11+, Git, a C++20 compiler, CMake, Ninja and FFmpeg. Install VCMI's
native build dependencies using its
[macOS build guide at the pinned revision](https://github.com/vcmi/vcmi/blob/6901839ae9760dfcd1d5ff2c4aa4ef6968fee752/docs/developers/Building_macOS.md).
Provide your own game data in VCMI's normal user-data directory; `--data` can
select another installation. HotA is not required for this scenario.

Clone this project, prepare its pinned dependency, and enable the bridge:

```sh
git clone https://github.com/tkorsi/heroes-rl.git
cd heroes-rl
python3 tools/prepare_vcmi.py
cmake -S vcmi -B vcmi/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/homebrew -DENABLE_GOLDMASTER=ON \
  -DENABLE_STRICT_COMPILATION=OFF -DENABLE_EXPLORER_RL=ON
cmake --build vcmi/build --target vcmiclient vcmiserver ExplorerRL Nullkiller --parallel 8
python3 experiments/explorer/create_scenario.py
python3 experiments/explorer/run_random.py \
  --map experiments/explorer/maps/random-baseline-v1.vmap \
  --output runs/my-first-agent --episode 1 --seed 12345 --policy-seed 1
```

This records both perspectives by default, with a 240-second wall limit,
40-game-day limit and 2,000-action limit. The default 0.15-second pause before
sending a choice makes actions visible; it changes timing, not action selection.
The manifest records actual AI-module routing, seeds, action metrics, terminal
evidence, and both videos. Use `--player blue` to swap agent/opponent sides.
The summary reel uses the same `video.py montage --view player|omniscient`
commands below and labels these clips **random baseline — no training**.

The shown Homebrew prefix is for Apple Silicon. Adjust it for your dependency
installation. To run the tooling tests without game assets or an engine build:

```sh
python3 -m unittest discover -s tests -v
```

Video integration tests require `ffmpeg` and `ffprobe`; the native shutdown test
requires a C++ compiler and the prepared VCMI sources. Missing prerequisites are
reported as skipped tests, not as gameplay validation. Historical local video
paths in the experiment results are not downloadable assets in this repository.

## Run random self-play

After preparing the dependency and enabling/building ExplorerRL as above:

```sh
python3 experiments/explorer/run_self_play.py \
  --output runs/my-self-play --episode 1 --seed 22345 \
  --red-policy-seed 1 --blue-policy-seed 101
```

Defaults are a 180-second wall limit, 40 game days, 2,000 actions per side,
4 captured samples per second, and a 0.03-second pause per external choice.
The pause controls presentation timing, not choice probabilities. Every output
directory must be new. Each run contains four MP4s, four frame ledgers, two
`agents/<color>/policy.jsonl` files, isolated logs/settings, and `manifest.json`.
The existing fixed-opponent launcher remains available for separate comparisons.

Create matching overview clips from a directory containing only the intended
self-play runs; `--player` selects both the camera and the outcome label:

```sh
python3 experiments/recording/video.py montage runs/my-self-play \
  --output videos/self-play-red-fog.mp4 --seconds 60 --samples 0 \
  --view player --player red
python3 experiments/recording/video.py montage runs/my-self-play \
  --output videos/self-play-blue-fog.mp4 --seconds 60 --samples 0 \
  --view player --player blue
python3 experiments/recording/video.py montage runs/my-self-play \
  --output videos/self-play-open.mp4 --seconds 60 --samples 0 \
  --view omniscient --player red
```

Use `--view omniscient --player blue` for Blue's matching open-map camera.
Self-play defaults to Red when no player is specified. Overlays explicitly label
the run **random self-play — no training**, including unknown outcomes for limits.

## Analyze recorded baseline attempts

The offline CLI reads explicit run directories and writes new reports without
modifying the inputs. It supports schema 2 `random-baseline` manifests; schema 3
self-play and other inputs are explicitly reported as unsupported. List every
attempt in a comparison, including failures, rather than selecting only successes.

```sh
python3 experiments/analysis/analyze_runs.py \
  runs/random-baseline-001 runs/random-baseline-002 runs/random-baseline-003 \
  runs/random-baseline-004 runs/random-baseline-005 runs/random-baseline-006 \
  runs/random-baseline-007 runs/random-baseline-008 \
  --json reports/baseline-analysis.json --markdown reports/baseline-analysis.md
```

Reports separate verified endings, truncations and failures, retain incompatible
provenance cohorts, and expose missing or contradictory evidence. Optional ledger
metrics require matching hashes, player/episode identity and monotonic timestamps.
Observed intervals without a hero do not count as defeat while a town remains.
Existing output paths are refused; omit output flags to print JSON to stdout.

## Recorder commands

Requirements: original game data, the VCMI native build dependencies, Python
3.11+, and `ffmpeg`/`ffprobe` on PATH. No Python package install is needed.

Prepare or verify the pinned VCMI dependency and our tracked modifications:

```sh
python3 tools/prepare_vcmi.py
python3 tools/prepare_vcmi.py --check
```

`--check` reconstructs the expected source from the upstream pin, patches, and
owned bridge and compares every file; the recorder runs this check before
launching. It accepts the existing local prepared checkout, including its
historical local commits, only when the content matches exactly.

When changing engine code, update the canonical patch/source in the root and
the corresponding generated source together, then run `--check`. To regenerate
instead, preserve the existing `vcmi/` directory outside this project and run
preparation again; reconfigure the build afterward. Preparation deliberately
refuses to reset or overwrite a checkout whose content does not match either
the current recipe or pristine upstream.

From this repository, build the client with the recorder changes:

```sh
# Only needed when creating the build directory for the first time:
cmake -S vcmi -B vcmi/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/homebrew -DENABLE_GOLDMASTER=ON \
  -DENABLE_STRICT_COMPILATION=OFF
cmake --build vcmi/build --target vcmiclient vcmiserver Nullkiller --parallel 8
```

Record a bounded game on the installed **Key to Victory** map. Both players use
the built-in Nullkiller on Easy with automatic map reveal disabled. The runner
opens a game window and follows the selected player's heroes; battles resolve
automatically. Every run requires a new directory:

```sh
python3 experiments/recording/record.py \
  --output runs/example-001 --episode 1 --seed 12345 --seconds 90 \
  --view both --player red
```

The runner defaults to 6 captured frames/second and a maximum of 3,600 frames.
`--view both` (default) creates two videos; `--view player` or `--view omniscient`
records just one. `--player red` (default) selects the side whose fog/camera is
used; other standard colors are supported when present on the map. `--map`,
`--fps`, `--max-frames`, `--binary`, and `--data` are configurable.
Settings, saves, logs, frames and a copied map use a fresh run-specific profile.
The original game assets are read through links. A wall/frame limit is reported
as an unscored truncation with unknown outcome; it is not a natural defeat.

The results are `runs/example-001/omniscient.mp4` and
`runs/example-001/player-red.mp4`, plus `manifest.json`, logs, and separate
`frames/omniscient/frames.jsonl` / `frames/player/frames.jsonl` ledgers. Schema v2
identifies each recording's perspective explicitly. Paired samples share frame
IDs, timestamps and game days, captured under one client-state lock. The window
returns to the player's fog view after recording. PNG timestamps preserve actual
elapsed capture time. Individual MP4 playback uses nominal capture FPS; skipped
render intervals therefore shorten idle/slow periods. This is a visual recording, not exact replay.

Create a 60-second overview from evenly spaced recorded episodes:

```sh
python3 experiments/recording/video.py montage runs \
  --output videos/overview-player.mp4 --seconds 60 --samples 12 --view player
```

Use `--view omniscient` for an open-map overview. Player view is the default;
missing player views are skipped explicitly, never replaced by open-map clips.
Legacy schema v1 recordings are available only as omniscient views. Each montage
labels the view and selected player, and retains the no-training label for smoke
runs and random baselines. Built-in recorder smoke runs report Red's outcome;
external-agent runs report the selected agent's outcome. Manifests and montages
always label the outcome owner.

Use `--samples 0` to show every available episode. At 30 output frames/second,
1,000 episodes fit in 60 seconds as 1–2 frames each, so the sampled overview is
usually more useful for understanding decisions. The montage's adjacent
`.manifest.json` records selected and omitted episodes, source hashes, speed
changes, labels, and coverage. Selection never favours winning episodes.

To encode an existing capture separately (for example, after `--no-encode`):

```sh
python3 experiments/recording/video.py encode \
  --frames runs/example-001/frames/player --fps 6 --output videos/player-001.mp4
python3 -m unittest discover -s tests -v
```

Outputs are never overwritten. Keep full recordings locally; share or archive
them explicitly when needed. The smoke launcher remains available for recorder checks; `run_random.py` connects
our external baseline agent. A trained policy and scheduled checkpoint evaluation
remain future work.
