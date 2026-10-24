# Heroes RL

An experimental environment for learning Heroes III adventure-map strategy with
reinforcement learning, using the real [VCMI engine](https://github.com/vcmi/vcmi).
It connects an external Python agent to the game and records synchronized
omniscient and player-fog videos.

**Current milestone:** two independent uniform-random agents play against each
other with separate fog, automatic battles, and four synchronized recordings.
The earlier fixed Easy-opponent baseline is retained. A separate, randomly
initialized recurrent actor-critic prototype and strict learner contracts now
exist; they are not connected to VCMI and have not been trained. Draft headless
PR #6 has separately passed bounded native-game, boundary and replay validation
on macOS at `4fe0d7c`; it remains unmerged. The next interface design is structured
map, hero-window and town-window interaction, specified below but not implemented.
Its agreed movement action selects a destination coordinate for engine routing;
the existing adjacent-step baseline stays separate. Other platforms are not yet
validated.

Repository: [tkorsi/heroes-rl](https://github.com/tkorsi/heroes-rl).
Original game assets are required locally and are not included or downloaded.

This is the canonical project document. Last agreement: **24 October 2026**.

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

**Claude owns the entire headless simulation, performance and offline replay
workstream**, tracked in [issue #4](https://github.com/tkorsi/heroes-rl/issues/4).
Codex owns the learning design, policy/value implementation and integration.
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

The live policy is uniform random over permitted attempts. An untrained policy
and value component is implemented under `experiments/learner/`; no rollout
collector, PPO update, opponent pool or training loop exists. The first candidate is a small
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

**Status:** the constraints in this section, including destination selection and
engine routing, are agreed requirements. The concrete v2 schemas, route execution
details, initial internal-window operations and implementation stages are
proposals, not implemented behavior. This revision changes documentation only
and does not authorize training. Keep the adjacent-step `movement-v1` baseline
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
| Live action interface | One selected owned hero, eight adjacent movement attempts and end-turn; automatic battles. | Explicit hero targeting, destination-coordinate engine routing and map/hero/town context navigation. |
| History and fog | PR #6 records per-player visible events and current map/own-state views; local replay and transient-sighting checks pass. The learner tile schema has terrain but no road field. | Context/field visibility audit, permitted road encoding, window observations and complete terminal/supersession history. |
| Memory and model | An isolated random-initialized shared GRU actor–critic, separate `PlayerSession` state, synthetic gradient checks. | Live contextual encoder/adapter; no window state or learned game behavior exists. |
| Candidate scoring | Generic typed scorer; movement adapter only. Opaque candidate keys are not features. | Typed entity/control references and parameter domains; history/trajectory v1 still accepts only indices 0–8. |
| Hero/town operations | No structured open/close, construction, recruitment, army editing or quantity-choice transport. | Implement and validate each declared operation against VCMI. Calling a generic scorer with a new type does not implement it. |
| Learning | No collector, PPO update, frozen-opponent pool or training loop. | Integrate only after the versioned environment passes its own gates. |

The action-consumption and stream-finalization findings are **resolved in the
reviewed draft head `4fe0d7c`**, not still-open design blockers. Integration must
use that fix or an equivalent revalidated descendant, not an older build. The
remaining history-v1 limitation is separate: final visible changes after a
superseded decision invalidate the projection. A natural game result does not
make that incomplete player trajectory eligible for learning.

#### Proposed observation and history boundary

Introduce a new environment identifier, for example `adventure-windows-v1`, with
`heroes-player-history-v2` and `heroes-context-actions-v2`. These are proposed
names. Store versions, supported operations, map/engine hashes and checkpoint
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

Construction/recruitment panels are a **proposed** refinement of the allowed town
window, not an already agreed blanket permission for extra windows. Audit their
native navigation and information scope before implementation. If the operation
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

**Proposed first execution contract — not yet agreed or implemented:**

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
- The first proposal does not speculate through unexplored terrain. The native
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
targets. Proposed stopping semantics are:

| Situation | Proposed execution and next boundary |
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

The smallest quantity implementation can enumerate the finite legal counts in
the bounded scenario. A later conditional distribution may select an offer and
then quantity; record every support and conditional probability so PPO recovers
the exact joint behavior probability. Do not let the environment fill parameters
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

Proposed rule: an unsupported dialog or malformed/unrepresentable transition
stops collection with a distinct `unsupported_interaction`/technical status,
preserves diagnostic evidence and performs bounded engine shutdown. It is not
a defeat, draw or eligible horizon: no terminal reward, value bootstrap or
training targets. Do not answer it randomly, select a default, or wait forever.
Keep this reason separate from crashes in reporting even though both are
ineligible. A later natural result cannot repair missing interface history.

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
5. **Add internal operations incrementally.** One hero army edit, then audited
   town construction/recruitment and quantity selection. Verify exact costs,
   stock, slot/count changes, rejections and no hidden-mask leakage. Each step
   needs real game evidence, an explicit coverage list and replay verification.
   Exercise unsupported-dialog termination without answering it. Retain failed
   attempts rather than selecting only successful recordings.
6. **Collect bounded random-interface validation, then review integration.**
   Use predetermined seeds, both colors and matching interface versions. Recheck
   reachable natural outcomes, action consumption, complete histories, rewards
   and player-local bootstrap boundaries. Report natural endings, deliberate
   cutoffs, technical failures and unsupported interactions separately. Only
   after review design the collector/frozen-opponent PPO experiment; this task
   starts no training campaign and purchases no compute.

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
The next proposed work is the versioned map/hero/town interface above. Destination
selection with engine routing is agreed; its detailed execution rules remain
proposed. First resolve those rules, the field/context audit and town-panel scope,
then implement navigation, complete histories and destination routing before
adding internal operations. No live contextual
adapter, collector or PPO loop exists. The bounded headless checks establish
reachable natural outcomes and validate the reviewed boundary fixes; they do
not pass the expanded interface's integration gates or authorize training.
The original three rendered self-play runs remain truncated and separate.
Verified video binding now passes at `4fe0d7c`; a clean throughput benchmark
has not been accepted. Do not start a training campaign.

## Git and local artifacts

This is one Git project. Open its root in SourceTree to see all Python code,
ExplorerRL C++ source, engine patches, dependency locks and documentation.
The public default branch is `main`; development branches use `codex/` names.
The root repository has no submodules.

| Path | Contents |
| --- | --- |
| `experiments/` | Agent policies, authored maps and recording tools |
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
- An isolated, randomly initialized shared recurrent policy/value component,
  strict player-history input and per-player trajectory validation now exist.
  It has no live bridge adapter, collector, PPO update or checkpoint league.
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
