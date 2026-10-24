# Heroes RL

An experimental environment for learning Heroes III adventure-map strategy with
reinforcement learning, using the real [VCMI engine](https://github.com/vcmi/vcmi).
It connects an external Python agent to the game and records synchronized
omniscient and player-fog videos.

**Current milestone:** two independent uniform-random agents play against each
other with separate fog, automatic battles, and four synchronized recordings.
The earlier fixed Easy-opponent baseline is retained. No policy training has
been performed yet. Native gameplay has been tested on macOS;
other platforms are not yet validated.

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
action replay is a separate capability and is not implemented yet; it is now
part of the delegated headless workstream below.

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

### Initial learner specification (proposed, not implemented)

The current policy is uniform random over permitted attempts, and no learned
policy, value function or training loop exists. The first candidate is a small
recurrent actor-critic trained with [PPO](https://arxiv.org/abs/1707.06347).
This is a starting hypothesis to test, not an established best algorithm for
Heroes III.

| Component | Initial specification |
| --- | --- |
| Reward | 0 on ordinary transitions; +1 for a verified natural win and -1 for a verified natural loss, from that player's perspective. No exploration, mine, gold, survival or invalid-move rewards/penalties. |
| Observation | Only the player's structured visible map, own heroes/army/towns/resources, attempt mask, and history of its observations, actions and results. No spectator state or opponent memory. |
| Policy `pi(a \| h)` | A map/state encoder and GRU memory feed nine masked action logits: eight adjacent attempts and end-turn. Sample during collection; record the mask and behavior log-probability. The server still validates attempts into unknown terrain. |
| Value `V(h)` | A scalar prediction of expected future terminal return under the current policy and frozen opponent distribution, using the same permitted information and memory. No separate Q-network is required initially. |
| Self-play | Share model weights across colors, with separate recurrent state and trajectories. Start each episode with a learner seat and a frozen opponent sampled from random play or saved policy checkpoints; alternate learner colors. |

Here `h` is the player's observation/action history, summarized by recurrent
memory. Use an undiscounted terminal objective (`gamma = 1`) initially, so
winning sooner is not a separate reward. For games that eventually end in a
win or loss, the ideal value is `P(win | h) - P(loss | h)`; it is a prediction
under the training opponent distribution, not an objective score for the map.
Both policy and value learn from experience. Mines and exploration are logged
as metrics; their strategic usefulness must be learned through eventual returns.
The existing uniform dialog-choice policy and automatic BattleAI remain fixed
subsystems for this first movement learner, with their versions recorded.

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

The three existing self-play runs contain no verified terminal outcomes.
Bootstrapping zero-reward trajectories alone does not supply missing win/loss
information. Before a learning campaign, establish reachable natural endings
and measure their frequency under the headless baseline. Evaluate frozen
checkpoints on held-out seeds, both colors and a fixed opponent set; do not use
win rate against the current copy of itself as the sole measure of progress.

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
The current bridge sends player-filtered snapshots at the agent's decision
points. It does not yet provide a complete history of visible opponent movement
between decisions. An enemy could appear and disappear during the opposing turn
without entering a snapshot; a recurrent network cannot recover an event it
never received. Issue #4 therefore includes sanitized player-visible events
between decisions as part of the environment contract.

Do not pass raw engine callbacks or numeric global object IDs into the learned
representation. Movement callbacks can include a hidden endpoint, and some
removal callbacks disclose hidden events. Filter each field against information
available to that player at that time. Keep engine IDs internal; if identity is
needed, assign opaque per-player episode-local handles only on legitimate first
observation. Do not enumerate hidden objects to populate memory or handles.

These are requirements for the future observation/history adapter. The current
random agents use only the attempt mask for movement; their diagnostic logs are
not learned memory. No learned policy, value or memory has been implemented.

### Continue in a new chat

Use English and read `AGENTS.md` plus this README first. The public `main`
contains merged self-play PR #3 and offline analysis PR #2, including evidence
validation fixes. At the 24 October handoff, Claude's **Build headless VCMI
simulation and replay** task was working on issue #4; no headless PR had been
published. Check its current status rather than treating this snapshot as live.

Codex's next bounded task is to specify the learner's observation/history
encoding and per-player trajectory interface, then implement the smallest
recurrent policy/value component against that contract. Keep engine, replay and
throughput work with Claude. Review staged engine PRs and run real-game checks
locally when available; the cloud checkout has no proprietary game assets.
Do not start a training campaign until headless validation establishes correct
episode boundaries and reachable natural outcomes. Keep unresolved gaps visible:
the current three self-play runs were truncated, schema 3 offline analysis is
pending, and no headless throughput or replay fidelity has been measured here.

## Headless self-play environment (issue #4, in progress)

Status: implemented in source and unit-tested in the cloud with synthetic
fixtures; **not yet validated against the real game**. No real headless
episode, natural ending, throughput or replay has been measured. Those checks
need the Mac's local game data and are tracked in the draft PR for issue #4.

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
`heroes-player-history-v1` frames (`experiments/headless/history_v1.py`):
a complete `view` at each visible change, a `decision` with its view and
attempt mask, the player's own `action` and its `result`. Sequence numbers
count only that player's frames, owners are relative to the player and object
identities are the player's opaque handles. Visible events that v1 cannot
represent (battle details, visits, dialogs and their answers) stay in the full
record stream and are counted per player in the manifest; a frame order that
v1 forbids marks only the projection invalid for that episode. The frames
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
  truncation. The action horizon is a runaway guard and stops at once.
- Only a stop the environment declares can be a truncation. Watchdog, crash,
  protocol violation, engine error, unclean shutdown and invalid engine
  evidence are `failed`, also during a horizon continuation. Neither
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
days. It also records step-latency percentiles, startup and shutdown time,
log volume, peak memory and end-to-end resolved actions per second, which
include relaunch time. Steady-state rates are reported separately and never
replace the end-to-end figure.

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
  packs. The log is flushed at each such boundary.
- `observations-p<N>.jsonl`: for each player, the client pack position and
  digest of every capture that produced a visible update and of the capture
  each policy message ended at.

`vcmireplay` loads the recorded StartInfo and seed into the real game handler,
feeds it the recorded packs and stops at the first difference in an outcome,
the sent-pack count or chain, a boundary, a terminal, a state digest or a
player capture, which it recomputes from the replayed state at the recorded
pack position. `experiments/headless/replay.py` adds the policy side: it folds
the visible updates each player actually received (from the bridge ledger) and
checks every capture and message digest with a Python mirror of the native
digest (`tests/test_native_visible_state.py` pins the two together). A
natural ending must also replay its game-end boundary and both terminals. An
episode the environment stopped replays as a prefix up to the last complete
record; observations after it are counted, not required.

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
`--trace` a per-input trace for rendering. `tamper` changes the seed, an input
byte, a boundary digest and a player capture digest, removes an input (and
renumbers the rest), swaps two inputs and alters a received visible update;
each must be rejected. `series` writes `replay-series.json`. All of this
material names engine IDs and hidden state and is never a policy input.

### Offline video

`experiments/headless/render_replay.py` draws a verified trace after the
episode, never during simulation: an omniscient view labelled as
verification-only, and each player's fog view, where tiles seen now are in
full color, tiles seen before are dimmed with that player's last-seen objects,
and never-seen tiles are black. Memory changes only through the player's own
sightings. Battles and boundaries are labelled. It uses only the standard
library and ffmpeg (H.264).

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

`series.json` reports natural endings, truncations and failures separately,
wins by color, episode lengths and any missing attempt. The benchmark defaults
to the evidence the scoring needs (engine info log and metadata); add
`--ledger full --replay record` to measure with replay evidence too.

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
- No Heroes learner training or 1,000-episode run has been performed here.
- Claude's offline baseline analyzer is integrated with additional evidence
  validation. All eight local baseline attempts classify as five natural
  endings, two technical failures and one truncation across six provenance
  cohorts; input files remain unchanged. Schema 3 self-play analysis is not yet
  supported by this CLI. A headless environment is implemented and unit-tested
  (see below), but no real headless episode, throughput or replay has been
  measured yet; exact replay remains pending in issue #4.

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
in this series, so the full two-terminal shutdown path remains unverified live.

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
