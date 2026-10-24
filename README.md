# Heroes III reinforcement learning experiment

This is the canonical project document. Last agreement: **24 October 2026**.

## Why this project exists

Build an agent that learns adventure-map strategy in Heroes III through
reinforcement learning, and make the development of its behaviour observable.
The original long-term idea was self-play from random initialization. The first
opponent is now a **fixed built-in computer opponent on Easy**, with battles
resolved automatically. Self-play remains a later research direction.

There are two separate tracks:

- **Experiment track — this workspace and chat:** implement the environment,
  record episodes, run controlled experiments, inspect evidence, and iterate.
- **Learning track — a separate chat:** explanations, exercises, and the
  Breakout project. Lessons do not need to block experiment engineering here.

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
- Only player-visible information may enter the learner. A spectator recording
  can show more for human inspection, but must never feed hidden state to it.
- Freeze opponent settings during comparisons. Use Easy (difficulty index 0)
  with an active built-in adventure AI; an idle EmptyAI is not a meaningful
  opponent. Easy is the lowest difficulty, not proof that one AI module is
  empirically the weakest of every available module.
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
action replay is a separate capability and is not implemented yet.

The player view means the map information available to a **side/team**, shared
by its heroes and towns; it is not a single hero's line of sight or a literal
visualization of the AI's internal thoughts. Preserve VCMI's normal fog rules.
Only the map perspective changes; spectator sidebars are still spectator UI.

The recorder disables engine cheats before game-state creation. This also
turns off Nullkiller's automatic map reveal and hidden-object graph. Record
actual initial fog counts and effective AI flags, then the chosen side's visible
tile count per sample. Treat these checks as evidence of configured fog, not a
complete audit of every built-in AI information access. The future learner still
needs its own validated player-visible observation interface.

## First experiment proposal after recording

**H1:** On a small fixed two-player map, an untrained policy selecting among
valid primitive adventure actions can leave its starting area and encounter
meaningful game events before an Easy opponent ends the game. First measure
the random-policy baseline and the rate of natural endings. This tests the
environment and whether terminal feedback is reachable; it is not training.

Then try a small recurrent policy with PPO and terminal win/loss rewards,
keeping the opponent, combat resolver, map rules, and action interface fixed.
Compare against the random baseline and frozen checkpoints. Log exploration,
neutral battles, mine ownership, survival, and wins as **metrics**, not added
rewards. If feedback is too sparse, discuss the smallest next experiment rather
than silently introducing reward shaping.

Before that experiment, build and validate the adventure interface and a map
with the agreed castle/neutral/mine interactions. The existing ExplorerRL draft
supports one hero, eight one-tile moves and end-turn, and explicitly rejects
several dialogs. Its old two-hero maps have **no towns, mines or neutral armies**.
Those drafts must not be represented as the requested full scenario.

## Git and local artifacts

Open **Heroes** in SourceTree. This is one Git project: all our Python code,
ExplorerRL C++ source, engine patches, dependency locks and documentation appear
in its normal diffs. Work is on `codex/experiment-recording`; no remote push is
implied. The root repository has no submodules.

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
with references to external game assets.

Raw recordings live under `runs/`; exported videos under `videos/`. Both are
ignored by Git, along with build products and proprietary Heroes III game data.
Compact validation manifests belong in `validation/`. Keep this document as
the single source of purpose and decisions; do not maintain competing charters.

## Current status

- VCMI 1.7.5 is installed locally and its source/build are available.
- An older, unvalidated ExplorerRL bridge and tiny duel-map generator exist.
- Stage 1 records synchronized open-map and player-fog views. A real Easy
  Nullkiller game produced 120 paired samples and two 20-second MP4s. Red's
  explored area grew from 65 to 221 of 2,592 tiles across days 1–3; both AIs'
  actual map-reveal and object-graph flags were off.
- Native player-only (Blue) and omniscient-only modes also passed bounded runs.
  All runs ended at frame limits with unknown outcomes, never scored defeats.
- Client/server/Nullkiller builds, 20 Python tests, exact dependency-source
  verification, and visual checks of map/minimap fog pass.
- Current previews: `videos/dual-view-005-player.mp4` and
  `videos/dual-view-005-omniscient.mp4`. Compact evidence:
  `validation/recording-dual-view-2026-10-24.json`.
- Historical `videos/recorder-demo-60s.mp4` and
  `validation/recording-2026-10-24.json` used Nullkiller's automatic open-map
  cheat (runs 003/004). They validate recording only, **not** play under fog.
- No Heroes learner training or 1,000-episode run has been performed here.

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
runs. The current test-map terminal detector reports Red's result; manifests and
montages label the outcome owner even when recording another side.

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
them explicitly when needed. The recorder is currently a built-in-AI smoke-run
launcher. Connecting it to the learner and scheduled checkpoint evaluations is
the next integration, after the adventure environment is validated.
