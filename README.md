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
2. Save a timestamped frame ledger, isolated engine logs/settings, and a run
   manifest. Set finite frame and wall-time limits.
3. Encode an individual episode to MP4 and produce a labelled, roughly
   one-minute montage across a set of episodes.
4. Verify the path on a real, bounded built-in-AI game with automatic battles.
   Label this **recorder smoke test**, with no learner and no training claim.

For future training, save episode records and checkpoints for the whole run;
record scheduled evaluation episodes from frozen checkpoints. This avoids
making every training step wait for rendering. A video of a later evaluation
is labelled as such, not presented as the original training episode. Exact
action replay is a separate capability and is not implemented yet.

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
- Stage 1 works: two real bounded games were captured (360 frames each), encoded
  to 60-second H.264 videos, and combined into a labelled 60-second overview.
  Both were stopped at the recording frame limit, with no scored outcome.
- The native build, nine Python tests, and exact dependency-source verification
  pass. Spectator startup, camera, palettes, resource display, and unattended
  shutdown were repaired as part of the real-game validation.
- Local preview: `videos/recorder-demo-60s.mp4`. Compact evidence:
  `validation/recording-2026-10-24.json`. This is built-in-AI recorder validation.
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
cmake --build vcmi/build --target vcmiclient vcmiserver --parallel 8
```

Record a bounded game on the installed **Key to Victory** map. Both players use
the built-in Nullkiller on Easy. The runner opens a game window and follows the
active heroes; battles resolve automatically. Every run requires a new directory:

```sh
python3 experiments/recording/record.py \
  --output runs/example-001 --episode 1 --seed 12345 --seconds 90
```

The runner defaults to 6 captured frames/second and a maximum of 3,600 frames.
`--map`, `--fps`, `--max-frames`, `--binary`, and `--data` are configurable.
Settings, saves, logs, frames and a copied map use a fresh run-specific profile.
The original game assets are read through links. A wall/frame limit is reported
as an unscored truncation with unknown outcome; it is not a natural defeat.

The result is `runs/example-001/episode.mp4`, plus its `manifest.json`, native
`frames/frames.jsonl`, and logs. PNG timestamps preserve actual elapsed capture
time. Individual MP4 playback uses nominal capture FPS; skipped render intervals
therefore shorten idle/slow periods. This is a visual recording, not exact replay.

Create a 60-second overview from evenly spaced recorded episodes:

```sh
python3 experiments/recording/video.py montage runs \
  --output videos/overview.mp4 --seconds 60 --samples 12
```

Use `--samples 0` to show every available episode. At 30 output frames/second,
1,000 episodes fit in 60 seconds as 1–2 frames each, so the sampled overview is
usually more useful for understanding decisions. The montage's adjacent
`.manifest.json` records selected and omitted episodes, source hashes, speed
changes, labels, and coverage. Selection never favours winning episodes.

To encode an existing capture separately (for example, after `--no-encode`):

```sh
python3 experiments/recording/video.py encode \
  --frames runs/example-001/frames --fps 6 --output videos/episode-001.mp4
python3 -m unittest discover -s tests -v
```

Outputs are never overwritten. Keep full recordings locally; share or archive
them explicitly when needed. The recorder is currently a built-in-AI smoke-run
launcher. Connecting it to the learner and scheduled checkpoint evaluations is
the next integration, after the adventure environment is validated.
