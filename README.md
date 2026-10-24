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

Open this folder in SourceTree. The experiment code and this document belong to
the parent repository. `vcmi/` is a submodule with its own local commits, also
viewable in SourceTree. Work is on `codex/experiment-recording`. No remote push
is implied. The pre-existing ExplorerRL draft was preserved in its own commit.

Raw recordings live under `runs/`; exported videos under `videos/`. Both are
ignored by Git, along with build products and proprietary Heroes III game data.
Compact validation manifests belong in `validation/`. Keep this document as
the single source of purpose and decisions; do not maintain competing charters.

## Current status

- VCMI 1.7.5 is installed locally and its source/build are available.
- An older, unvalidated ExplorerRL bridge and tiny duel-map generator exist.
- Stage 1 recording implementation and validation are in progress.
- No Heroes learner training or 1,000-episode run has been performed here.

## Recorder commands

Requirements: the existing local VCMI build, original game data, Python 3.11+,
and `ffmpeg`/`ffprobe` on PATH. No Python package install is needed.

From this repository, build the client with the recorder changes:

```sh
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
