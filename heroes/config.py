"""Everything a training run can be told, in one place.

Edit a default here to change it for good; the command line overrides the few
values worth flipping per run. Paths default to the layout on the training Mac
and follow ``HEROES_ARTIFACTS`` and ``HEROES_VCMI`` when those are set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from .engine.env import Limits
from .engine.process import Engine
from .game import guard
from .learner.ppo import PPOConfig

REPOSITORY = Path(__file__).resolve().parents[1]
PUBLISHED_CHECKPOINT = REPOSITORY / "checkpoints" / "update-002025.pt"
# The update by whole games: a step takes all windows of ten trajectories (a game gives one, a 365-day game
# two), so a batch of 100 games is some sixteen steps a pass and a batch of ten games one. It makes one pass;
# given more epochs, a further pass is made only while the first games of it show a mean KL below the target.
# Its time grows with the number of decisions, where the window updates above also grow with the length of a game.
#
# Adam's first moment is not used (``momentum`` 0): each step follows its own ten trajectories. A checkpoint
# keeps the moment of the update that made it, and with the usual 0.9 the few steps of this update spent three
# quarters of their movement on that kept moment, a direction no game of the batch had a part in. The second
# moment a checkpoint keeps fits the steps that made it as well: a run that changes the update's settings
# starts a new optimizer (``Config.new_optimizer``), and the rate here is for an optimizer of this update's
# own: one pass then moves the policy by a KL of about 0.001 on games it was not trained on.
#
# The value loss trains the value head alone (``value_trunk`` 0), so that nothing the policy reads is moved
# for the value's sake. Journal, 2026-11-22.
WHOLE_GAMES_MLX_PPO = PPOConfig(games_per_minibatch=10, epochs=1, learning_rate=1e-3, target_kl=0.01,
                                value_trunk=0.0, momentum=0.0)


def _artifacts():
    return Path(os.environ.get("HEROES_ARTIFACTS", "~/Projects/Heroes-artifacts")).expanduser()


def _vcmi():
    return Path(os.environ.get("HEROES_VCMI", _artifacts() / "runtime/local-v8-quiet-captures-v1/bin/vcmiclient")).expanduser()


def _training_state():
    directory = "training-codex-2025-fast" if "Heroes-codex" in REPOSITORY.parts else "training-2025-fast"
    return _artifacts() / directory


@dataclass(frozen=True)
class Config:
    # Where things are.
    vcmi: Path = field(default_factory=_vcmi)  # the patched vcmiclient, built by tools/prepare_vcmi.py
    game_data: Path = field(default_factory=lambda: _artifacts() / "game-data")  # holds the original game's Data
    game_map: Path = REPOSITORY / "maps" / "hero-management-v4.vmap"
    state: Path = field(default_factory=_training_state)  # checkpoints, log, running games
    legacy: Path = field(default_factory=lambda: _artifacts() / "selfplay-ten-then-check")  # imported once

    # How games are played.
    workers: int = 14  # games in play at once, one process each; 16 cores carry no more
    limits: Limits = Limits()
    guard: str | None = guard.VERSION  # repeated no-progress actions the policy may not sample; None for none
    keep_game_files: bool = False  # keep every game's engine folder instead of deleting it
    # Practice games: a file listing start positions (heroes/starts.py). Every game then begins from one of
    # them, the training map with the heroes elsewhere, instead of from the training map as it is.
    starts: Path | None = None

    # What a game is trained on.
    both_seats: bool = False  # train on the opponent seat's experience too: two trajectories per game
    play_during_update: bool = False  # collect the next batch while the update runs, one update behind; torch backend only
    # Start this run's updates with a new Adam optimizer instead of the checkpoint's. A checkpoint keeps the
    # moments of the update that made it; they fit that update's steps (its minibatch, its rate) and no other's.
    new_optimizer: bool = False

    # How the policy learns.
    games_per_update: int | None = None  # one update after all requested games unless a batch size is set
    backend: str = "mlx"  # "mlx" on Apple GPUs, "torch" for the slow reference
    ppo: PPOConfig = PPOConfig()
    initial_checkpoint: Path | None = None  # use when the chosen state has no saved checkpoint

    def engine(self):
        return Engine(self.vcmi, self.game_data, self.game_map, replay_ledger=self.keep_game_files)
