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
GROUPED_MLX_PPO = PPOConfig(windows_per_minibatch=16, windows_per_game=8, learning_rate=8.5e-4)


def _artifacts():
    return Path(os.environ.get("HEROES_ARTIFACTS", "~/Projects/Heroes-artifacts")).expanduser()


def _vcmi():
    return Path(os.environ.get("HEROES_VCMI", _artifacts() / "runtime/local-v8-tile-changes-v1/bin/vcmiclient")).expanduser()


def _training_state():
    # Claude keeps the original training directory and its checkpoint history.
    directory = "training-codex" if REPOSITORY.name == "Heroes-codex" else "training"
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

    # What a game is trained on.
    both_seats: bool = False  # train on the opponent seat's experience too: two trajectories per game
    play_during_update: bool = False  # collect the next batch while the update runs, one update behind; torch backend only

    # How the policy learns.
    games_per_update: int | None = None  # one update after all requested games unless a batch size is set
    backend: str = "mlx"  # "mlx" on Apple GPUs, "torch" for the slow reference
    ppo: PPOConfig = PPOConfig()

    def engine(self):
        return Engine(self.vcmi, self.game_data, self.game_map, replay_ledger=self.keep_game_files)
