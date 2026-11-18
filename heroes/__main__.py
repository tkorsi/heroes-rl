"""Command line: ``python -m heroes train 1000`` or ``python -m heroes check``."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import Config


def _path(text):
    return Path(text).expanduser().resolve()  # "--state=~/x" reaches us with the tilde unexpanded


def _config(arguments):
    config = Config()
    changes = {name: getattr(arguments, name) for name in ("state", "legacy", "vcmi", "game_data", "game_map",
                                                           "workers", "games_per_update", "backend")
               if getattr(arguments, name, None) is not None}
    if getattr(arguments, "keep_game_files", False):
        changes["keep_game_files"] = True
    return replace(config, **changes)


def _check(config):
    """Report what a training run would start from, without playing or writing anything."""
    from . import checkpoint
    from .train import restore, update_function
    config.engine().check()
    print(f"engine     {config.vcmi}")
    print(f"map        {config.game_map}")
    update_function(config)
    print(f"backend    {config.backend} is available")
    state = restore(config, save=False)
    empty = checkpoint.latest(config.state) is None
    print(f"state      {config.state}" + (f" (empty; starts from {state.origin})" if empty else ""))
    print(f"continues  {state.progress}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="heroes", description="Self-play training for Heroes III on headless VCMI.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("train", "Play games and train on them."),
                       ("check", "Show what a run would start from. Plays and writes nothing.")):
        command = commands.add_parser(name, help=text, description=text)
        if name == "train":
            command.add_argument("games", type=int, help="number of games to play in this run")
            command.add_argument("--fresh", action="store_true",
                                 help="start from a random policy when the state folder is empty")
            command.add_argument("--games-per-update", type=int, help="games per PPO update (default 100)")
            command.add_argument("--workers", type=int, help="games in play at once (default 10)")
            command.add_argument("--keep-game-files", action="store_true", help="keep every game's engine folder")
        command.add_argument("--backend", choices=("mlx", "torch"), help="PPO backend (default mlx)")
        command.add_argument("--state", type=_path, help="folder for checkpoints and the log")
        command.add_argument("--legacy", type=_path, help="folder of the previous pipeline, imported once")
        command.add_argument("--vcmi", type=_path, help="path of the patched vcmiclient")
        command.add_argument("--game-data", type=_path, help="folder holding the original game's Data")
        command.add_argument("--game-map", type=_path, help="map both seats play")
    arguments = parser.parse_args(argv)
    config = _config(arguments)
    if arguments.command == "train" and arguments.games < 1:
        parser.error("games must be positive")
    if config.workers < 1 or config.games_per_update < 1:
        parser.error("--workers and --games-per-update must be at least 1")
    try:
        if arguments.command == "check":
            _check(config)
        else:
            from .train import train
            train(config, arguments.games, fresh=arguments.fresh)
    except FileNotFoundError as error:
        raise SystemExit(str(error))  # a missing runtime path needs no traceback


if __name__ == "__main__":
    main()
