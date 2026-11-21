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
    for name in ("keep_game_files", "both_seats", "play_during_update"):
        if getattr(arguments, name, False):
            changes[name] = True
    if getattr(arguments, "stopped_game_reward", None) is not None:
        changes["stopped_game_reward"] = arguments.stopped_game_reward
    budgets = {name: getattr(arguments, name, None) for name in ("stack_moves", "windows", "decisions")}
    if any(value is not None for value in budgets.values()):
        from .game import guard
        changes["guard"] = guard.version(**budgets)
    ppo = {name: getattr(arguments, name) for name in ("windows_per_minibatch", "windows_per_game", "learning_rate",
                                                       "epochs", "max_kl")
           if getattr(arguments, name, None) is not None}
    if ppo:
        changes["ppo"] = replace(config.ppo, **ppo)
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
            command.add_argument("--workers", type=int, help="games in play at once (default 14)")
            command.add_argument("--keep-game-files", action="store_true", help="keep every game's engine folder")
            command.add_argument("--both-seats", action="store_true",
                                 help="train on both seats of every game: two trajectories per game")
            command.add_argument("--play-during-update", action="store_true",
                                 help="play the next batch while the update runs; its first games use the weights "
                                      "from before that update (with --backend torch only)")
            command.add_argument("--stopped-game-reward", type=int, choices=(-1, 0),
                                 help="what a game stopped at the day limit is worth to a seat: -1 a loss for both "
                                      "seats, 0 a draw (default: scored with the policy's own value estimate)")
            command.add_argument("--stack-moves", type=int,
                                 help="stack rearrangements a hero or town may do per game day (default 32)")
            command.add_argument("--windows", type=int,
                                 help="windows a hero or town may open per game day (default: no daily budget)")
            command.add_argument("--decisions", type=int,
                                 help="decisions a seat may take per game day before it can only close windows "
                                      "and end its turn (default: no budget)")
            command.add_argument("--windows-per-minibatch", type=int,
                                 help="loss windows of 16 decisions per optimizer step (default 2)")
            command.add_argument("--windows-per-game", type=int,
                                 help="windows of a minibatch drawn from the same game, sharing one replay of it "
                                      "(default 1; must divide --windows-per-minibatch)")
            command.add_argument("--learning-rate", type=float,
                                 help="Adam step size (default 3e-4, for 2 windows per optimizer step; "
                                      "8.5e-4 goes with 16)")
            command.add_argument("--epochs", type=int, help="passes of an update over its games (default 1)")
            command.add_argument("--max-kl", type=float,
                                 help="stop an update once its latest steps find the policy this far from the one "
                                      "that played the games, for example 0.02 (default: never)")
        command.add_argument("--backend", choices=("mlx", "torch"), help="PPO backend (default mlx)")
        command.add_argument("--state", type=_path, help="folder for checkpoints and the log")
        command.add_argument("--legacy", type=_path, help="folder of the previous pipeline, imported once")
        command.add_argument("--vcmi", type=_path, help="path of the patched vcmiclient")
        command.add_argument("--game-data", type=_path, help="folder holding the original game's Data")
        command.add_argument("--game-map", type=_path, help="map both seats play")
    arguments = parser.parse_args(argv)
    try:
        config = _config(arguments)
    except ValueError:
        parser.error("--windows-per-minibatch, --windows-per-game and --epochs must be at least 1, "
                     "--windows-per-game must divide --windows-per-minibatch, --learning-rate and --max-kl must "
                     "be positive, and the daily budgets --stack-moves, --windows and --decisions at least 1")
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
