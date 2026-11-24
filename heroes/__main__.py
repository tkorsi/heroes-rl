"""Command line: ``python -m heroes train 1000`` or ``python -m heroes check``."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import Config, PUBLISHED_CHECKPOINT, WHOLE_GAMES_MLX_PPO


def _path(text):
    return Path(text).expanduser().resolve()  # "--state=~/x" reaches us with the tilde unexpanded


def _config(arguments):
    config = Config()
    changes = {name: getattr(arguments, name) for name in ("state", "legacy", "vcmi", "game_data", "game_map",
                                                           "workers", "games_per_update", "backend", "starts")
               if getattr(arguments, name, None) is not None}
    if "state" not in changes and "legacy" not in changes:
        changes["initial_checkpoint"] = PUBLISHED_CHECKPOINT
    if getattr(arguments, "max_days", None) is not None:
        changes["limits"] = replace(config.limits, max_days=arguments.max_days)
    for name in ("keep_game_files", "both_seats", "play_during_update", "new_optimizer"):
        if getattr(arguments, name, False):
            changes[name] = True
    budgets = {name: getattr(arguments, name, None) for name in ("stack_moves", "windows", "decisions")}
    if any(value is not None for value in budgets.values()):
        from .game import guard
        changes["guard"] = guard.version(**budgets)
    ppo = {name: getattr(arguments, name) for name in ("windows_per_minibatch", "windows_per_game", "learning_rate",
                                                         "games_per_minibatch", "epochs", "target_kl", "value_trunk", "momentum",
                                                         "gae_lambda")
           if getattr(arguments, name, None) is not None}
    windows = any(name.startswith("windows_") for name in ppo)
    if windows and any(name in ppo for name in ("games_per_minibatch", "target_kl")):
        raise ValueError("An update takes windows or whole games per step, not both")
    defaults = config.ppo
    if "games_per_minibatch" in ppo or (changes.get("backend", config.backend) == "mlx" and not windows):
        # The MLX update is by whole games unless window counts are given.
        defaults = WHOLE_GAMES_MLX_PPO
    changes["ppo"] = replace(defaults, **ppo)
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
    source = config.initial_checkpoint or state.origin
    print(f"state      {config.state}" + (f" (empty; starts from {source})" if empty else ""))
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
            command.add_argument("--games-per-update", type=int, help="games per PPO update (default: all requested games)")
            command.add_argument("--workers", type=int, help="games in play at once (default 14)")
            command.add_argument("--keep-game-files", action="store_true", help="keep every game's engine folder")
            command.add_argument("--both-seats", action="store_true",
                                 help="train on both seats of every game: two trajectories per game")
            command.add_argument("--play-during-update", action="store_true",
                                 help="play the next batch while the update runs; its first games use the weights "
                                      "from before that update (with --backend torch only)")
            command.add_argument("--stack-moves", type=int,
                                 help="stack rearrangements a hero or town may do per game day (default 32)")
            command.add_argument("--windows", type=int,
                                 help="windows a hero or town may open per game day (default: no daily budget)")
            command.add_argument("--decisions", type=int,
                                 help="decisions a seat may take per game day before it can only close windows "
                                      "and end its turn (default: no budget)")
            command.add_argument("--games-per-minibatch", type=int,
                                 help="trajectories per optimizer step, 0 for every one of the batch (default 10): "
                                      "a step uses all loss windows of its games; a game gives one trajectory, "
                                      "a 365-day game two")
            command.add_argument("--epochs", type=int,
                                 help="passes of the update over the batch (default 1)")
            command.add_argument("--target-kl", type=float,
                                 help="by whole games with more than one pass: make no further pass once the first "
                                      "games of it show a mean KL above this from the policy that played them "
                                      "(default 0.01)")
            command.add_argument("--learning-rate", type=float,
                                 help="Adam step size (default 1e-3 by whole games, 3e-4 by windows)")
            command.add_argument("--value-trunk", type=float,
                                 help="share of the value loss's gradient that reaches the network the policy "
                                      "reads, 0 to 1 (default 0 by whole games: the value loss trains the value "
                                      "head alone; 1 by windows)")
            command.add_argument("--new-optimizer", action="store_true",
                                 help="start this run's updates with a new Adam optimizer, not the checkpoint's: "
                                      "for the first run after the update's settings changed (later runs "
                                      "continue the optimizer the checkpoint then holds)")
            command.add_argument("--momentum", type=float,
                                 help="Adam's first-moment coefficient, 0 to below 1 (default 0 by whole games: a "
                                      "step follows its own minibatch and nothing kept from an earlier update; "
                                      "0.9 by windows)")
            command.add_argument("--gae-lambda", type=float,
                                 help="how far back an advantage reaches, 0 to 1 (default 1): with 1 it is every "
                                      "later reward less the value the network gave at the decision; below 1 a "
                                      "later reward, or a later change of the network's value, counts by that "
                                      "factor per decision between")
            command.add_argument("--windows-per-minibatch", type=int,
                                 help="instead of whole games: loss windows of 16 decisions per optimizer step, "
                                      "each replayed from the start of its game (the earlier update; default 2)")
            command.add_argument("--windows-per-game", type=int,
                                 help="with --windows-per-minibatch: windows drawn from the same game per "
                                      "minibatch (default 1; must divide --windows-per-minibatch)")
            command.add_argument("--starts", type=_path,
                                 help="practice games: a file listing start positions (heroes/starts.py); every "
                                      "game begins from one of them instead of from the training map as it is")
            command.add_argument("--max-days", type=int,
                                 help="stop a game at its first decision after this game day (default 365, where a "
                                      "stopped game is a draw that trains both seats with -1; a game stopped "
                                      "earlier is cut off, and the value the network gave at the next decision "
                                      "stands in for the rest of it)")
            command.add_argument("--seed", type=int,
                                 help="for measurements: draw the games' seeds from this number, so that two runs "
                                      "play the same games")
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
                     "--games-per-minibatch at least 0, --windows-per-game must divide --windows-per-minibatch, "
                     "--learning-rate and --target-kl must be positive, --value-trunk and --gae-lambda from 0 to 1, "
                     "--momentum from 0 to below 1, the window "
                     "counts do not go with --games-per-minibatch or --target-kl, and the daily budgets "
                     "--stack-moves, --windows and --decisions must be at least 1")
    if arguments.command == "train" and arguments.games < 1:
        parser.error("games must be positive")
    if config.workers < 1 or (config.games_per_update is not None and config.games_per_update < 1):
        parser.error("--workers and --games-per-update must be at least 1")
    if config.limits.max_days < 1:
        parser.error("--max-days must be at least 1")
    try:
        if arguments.command == "check":
            _check(config)
        else:
            from .train import train
            train(config, arguments.games, fresh=arguments.fresh, seed=arguments.seed)
    except FileNotFoundError as error:
        raise SystemExit(str(error))  # a missing runtime path needs no traceback


if __name__ == "__main__":
    main()
