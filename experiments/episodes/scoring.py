"""Episode identity, effective-state validation and scoring shared by recorded and headless runs.

Nothing here reads frames or videos. A recorded run additionally validates its
recordings; a headless run is scored from the same engine and socket evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

PLAYER_COLORS = ("red", "blue", "tan", "green", "orange", "purple", "teal", "pink")
SELF_PLAY_COLORS = ("red", "blue")
# Limits the recorded path reports as "timeout"; they never produce a reward.
RECORDING_LIMITS = {"wall_time_limit", "frame_limit", "agent_action_limit", "game_day_limit"}
# Explicit horizons of a headless episode, including the end of a declared
# collection or benchmark window. Everything else that stops an unfinished
# game (watchdog, crash, protocol error) is a technical failure.
HEADLESS_TRUNCATIONS = {"game_day_limit", "agent_action_limit", "collection_window"}
# Root-owned sources that define the prepared engine.
ENGINE_RECIPE = ("dependencies/vcmi.json", "engine/patches", "engine/ExplorerRL", "engine/ExperimentReplay")


def digest(path: Path) -> str:
    # Also loaded by file path (heroes_episode_scoring); without the repository root on
    # sys.path, hash directly as before.
    try:
        from experiments.headless.digest_scope import sha256_file
    except ImportError:
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    return sha256_file(path)[0]


def git_state(path: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def engine_recipe_digests(root: Path) -> dict:
    """Hashes of every root-owned input that defines the prepared engine source.

    Only files git counts as source (tracked, or untracked and not ignored) are
    hashed, as they are in the working tree, so an ignored .DS_Store never
    changes the recipe. Without git, hidden files are skipped instead.
    """
    try:
        listed = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "-co", "--exclude-standard", "--",
                                 *ENGINE_RECIPE], capture_output=True, check=True).stdout
        paths = [root / name for name in listed.decode().split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        paths = [path for name in ENGINE_RECIPE for path in [root / name, *(root / name).rglob("*")]
                 if not any(part.startswith(".") for part in path.relative_to(root).parts)]
    return {str(path.relative_to(root)): digest(path) for path in sorted(paths) if path.is_file()}


def classify_run(returncode: int, reason: str | None, log: str, outcome_player: int = 0) -> tuple[str, str | None, str]:
    player_label = "Red player" if outcome_player == 0 else f"Experiment player {outcome_player}"
    outcomes = set(re.findall(re.escape(player_label) + r" (won|lost)\. Ending game\.", log))
    outcome = ({"won": "win", "lost": "loss"}[next(iter(outcomes))]
               if len(outcomes) == 1 else None)
    if reason:
        # Even a delayed terminal log cannot turn an interrupted run into a scored game.
        return ("timeout" if reason in RECORDING_LIMITS else "failed", None, reason)
    if returncode == 0 and outcome:
        return "completed", outcome, "natural_terminal"
    return "failed", None, "process_exit_without_verified_terminal"


def classify_agent_run(returncode: int, reason: str | None, log: str, player_index: int,
                       execution: dict) -> tuple[str, str | None, str, int | None]:
    """Score only agreeing natural results after the external agent has acted."""
    status, outcome, termination = classify_run(returncode, reason, log, player_index)
    terminal = execution["terminal"]
    if execution["error"]:
        status, outcome, termination = "failed", None, "policy_error"
    elif not execution["ready"] or execution["actions_sent"] == 0:
        status, outcome, termination = "failed", None, "agent_never_acted"
    elif status == "completed" and (not terminal or terminal["outcome"] != outcome):
        status, outcome, termination = "failed", None, "unverified_agent_terminal"
    reward = (1 if outcome == "win" else -1) if status == "completed" else None
    return status, outcome, termination, reward


def _unscored_self_play(status: str, termination: str) -> tuple[str, dict, str, dict]:
    return status, {color: None for color in SELF_PLAY_COLORS}, termination, {color: None for color in SELF_PLAY_COLORS}


def _policy_failure(log: str, executions: dict[str, dict]) -> bool:
    return (any(state.get("error") for state in executions.values())
            or "ExplorerRL: " in log or "ExplorerRL terminal delivery failed" in log)


def _verified_self_play_terminals(log: str, executions: dict[str, dict]) -> tuple[dict | None, str | None]:
    """Both sides' engine terminal logs and socket terminals must agree and complement."""
    if any(not state.get("ready") or state.get("actions_sent", 0) < 1 for state in executions.values()):
        return None, "agent_never_acted"
    if log.count("Experiment self-play complete. Ending game.") != 1:
        return None, "unverified_self_play_terminal"
    outcomes = {}
    for index, color in enumerate(SELF_PLAY_COLORS):
        matches = re.findall(rf"Experiment player {index} (won|lost)\. Terminal received\.", log)
        terminal = executions[color].get("terminal")
        if len(matches) != 1 or not terminal:
            return None, "unverified_agent_terminal"
        outcome = {"won": "win", "lost": "loss"}[matches[0]]
        if terminal.get("outcome") != outcome:
            return None, "unverified_agent_terminal"
        outcomes[color] = outcome
    if set(outcomes.values()) != {"win", "loss"}:
        return None, "noncomplementary_agent_terminals"
    return outcomes, None


def classify_self_play_run(returncode: int, reason: str | None, log: str,
                           executions: dict[str, dict]) -> tuple[str, dict, str, dict]:
    """Require two independent, complementary terminal deliveries before scoring."""
    if set(executions) != set(SELF_PLAY_COLORS):
        return _unscored_self_play("failed", "missing_agent_evidence")
    if _policy_failure(log, executions):
        return _unscored_self_play("failed", "policy_error")
    limits = {state.get("limit") for state in executions.values()} - {None}
    if reason or limits:
        termination = reason or sorted(limits)[0]
        return _unscored_self_play("timeout" if termination in RECORDING_LIMITS else "failed", termination)
    if returncode != 0:
        return _unscored_self_play("failed", "process_exit_without_verified_terminal")
    outcomes, problem = _verified_self_play_terminals(log, executions)
    if problem:
        return _unscored_self_play("failed", problem)
    return "completed", outcomes, "natural_terminal", {color: 1 if outcome == "win" else -1 for color, outcome in outcomes.items()}


def classify_headless_self_play(returncode: int | None, reason: str | None, log: str,
                                executions: dict[str, dict]) -> tuple[str, dict, str, dict]:
    """Score a headless episode without any recording evidence.

    Status is ``completed`` (natural, verified, rewarded), ``truncated`` (an
    explicit day/action horizon, unscored) or ``failed`` (watchdog, crash,
    protocol or evidence failure, unscored). Hero death never implies a loss:
    only the engine's own terminal packets, delivered to both policies, count.
    """
    if set(executions) != set(SELF_PLAY_COLORS):
        return _unscored_self_play("failed", "missing_agent_evidence")
    if _policy_failure(log, executions):
        return _unscored_self_play("failed", "policy_error")
    # Only a stop the environment itself declared can be a truncation; an
    # engine that exits on its own is judged by its exit status and terminals.
    if reason:
        return _unscored_self_play("truncated" if reason in HEADLESS_TRUNCATIONS else "failed", reason)
    if returncode != 0:
        delivered = all(state.get("terminal") for state in executions.values())
        return _unscored_self_play("failed", "unclean_exit_after_terminals" if delivered
                                   else "process_exit_without_verified_terminal")
    outcomes, problem = _verified_self_play_terminals(log, executions)
    if problem:
        return _unscored_self_play("failed", problem)
    return "completed", outcomes, "natural_terminal", {color: 1 if outcome == "win" else -1 for color, outcome in outcomes.items()}


def validate_experiment_metadata(metadata: dict, player_index: int, log: str, agent_player: int | None = None,
                                 agent_players: list[int] | None = None) -> dict:
    """Verify effective engine state and actual AI flags, independently of requested settings."""
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Missing or unsupported effective experiment metadata")
    if metadata.get("cheats_allowed") is not False or metadata.get("requested_no_cheats") is not True:
        raise ValueError("Actual engine cheats must be disabled with an explicit no-cheats request")
    if type(metadata.get("difficulty_index")) is not int or metadata["difficulty_index"] != 0:
        raise ValueError("Actual engine difficulty must be Easy (index 0)")
    if metadata.get("phase") != "before_first_turn" or type(metadata.get("game_day")) is not int or metadata["game_day"] < 0:
        raise ValueError("Missing initial game-state phase/day evidence")
    sizes = [metadata.get(key) for key in ("map_width", "map_height", "map_levels")]
    if any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError("Invalid effective map dimensions")
    total_tiles = sizes[0] * sizes[1] * sizes[2]
    players = metadata.get("players")
    if not isinstance(players, list) or not players:
        raise ValueError("Missing initial player visibility evidence")
    indexed = {}
    for player in players:
        if not isinstance(player, dict):
            raise ValueError("Invalid initial player evidence")
        index = player.get("player_index")
        visible = player.get("visible_tiles")
        if type(index) is not int or not 0 <= index < len(PLAYER_COLORS) or index in indexed:
            raise ValueError("Invalid or duplicate initial player index")
        if type(player.get("human")) is not bool or type(player.get("team_index")) is not int or player["team_index"] < 0:
            raise ValueError("Missing initial player identity")
        if type(visible) is not int or not 0 <= visible <= total_tiles or player.get("total_tiles") != total_tiles:
            raise ValueError("Invalid initial player visibility count")
        indexed[index] = player
    if player_index not in indexed or indexed[player_index]["human"]:
        raise ValueError("Requested observer player is absent or is not controlled by AI")
    ai_players = {index for index, player in indexed.items() if not player["human"]}
    if agent_players is not None:
        if agent_player is not None or agent_players != [0, 1] or metadata.get("requested_agent_players") != agent_players:
            raise ValueError("Actual engine did not route both requested self-play sides")
        if set(indexed) != set(agent_players) or ai_players != set(agent_players) or len({p["team_index"] for p in indexed.values()}) != 2:
            raise ValueError("Self-play requires exactly two opposing AI players")
        if any(indexed[i].get("adventure_ai") != "ExplorerRL" for i in agent_players):
            raise ValueError("Both self-play sides must actually use ExplorerRL")
        ai_players = set()
    elif agent_player is not None:
        if metadata.get("requested_agent_player") != agent_player or agent_player not in ai_players:
            raise ValueError("Actual engine did not route the requested agent side")
        if indexed[agent_player].get("adventure_ai") != "ExplorerRL":
            raise ValueError("Agent side is not actually controlled by ExplorerRL")
        if any(indexed[i].get("adventure_ai") != "Nullkiller" for i in ai_players - {agent_player}):
            raise ValueError("Opponent is not the fixed Nullkiller AI")
        if len(indexed) != 2 or len(ai_players) != 2 or len({p["team_index"] for p in indexed.values()}) != 2:
            raise ValueError("First random-agent experiment requires exactly two opposing AI players")
        ai_players = ai_players - {agent_player}
    ai_evidence = {}
    pattern = r"Experiment AI visibility: player=(\d+) cheats_allowed=(\d+) open_map=(\d+) use_object_graph=(\d+)"
    for values in re.findall(pattern, log):
        index, cheats, open_map, graph = map(int, values)
        if index not in ai_players or cheats or open_map or graph:
            raise ValueError("Actual AI visibility flags permit cheats/open-map access or name an unexpected player")
        ai_evidence[index] = {"player_index": index, "cheats_allowed": False,
                              "open_map": False, "use_object_graph": False}
    if set(ai_evidence) != ai_players:
        raise ValueError("Missing actual Nullkiller visibility flags for one or more AI players")
    return {"validated": True, "cheats_allowed": False, "requested_no_cheats": True,
            "difficulty_index": 0, "observer_player_index": player_index,
            "initial_visible_tiles": indexed[player_index]["visible_tiles"], "total_tiles": total_tiles,
            "ai_players": [ai_evidence[index] for index in sorted(ai_evidence)],
            "initial_visibility_by_player": {PLAYER_COLORS[i]: indexed[i]["visible_tiles"] for i in sorted(indexed)},
            "scope": "Effective engine settings, AI flags and fog counts; pixel rendering requires separate visual verification"}


def engine_seed_evidence(log: str, seed: int) -> dict:
    """The server logs the configured seed it actually used; require exactly one matching entry."""
    seeds = re.findall(r"Experiment engine RNG: initial_seed=(\d+) initial_sample=(\d+)", log)
    if len(seeds) != 1 or int(seeds[0][0]) != seed:
        raise ValueError("Missing or mismatched authoritative engine seed evidence")
    return {"initial_seed": int(seeds[0][0]), "initial_sample": int(seeds[0][1]),
            "deterministic_replay_verified": False}
