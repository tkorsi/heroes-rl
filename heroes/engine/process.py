"""Launch one headless VCMI game and stop it again.

Every game is a fresh ``vcmiclient`` process with its own profile directory.
Both seats are played by the ExplorerRL AI plugin, which connects back to two
loopback TCP ports and asks the policy for every adventure-map decision.
Battles are fought by VCMI's BattleAI.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from ..game import contract

_OWNED_VARIABLES = ("EXPLORER_RL_", "VCMI_RECORD_", "VCMI_EXPERIMENT_", "VCMI_USER_DATA_DIR", "XDG_")


@dataclass(frozen=True)
class Engine:
    binary: Path  # the patched vcmiclient
    data: Path  # folder holding the original game's Data directory
    game_map: Path  # the .vmap or .h3m both seats play
    interface: str = contract.EXPANDED_ENVIRONMENT
    army_information: str | None = contract.OWN_ARMY_INFORMATION
    equipment_policy: str | None = "no-artifacts-v1"
    # The engine logs every network packet and observation digest for offline
    # replay. Training never reads these logs, and they reach gigabytes in long
    # games, but the engine has only ever been run with them switched on.
    record_replay: bool = True

    def check(self):
        """Fail early, with the path, when the runtime is not where the configuration says."""
        binary = Path(self.binary)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise FileNotFoundError(f"VCMI client not found or not executable: {binary}")
        if not (Path(self.data) / "Data").is_dir():
            raise FileNotFoundError(f"Original game assets not found: {Path(self.data) / 'Data'}")
        if not Path(self.game_map).is_file():
            raise FileNotFoundError(f"Map not found: {self.game_map}")

    def settings(self, seed):
        server = {"seed": seed, "useProcess": False, "localPort": 0}
        if self.equipment_policy is not None:
            server["experimentEquipmentPolicy"] = self.equipment_policy
        return {
            "general": {"lastDifficulty": 0, "music": 0, "sound": 0},
            "server": server,
            "ai": {"adventureEnemyAI": "Nullkiller", "adventureAlliedAI": "Nullkiller",
                   "combatEnemyAI": "BattleAI", "combatAlliedAI": "BattleAI", "combatNeutralAI": "BattleAI"},
            # Outcomes are read from info-level log lines; trace logging is far slower.
            "logging": {"loggers": [{"domain": "global", "level": "info"}, {"domain": "rng", "level": "info"}]},
        }

    def launch(self, directory, episode, seed, ports, log):
        """Start the game in ``directory``; ``ports`` are the two policy listeners."""
        directory = Path(directory)
        profile, logs, engine = directory / "profile", directory / "logs", directory / "engine"
        for folder in (logs, engine):
            folder.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            data_root, config_root = profile, profile / "config"
            environment = {"VCMI_USER_DATA_DIR": str(profile)}
        else:
            data_root, config_root = profile / "data" / "vcmi", profile / "config" / "vcmi"
            environment = {"XDG_DATA_HOME": str(profile / "data"), "XDG_CONFIG_HOME": str(profile / "config"),
                           "XDG_CACHE_HOME": str(profile / "cache")}
        for folder in (data_root / "Maps", config_root):
            folder.mkdir(parents=True, exist_ok=True)
        (data_root / "Data").symlink_to(Path(self.data).resolve() / "Data", target_is_directory=True)
        local_map = data_root / "Maps" / ("headless" + Path(self.game_map).suffix.lower())
        shutil.copyfile(self.game_map, local_map)
        (config_root / "settings.json").write_text(json.dumps(self.settings(seed), indent=2, sort_keys=True) + "\n")
        environment.update({
            "VCMI_EXPERIMENT_DIR": str(engine), "VCMI_EXPERIMENT_NO_CHEATS": "1",
            "EXPLORER_RL_PLAYERS": "0,1", "EXPLORER_RL_EPISODE": str(episode),
            "EXPLORER_RL_TIMEOUT": "0",  # the engine waits for each answer without a deadline
            "EXPLORER_RL_PROFILE": "0", "EXPLORER_RL_INTERFACE": self.interface,
            # The contextual evidence mode also selects the engine's orderly
            # end-of-game shutdown, which the outcome check relies on.
            "VCMI_EXPERIMENT_CONTEXTUAL_EVIDENCE": "1",
            "EXPLORER_RL_PORT_0": str(ports[0]), "EXPLORER_RL_PORT_1": str(ports[1]),
        })
        if self.record_replay:
            environment["VCMI_EXPERIMENT_REPLAY"] = "1"
        if self.army_information is not None:
            environment["EXPLORER_RL_ARMY_INFORMATION"] = self.army_information
        inherited = {key: value for key, value in os.environ.items() if not key.startswith(_OWNED_VARIABLES)}
        binary = Path(self.binary).resolve()
        command = [str(binary), "--headless", "--testmap", "Maps/" + local_map.name,
                   "--nointro", "--disable-video", "--logLocation", str(logs)]
        # Its own session, so the whole process group can be stopped at once.
        return subprocess.Popen(command, cwd=binary.parent, env={**inherited, **environment},
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)


def stop(process):
    """Stop the game's process group, stragglers included."""
    # macOS answers EPERM, not ESRCH, while the group's only member is an unreaped zombie.
    gone = (ProcessLookupError, PermissionError)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except gone:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except gone:
        pass
    process.wait(timeout=5)
