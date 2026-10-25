"""Seeded uniform policy over player-visible primitive attempts; no learning."""
from __future__ import annotations

import json
from pathlib import Path
import random
import socket
import threading
import time


class RandomPolicy:
    """One local bridge connection. Policy decisions never read engine/spectator state."""
    def __init__(self, seed: int, max_actions: int = 2000, max_days: int = 40, step_delay: float = 0.15):
        if (type(seed) is not int or type(max_actions) is not int or type(max_days) is not int
                or max_actions < 1 or max_days < 1 or not 0 <= step_delay <= 5):
            raise ValueError("Use an integer policy seed, positive limits and step delay 0..5 seconds")
        self.seed, self.max_actions, self.max_days, self.step_delay = seed, max_actions, max_days, step_delay
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.listener = None
        self.connection = None
        self.shutdown_expected = False
        self.error = None
        self.limit = None
        self.terminal = None
        self.ready = False
        self.observations = 0
        self.actions = 0
        self.actions_sent = 0
        self.decisions_sent = 0
        self.decisions = 0
        self.accepted = 0
        self.rejected = 0
        self.moves = 0
        self.turns = 0
        self.last_day = 0
        self.initial_observation = None
        self.last_observation = None
        self.visited = set()
        self.last_action_id = -1
        self.pending_action = None
        self.awaiting_turn_end = False
        self.last_decision_id = -1
        self.started = None

    @property
    def description(self) -> dict:
        return {"label": "Uniform random agent (no learning)", "checkpoint": None,
                "algorithm": "uniform_random", "seed": self.seed, "learning": False,
                "action_space": "8 adjacent move attempts + end turn; uniform over permitted attempts",
                "dialog_policy": "uniform over choices supplied by the player interface",
                "rewards": "natural win +1 / loss -1 only; truncation and errors unscored",
                "step_delay_seconds": self.step_delay}

    def start(self, output: Path, episode: str, player: int) -> dict:
        self.episode, self.player = episode, player
        self.path = output / "policy.jsonl"
        if self.path.exists():
            raise FileExistsError(f"Policy ledger already exists: {self.path}")
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(0.2)
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._run, name="random-policy", daemon=True)
        self.thread.start()
        return {"EXPLORER_RL_PORT": str(self.listener.getsockname()[1]),
                "EXPLORER_RL_PLAYER": str(player), "EXPLORER_RL_EPISODE": episode,
                "EXPLORER_RL_TIMEOUT": "60"}

    @property
    def stop_reason(self) -> str | None:
        with self.lock:
            return "policy_error" if self.error else self.limit

    def snapshot(self) -> dict:
        with self.lock:
            return {"ready": self.ready, "error": self.error, "limit": self.limit,
                    "terminal": self.terminal, "observations": self.observations,
                    "actions_sent": self.actions_sent, "actions_chosen": self.actions,
                    "decisions_sent": self.decisions_sent, "decisions_chosen": self.decisions,
                    "actions_accepted": self.accepted, "actions_rejected": self.rejected,
                    "successful_displacements": self.moves, "unique_hero_positions": len(self.visited),
                    "completed_turns": self.turns, "last_day": self.last_day,
                    "initial_observation": self.initial_observation, "last_observation": self.last_observation}

    def prepare_stop(self) -> None:
        # Deliberate truncation closes sockets, but is never a policy error or a loss.
        self.shutdown_expected = True

    def close(self) -> None:
        self.prepare_stop()
        self.stop.set()
        for sock in (self.connection, self.listener):
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
        if self.thread is not None:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                raise RuntimeError("Policy thread did not stop")

    @staticmethod
    def observation_summary(message: dict) -> dict:
        # Only data delivered through the player's callback. No omniscient metadata here.
        state = message.get("state", message)
        keys = ("day", "hero_id", "x", "y", "z", "movement", "movement_max", "own_towns",
                "own_mines", "resources", "completed_battles", "own_heroes", "army")
        summary = {k: state[k] for k in keys if k in state}
        summary["visible_tiles"] = state.get("visible_tiles", len(message.get("tiles", [])))
        return summary

    def choose_action(self, message: dict) -> int:
        mask = message.get("action_mask")
        if not isinstance(mask, list) or len(mask) != 9 or any(type(x) is not bool for x in mask) or not mask[8]:
            raise ValueError("Expected nine boolean attempt-mask entries with end-turn enabled")
        return self.rng.choice([i for i, allowed in enumerate(mask) if allowed])

    def _handle(self, message: dict) -> dict | None:
        if (not isinstance(message, dict) or type(message.get("player")) is not int
                or message.get("player") != self.player or message.get("episode") != self.episode):
            raise ValueError("Policy message has wrong episode/player identity")
        kind = message.get("type")
        with self.lock:
            if self.terminal:
                raise ValueError("Message received after terminal")
            if kind == "ready":
                if self.ready or message.get("protocol_version") != 3:
                    raise ValueError("Unexpected or unsupported bridge handshake")
                self.ready = True
                return None
            if not self.ready:
                raise ValueError("Bridge must send protocol-v3 ready before any game message")
            if kind == "observation":
                action_id, day = message.get("action_id"), message.get("day")
                if type(action_id) is not int or action_id != self.last_action_id + 1 or self.pending_action is not None or self.awaiting_turn_end:
                    raise ValueError("Observation action sequence is incomplete or out of order")
                if type(day) is not int or day < self.last_day or day < 1:
                    raise ValueError("Observation game day is invalid")
                if not isinstance(message.get("tiles"), list):
                    raise ValueError("Observation must explicitly contain player-visible tiles")
                self.observations += 1
                self.last_day = day
                summary = self.observation_summary(message)
                if self.initial_observation is None:
                    self.initial_observation = summary
                self.last_observation = summary
                pos = tuple(message.get(k) for k in ("x", "y", "z"))
                if all(type(x) is int for x in pos):
                    self.visited.add(pos)
                if day > self.max_days or self.actions >= self.max_actions:
                    self.limit = "game_day_limit" if day > self.max_days else "agent_action_limit"
                    return None
                action = self.choose_action(message)
                self.pending_action = (action_id, action, pos)
                self.last_action_id = action_id
                self.actions += 1
                return {"action_id": action_id, "action": action}
            if kind == "action_result":
                if (type(message.get("action_id")) is not int or type(message.get("action")) is not int
                        or self.pending_action is None or (message.get("action_id"), message.get("action")) != self.pending_action[:2]):
                    raise ValueError("Action result does not match the submitted action")
                if type(message.get("accepted")) is not bool:
                    raise ValueError("Action result is missing an acceptance flag")
                if message["accepted"]:
                    self.accepted += 1
                else:
                    self.rejected += 1
                after = message.get("after", {})
                if not isinstance(after, dict):
                    raise ValueError("Action result state must be an object")
                pos = tuple(after.get(k) for k in ("x", "y", "z"))
                if self.pending_action[1] < 8 and all(type(x) is int for x in pos) and pos != self.pending_action[2]:
                    self.moves += 1
                    self.visited.add(pos)
                self.awaiting_turn_end = self.pending_action[1] == 8 and message["accepted"]
                self.pending_action = None
                return None
            if kind == "decision":
                decision_id, choices = message.get("decision_id"), message.get("choices")
                if type(decision_id) is not int or decision_id <= self.last_decision_id:
                    raise ValueError("Decision ID must strictly increase")
                if not isinstance(choices, list) or not choices or any(not isinstance(x, dict) or type(x.get("choice")) is not int for x in choices):
                    raise ValueError("Decision must provide explicit integer choices")
                if len({x["choice"] for x in choices}) != len(choices):
                    raise ValueError("Duplicate decision choices")
                self.last_decision_id = decision_id
                self.decisions += 1
                return {"decision_id": decision_id, "choice": self.rng.choice(choices)["choice"]}
            if kind == "turn_end":
                if not self.awaiting_turn_end or self.pending_action is not None:
                    raise ValueError("Turn end must follow an accepted end-turn action")
                self.awaiting_turn_end = False
                self.turns += 1
                return None
            if kind == "terminal":
                if message.get("outcome") not in {"win", "loss"}:
                    raise ValueError("Terminal must contain a natural win or loss")
                # This is reported engine evidence, not a scored episode. The
                # recorder assigns reward only after process/recording checks.
                self.terminal = {"outcome": message["outcome"], "state": message.get("state"),
                                 "events": message.get("events", [])}
                return None
            if kind == "error":
                raise ValueError("ExplorerRL bridge: " + str(message.get("message")))
            raise ValueError(f"Unsupported bridge message: {kind}")

    def _run(self) -> None:
        try:
            with self.path.open("x") as ledger:
                def write(direction, message):
                    ledger.write(json.dumps({"elapsed_seconds": time.monotonic() - self.started,
                                             "direction": direction, "message": message}, separators=(",", ":")) + "\n")
                    ledger.flush()
                while not self.stop.is_set():
                    try:
                        self.connection, _ = self.listener.accept()
                        break
                    except socket.timeout:
                        continue
                if self.stop.is_set():
                    return
                self.connection.settimeout(0.2)
                buffer = b""
                while not self.stop.is_set():
                    try:
                        chunk = self.connection.recv(65536)
                    except socket.timeout:
                        continue
                    if not chunk:
                        if not self.terminal and not self.shutdown_expected:
                            raise ValueError("Bridge disconnected without a natural terminal message")
                        return
                    buffer += chunk
                    if len(buffer) > 16 * 1024 * 1024:
                        raise ValueError("Bridge message exceeds 16 MiB")
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        message = json.loads(line)
                        write("engine", message)
                        response = self._handle(message)
                        if response is not None:
                            if self.stop.wait(self.step_delay):
                                return
                            self.connection.sendall((json.dumps(response) + "\n").encode())
                            with self.lock:
                                if "action" in response:
                                    self.actions_sent += 1
                                else:
                                    self.decisions_sent += 1
                            write("policy", response)
        except Exception as error:
            if not self.stop.is_set():
                with self.lock:
                    self.error = str(error)
