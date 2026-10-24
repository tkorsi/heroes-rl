"""A miniature stand-in for the headless client, for environment tests only.

It speaks bridge protocol 3 to both players with a tiny hidden world, per-team
persistent fog and the same snapshot-diff event semantics as VisibleState.h.
It is not VCMI and proves nothing about real gameplay or throughput.
"""
import json
import os
from pathlib import Path
import socket
import sys
import time

DIRECTIONS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
WIDTH, HEIGHT = 8, 6
SCENARIO = json.loads(os.environ.get("FAKE_ENGINE_SCENARIO", "{}"))
MODE = SCENARIO.get("mode", "natural")
EPISODE = os.environ["EXPLORER_RL_EPISODE"]
SEED = int(os.environ.get("FAKE_ENGINE_SEED", "1"))


class World:
    def __init__(self):
        start = SCENARIO.get("blue_start", [6, 4])
        self.objects = {  # engine id -> fact; the complete hidden truth
            10: {"x": 1, "y": 1, "type": 34, "subtype": 0, "owner": 0},
            20: {"x": start[0], "y": start[1], "type": 34, "subtype": 1, "owner": 1},
            11: {"x": 0, "y": 0, "type": 98, "subtype": 0, "owner": 0},
            21: {"x": 7, "y": 5, "type": 98, "subtype": 1, "owner": 1},
            30: {"x": 3, "y": 3, "type": 53, "subtype": 0, "owner": 255},
            40: {"x": 2, "y": 1, "type": 79, "subtype": 6, "owner": 255},
        }
        self.fog = {0: {(x, y, 0) for x in range(4) for y in range(HEIGHT)},
                    1: {(x, y, 0) for x in range(4, WIDTH) for y in range(HEIGHT)}}
        self.resources = {0: [0] * 7, 1: [0] * 7}
        self.movement = {0: 3, 1: 3}
        self.day = 1

    def hero(self, player):
        return 10 if player == 0 else 20

    def visible_objects(self, player):
        result = {}
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if (x, y, 0) in self.fog[player]:
                    for oid in sorted(self.objects):
                        fact = self.objects[oid]
                        if (fact["x"], fact["y"]) == (x, y):
                            result.setdefault(oid, {"id": oid, "z": 0, **fact})
        return result

    def state(self, player):
        oid = self.hero(player)
        hero = self.objects.get(oid)
        return {"day": self.day, "own_heroes": 1 if hero else 0,
                "own_towns": sum(1 for f in self.objects.values() if f["type"] == 98 and f["owner"] == player),
                "own_mines": sum(1 for f in self.objects.values() if f["type"] == 53 and f["owner"] == player),
                "resources": list(self.resources[player]), "hero_id": oid if hero else -1,
                "movement": self.movement[player], "movement_max": 3,
                "x": hero["x"] if hero else None, "y": hero["y"] if hero else None, "z": 0 if hero else None,
                "visible_tiles": len(self.fog[player]), "completed_battles": 0}


class Tracker:
    def __init__(self):
        self.previous = None
        self.sequence = 0

    def update(self, world, player):
        tiles, objects = set(world.fog[player]), world.visible_objects(player)
        before_tiles, before_objects = self.previous or (set(), {})
        change = {"kind": "visible_update", "sequence": self.sequence, "day": world.day,
                  "appeared": [o for i, o in sorted(objects.items()) if i not in before_objects],
                  "changed": [o for i, o in sorted(objects.items()) if i in before_objects and before_objects[i] != o],
                  "disappeared": [i for i in sorted(before_objects) if i not in objects],
                  "revealed": [list(t) for t in sorted(tiles - before_tiles, key=lambda t: (t[2], t[1], t[0]))],
                  "hidden": [list(t) for t in sorted(before_tiles - tiles, key=lambda t: (t[2], t[1], t[0]))]}
        self.previous = (tiles, objects)
        if any(change[k] for k in ("appeared", "changed", "disappeared", "revealed", "hidden")):
            self.sequence += 1
            return [change]
        return []


class Bridge:
    def __init__(self, player, port):
        self.player = player
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.buffer = b""
        self.events = []
        self.tracker = Tracker()
        self.action_id = 0
        self.decision_id = 0

    def message(self, message_type, **fields):
        return {"type": message_type, "protocol_version": SCENARIO.get("protocol_version", 3), "player": self.player,
                "episode": EPISODE, **fields}

    def send(self, message):
        self.sock.sendall((json.dumps(message) + "\n").encode())

    def receive(self):
        while b"\n" not in self.buffer:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise SystemExit(f"policy {self.player} disconnected")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def note(self, world):
        self.events.extend(self.tracker.update(world, self.player))

    def take(self):
        events, self.events = self.events, []
        return events


def observation(world, bridge):
    player = bridge.player
    bridge.note(world)
    hero = world.objects.get(world.hero(player))
    tiles = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if (x, y, 0) in world.fog[player]:
                objects = [{"id": oid, "type": f["type"], "subtype": f["subtype"], "owner": f["owner"]}
                           for oid, f in sorted(world.objects.items()) if (f["x"], f["y"]) == (x, y)]
                tiles.append({"x": x, "y": y, "z": 0, "terrain": 2, "blocked": False, "visitable": bool(objects),
                              "objects": objects})
    mask = [False] * 9
    mask[8] = True
    if hero and world.movement[player] > 0:
        for index, (dx, dy) in enumerate(DIRECTIONS):
            mask[index] = 0 <= hero["x"] + dx < WIDTH and 0 <= hero["y"] + dy < HEIGHT
    action_id = bridge.action_id + SCENARIO.get("action_id_offset", 0)
    return bridge.message("observation", action_id=action_id, day=world.day,
                          hero_id=world.hero(player) if hero else -1, x=hero["x"] if hero else None,
                          y=hero["y"] if hero else None, z=0 if hero else None, width=WIDTH, height=HEIGHT,
                          levels=1, movement=world.movement[player], movement_max=3, tiles=tiles,
                          action_mask=mask, state=world.state(player), events=bridge.take())


def move(world, bridges, player, action):
    hero = world.objects[world.hero(player)]
    dx, dy = DIRECTIONS[action]
    hero["x"] += dx
    hero["y"] += dy
    world.movement[player] -= 1
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            x, y = hero["x"] + ox, hero["y"] + oy
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                world.fog[player].add((x, y, 0))
    for oid, fact in list(world.objects.items()):
        if oid not in (10, 20) and (fact["x"], fact["y"]) == (hero["x"], hero["y"]):
            bridges[player].events.append({"kind": "visit", "object_id": oid, "object_type": fact["type"],
                                           "object_subtype": fact["subtype"], "owner": fact["owner"]})
            if fact["type"] == 79:
                del world.objects[oid]
                world.resources[player][6] += 500
            elif fact["type"] == 53:
                fact["owner"] = player
    for bridge in bridges.values():  # each step is applied, then every interface is notified
        bridge.note(world)


def write_metadata(world):
    directory = Path(os.environ["VCMI_EXPERIMENT_DIR"])
    players = [{"player_index": p, "team_index": p, "human": False, "adventure_ai": "ExplorerRL",
                "visible_tiles": len(world.fog[p]), "total_tiles": WIDTH * HEIGHT} for p in (0, 1)]
    metadata = {"schema_version": 1, "phase": "before_first_turn", "cheats_allowed": False,
                "requested_no_cheats": True, "difficulty_index": 0, "game_day": 0, "map_width": WIDTH,
                "map_height": HEIGHT, "map_levels": 1, "requested_agent_player": None,
                "requested_agent_players": [0, 1], "players": players}
    if MODE != "missing_metadata":
        (directory / "experiment-metadata.json").write_text(json.dumps(metadata))
    print(f"Experiment engine RNG: initial_seed={SEED} initial_sample=7", flush=True)


def end_game(world, bridges, winner):
    for player in (0, 1):
        outcome = "win" if player == winner else "loss"
        bridges[player].note(world)
        if not (MODE == "missing_terminal" and player == 1):
            bridges[player].send(bridges[player].message("terminal", outcome=outcome, events=bridges[player].take(),
                                                         state=world.state(player)))
        print(f"Experiment player {player} {'won' if outcome == 'win' else 'lost'}. Terminal received.", flush=True)
    print("Experiment self-play complete. Ending game.", flush=True)
    if MODE == "no_shutdown":
        time.sleep(3600)
    raise SystemExit(1 if MODE == "unclean_exit" else 0)


def play_turn(world, bridges, player):
    bridge = bridges[player]
    actions = 0
    while True:
        bridge.send(observation(world, bridge))
        response = bridge.receive()
        if response.get("action_id") != bridge.action_id:
            raise SystemExit("mismatched response")
        action = response["action"]
        actions += 1
        before = world.state(player)
        if MODE == "crash" and world.day == SCENARIO.get("crash_day", 2):
            os.abort()
        if MODE == "hang" and world.day == SCENARIO.get("hang_day", 2):
            time.sleep(3600)
        if MODE == "engine_error" and player == 1:
            bridge.send(bridge.message("error", message="Unsupported scenario interaction: market"))
            time.sleep(3600)
        if MODE == "decision" and player == 0 and actions == 1:
            other = bridges[1]
            other.note(world)
            other.send(other.message("decision", kind="hero_level_up", hero_id=20, primary_skill=2,
                                     choices=[{"choice": 0, "secondary_skill": 7}, {"choice": 1, "secondary_skill": 9}],
                                     decision_id=other.decision_id, events=other.take()))
            answer = other.receive()
            if answer != {"decision_id": other.decision_id, "choice": answer.get("choice")} or answer["choice"] not in (0, 1):
                raise SystemExit("bad decision answer")
            other.events.append({"kind": "decision", "decision_kind": "hero_level_up",
                                 "decision_id": other.decision_id, "choice": answer["choice"]})
            other.decision_id += 1
        if MODE == "terminal_race" and world.day == SCENARIO.get("end_day", 2) and player == 0:
            if action < 8:
                move(world, bridges, player, action)
            end_game(world, bridges, 0)  # the action result is suppressed; its events ride on the terminal
        if action < 8:
            move(world, bridges, player, action)
        result = bridge.message("action_result", action_id=bridge.action_id, action=action, accepted=True,
                                before=before, after=world.state(player))
        bridge.note(world)
        result["events"] = bridge.take()
        bridge.action_id += 1
        bridge.send(result)
        if action == 8:
            bridge.send(bridge.message("turn_end"))
            return


def main():
    world = World()
    bridges = {p: Bridge(p, int(os.environ[f"EXPLORER_RL_PORT_{p}"])) for p in (0, 1)}
    for bridge in bridges.values():
        bridge.send(bridge.message("ready", adventure_actions=9, battle_ai="BattleAI",
                                   observation_events=SCENARIO.get("observation_events", "visible_update-v1"),
                                   profiling=False))
    write_metadata(world)
    for bridge in bridges.values():
        bridge.note(world)  # the first capture is reported against an empty view
    if MODE == "leak":
        bridges[0].events.append({"kind": "visible_update", "sequence": bridges[0].tracker.sequence, "day": 1,
                                  "appeared": [{"id": 20, "z": 0, **world.objects[20]}], "changed": [],
                                  "disappeared": [], "revealed": [], "hidden": []})
        bridges[0].tracker.sequence += 1
    if MODE == "exit_early":
        raise SystemExit(0)
    while True:
        for player in (0, 1):
            if MODE in {"natural", "unclean_exit", "no_shutdown", "missing_terminal", "missing_metadata"} \
                    and world.day == SCENARIO.get("end_day", 2) and player == 1:
                end_game(world, bridges, 0)
            play_turn(world, bridges, player)
            world.movement[player] = 3
        world.day += 1
        if MODE == "decision" and world.day == 2:
            end_game(world, bridges, 1)


if __name__ == "__main__":
    main()
