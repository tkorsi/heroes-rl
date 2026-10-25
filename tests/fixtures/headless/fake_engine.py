"""A miniature stand-in for the headless client, for environment tests only.

It speaks bridge protocol 3 to both players with a tiny hidden world in one
corner of a 20x20 map, per-team persistent fog and the same snapshot-diff event
semantics as VisibleState.h, including tile facts and the own state.
It is not VCMI and proves nothing about real gameplay or throughput.
"""
import json
import os
from pathlib import Path
import socket
import sys
import time

DIRECTIONS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
WIDTH, HEIGHT = 20, 20
ROCK = (2, 4)  # a static blocking tile
SCENARIO = json.loads(os.environ.get("FAKE_ENGINE_SCENARIO", "{}"))
MODE = SCENARIO.get("mode", "natural")
EPISODE = os.environ["EXPLORER_RL_EPISODE"]
SNAPSHOT_EVERY = int(os.environ.get("EXPLORER_RL_SNAPSHOT_EVERY", "1"))
SEED = int(os.environ.get("FAKE_ENGINE_SEED", "1"))
REPLAY = os.environ.get("VCMI_EXPERIMENT_REPLAY") == "1"
POLICY_MESSAGES = {"observation", "action_result", "decision", "terminal"}


def view_digest(world, player):
    """FNV-1a of one player's current view, computed from the world directly
    (not from the emitted events), as VisibleState.h snapshotDigest orders it."""
    state = 1469598103934665603

    def add(value):
        nonlocal state
        bits = int(value) & (2 ** 64 - 1)
        for _ in range(8):
            state = ((state ^ (bits & 0xFF)) * 1099511628211) & (2 ** 64 - 1)
            bits >>= 8

    for value in (WIDTH, HEIGHT, 1):
        add(value)
    for x, y, _ in sorted(world.fog[player], key=lambda t: t[0] + WIDTH * t[1]):
        tile = world.tile(x, y)
        for value in (x + WIDTH * y, tile["terrain"], tile["blocked"], tile["visitable"]):
            add(value)
    objects = world.visible_objects(player)
    add(len(objects))
    for oid in sorted(objects):
        for key in ("id", "x", "y", "z", "type", "subtype", "owner"):
            add(objects[oid][key])
    own = world.own(player)
    add(own["hero"] is not None)
    for value in own["hero"] or []:
        add(value)
    add(own["movement"])
    add(own["movement_max"])
    add(len(own["resources"]))
    for value in own["resources"]:
        add(value)
    add(len(own["army"]))
    for stack in own["army"]:
        for key in ("slot", "creature", "count"):
            add(stack[key])
    add(len(own["towns"]))
    for town in own["towns"]:
        for value in town:
            add(value)
    add(own["mines"])
    return f"{state:016x}"


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
        self.fog = {0: {(x, y, 0) for x in range(4) for y in range(6)},
                    1: {(x, y, 0) for x in range(4, 8) for y in range(6)}}
        self.resources = {0: [0] * 7, 1: [0] * 7}
        self.movement = {0: 3, 1: 3}
        self.day = 1
        self.packs = 1  # client packs applied; the first day's new-turn pack precedes every capture

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

    def tile(self, x, y):
        occupied = any((f["x"], f["y"]) == (x, y) for f in self.objects.values())
        return {"terrain": (x + y) % 3, "blocked": (x, y) == ROCK, "visitable": occupied}

    def own(self, player):
        hero = self.objects.get(self.hero(player))
        return {"hero": [hero["x"], hero["y"], 0] if hero else None,
                "movement": self.movement[player] if hero else 0, "movement_max": 3 if hero else 0,
                "resources": list(self.resources[player]),
                "army": [{"slot": 0, "creature": 1, "count": 10}] if hero else [],
                "towns": sorted([f["x"], f["y"], 0] for f in self.objects.values() if f["type"] == 98 and f["owner"] == player),
                "mines": sum(1 for f in self.objects.values() if f["type"] == 53 and f["owner"] == player)}

    def state(self, player):
        oid = self.hero(player)
        hero = self.objects.get(oid)
        own = self.own(player)
        return {"day": self.day, "own_heroes": 1 if hero else 0, "army": own["army"], "towns": own["towns"],
                "own_towns": sum(1 for f in self.objects.values() if f["type"] == 98 and f["owner"] == player),
                "own_mines": sum(1 for f in self.objects.values() if f["type"] == 53 and f["owner"] == player),
                "resources": list(self.resources[player]), "hero_id": oid if hero else -1,
                "movement": own["movement"], "movement_max": own["movement_max"],
                "x": hero["x"] if hero else None, "y": hero["y"] if hero else None, "z": 0 if hero else None,
                "visible_tiles": len(self.fog[player]), "completed_battles": 0}


class Tracker:
    def __init__(self):
        self.previous = None
        self.sequence = 0
        self.ledger = None  # engine-side capture digests when replay evidence is recorded
        self.digest = None
        self.pack = None
        self.messages = 0

    def update(self, world, player):
        tiles = {t: world.tile(t[0], t[1]) for t in world.fog[player]}
        objects, own = world.visible_objects(player), world.own(player)
        before_tiles, before_objects, before_own = self.previous or ({}, {}, None)
        order = lambda t: (t[2], t[1], t[0])  # noqa: E731
        change = {"kind": "visible_update", "sequence": self.sequence, "day": world.day,
                  "appeared": [o for i, o in sorted(objects.items()) if i not in before_objects],
                  "changed": [o for i, o in sorted(objects.items()) if i in before_objects and before_objects[i] != o],
                  "disappeared": [i for i in sorted(before_objects) if i not in objects],
                  "revealed": [{"x": t[0], "y": t[1], "z": t[2], **tiles[t]}
                               for t in sorted(tiles.keys() - before_tiles.keys(), key=order)],
                  "tiles_changed": [{"x": t[0], "y": t[1], "z": t[2], **tiles[t]}
                                    for t in sorted(tiles.keys() & before_tiles.keys(), key=order) if tiles[t] != before_tiles[t]],
                  "hidden": [list(t) for t in sorted(before_tiles.keys() - tiles.keys(), key=order)],
                  "own": own if own != before_own else None}
        self.previous = (tiles, objects, own)
        if self.ledger is not None:
            self.digest, self.pack = view_digest(world, player), world.packs
        if any(change[k] for k in ("appeared", "changed", "disappeared", "revealed", "tiles_changed", "hidden", "own")):
            if self.ledger is not None:
                self.ledger.write(json.dumps({"kind": "capture", "sequence": self.sequence, "pack": self.pack,
                                              "pack_type": 0, "digest": self.digest}) + "\n")
            self.sequence += 1
            return [change]
        return []

    def note_message(self, message_type):
        if self.ledger is not None and self.digest is not None and message_type in POLICY_MESSAGES:
            self.ledger.write(json.dumps({"kind": "message", "type": message_type, "index": self.messages,
                                          "pack": self.pack, "pack_type": 0, "digest": self.digest}) + "\n")
            self.messages += 1
            self.ledger.flush()


class Bridge:
    def __init__(self, player, port):
        self.player = player
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # as ExplorerRL does
        self.buffer = b""
        self.events = []
        self.tracker = Tracker()
        if REPLAY:
            self.tracker.ledger = (Path(os.environ["VCMI_EXPERIMENT_DIR"]) / f"observations-p{player}.jsonl").open("w")
        self.action_id = 0
        self.consumed = -1  # the last action the engine took, as the terminal's watermark reports it
        self.decision_id = 0
        self.observations = 0

    def message(self, message_type, **fields):
        return {"type": message_type, "protocol_version": SCENARIO.get("protocol_version", 3), "player": self.player,
                "episode": EPISODE, **fields}

    def send(self, message):
        self.tracker.note_message(message["type"])
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
                tiles.append({"x": x, "y": y, "z": 0, **world.tile(x, y), "objects": objects})
    mask = [False] * 9
    mask[8] = True
    if hero and world.movement[player] > 0:
        for index, (dx, dy) in enumerate(DIRECTIONS):
            mask[index] = 0 <= hero["x"] + dx < WIDTH and 0 <= hero["y"] + dy < HEIGHT
    action_id = bridge.action_id + SCENARIO.get("action_id_offset", 0)
    message = bridge.message("observation", action_id=action_id, day=world.day,
                             hero_id=world.hero(player) if hero else -1, x=hero["x"] if hero else None,
                             y=hero["y"] if hero else None, z=0 if hero else None, width=WIDTH, height=HEIGHT,
                             levels=1, movement=world.movement[player], movement_max=3, tiles=tiles,
                             action_mask=mask, state=world.state(player), events=bridge.take())
    if not (SNAPSHOT_EVERY and bridge.observations % SNAPSHOT_EVERY == 0):
        del message["tiles"]
    bridge.observations += 1
    return message


def move(world, bridges, player, action):
    hero = world.objects[world.hero(player)]
    dx, dy = DIRECTIONS[action]
    hero["x"] += dx
    hero["y"] += dy
    world.movement[player] -= 1
    world.packs += 1
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
                                                         state=world.state(player), last_action_id=bridges[player].consumed))
        print(f"Experiment player {player} {'won' if outcome == 'win' else 'lost'}. Terminal received.", flush=True)
    if MODE == "trailing_partial":
        bridges[1].sock.sendall(b'{"type": "observation"')  # a stream that ends inside a message
    print("Experiment self-play complete. Ending game.", flush=True)
    if SCENARIO.get("linger") and os.fork() == 0:
        time.sleep(SCENARIO["linger"])  # a child keeps the bridge sockets open after the engine exits
        os._exit(0)
    time.sleep(SCENARIO.get("exit_delay", 0))  # shutdown work after the terminals
    if MODE == "no_shutdown":
        time.sleep(3600)
    raise SystemExit(1 if MODE == "unclean_exit" else 0)


def play_turn(world, bridges, player):
    bridge = bridges[player]
    actions = 0
    while True:
        bridge.send(observation(world, bridge))
        ending = world.day == SCENARIO.get("end_day", 2) and player == 0
        if MODE == "terminal_before_reply" and ending:
            end_game(world, bridges, 1)  # the game ends while the policy is still choosing
        if MODE == "crash_after_observation" and ending:
            os.abort()  # the engine dies while the policy is still choosing
        if MODE == "garbage_after_observation" and ending:
            bridge.sock.sendall(b"{not json\n")  # a malformed message while the policy is still choosing
        response = bridge.receive()
        if response.get("action_id") != bridge.action_id:
            raise SystemExit("mismatched response")
        if MODE == "reply_unconsumed" and ending:
            end_game(world, bridges, 1)  # the reply arrived, but the game ended before the engine took it
        action = response["action"]
        bridge.consumed = bridge.action_id
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
            # A later level-up offering one skill is applied without a query.
            other.events.append({"kind": "level_up", "primary_skill": 1, "secondary_skills": [7]})
            other.decision_id += 1
        if MODE == "terminal_race" and ending:
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
    time.sleep(SCENARIO.get("startup_delay", 0))  # map loading before the bridges connect
    world = World()
    bridges = {p: Bridge(p, int(os.environ[f"EXPLORER_RL_PORT_{p}"])) for p in (0, 1)}
    for bridge in bridges.values():
        bridge.send(bridge.message("ready", adventure_actions=9, battle_ai="BattleAI",
                                   observation_events=SCENARIO.get("observation_events", "visible_update-v2"),
                                   profiling=False, map_size=[WIDTH, HEIGHT, 1], snapshot_every=SNAPSHOT_EVERY))
    write_metadata(world)
    for bridge in bridges.values():
        bridge.note(world)  # the first capture is reported against an empty view
    if MODE == "leak":
        bridges[0].events.append({"kind": "visible_update", "sequence": bridges[0].tracker.sequence, "day": 1,
                                  "appeared": [{"id": 20, "z": 0, **world.objects[20]}], "changed": [],
                                  "disappeared": [], "revealed": [], "tiles_changed": [], "hidden": [], "own": None})
        bridges[0].tracker.sequence += 1
    if MODE == "exit_early":
        raise SystemExit(0)
    while True:
        for player in (0, 1):
            if MODE in {"natural", "unclean_exit", "no_shutdown", "missing_terminal", "missing_metadata", "trailing_partial"} \
                    and world.day == SCENARIO.get("end_day", 2) and player == 1:
                end_game(world, bridges, 0)
            play_turn(world, bridges, player)
            world.movement[player] = 3
            world.packs += 1
        world.day += 1
        if MODE == "decision" and world.day == 2:
            end_game(world, bridges, 1)


if __name__ == "__main__":
    main()
