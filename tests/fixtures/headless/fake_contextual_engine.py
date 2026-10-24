"""A miniature stand-in for the contextual (adventure-windows-v1) headless client.

It speaks bridge protocol 4 to both players over a tiny hidden world with
per-player persistent fog: context updates as ContextualState.h orders them,
hero and town windows, destination routes over each player's known tiles walked
one acknowledged step per command, visits with an audited notice, a battle,
turn ends and natural terminals. It is not VCMI and proves nothing about real
gameplay, routing quality or throughput.
"""
import json
import os
from pathlib import Path
import socket
import time

WIDTH, HEIGHT = 12, 10
ROCK = (2, 3)
MOVEMENT = 4
SCENARIO = json.loads(os.environ.get("FAKE_ENGINE_SCENARIO", "{}"))
MODE = SCENARIO.get("mode", "natural")
END_DAY = SCENARIO.get("end_day", 3)
EPISODE = os.environ["EXPLORER_RL_EPISODE"]
SNAPSHOT_EVERY = int(os.environ.get("EXPLORER_RL_SNAPSHOT_EVERY", "1"))
SEED = int(os.environ.get("FAKE_ENGINE_SEED", "1"))
GOLD_NOTICE = {"display_mode": "info", "text": [
    {"op": "append_text", "text_id": "core.advevent.113", "number": None, "owner": None},
    {"op": "replace_text", "text_id": "core.restypes.6", "number": None, "owner": None}],
    "components": [{"kind": "resource", "resource": "gold", "amount": 500, "owner": None}]}


class World:
    def __init__(self):
        self.objects = {  # engine id -> fact; the complete hidden truth
            10: {"x": 1, "y": 1, "type": 34, "subtype": 0, "owner": 0},
            11: {"x": 0, "y": 0, "type": 98, "subtype": 2, "owner": 0},
            20: {"x": 9, "y": 7, "type": 34, "subtype": 1, "owner": 1},
            21: {"x": 10, "y": 8, "type": 98, "subtype": 3, "owner": 1},
            40: {"x": 3, "y": 1, "type": 79, "subtype": 6, "owner": 255},
            30: {"x": 1, "y": 4, "type": 54, "subtype": 12, "owner": 255},
        }
        self.known = {0: set(), 1: set()}
        self.resources = {0: [10, 0, 10, 0, 0, 0, 1000], 1: [10, 0, 10, 0, 0, 0, 1000]}
        self.movement = {10: MOVEMENT, 20: MOVEMENT}
        self.day = 1
        for player in (0, 1):
            self.reveal(player)

    def heroes(self, player):
        return [oid for oid, f in sorted(self.objects.items()) if f["type"] == 34 and f["owner"] == player]

    def towns(self, player):
        return [oid for oid, f in sorted(self.objects.items()) if f["type"] == 98 and f["owner"] == player]

    def reveal(self, player):
        for oid in self.heroes(player) + self.towns(player):
            fact = self.objects[oid]
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    x, y = fact["x"] + dx, fact["y"] + dy
                    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                        self.known[player].add((x, y, 0))

    def at(self, x, y):
        return [oid for oid, f in sorted(self.objects.items()) if (f["x"], f["y"]) == (x, y)]

    def tile(self, x, y):
        return {"terrain": (x * 3 + y) % 4, "road": 1 if y == 1 else 0, "blocked": (x, y) == ROCK,
                "visitable": bool(self.at(x, y))}

    def capture(self, player):
        tiles = {t: self.tile(t[0], t[1]) for t in self.known[player]}
        objects = {oid: {"id": oid, "x": f["x"], "y": f["y"], "z": 0, "type": f["type"], "subtype": f["subtype"],
                         "owner": f["owner"]}
                   for oid, f in self.objects.items() if (f["x"], f["y"], 0) in self.known[player]}
        own = {"resources": list(self.resources[player]),
               "heroes": [{"id": h, "position": [self.objects[h]["x"], self.objects[h]["y"], 0],
                           "movement": self.movement[h], "movement_max": MOVEMENT} for h in self.heroes(player)],
               "towns": [{"id": t, "position": [self.objects[t]["x"], self.objects[t]["y"], 0]} for t in self.towns(player)]}
        return tiles, objects, own

    def route(self, player, hero, destination):
        """Breadth-first over the player's known, unblocked tiles; objects only as the destination."""
        fact = self.objects[hero]
        start = (fact["x"], fact["y"])
        if (*destination, 0) not in self.known[player]:
            return None
        previous, frontier = {start: None}, [start]
        while frontier:
            following = []
            for x, y in frontier:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        step = (x + dx, y + dy)
                        if step in previous or (*step, 0) not in self.known[player] or step == ROCK:
                            continue
                        if self.at(*step) and step != destination:
                            continue
                        previous[step] = (x, y)
                        following.append(step)
            frontier = following
        if destination not in previous:
            return None
        path = []
        while destination != start:
            path.append(destination)
            destination = previous[destination]
        return list(reversed(path))


class Bridge:
    def __init__(self, player, port):
        self.player = player
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.buffer = b""
        self.events = []
        self.previous = ({}, {}, None)
        self.sequence = 0
        self.decision_id = 0
        self.revision = 0
        self.window = None
        self.decisions = 0

    def message(self, kind, **fields):
        return {"type": kind, "protocol_version": 4, "player": self.player, "episode": EPISODE, **fields}

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
        tiles, objects, own = world.capture(self.player)
        before_tiles, before_objects, before_own = self.previous
        order = lambda t: (t[2], t[1], t[0])  # noqa: E731
        fact = lambda t: {"x": t[0], "y": t[1], "z": t[2], **tiles[t]}  # noqa: E731
        change = {"kind": "context_update", "sequence": self.sequence, "day": world.day,
                  "appeared": [o for i, o in sorted(objects.items()) if i not in before_objects],
                  "changed": [o for i, o in sorted(objects.items()) if i in before_objects and before_objects[i] != o],
                  "disappeared": [i for i in sorted(before_objects) if i not in objects],
                  "revealed": [fact(t) for t in sorted(tiles.keys() - before_tiles.keys(), key=order)],
                  "tiles_changed": [fact(t) for t in sorted(tiles.keys() & before_tiles.keys(), key=order)
                                    if tiles[t] != before_tiles[t]],
                  "hidden": [], "own": own if own != before_own else None}
        lost = [h["id"] for h in (before_own or {}).get("heroes", []) if h["id"] not in {x["id"] for x in own["heroes"]}]
        self.previous = (tiles, objects, own)
        if any(change[k] for k in ("appeared", "changed", "disappeared", "revealed", "tiles_changed", "own")):
            self.sequence += 1
            self.events.append(change)
            self.events.extend({"kind": "hero_lost", "hero_id": hero} for hero in lost)

    def take(self):
        events, self.events = self.events, []
        return events

    def state(self):
        tiles, objects, own = self.previous
        return {"tiles": [{"x": t[0], "y": t[1], "z": t[2], **f} for t, f in sorted(tiles.items())],
                "objects": [o for _, o in sorted(objects.items())], "own": own}


def write_metadata(world):
    directory = Path(os.environ["VCMI_EXPERIMENT_DIR"])
    players = [{"player_index": p, "team_index": p, "human": False, "adventure_ai": "ExplorerRL",
                "visible_tiles": len(world.known[p]), "total_tiles": WIDTH * HEIGHT} for p in (0, 1)]
    metadata = {"schema_version": 1, "phase": "before_first_turn", "cheats_allowed": False,
                "requested_no_cheats": True, "difficulty_index": 0, "game_day": 0, "map_width": WIDTH,
                "map_height": HEIGHT, "map_levels": 1, "requested_agent_player": None,
                "requested_agent_players": [0, 1], "players": players}
    (directory / "experiment-metadata.json").write_text(json.dumps(metadata))
    print(f"Experiment engine RNG: initial_seed={SEED} initial_sample=7", flush=True)


def end_game(world, bridges, winner, results=None):
    for player in (0, 1):
        bridge = bridges[player]
        bridge.note(world)
        message = bridge.message("terminal", outcome="win" if player == winner else "loss", events=bridge.take())
        if results and player in results:
            message["result"] = results[player]
        bridge.send(message)
        print(f"Experiment player {player} {'won' if player == winner else 'lost'}. Terminal received.", flush=True)
    print("Experiment self-play complete. Ending game.", flush=True)
    time.sleep(SCENARIO.get("exit_delay", 0))
    raise SystemExit(0)


def walk(world, bridges, bridge, hero, destination, route, commands):
    """Walks the current-turn prefix one acknowledged step per command; returns the result."""
    stopped = False
    rejected = accepted = 0
    for step in route[:world.movement[hero]]:
        index = commands
        commands += 1
        bridge.events.append({"kind": "issued", "command_index": index})
        if MODE == "reject_step" and index == 1 and bridge.player == 0:
            bridge.events.append({"kind": "acknowledged", "command_index": index, "accepted": False})
            rejected += 1
            break
        fact = world.objects[hero]
        moves = True
        for target in [oid for oid in world.at(*step) if oid != hero]:
            if world.objects[target]["type"] == 79:  # picked up: the hero steps onto its tile
                bridge.events.append({"kind": "visit", "object_id": target, "object_type": 79, "object_subtype": 6,
                                      "owner": 255})
                bridge.events.append({"kind": "notice", "notice": GOLD_NOTICE})
                world.resources[bridge.player][6] += 500
            else:  # a guard: the hero fights from where it stands
                bridge.events.append({"kind": "battle_start", "side": "attacker"})
                bridge.events.append({"kind": "battle_complete", "won": True})
                moves = False
            del world.objects[target]
            stopped = True
        if moves:
            fact["x"], fact["y"] = step
        world.movement[hero] -= 1
        world.reveal(bridge.player)
        for each in bridges.values():
            each.note(world)
        if MODE == "terminal_mid_route" and bridge.player == 0 and index == 0:
            # The game ends after this step's acknowledgement: the terminal carries the result.
            bridge.events.append({"kind": "acknowledged", "command_index": index, "accepted": True})
            fact = world.objects[hero]
            result = {"status": "interrupted", "accepted": True, "position": [fact["x"], fact["y"], 0],
                      "movement": world.movement[hero], "destination_reached": (fact["x"], fact["y"]) == destination}
            end_game(world, bridges, 1, {0: result})
        bridge.events.append({"kind": "acknowledged", "command_index": index, "accepted": True})
        accepted += 1
        if stopped:
            break
    fact = world.objects[hero]
    position = (fact["x"], fact["y"])
    status = ("rejected" if not accepted else "interrupted") if rejected else "interaction" if stopped else \
        "reached" if position == destination else "movement_limit"
    return {"status": status, "accepted": accepted > 0, "position": [fact["x"], fact["y"], 0],
            "movement": world.movement[hero], "destination_reached": position == destination}


def act(world, bridges, bridge, action):
    """Executes one contextual action; returns (result, window, turn ended)."""
    kind = action["kind"]
    if bridge.window is not None:
        assert kind == "close"
        bridge.events.append({"kind": "consumed", "consumer": "adapter"})
        bridge.revision += 1
        bridge.events.append({"kind": "context", "context": {"kind": "map", "subject": None}, "revision": bridge.revision})
        bridge.window = None
        return {"status": "closed", "accepted": True}, None, False
    if kind == "end_turn":
        bridge.events.append({"kind": "consumed", "consumer": "engine"})
        bridge.events.append({"kind": "issued", "command_index": 0})
        bridge.events.append({"kind": "acknowledged", "command_index": 0, "accepted": True})
        bridge.events.append({"kind": "turn_end"})
        return {"status": "turn_ended", "accepted": True}, None, True
    if kind in ("open_hero", "open_town"):
        subject = action["subject"]
        assert subject in (world.heroes(bridge.player) if kind == "open_hero" else world.towns(bridge.player))
        if MODE == "unsupported" and kind == "open_town":
            bridge.send(bridge.message("error", message="Unsupported scenario interaction: market"))
            time.sleep(3600)
        bridge.events.append({"kind": "consumed", "consumer": "adapter"})
        bridge.revision += 1
        context = "hero" if kind == "open_hero" else "town"
        bridge.events.append({"kind": "context", "context": {"kind": context, "subject": subject}, "revision": bridge.revision})
        bridge.window = subject
        army = [{"slot": 0, "creature": 1, "count": 10}]
        window = ({"kind": "hero", "subject": subject, "army": army, "primary_skills": [1, 1, 1, 1], "mana": 10, "level": 1}
                  if context == "hero" else {"kind": "town", "subject": subject, "army": [], "faction": 2})
        return {"status": "opened", "accepted": True}, window, False
    assert kind == "move_to"
    hero, destination = action["subject"], (action["x"], action["y"])
    assert hero in world.heroes(bridge.player)
    fact = world.objects[hero]
    here = {"position": [fact["x"], fact["y"], 0], "movement": world.movement[hero]}
    if (fact["x"], fact["y"]) == destination:
        bridge.events.append({"kind": "consumed", "consumer": "adapter"})
        return {"status": "reached", "accepted": True, **here, "destination_reached": True}, None, False
    if world.movement[hero] <= 0:
        bridge.events.append({"kind": "consumed", "consumer": "adapter"})
        return {"status": "movement_limit", "accepted": True, **here, "destination_reached": False}, None, False
    route = world.route(bridge.player, hero, destination)
    if route is None:
        bridge.events.append({"kind": "consumed", "consumer": "adapter"})
        return {"status": "no_route_known", "accepted": True, **here, "destination_reached": False}, None, False
    bridge.events.append({"kind": "consumed", "consumer": "engine"})
    return walk(world, bridges, bridge, hero, destination, route, 0), None, False


def play_turn(world, bridges, player):
    bridge = bridges[player]
    while True:
        bridge.note(world)
        context = {"kind": "map", "subject": None} if bridge.window is None else \
            {"kind": "hero" if bridge.window in world.heroes(player) else "town", "subject": bridge.window}
        bridge.revision += 1
        message = bridge.message("decision", decision_id=bridge.decision_id, revision=bridge.revision, context=context,
                                 day=world.day, events=bridge.take())
        if SNAPSHOT_EVERY and bridge.decisions % SNAPSHOT_EVERY == 0:
            message["state"] = bridge.state()
        if MODE == "leak" and player == 0 and world.day == 2:
            message["state"]["objects"].append({"id": 20, "x": 9, "y": 7, "z": 0, "type": 34, "subtype": 1, "owner": 1})
        bridge.decisions += 1
        bridge.send(message)
        if MODE == "hang_after_decision" and world.day == 2 and player == 0:
            time.sleep(3600)  # the test kills the engine while the policy is choosing
        response = bridge.receive()
        if response.get("decision_id") != bridge.decision_id or response.get("revision") != bridge.revision:
            raise SystemExit("mismatched response")
        bridge.decision_id += 1
        result, window, ended = act(world, bridges, bridge, response["action"])
        bridge.note(world)
        reply = bridge.message("result", decision_id=message["decision_id"], events=bridge.take(), result=result)
        if window is not None:
            reply["window"] = window
        bridge.send(reply)
        if ended:
            return


def main():
    world = World()
    bridges = {p: Bridge(p, int(os.environ[f"EXPLORER_RL_PORT_{p}"])) for p in (0, 1)}
    for bridge in bridges.values():
        bridge.send(bridge.message("ready", interface="adventure-windows-v1", action_schema="heroes-context-actions-v2",
                                   observation_events="context_update-v1", routing="player-knowledge-current-turn-v1",
                                   notices="audited-notice-v1", battle_ai="BattleAI", profiling=False,
                                   snapshot_every=SNAPSHOT_EVERY, map_size=[WIDTH, HEIGHT, 1]))
    write_metadata(world)
    while True:
        for player in (0, 1):
            if world.day == END_DAY and player == 0 and MODE != "endless":
                end_game(world, bridges, 0)
            play_turn(world, bridges, player)
            for hero in world.heroes(player):
                world.movement[hero] = MOVEMENT
        world.day += 1
        for player in (0, 1):
            world.resources[player][6] += 1000


if __name__ == "__main__":
    main()
