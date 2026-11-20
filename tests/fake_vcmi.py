"""A stand-in for ``vcmiclient`` that plays a scripted two-seat game.

It speaks the real wire protocol over the real sockets and writes the log lines
the environment reads, so the environment, the actor and the training loop can
be exercised without the engine. It is not VCMI: the "game" is two seats
ending their turns on a one-tile map until the script says who wins.

The script is the JSON in ``FAKE_VCMI``:
  days     full days to play before the game ends (default 3)
  winner   seat that wins, 0 or 1 (default 0)
  map      side of a square map whose objects and resources change from day
           to day, so every decision shows a different observation (default:
           the same one-tile view every day)
  end      "natural" (default), "closure" (the last action is closed by the
           terminal instead of a result), "endless" (never ends), "crash"
           (exits with an error on day 2), "hang" (goes silent on day 2 until
           the policy side hangs up) or "abort" (fails before connecting)
  still    true for a map whose objects stay where they are, so only the
           resources change from day to day (default false)
  abort_from  game number from which every game fails before connecting, like
           "abort" (default: none)
  repeat   true for an engine that ignores EXPLORER_RL_DECISION_OBSERVATION=0
           and sends the observation again in every decision (default false)
  flicker  true for a map on which two tiles change from day to day (default false)
  repeat_map  true for an engine that ignores EXPLORER_RL_REPEATED_MAP=0 and
           EXPLORER_RL_TILE_CHANGES=1 and sends tiles and objects in every
           view (default false)

Like the engine it sends a view whenever what the seat sees has changed, and
then the decision; told EXPLORER_RL_DECISION_OBSERVATION=0, the decision
carries no second copy of that observation. Told EXPLORER_RL_REPEATED_MAP=0, a
view whose tiles and objects are those of the seat's previous view carries
neither; told EXPLORER_RL_TILE_CHANGES=1 as well, a view that changed the map
on the same tiles carries its objects and only the tiles that changed.

Like the engine it runs in its own folder, so it finds its profile only when
the paths it was given are absolute.
"""
import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heroes.game import contract as c  # noqa: E402
from heroes.engine.protocol import RECORD_FIELDS  # noqa: E402

SCRIPT = {"days": 3, "winner": 0, "end": "natural", **json.loads(os.environ.get("FAKE_VCMI", "{}"))}
CONFIG = (Path(os.environ["VCMI_USER_DATA_DIR"]) / "config" if "VCMI_USER_DATA_DIR" in os.environ
          else Path(os.environ["XDG_CONFIG_HOME"]) / "vcmi")
EPISODE = os.environ["EXPLORER_RL_EPISODE"]
PROFILE = c.profile_for(os.environ["EXPLORER_RL_INTERFACE"])
ARMIES = os.environ.get("EXPLORER_RL_ARMY_INFORMATION")
OBSERVATION = c.Observation((2, 2, 1), (c.Tile((0, 0, 0), 1, 0, False, False),), (), (0,) * 7, (), (),
                            army_information=ARMIES, own_armies=() if ARMIES else None)
SIDE = SCRIPT.get("map")
REPEAT = SCRIPT.get("repeat", False) or os.environ.get("EXPLORER_RL_DECISION_OBSERVATION") != "0"
REPEAT_MAP = SCRIPT.get("repeat_map", False) or os.environ.get("EXPLORER_RL_REPEATED_MAP") != "0"
TILE_CHANGES = not REPEAT_MAP and os.environ.get("EXPLORER_RL_TILE_CHANGES") == "1"
STILL = SCRIPT.get("still", False)
FLICKER = SCRIPT.get("flicker", False)
TILES = SIDE and tuple(c.Tile((x, y, 0), 1 + (x * 7 + y * 3) % 5, int((x + 2 * y) % 5 == 0), (x * y) % 11 == 3, False)
                       for y in range(SIDE) for x in range(SIDE))


def observation(day, player):
    """What the seat sees on ``day``: the fixed view, or a full map with wandering objects."""
    if not SIDE:
        return OBSERVATION
    moved = 0 if STILL else day
    objects = tuple(c.ObjectView(f"object-{index}", ((index * 5 + moved * (index % 3) + player) % SIDE,
                                                      (index * 11 + (moved * (index % 4)) // 2) % SIDE, 0),
                                 10 + index % 7, index % 4, ("self", "opponent", "neutral")[index % 3])
                    for index in range(14))
    resources = (1000 * day % 20011, day % 17, day % 13, (day + player) % 11, day % 7, day % 5, day % 3)
    tiles = TILES
    if FLICKER:  # the tiles at two places that move with the day are blocked
        marked = {(day * 3 + player) % len(TILES), (day * 5 + 1) % len(TILES)}
        tiles = tuple(c.Tile(tile.position, tile.terrain, tile.road, True, tile.visitable) if place in marked else tile
                      for place, tile in enumerate(TILES))
    return c.Observation((SIDE, SIDE, 1), tiles, objects, resources, (), (),
                         army_information=ARMIES, own_armies=() if ARMIES else None)


class Seat:
    def __init__(self, player):
        self.player = player
        self.history = c.PlayerHistory(EPISODE, player, schema=PROFILE.history_schema)
        self.socket = socket.create_connection(("127.0.0.1", int(os.environ[f"EXPLORER_RL_PORT_{player}"])))
        self.reader = self.socket.makefile("rb")
        ready = {"interface": PROFILE.environment, "history_schema": PROFILE.history_schema,
                 "action_schema": PROFILE.action_schema}
        if ARMIES:
            ready["army_information"] = ARMIES
        self.send("ready", **ready)
        self.batch = []
        self.seen = None  # (observation, day) of the last view sent

    def send(self, kind, **fields):
        message = {"protocol_version": PROFILE.protocol, "type": kind, "episode": EPISODE, "player": self.player, **fields}
        self.socket.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode())

    def emit(self, kind, **payload):
        """Add one frame to this seat's history and queue its wire record."""
        frame = self.history.emit(kind, **payload)
        encoded = c.encode_frame(frame)
        record = {"kind": kind, "day": frame.day}
        for name in RECORD_FIELDS[kind]:
            record[name] = (frame.seq if kind == "decision" else frame.decision_seq) if name == "decision_id" else encoded[name]
        self.batch.append(record)
        return frame

    def flush(self):
        if self.batch:
            self.send("contextual", records=self.batch)
            self.batch = []

    def decide(self, day):
        """Offer a decision and wait for the policy's answer."""
        seen = observation(day, self.player)
        if self.seen != (seen, day):
            self.emit("view", day=day, observation=seen)
            wire = self.batch[-1]["observation"]
            if REPEAT_MAP or self.seen is None:
                pass
            elif (seen.tiles, seen.objects) == (self.seen[0].tiles, self.seen[0].objects):
                self.batch[-1]["observation"] = {name: wire[name] for name in wire if name not in ("tiles", "objects")}
            elif TILE_CHANGES and [tile.position for tile in seen.tiles] == [tile.position for tile in self.seen[0].tiles]:
                self.batch[-1]["observation"] = {name: wire[name] for name in wire if name != "tiles"} | {
                    "tile_changes": [item for item, tile, before in zip(wire["tiles"], seen.tiles, self.seen[0].tiles)
                                     if tile != before]}
            self.seen = seen, day
        offers = c.Offers.from_observation(seen, schema=PROFILE.action_schema)
        decision = self.history.emit("decision", day=day, observation=seen, offers=offers,
                                     revision=self.history.revision + 1)
        record = {"kind": "decision", "day": day, "decision_id": decision.seq, "revision": decision.revision}
        if REPEAT:
            record["observation"] = c.encode_frame(decision)["observation"]
        self.batch.append(record)
        self.flush()
        answer = json.loads(self.reader.readline())
        assert answer["type"] == "context_action" and answer["decision_id"] == decision.seq, answer
        action = c.Action(**answer["action"])
        self.history.emit("submitted", decision_seq=decision.seq, action=action)  # the policy's own frame
        self.emit("consumed", decision_seq=decision.seq, consumer="engine")
        return decision.seq


def main():
    if not (CONFIG / "settings.json").is_file() or not Path(sys.argv[sys.argv.index("--logLocation") + 1]).is_dir():
        sys.exit("Headless game start failed: no profile at " + str(CONFIG))
    if SCRIPT["end"] == "abort" or "abort_from" in SCRIPT and int(EPISODE) >= SCRIPT["abort_from"]:
        print("Headless game start failed: scripted", flush=True)
        sys.exit(134)
    seats = [Seat(0), Seat(1)]
    day = 1
    while True:
        for seat in seats:
            seq = seat.decide(day)
            last = SCRIPT["end"] == "closure" and day == SCRIPT["days"] and seat.player == 1
            if last:
                seat.emit("superseded", decision_seq=seq, reason="terminal")
            else:
                seat.emit("issued", decision_seq=seq, command_index=0)
                seat.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
                seat.emit("result", decision_seq=seq, result=c.Result("turn_ended", True))
            seat.flush()
        if SCRIPT["end"] == "crash" and day == 2:
            sys.exit(3)
        if SCRIPT["end"] == "hang" and day == 2:
            seats[0].reader.read()  # until the policy side closes the connection
            sys.exit(4)
        if SCRIPT["end"] != "endless" and day == SCRIPT["days"]:
            break
        day += 1
    for seat in seats:
        won = seat.player == SCRIPT["winner"]
        seat.emit("terminal", outcome="win" if won else "loss")
        seat.flush()
        print(f"Experiment player {seat.player} {'won' if won else 'lost'}. Terminal received.", flush=True)
    print("Experiment self-play complete. Ending game.", flush=True)
    for seat in seats:
        seat.socket.shutdown(socket.SHUT_WR)


if __name__ == "__main__":
    main()
