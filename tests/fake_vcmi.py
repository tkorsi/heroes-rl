"""A stand-in for ``vcmiclient`` that plays a scripted two-seat game.

It speaks the real wire protocol over the real sockets and writes the log lines
the environment reads, so the environment, the actor and the training loop can
be exercised without the engine. It is not VCMI: the "game" is two seats
ending their turns on a one-tile map until the script says who wins.

The script is the JSON in ``FAKE_VCMI``:
  days     full days to play before the game ends (default 3)
  winner   seat that wins, 0 or 1 (default 0)
  end      "natural" (default), "closure" (the last action is closed by the
           terminal instead of a result), "endless" (never ends), "crash"
           (exits with an error on day 2), "hang" (goes silent on day 2 until
           the policy side hangs up) or "abort" (fails before connecting)

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
        offers = c.Offers.from_observation(OBSERVATION, schema=PROFILE.action_schema)
        decision = self.history.emit("decision", day=day, observation=OBSERVATION, offers=offers,
                                     revision=self.history.revision + 1)
        self.batch.append({"kind": "decision", "day": day, "decision_id": decision.seq, "revision": decision.revision,
                           "observation": c.encode_frame(decision)["observation"]})
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
    if SCRIPT["end"] == "abort":
        print("Headless game start failed: scripted", flush=True)
        os.abort()
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
