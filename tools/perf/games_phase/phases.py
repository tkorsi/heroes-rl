"""Where the actor's time goes in a replay, without a profiler: wall time inside ingest, observe and act.
    python phases.py TREE REPLAY"""
import sys, time, runpy
tree, replay = sys.argv[1], sys.argv[2]
sys.path.insert(0, tree)
from heroes import actor
from heroes.engine import protocol
totals = {"ingest": 0.0, "observe": 0.0, "act": 0.0}
def timed(owner, name, key):
    original = getattr(owner, name)
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return original(*a, **k)
        finally:
            totals[key] += time.perf_counter() - t
    setattr(owner, name, wrapper)
timed(protocol.Producer, "ingest", "ingest"); timed(actor.Seat, "observe", "observe"); timed(actor.Seat, "act", "act")
sys.argv = [f"{sys.path[0]}/../scripts/actor_replay.py", "replay", f"{tree}/../prof", replay, "--tree", tree]
import os
G = os.path.expanduser("~/Projects/Heroes-artifacts/claude-g10")
sys.argv = [f"{G}/scripts/actor_replay.py", "replay", f"{G}/prof", replay, "--tree", tree]
runpy.run_path(f"{G}/scripts/actor_replay.py", run_name="__main__")
print({k: round(v, 2) for k, v in totals.items()}, file=sys.stderr)
