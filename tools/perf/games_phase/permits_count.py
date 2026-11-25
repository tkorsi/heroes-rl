import sys, runpy, atexit, collections, os
tree = sys.argv[sys.argv.index("--tree") + 1]
sys.path.insert(0, tree)
from heroes.game import guard
counts = collections.Counter()
original = guard.GuardSupport.permits
def counting(self, action, observation, offers):
    result = original(self, action, observation, offers)
    blocked = bool(self.blocked_management_subjects)
    counts[(action.kind, result, blocked)] += 1
    return result
guard.GuardSupport.permits = counting
atexit.register(lambda: print(sum(counts.values()), counts.most_common(14), file=sys.stderr))
G = os.path.expanduser("~/Projects/Heroes-artifacts/claude-g10")
sys.argv = [f"{G}/scripts/actor_replay.py"] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
