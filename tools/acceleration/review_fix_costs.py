"""Costs of two review fixes, measured over recorded games on the accelerated code.

Replays the games through the collector with typed frames (as controller_replay.py
does) to fill the observation memo, then reports: the memo's charged bytes against
an independent deep measurement of what its entries keep (tensors counted as their
data plus 450 bytes), the time of the memo charge per stored entry, and the time of
the exact-JSON-type check per decision over every delivered record.

Usage: python tools/acceleration/review_fix_costs.py [--interface NAME] EPISODE_DIR...
"""
import argparse
import json
import sys
import time
from pathlib import Path
from sys import getsizeof
sys.path.insert(0, str(Path(__file__).resolve().parent))
import controller_replay as replayer  # sets up the checkout path and torch threads
import torch
from experiments.learner import contextual as c
from experiments.learner import contextual_model as m
from experiments.learner.builtin_rollout import load_builtin_episode
from experiments.learner.contextual_rollout import _json_native
from validation.probes.validate_embedded_engine import random_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--interface", default=c.DIALOGS_ENVIRONMENT)
    args = parser.parse_args()
    loaded, _ = random_policy(args.seed, args.interface)
    replayer.TYPED = True
    for folder in args.episodes:
        replayer.replay(folder, loaded.model, loaded.checkpoint)
    store = loaded.model.__dict__["_observation_memo_state"]
    seen = {id(x) for x in (m._EXACT_CACHE, m._TRUE, m._FALSE, m._TUPLE, "node", "text", "number")}

    def kept(value):
        if (id(value) in seen or isinstance(value, type) or value is None or type(value) is bool
                or (type(value) is int and -5 <= value <= 256)):
            return 0
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size() + 450
        size = getsizeof(value)
        if type(value) in (tuple, list):
            size += sum(kept(item) for item in value)
        elif type(value) is dict:
            size += sum(kept(key) + kept(item) for key, item in value.items())
        return size

    entries = store["entries"]
    measured = sum(kept(key) + kept(entry) for key, entry in entries.items())
    begun = time.perf_counter()
    for key, entry in entries.items():
        m._observation_entry_bytes({"charged": set()}, key, entry)
    charge_ms = 1e3 * (time.perf_counter() - begun) / len(entries)
    records, decisions = [], 0
    for folder in args.episodes:
        collected = load_builtin_episode(folder / "builtin-rollout.json")
        records += [json.loads(c.canonical_frame(frame)) for frame in collected.history]
        decisions += len(collected.decisions)
    begun = time.perf_counter()
    if not all(_json_native(record) for record in records):
        raise SystemExit("a delivered record is not of exact JSON types")
    check_ms = 1e3 * (time.perf_counter() - begun) / decisions
    print(json.dumps({"observation_memo_entries": len(entries), "observation_memo_measured_bytes": measured,
                      "observation_memo_charged_bytes": store["bytes"],
                      "observation_memo_charged_over_measured": store["bytes"] / measured,
                      "observation_memo_charge_ms_per_entry": charge_ms, "delivered_records": len(records),
                      "decisions": decisions, "exact_json_type_check_ms_per_decision": check_ms}))


if __name__ == "__main__":
    main()
