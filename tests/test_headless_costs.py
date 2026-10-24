"""Cost breakdown attribution on synthetic profiled episodes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("headless_cost_breakdown", ROOT / "experiments/headless/cost_breakdown.py")
cost_breakdown = importlib.util.module_from_spec(_SPEC)
sys.modules["headless_cost_breakdown"] = cost_breakdown
_SPEC.loader.exec_module(cost_breakdown)

ACTION = ("policy", {"action_id": 0, "action": 1})
CHOICE = ("policy", {"decision_id": 0, "choice": 0})
RESET = "reset (last terminal to exit, replay tail wait, runner relaunch)"


def engine(kind, events=(), request_us=None, before=None, after=None):
    message = {"type": kind, "events": [{"kind": event} for event in events], "perf": {}}
    if request_us is not None:
        message["perf"]["request_us"] = request_us
    if kind == "action_result":
        message.update(before=before or {}, after=after or {})
    return "engine", message


def write_episode(run: Path, episode: int, red=(), *, shutdown=None, waited=0.0, adapter=None) -> None:
    directory = run / "episodes" / f"episode-{episode:06d}"
    (directory / "agents" / "red").mkdir(parents=True)
    (directory / "agents" / "red" / "bridge.jsonl").write_text("".join(
        json.dumps({"t": 0.0, "direction": direction, "message": message}) + "\n" for direction, message in red))
    players = {color: {"actions_accepted": 1, "actions_rejected": 0, "battles_started": 0} for color in ("red", "blue")}
    (directory / "manifest.json").write_text(json.dumps({
        "episode": episode, "status": "truncated", "players": players,
        "timing": {"wall_seconds": 10.0, "startup_seconds": 1.0, "shutdown_seconds": shutdown,
                   "engine_wait_seconds": 2.0, "agent_seconds": 0.5, "adapter_seconds": adapter or {}},
        "replay": {"tail": {"observation_pack": 4, "replay_client_out": 4, "covered": True, "waited_seconds": waited}}}))


class CostBreakdownTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="heroes-costs-test-")
        self.run = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_battle_events_moved_into_a_decision_still_mark_the_action(self):
        write_episode(self.run, 1, [
            engine("observation"), ACTION,
            # A level-up after the fight takes the queued battle_start; the result keeps only battle_complete.
            engine("decision", ["battle_start", "decision"]), CHOICE,
            engine("action_result", ["battle_complete"], request_us=5000),
            engine("turn_end"),
            # A defensive battle during the opponent's turn belongs to no action of this side.
            engine("decision", ["battle_start", "battle_complete", "decision"]), CHOICE,
            engine("observation"), ACTION,
            engine("action_result", ["visible_update"], request_us=200)])
        native = cost_breakdown.episode_costs(self.run / "episodes" / "episode-000001")["native"]
        self.assertEqual((native["battle_request_us"], native["battle_actions"]), (5000, 1))
        self.assertEqual((native["action_request_us"], native["actions"]), (200, 1))

    def test_the_bridge_battle_counter_marks_an_action_without_events(self):
        write_episode(self.run, 1, [
            engine("observation"), ACTION,
            engine("action_result", request_us=7000, before={"completed_battles": 1}, after={"completed_battles": 2}),
            engine("observation"), ACTION,
            engine("action_result", request_us=100, before={"completed_battles": 2}, after={"completed_battles": 2})])
        native = cost_breakdown.episode_costs(self.run / "episodes" / "episode-000001")["native"]
        self.assertEqual((native["battle_request_us"], native["battle_actions"]), (7000, 1))
        self.assertEqual((native["action_request_us"], native["actions"]), (100, 1))

    def test_adapter_intervals_and_the_replay_tail_wait_are_each_counted_once(self):
        adapter = {"ingest": 0.001, "act": 0.002, "ledger": 0.004, "respond": 0.008, "decode": 0.016}
        write_episode(self.run, 1, shutdown=None, waited=0.5, adapter=adapter)  # stopped by the environment
        write_episode(self.run, 2, shutdown=1.0, waited=0.25, adapter=adapter)  # the wait lies inside shutdown
        (self.run / "summary.json").write_text(json.dumps({"episodes": [{"episode": 1, "wall_seconds": 10.5},
                                                                        {"episode": 2, "wall_seconds": 10.5}]}))
        result = cost_breakdown.breakdown(self.run)
        categories = result["categories_seconds"]
        self.assertAlmostEqual(categories[RESET], 1.0 + 0.5 + 2 * 0.5)
        self.assertAlmostEqual(categories["transport (adapter decode and response send)"], 2 * (0.008 + 0.016))
        self.assertAlmostEqual(categories["logging (adapter bridge ledger)"], 2 * 0.004)
        self.assertAlmostEqual(categories["actions (adapter history fold of submitted actions)"], 2 * 0.002)
        self.assertAlmostEqual(sum(value for key, value in categories.items() if "(adapter " in key),
                               sum(result["adapter_totals"].values()))


if __name__ == "__main__":
    unittest.main()
