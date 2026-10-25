"""Pure v5 mutation/wire-binding tests; fixtures are not native engine evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.headless import contextual_replay as replay
from experiments.learner import contextual as c
from tests.test_contextual_heroes import observation, tavern, hired_observation
from tests.test_contextual_armies import army
from dataclasses import replace


HERO_CASES = {
    "tavern_gold_price", "tavern_offer_portrait", "tavern_offer_name", "tavern_selected_offer",
    "tavern_selected_level", "tavern_selected_class", "tavern_selected_artifact_count",
    "tavern_open_navigation", "tavern_hero_selection", "tavern_hire_selection", "hire_acknowledgement",
    "hired_hero_creation", "hired_hero_gold_effect", "hired_hero_position", "hired_hero_army_count",
}


def mutation_fixture():
    """Only fields consumed by the generator; never passed as a native result."""
    before = observation(tavern(selected="offer-b"))
    after = hired_observation(before)
    exchange = after.window.town_panel.armies
    visitor = army("new-army", "visiting_hero", "new-hero", {0: (0, 7)})
    after = replace(after, window=replace(after.window,
                    town_panel=replace(after.window.town_panel, armies=replace(exchange, destination=visitor))))
    def view(obs):
        return {"kind": "view", "day": 1, "observation": c.encode_observation(obs, schema=c.HEROES_SCHEMA)}
    def message(records, player=0):
        return {"kind": "message", "player": player, "message": {"records": records}}
    def response(kind, decision, **kwargs):
        return {"kind": "response", "player": 0, "response": {"decision_id": decision,
                "action": c.encode_action(c.Action(kind, "town", schema=c.HEROES_ACTION_SCHEMA, **kwargs))}}
    tape = [{"kind": "init", "schema": replay.SCHEMA, "episode": "synthetic-fixture", "interface": c.HEROES_ENVIRONMENT,
             "protocol_version": c.HEROES_PROFILE.protocol, "history_schema": c.HEROES_SCHEMA, "action_schema": c.HEROES_ACTION_SCHEMA},
            message([view(observation("overview"))]),
            response("open_panel", 1, panel="tavern"),
            message([view(observation(tavern()))]),
            response("select_hero", 2, selection="offer-b"),
            message([view(before)]),
            response("hire_hero", 3, selection="offer-b"),
            message([view(after)], player=1),  # Different player must never supply hire-effect evidence.
            message([{"kind": "acknowledged", "day": 1, "decision_id": 3, "command_index": 0, "accepted": True},
                     view(after), {"kind": "result", "day": 1, "decision_id": 3,
                                   "result": {"status": "hired", "accepted": True}}])]
    for seq, row in enumerate(tape[1:]): row["seq"] = seq
    return tape


def hero_cases(tape):
    return {name: changed for name, changed in replay.tamper_cases(tape) if name in HERO_CASES}


class HeroReplayMutationTests(unittest.TestCase):
    def test_independent_mutations_cover_only_recorded_display_and_native_effect_fields(self):
        tape = mutation_fixture(); original = deepcopy(tape)
        cases = hero_cases(tape)
        self.assertEqual(set(cases), HERO_CASES)
        self.assertEqual(tape, original)
        for name, changed in cases.items():
            self.assertNotEqual(changed, original, name)
            self.assertEqual([row["seq"] for row in changed[1:]], list(range(len(changed) - 1)))
            # All semantic mutations leave the other player's message intact.
            self.assertEqual(changed[7], tape[7], name)
            self.assertEqual(changed[0], tape[0], name)
        card = lambda changed: changed[3]["message"]["records"][0]["observation"]["window"]["town_panel"]["tavern"]
        self.assertEqual(card(cases["tavern_gold_price"])["gold_cost"], 2501)
        self.assertEqual(card(cases["tavern_selected_offer"])["selected_offer_ref"], "offer-b")
        self.assertEqual(card(cases["tavern_offer_portrait"])["offers"][0]["portrait"], 1)
        self.assertEqual(card(cases["tavern_offer_name"])["offers"][0]["name_text_id"], "hero.core.valeska.name")
        for field, case in (("level", "tavern_selected_level"), ("hero_class", "tavern_selected_class"),
                            ("artifact_count", "tavern_selected_artifact_count")):
            self.assertEqual(card(cases[case])["selected_detail"][field], card(tape)["selected_detail"][field] + 1)
        self.assertEqual(cases["tavern_hero_selection"][4]["response"]["action"]["selection"], "offer-a")
        self.assertEqual(cases["tavern_hire_selection"][6]["response"]["action"]["selection"], "offer-a")
        self.assertEqual(cases["tavern_open_navigation"][2]["response"]["action"]["panel"], "hall")
        self.assertFalse(cases["hire_acknowledgement"][8]["message"]["records"][0]["accepted"])
        after = lambda changed: changed[8]["message"]["records"][1]["observation"]
        self.assertEqual(len(after(cases["hired_hero_creation"])["heroes"]), 1)
        self.assertEqual(after(cases["hired_hero_gold_effect"])["resources"][6], 4501)
        self.assertEqual(after(cases["hired_hero_position"])["heroes"][1]["position"], [2, 1, 0])
        self.assertEqual(after(cases["hired_hero_army_count"])["window"]["town_panel"]["armies"]["destination"]["slots"][0]["count"], 8)

    def test_v6_uses_its_current_window_armies_for_hired_hero_effect(self):
        tape = mutation_fixture()
        profile = c.MANAGEMENT_PROFILE
        tape[0].update(interface=profile.environment, protocol_version=profile.protocol,
                       history_schema=profile.history_schema, action_schema=profile.action_schema)
        for row in tape:
            for record in row.get("message", {}).get("records", []):
                window = (record.get("observation") or {}).get("window")
                if window is None: continue
                old = window.get("town_panel", {}).pop("armies", None)
                window["armies"] = [army for army in (old.get("source"), old.get("destination")) if army] if old else []
        cases = hero_cases(tape)
        self.assertEqual(set(cases), HERO_CASES)
        visitor = cases["hired_hero_army_count"][8]["message"]["records"][1]["observation"]["window"]["armies"][1]
        self.assertEqual(visitor["slots"][0]["count"], 8)

    def test_new_cases_require_exact_v5_capture_profile(self):
        tape = mutation_fixture()
        for field, value in (("interface", c.ARMIES_ENVIRONMENT), ("protocol_version", 6), ("protocol_version", 7.0),
                             ("history_schema", c.ARMIES_SCHEMA), ("action_schema", c.ARMIES_ACTION_SCHEMA),
                             ("schema", "unknown")):
            changed = deepcopy(tape); changed[0][field] = value
            with self.subTest(field=field, value=value): self.assertEqual(hero_cases(changed), {})
        changed = deepcopy(tape); changed[0] = {"kind": "init"}
        self.assertEqual(hero_cases(changed), {})

    def test_rejected_or_superseded_hire_cannot_claim_creation_effect_coverage(self):
        for ending in ({"kind": "result", "decision_id": 3, "result": {"status": "rejected", "accepted": False}},
                       {"kind": "superseded", "decision_id": 3, "reason": "terminal"},
                       {"kind": "result", "decision_id": 99, "result": {"status": "hired", "accepted": True}}):
            tape = mutation_fixture(); tape[8]["message"]["records"][-1] = ending
            names = hero_cases(tape)
            self.assertIn("tavern_hire_selection", names)
            self.assertFalse(any(name.startswith("hired_") or name == "hire_acknowledgement" for name in names))
        tape = mutation_fixture(); tape.pop(8)  # Opponent-only creation is insufficient.
        self.assertFalse(any(name.startswith("hired_") for name in hero_cases(tape)))

    def test_no_witnessed_hiring_or_tavern_does_not_invent_coverage(self):
        tape = mutation_fixture()[:2]
        self.assertEqual(hero_cases(tape), {})
        tape = mutation_fixture()[:6]
        names = hero_cases(tape)
        self.assertIn("tavern_hero_selection", names)
        self.assertNotIn("tavern_hire_selection", names)
        self.assertFalse(any(name.startswith("hired_") for name in names))

    def test_other_player_and_closed_tavern_cannot_supply_action_selection_basis(self):
        tape = mutation_fixture()
        tape[5]["player"] = 1
        # Replace the last own snapshot with a closed window. A stale earlier
        # tavern must not be searched for to invent a currently displayed card.
        tape[3]["message"]["records"][0]["observation"] = c.encode_observation(observation(), schema=c.HEROES_SCHEMA)
        names = hero_cases(tape)
        self.assertNotIn("tavern_hire_selection", names)
        self.assertNotIn("tavern_hero_selection", names)
        self.assertFalse(any(name.startswith("hired_") for name in names))

    def test_wire_binding_covers_v5_profile_and_rejects_semantic_mutations(self):
        tape = mutation_fixture()
        manifest = {"episode": "synthetic-fixture", "environment": c.HEROES_ENVIRONMENT, "status": "truncated"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for player, color in enumerate(("red", "blue")):
                folder = root / "agents" / color; folder.mkdir(parents=True)
                bridge = []
                for row in tape[1:]:
                    if row.get("player") != player: continue
                    if row["kind"] == "message": bridge.append({"direction": "engine", "message": row["message"]})
                    if row["kind"] == "response": bridge.append({"direction": "policy", "message": {
                        **row["response"], "delivered": True, "outside_learner_prefix": False}})
                (folder / "bridge.jsonl").write_text("".join(json.dumps(row) + "\n" for row in bridge))
            result = replay.cross_check_wire(root, manifest, tape)
            self.assertEqual(result["responses"], 3)
            for name, changed in hero_cases(tape).items():
                with self.subTest(case=name), self.assertRaises(ValueError): replay.cross_check_wire(root, manifest, changed)
            for field, value in (("interface", c.ARMIES_ENVIRONMENT), ("protocol_version", 6), ("protocol_version", 7.0),
                                 ("history_schema", c.ARMIES_SCHEMA), ("action_schema", c.ARMIES_ACTION_SCHEMA)):
                changed = deepcopy(tape); changed[0][field] = value
                with self.subTest(header=field), self.assertRaises(ValueError): replay.cross_check_wire(root, manifest, changed)


if __name__ == "__main__": unittest.main()
