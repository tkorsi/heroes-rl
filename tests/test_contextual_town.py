"""Synthetic town contract/transport tests; these are not native game evidence."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/headless"))
from experiments.learner import contextual as c
from contextual_protocol import ContextualProducer, clone_history
from env import HeadlessSelfPlayEnv
from replay import ReplayError, require_legacy_interface
from validation.probes.audit_contextual_histories import audit

NAME = "building.core.castle.dwellingLvl1.name"
DESCRIPTION = "building.core.castle.dwellingLvl1.description"
COST = (0, 0, 0, 0, 0, 0, 500)


def action(kind, **kwargs):
    return c.Action(kind, schema=c.TOWN_ACTION_SCHEMA, **kwargs)


def observation(context=None, *, status="allowed", resources=(1, 2, 3, 4, 5, 6, 7000), requirements=None):
    context = context or c.Context()
    window = None
    if context.kind == "town":
        if context.panel == "overview":
            panel = c.TownPanel("overview", (c.PanelControl("hall", "panel-0"),))
            window = c.Window(context, (), faction=0, town_panel=panel)
        elif context.panel == "hall":
            panel = c.TownPanel("hall", buildings=(c.BuildingChoice("building-local", NAME, status),))
            window = c.Window(context, None, town_panel=panel)
        else:
            detail = c.BuildingDetails("building-local", NAME, DESCRIPTION, status, COST, requirements)
            window = c.Window(context, None, town_panel=c.TownPanel("building", building=detail))
    return c.Observation((4, 4, 1), (c.Tile((1, 1, 0), 2, 0, False, False),), (), resources,
                         (c.Hero("hero", (1, 1, 0), 100),), (c.Town("town", (1, 1, 0)),), window)


def decide(history, obs=None):
    obs = obs or observation(history.context)
    return history.emit("decision", observation=obs, offers=c.Offers.from_observation(obs, schema=history.profile.action_schema),
                        revision=history.revision + 1)


def submit(history, selected):
    seq = history.decision.seq
    history.emit("submitted", action=selected, decision_seq=seq)
    history.emit("consumed", consumer="engine" if selected.kind == "construct" else "adapter", decision_seq=seq)
    return seq


def navigate(history, selected, target, *, status="opened", obs=None):
    seq = submit(history, selected)
    history.emit("context", context=target, revision=history.revision + 1, decision_seq=seq)
    obs = obs or observation(target)
    history.emit("view", observation=obs)
    history.emit("result", result=c.Result(status, True), decision_seq=seq)
    return decide(history, obs)


def building_history(player=0, *, status="allowed", requirements=None):
    history = c.PlayerHistory("town-test", player, schema=c.TOWN_SCHEMA)
    decide(history)
    navigate(history, action("open_town", subject="town"), c.Context("town", "town", "overview"))
    navigate(history, action("open_panel", subject="town", panel="hall"), c.Context("town", "town", "hall"))
    context = c.Context("town", "town", "building", "building-local")
    navigate(history, action("open_building", subject="town", selection="building-local"), context,
             obs=observation(context, status=status, requirements=requirements))
    return history


def build(history, *, accepted=True):
    seq = submit(history, action("construct", subject="town", selection="building-local"))
    history.emit("issued", decision_seq=seq, command_index=0)
    history.emit("context", context=c.Context("town", "town", "overview"), revision=history.revision + 1, decision_seq=seq)
    history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=accepted)
    history.emit("view", observation=observation(history.context, resources=(1, 2, 3, 4, 5, 6, 6500 if accepted else 7000)))
    history.emit("result", decision_seq=seq, result=c.Result("built" if accepted else "rejected", accepted))
    decide(history)


class TownContractTests(unittest.TestCase):
    def test_v2_navigation_bytes_are_unchanged(self):
        wire = []
        for window in (None, c.Window(c.Context("hero", "hero"), (c.Stack(0, 0, 25),), (1, 2, 3, 4), 10, 1),
                       c.Window(c.Context("town", "town"), (), faction=0)):
            obs = replace(observation(), window=window)
            frame = c.Frame("legacy-golden", 0, 0, 1, "decision", observation=obs,
                            offers=c.Offers.from_observation(obs), revision=0)
            wire.append(c.encode_frame(frame))
            self.assertEqual(c.decode_frame(wire[-1]), frame)
        # Captured from the unmodified 48d8e1a v2 implementation, including null fields.
        self.assertEqual(hashlib.sha256(json.dumps(wire, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                         "f37f4c6cc9b63561eff1458cdd6350784d40607b98cf8f7c00b5f6b29db28989")
        self.assertEqual(c.encode_action(c.Action("end_turn")),
                         dict(kind="end_turn", subject=None, x=None, y=None, schema=c.ACTION_SCHEMA))

    def test_roundtrip_navigation_and_build_preserves_delivered_prefix(self):
        history = building_history()
        prefix = tuple(c.encode_frame(frame) for frame in history.frames)
        build(history)
        self.assertEqual(tuple(c.encode_frame(frame) for frame in history.frames[:len(prefix)]), prefix)
        restored = c.PlayerHistory(history.episode, 0, schema=c.TOWN_SCHEMA)
        for frame in history.frames:
            restored.append(c.decode_frame(c.encode_frame(frame)))
        self.assertEqual(restored.digest(), history.digest())
        self.assertEqual(restored.context, c.Context("town", "town", "overview"))
        self.assertTrue(all(t.reward is None and not t.eligible for t in restored.transitions()))

    def test_panels_are_immutable_and_exclusive(self):
        detail = observation(c.Context("town", "town", "building", "building-local")).window.town_panel
        with self.assertRaises(FrozenInstanceError):
            detail.building.status = "forbidden"
        for invalid in (lambda: c.TownPanel("overview", building=detail.building),
                        lambda: c.TownPanel("hall", controls=(c.PanelControl("hall", "p"),)),
                        lambda: c.TownPanel("hall", buildings=[]),
                        lambda: c.Window(c.Context("town", "town", "building", "building-local"), (), faction=0, town_panel=detail),
                        lambda: c.Window(c.Context("town", "town"), (), faction=0, town_panel=detail)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): invalid()

    def test_v2_rejects_new_fields_even_when_null(self):
        obs = replace(observation(), window=c.Window(c.Context("town", "town"), (), faction=0))
        for mutate in (lambda raw: raw["window"].update(town_panel=None),
                       lambda raw: raw["window"]["context"].update(panel=None),
                       lambda raw: raw["window"]["context"].update(selection=None)):
            raw = c.encode_observation(obs)
            mutate(raw)
            with self.assertRaises(ValueError): c.decode_observation(raw)
        history = c.PlayerHistory("v2", 0)
        decide(history)
        frame = history.emit("submitted", action=c.Action("end_turn"), decision_seq=0)
        for field in ("panel", "selection", "count", "destination"):
            raw = c.encode_frame(frame); raw["action"][field] = None
            with self.assertRaises(ValueError): c.decode_frame(raw)
        with self.assertRaises(ValueError): c.Action("construct", "town", selection="building-local")
        with self.assertRaises(ValueError): c.Frame("v2", 0, 1, 1, "result", decision_seq=0, result=c.Result("built", True))

    def test_versions_do_not_mix_at_frame_history_or_finalizer(self):
        history = building_history()
        with self.assertRaises(ValueError): history.emit("submitted", action=c.Action("close"), decision_seq=history.decision.seq)
        with self.assertRaises(ValueError): c.PlayerHistory("town-test", 0).append(history.frames[0])
        with self.assertRaises(ValueError): replace(history.frames[0], schema=c.SCHEMA)
        pair = (c.PlayerHistory("mixed", 0), c.PlayerHistory("mixed", 1, schema=c.TOWN_SCHEMA))
        for item in pair: item.emit("terminal", outcome="win" if item.player == 0 else "loss")
        with self.assertRaises(ValueError):
            c.finalize_natural(pair, engine_outcomes=("win", "loss"), socket_outcomes=("win", "loss"),
                               process_exit_code=0, evidence_ref="synthetic")
        self.assertTrue(all(item.boundary is None for item in pair))

    def test_disabled_unknown_and_wrong_town_actions_have_no_history_effect(self):
        history = building_history(status="forbidden")
        before = history.digest()
        for selected in (action("construct", subject="town", selection="building-local"),
                         action("construct", subject="opponent-town", selection="building-local"),
                         action("open_building", subject="town", selection="hidden-building"), action("close")):
            self.assertFalse(history.decision.offers.permits(selected))
            with self.assertRaises(ValueError):
                history.emit("submitted", action=selected, decision_seq=history.decision.seq)
            self.assertEqual(history.digest(), before)

    def test_build_cannot_claim_success_without_one_accepted_command(self):
        history = building_history()
        seq = submit(history, action("construct", subject="town", selection="building-local"))
        with self.assertRaises(ValueError):
            history.emit("context", context=c.Context("town", "town", "overview"), revision=history.revision + 1, decision_seq=seq)
        with self.assertRaises(ValueError): history.emit("result", result=c.Result("built", True), decision_seq=seq)
        history.emit("issued", decision_seq=seq, command_index=0)
        history.emit("context", context=c.Context("town", "town", "overview"), revision=history.revision + 1, decision_seq=seq)
        history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=False)
        with self.assertRaises(ValueError): history.emit("result", result=c.Result("built", True), decision_seq=seq)
        history.emit("result", result=c.Result("rejected", False), decision_seq=seq)
        self.assertEqual(history.context.panel, "overview")

    def test_one_command_only_and_navigation_never_issues(self):
        history = building_history()
        seq = submit(history, action("back", subject="town"))
        with self.assertRaises(ValueError): history.emit("issued", decision_seq=seq, command_index=0)
        history = building_history()
        seq = submit(history, action("construct", subject="town", selection="building-local"))
        history.emit("issued", decision_seq=seq, command_index=0)
        history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
        with self.assertRaises(ValueError): history.emit("issued", decision_seq=seq, command_index=1)

    def test_back_graph_and_overlay_close_rejection(self):
        history = building_history()
        navigate(history, action("back", subject="town"), c.Context("town", "town", "hall"), status="closed")
        self.assertFalse(history.decision.offers.permits(action("close")))
        navigate(history, action("back", subject="town"), c.Context("town", "town", "overview"), status="closed")
        navigate(history, action("close"), c.Context(), status="closed")
        self.assertIsNone(history.decision.observation.window)

    def test_nested_decoder_rejects_hidden_fields_and_bounds(self):
        obs = observation(c.Context("town", "town", "building", "building-local"))
        for mutate in (lambda raw: raw["window"]["town_panel"]["building"].update(hidden_owner=1),
                       lambda raw: raw["window"]["town_panel"].update(inventory=[]),
                       lambda raw: raw["window"]["town_panel"]["building"].update(name_text_id="building.enemy.secret.name")):
            raw = c.encode_observation(obs, schema=c.TOWN_SCHEMA); mutate(raw)
            with self.assertRaises(ValueError): c.decode_observation(raw, schema=c.TOWN_SCHEMA)
        leaf = c.Requirement("building", NAME)
        expression = c.Requirement("none", children=(leaf, c.Requirement("any", children=(leaf,))))
        obs = observation(c.Context("town", "town", "building", "building-local"), status="prerequires", requirements=expression)
        self.assertEqual(c.decode_observation(c.encode_observation(obs, schema=c.TOWN_SCHEMA), schema=c.TOWN_SCHEMA), obs)
        with self.assertRaises(ValueError): c.BuildingDetails("b", NAME, DESCRIPTION, "allowed", COST, leaf)
        with self.assertRaises(ValueError): c.Requirement("all", children=[leaf])
        with self.assertRaises(ValueError):
            for _ in range(18): leaf = c.Requirement("all", children=(leaf,))

    def test_terminal_and_cutoff_rewards_keep_original_semantics(self):
        pair = tuple(building_history(player) for player in (0, 1))
        for history in pair:
            build(history)
            history.emit("superseded", decision_seq=history.decision.seq, reason="terminal")
            history.emit("terminal", outcome="win" if history.player == 0 else "loss")
        c.finalize_natural(pair, engine_outcomes=("win", "loss"), socket_outcomes=("win", "loss"), process_exit_code=0,
                           evidence_ref="synthetic-not-a-game")
        self.assertEqual([h.transitions()[-1].reward for h in pair], [1, -1])
        self.assertTrue(all(t.reward == 0 for h in pair for t in h.transitions()[:-1]))


class TownProtocolTests(unittest.TestCase):
    def producer(self, player=0):
        producer = ContextualProducer("town-test", player, interface=c.TOWN_ENVIRONMENT)
        producer.ingest(dict(protocol_version=5, type="ready", episode="town-test", player=player,
                             interface=c.TOWN_ENVIRONMENT, history_schema=c.TOWN_SCHEMA, action_schema=c.TOWN_ACTION_SCHEMA))
        return producer

    def test_handshake_versions_and_action_wire_are_strict(self):
        p = self.producer()
        p.ingest(dict(protocol_version=5, type="contextual", episode="town-test", player=0,
                      records=[dict(kind="decision", day=1, decision_id=0, revision=0,
                                    observation=c.encode_observation(observation(), schema=c.TOWN_SCHEMA))]))
        wire = p.response(action("open_town", subject="town"))
        self.assertEqual(wire["protocol_version"], 5)
        self.assertEqual(set(wire["action"]), {"schema", "kind", "subject", "x", "y", "panel", "selection", "count", "destination"})
        before = p.history.digest()
        with self.assertRaises(ValueError): p.response(c.Action("open_town", subject="town"))
        with self.assertRaises(ValueError): p.ingest(dict(protocol_version=4, type="contextual", episode="town-test", player=0, records=[]))
        self.assertEqual(p.history.digest(), before)
        self.assertEqual(clone_history(p.history).schema, c.TOWN_SCHEMA)

    def test_env_accessors_and_replay_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            town = HeadlessSelfPlayEnv(SimpleNamespace(), Path(temp), interface=c.TOWN_ENVIRONMENT)
            windows = HeadlessSelfPlayEnv(SimpleNamespace(), Path(temp), interface=c.ENVIRONMENT)
            with self.assertRaises(ValueError): town.history_v2("red")
            with self.assertRaises(ValueError): windows.history_v3("red")
        for metadata in ({"environment": c.TOWN_ENVIRONMENT}, {"interface": c.TOWN_ENVIRONMENT},
                         {"history_schema": c.TOWN_SCHEMA}, {"action_schema": c.TOWN_ACTION_SCHEMA}, {"protocol_version": 5}):
            with self.assertRaises(ReplayError): require_legacy_interface(metadata)

    def test_v4_env_requires_explicit_interface_and_legacy_replay_refuses_it(self):
        with tempfile.TemporaryDirectory() as temp:
            armies = HeadlessSelfPlayEnv(SimpleNamespace(), Path(temp), interface=c.ARMIES_ENVIRONMENT)
            town = HeadlessSelfPlayEnv(SimpleNamespace(), Path(temp), interface=c.TOWN_ENVIRONMENT)
            self.assertEqual(armies.context_profile, c.ARMIES_PROFILE)
            with self.assertRaises(ValueError): armies.history_v3("red")
            with self.assertRaises(ValueError): town.history_v4("red")
            legacy = HeadlessSelfPlayEnv(SimpleNamespace(), Path(temp))
            with self.assertRaises(ValueError): legacy.history_v4("red")
        for metadata in ({"environment": c.ARMIES_ENVIRONMENT}, {"interface": c.ARMIES_ENVIRONMENT},
                         {"history_schema": c.ARMIES_SCHEMA}, {"action_schema": c.ARMIES_ACTION_SCHEMA}, {"protocol_version": 6}):
            with self.assertRaises(ReplayError): require_legacy_interface(metadata)

    def test_v3_manifest_binding_and_mixed_pair_rejection(self):
        pair = tuple(building_history(player) for player in (0, 1))
        for history in pair: history.emit("cutoff", decision_seq=history.decision.seq, reason="synthetic-budget")
        c.finalize_cutoff(pair, evidence_ref="synthetic-not-a-game", histories_complete=(True, True), failure_free=True,
                          shutdown_verified=True, continuation_limit=8, continuation_steps=(4, 0))
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / name for name in ("red.jsonl", "blue.jsonl")]
            manifest = dict(schema_version=1, kind="contextual-headless-self-play", environment=c.TOWN_ENVIRONMENT,
                            history_schema=c.TOWN_SCHEMA, action_schema=c.TOWN_ACTION_SCHEMA, episode="town-test",
                            status="truncated", termination_reason="synthetic-budget", returncode=-15,
                            evidence_ref="synthetic-not-a-game", outcomes_by_player=dict(red=None, blue=None),
                            rewards_by_player=dict(red=None, blue=None),
                            identity=dict(engine_revision="synthetic", bridge_source_sha256="a" * 64, map_sha256="b" * 64), players={})
            for path, color, history in zip(paths, ("red", "blue"), pair):
                path.write_text("".join(json.dumps(c.encode_frame(frame)) + "\n" for frame in history.frames))
                manifest["players"][color] = {"history_v3": dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    canonical_sha256=history.digest(), prefix_sha256=history.frames[-1].cutoff_verification.history_sha256[history.player],
                    boundary="cutoff", complete=True)}
            report = audit(*paths, manifest)
            self.assertTrue(report["consistent"], report["failures"])
            self.assertTrue(report["record_contract_eligible"])
            bad = deepcopy(manifest); bad["players"]["blue"]["history_v2"] = bad["players"]["blue"]["history_v3"]
            self.assertFalse(audit(*paths, bad)["consistent"])
            bad = deepcopy(manifest); bad["history_schema"] = c.SCHEMA
            self.assertFalse(audit(*paths, bad)["consistent"])


if __name__ == "__main__":
    unittest.main()
