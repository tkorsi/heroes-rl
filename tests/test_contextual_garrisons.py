"""Synthetic v8 garrison query/permission checks; no native games or PPO."""
from dataclasses import fields, replace
import json
import unittest

from experiments.learner import contextual as c
from tests.test_contextual_capabilities import action, begin, decide, navigate, observation
from tests.test_contextual_armies import army


def row(role, *, stacks=None):
    return replace(army("object", "town_garrison", "mine", stacks or {}), role=role)


def pending(*, object_kind=53, owner="self", locked=False):
    obj = c.ObjectView("mine", (2, 1, 0), object_kind, 0, "opponent" if owner == "ally" else owner)
    obs = replace(observation(), objects=(obj,), tiles=(*observation().tiles, c.Tile((2, 1, 0), 1, 0, False, True)))
    history = c.PlayerHistory("synthetic-garrison", 0, schema=c.EXPANDED_SCHEMA)
    decide(history, obs)
    seq = begin(history, action("move_to", "hero", x=2, y=1), "engine")
    history.emit("issued", decision_seq=seq, command_index=0)
    event = c.VisibleEvent("garrison_locked_requested" if locked else "garrison_requested", "hero",
                           partner="mine", object_kind=object_kind, owner=owner)
    return history, obs, seq, event


def at_garrison(*, object_kind=53, owner="self", locked=False, stacks=None):
    history, obs, seq, event = pending(object_kind=object_kind, owner=owner, locked=locked)
    history.emit("event", event=event)
    history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
    history.emit("result", decision_seq=seq, result=c.Result("interaction", True, (1, 1, 0), 100, False))
    context = c.Context("exchange", "hero", None, "mine")
    history.emit("context_opened", context=context, revision=history.revision + 1, reason="object_garrison")
    own = army("own", "hero", "hero", {0: (0, 4), 1: (2, 1)})
    role = c._object_garrison_role(owner, locked)
    shown = replace(obs, window=c.Window(context, None, armies=(own, row(role, stacks=stacks)), capability_controls=()))
    decide(history, shown)
    return history


class GarrisonContractTests(unittest.TestCase):
    def test_all_supported_native_queries_roundtrip_and_close_explicitly(self):
        cases = [(53, "self", False)] + [(kind, owner, locked) for kind in (33, 219)
                                         for owner in ("self", "ally") for locked in (False, True)]
        for kind, owner, locked in cases:
            with self.subTest(kind=kind, owner=owner, locked=locked):
                history = at_garrison(object_kind=kind, owner=owner, locked=locked)
                self.assertTrue(history.decision.offers.permits(action("close")))
                seq = begin(history, action("close"), "engine")
                history.emit("issued", decision_seq=seq, command_index=0)
                history.emit("context", decision_seq=seq, context=c.Context(), revision=history.revision + 1)
                history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
                history.emit("result", decision_seq=seq, result=c.Result("closed", True))
                restored = c.PlayerHistory(history.episode, 0, schema=c.EXPANDED_SCHEMA)
                for frame in history.frames:
                    restored.append(c.decode_frame(json.loads(json.dumps(c.encode_frame(frame)))))
                self.assertEqual(restored.digest(), history.digest())
                self.assertIsNone(restored.active_garrison)

    def test_owned_rows_allow_deposit_withdraw_reorder_and_protect_hero_last_stack(self):
        history = at_garrison(stacks={0: (0, 3), 1: (2, 2)})
        rows = history.decision.offers.armies
        self.assertEqual(c.army_operation_maximum(rows, "split_stack", "object-0", "own-0"), 3)
        self.assertEqual(c.army_operation_maximum(rows, "split_stack", "own-0", "object-0"), 4)
        self.assertEqual(c.army_operation_maximum(rows, "swap_stacks", "object-0", "object-2"), 1)
        single = replace(rows[0], slots=tuple(replace(s, creature=None, count=0) if s.slot else s for s in rows[0].slots))
        self.assertEqual(c.army_operation_maximum((single, rows[1]), "split_stack", "own-0", "object-0"), 3)
        self.assertEqual(c.army_operation_maximum((single, rows[1]), "swap_stacks", "own-0", "object-2"), 0)

    def test_locked_row_keeps_deposit_and_internal_reorder_but_never_withdraws(self):
        history = at_garrison(object_kind=33, locked=True, stacks={0: (0, 3), 1: (2, 2)})
        rows = history.decision.offers.armies
        for kind in ("split_stack", "merge_stacks", "swap_stacks"):
            self.assertEqual(c.army_operation_maximum(rows, kind, "object-0", "own-0"), 0)
        self.assertEqual(c.army_operation_maximum(rows, "swap_stacks", "own-0", "object-1"), 0)
        self.assertEqual(c.army_operation_maximum(rows, "split_stack", "own-0", "object-0"), 4)
        self.assertEqual(c.army_operation_maximum(rows, "merge_stacks", "own-0", "object-0"), 1)
        self.assertEqual(c.army_operation_maximum(rows, "swap_stacks", "own-0", "object-2"), 1)
        self.assertEqual(c.army_operation_maximum(rows, "split_stack", "object-0", "object-2"), 3)
        self.assertEqual(c.army_operation_maximum(rows, "swap_stacks", "object-0", "object-1"), 1)

    def test_allied_row_allows_gifts_but_not_taking_or_rearrangement(self):
        for locked in (False, True):
            rows = at_garrison(object_kind=219, owner="ally", locked=locked, stacks={0: (0, 3)}).decision.offers.armies
            for kind in ("split_stack", "merge_stacks", "swap_stacks"):
                self.assertEqual(c.army_operation_maximum(rows, kind, "object-0", "own-0"), 0)
                self.assertEqual(c.army_operation_maximum(rows, kind, "object-0", "object-1"), 0)
            self.assertEqual(c.army_operation_maximum(rows, "swap_stacks", "own-0", "object-0"), 0)
            self.assertEqual(c.army_operation_maximum(rows, "split_stack", "own-0", "object-0"), 4)
            self.assertEqual(c.army_operation_maximum(rows, "merge_stacks", "own-0", "object-0"), 1)
            self.assertEqual(c.army_operation_maximum(rows, "swap_stacks", "own-0", "object-1"), 1)

    def test_missing_query_or_forged_role_cannot_expose_object_army(self):
        history = at_garrison(object_kind=33, locked=True)
        shown = history.decision.observation
        # The current ownership map cannot grant a locked row removable rights.
        forged = replace(shown, window=replace(shown.window, armies=(shown.window.armies[0],
                           replace(shown.window.armies[1], role="object_garrison"))))
        history.emit("superseded", decision_seq=history.decision.seq, reason="visible_state_changed")
        with self.assertRaisesRegex(ValueError, "recorded live native query"):
            history.emit("view", observation=forged)
        history.active_garrison = None
        with self.assertRaisesRegex(ValueError, "recorded live native query"):
            history.emit("view", observation=shown)

    def test_query_must_bind_visible_kind_owner_and_current_native_action(self):
        for mutation in ({"partner": "unknown"}, {"object_kind": 33}):
            history, _, _, event = pending()
            with self.assertRaises(ValueError): history.emit("event", event=replace(event, **mutation))
        with self.assertRaises(ValueError): pending(owner="ally")
        with self.assertRaises(ValueError): pending(locked=True)
        with self.assertRaises(ValueError): pending(object_kind=98)
        history, _, _, event = pending(object_kind=33)
        with self.assertRaises(ValueError): history.emit("event", event=replace(event, owner="ally"))

    def test_deposit_receipt_replays_exact_slots_and_rejects_missing_effect(self):
        for owner, locked in (("self", False), ("self", True), ("ally", False), ("ally", True)):
            for correct in (True, False):
                with self.subTest(owner=owner, locked=locked, correct=correct):
                    history = at_garrison(object_kind=33, owner=owner, locked=locked, stacks={0: (0, 3)})
                    before = history.decision.observation
                    seq = begin(history, action("split_stack", "hero", selection="own-0", destination="object-0", count=1), "engine")
                    history.emit("issued", decision_seq=seq, command_index=0)
                    history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
                    own, target = before.window.armies
                    own = replace(own, slots=(replace(own.slots[0], count=3), *own.slots[1:]))
                    target = replace(target, slots=(replace(target.slots[0], count=4), *target.slots[1:]))
                    after = replace(before, window=replace(before.window, armies=(own, target))) if correct else before
                    history.emit("view", observation=after)
                    if not correct:
                        with self.assertRaisesRegex(ValueError, "exact displayed slot effect"):
                            history.emit("result", decision_seq=seq, result=c.Result("split", True))
                        continue
                    history.emit("result", decision_seq=seq, result=c.Result("split", True))
                    restored = c.PlayerHistory(history.episode, 0, schema=c.EXPANDED_SCHEMA)
                    for frame in history.frames:
                        restored.append(c.decode_frame(json.loads(json.dumps(c.encode_frame(frame)))))
                    self.assertEqual(restored.digest(), history.digest())

    def test_stack_inspection_is_query_bound_and_read_only(self):
        for owner in ("self", "ally"):
            history = at_garrison(object_kind=33, owner=owner, stacks={0: (0, 3)})
            shown = history.decision.observation
            other = shown.window.armies[1]
            panel = c.StackPanel(shown.context, other.slots[0], (), None, None, None)
            details = replace(shown, window=c.Window(c.Context("stack", "mine", "overview", "object-0"), None,
                              armies=(), stack_panel=panel, capability_controls=()))
            navigate(history, action("open_stack", "hero", selection="object-0"), details)
            self.assertFalse(history.decision.offers.permits(action("open_capability", "mine", panel="artifacts")))
            navigate(history, action("back", "mine"), shown, "closed")
            self.assertTrue(history.decision.offers.permits(action("close")))
            navigate(history, action("open_stack", "hero", selection="object-0"), details)
            history.active_garrison = None
            # Remove the held decision with an explicit supersession.
            history.emit("superseded", decision_seq=history.decision.seq, reason="visible_state_changed")
            with self.assertRaises(ValueError): history.emit("view", observation=details)

    def test_legacy_army_wire_fields_stay_identical(self):
        self.assertEqual(tuple(field.name for field in fields(c.ArmyView)),
                         ("ref", "role", "subject", "requires_nonempty", "slots"))
        legacy = army("old", "hero", "hero", {0: (0, 2)})
        self.assertEqual(c.army_operation_maximum((legacy,), "split_stack", "old-0", "old-1"), 2)


class GarrisonModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.threads)

    def test_unchanged_model_normalizes_legal_garrison_controls_and_reconstructs(self):
        import torch
        from experiments.learner import contextual_model as m
        from experiments.learner import interaction_guard as g
        from tests.test_contextual_management_model import all_actions
        torch.manual_seed(771)
        model = m.ContextualActorCritic(m.ModelConfig(schema=m.model_schema_for(c.EXPANDED_SCHEMA),
                                                       hidden_size=8, feature_size=4))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        for owner, locked in (("self", False), ("self", True), ("ally", False), ("ally", True)):
            history = at_garrison(object_kind=33, owner=owner, locked=locked, stacks={0: (0, 2)})
            for guard_version in (None, g.VERSION):
                encoder = m.ContextualEncoder(history.episode, history.player, schema=c.EXPANDED_SCHEMA, guard_version=guard_version)
                state = model.initial_state()
                with torch.no_grad():
                    for frame in history.frames:
                        encoded = encoder.encode(frame)
                        output = model.forward_step(encoded, state)
                        state = output.state
                    policy = output.decision.distribution
                    legal = [replace(selected, schema=c.EXPANDED_ACTION_SCHEMA)
                             for selected in all_actions(history.decision)]
                    self.assertAlmostEqual(sum(policy.log_prob(selected).exp().item() for selected in legal), 1., places=5)
                    generator = torch.Generator().manual_seed(799)
                    for _ in range(16):
                        selected, value = policy.sample_evaluated(generator)
                        self.assertIn(selected, legal)
                        self.assertTrue(history.decision.offers.permits(selected))
                        self.assertTrue(torch.equal(value.log_prob, policy.log_prob(selected)))
                    self.assertIn(action("close"), legal)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
