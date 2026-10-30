"""Actual C++ parity for temporary army support, using disposable typed fixtures.

Set HEROES_TEST_NATIVE_MANIFEST explicitly. These tests neither build a backend
nor start VCMI, reserve game seeds, create an optimizer or admit experience.
"""
from dataclasses import replace
import os
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("Optional PyTorch is not installed")

from experiments.learner import contextual as c
from experiments.learner import contextual_model as m
from experiments.learner import interaction_guard as g
from experiments.learner.native_backend import load_backend, sha256
from experiments.learner.native_recipes import compile_history
from tests.test_contextual_armies import history_at_armies, selected as transfer
from tests.test_contextual_capabilities import action, decide, navigate, observation
from tests.test_contextual_dialogs import history_at as dialog_history, panel as dialog_panel
from tests.test_contextual_garrisons import at_garrison
from tests.test_contextual_management import history_at, at_stack, at_confirmation
from tests.test_contextual_model import recruitment, town_history


MANIFEST = os.environ.get("HEROES_TEST_NATIVE_MANIFEST")


def expanded_history(kind="map", player=0):
    history = c.PlayerHistory("synthetic-recruit-only-native", player, schema=c.EXPANDED_SCHEMA)
    base = observation()
    base = replace(base, tiles=(*base.tiles, c.Tile((3, 2, 0), 1, 0, False, False)))
    decide(history, base)
    if kind != "map":
        opened = replace(observation(kind), tiles=base.tiles)
        navigate(history, action("open_hero" if kind == "hero" else "open_town",
                                 "hero" if kind == "hero" else "town"), opened)
    return history


@unittest.skipUnless(MANIFEST, "Requires an explicitly built native learner manifest")
class RecruitOnlyNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.backend = load_backend(MANIFEST, sha256(MANIFEST))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self, history):
        with torch.random.fork_rng():
            torch.manual_seed(1974031)
            model = m.ContextualActorCritic(m.ModelConfig(
                schema=m.model_schema_for(history.schema), hidden_size=8, feature_size=4))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 15460)
        return model

    def reference(self, history, model, version):
        inputs = m.encode_history(history.frames, guard_version=version)
        state = model.initial_state()
        with torch.no_grad():
            for frame in inputs:
                output = model.forward_step(frame, state)
                state = output.state
        return inputs, output

    def assert_exact(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def parity(self, history, actions, *, version=g.RECRUIT_ONLY_VERSION):
        model = self.model(history)
        before = m.weights_sha256(model)
        rng = torch.get_rng_state().clone()
        inputs, expected = self.reference(history, model, version)
        distribution = expected.decision.distribution
        self.assertTrue(all(history.decision.offers.permits(a) for a in actions))
        self.assertFalse(set(distribution._kinds()) & g.RECRUIT_ONLY_BLOCKED_KINDS
                         if version == g.RECRUIT_ONLY_VERSION else False)
        native = self.backend.NativeModel(dict(model.named_parameters()), 4, 8)
        try:
            for selected in actions:
                with self.subTest(action=selected):
                    evaluated = distribution.evaluate(selected)
                    record = SimpleNamespace(decision_seq=history.decision.seq, action=selected,
                        support=distribution.support, conditionals=evaluated.conditionals)
                    compiled = compile_history(model, inputs, (record,), self.backend)
                    with torch.no_grad():
                        actual = native.evaluate_window_details(compiled, 0, record.decision_seq,
                            [record.decision_seq], model.initial_state())
                        self.assert_exact(actual[0], evaluated.log_prob.reshape(1))
                        self.assert_exact(actual[1], expected.decision.value.reshape(1))
                        self.assert_exact(actual[2], expected.state)
                        self.assertEqual(len(actual[3]), 1)
                        self.assert_exact(actual[3][0], torch.stack([v.log_prob for v in evaluated.conditionals]))
                        # The real C++ prefix path also retains every frame and
                        # stays exact when independent prefix work is parallel.
                        for workers in (1, 4):
                            stop = len(inputs)
                            state = native.prefix(compiled, [stop], workers)[stop]
                            self.assert_exact(state, expected.state)
            first, second = torch.Generator().manual_seed(19), torch.Generator().manual_seed(19)
            for _ in range(24):
                self.assertEqual(distribution.sample(first), distribution.sample(second))
            self.assertTrue(torch.equal(first.get_state(), second.get_state()))
        finally:
            native.close()
        self.assertEqual(m.weights_sha256(model), before)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        return distribution

    def test_both_seats_keep_exact_map_and_army_window_probabilities(self):
        for player in (0, 1):
            for kind, actions in (
                ("map", [action("end_turn"), action("open_hero", "hero"), action("open_town", "town"),
                         action("move_to", "hero", x=3, y=2)]),
                ("hero", [action("close"), action("open_capability", "hero", panel="artifacts"),
                          action("open_capability", "hero", panel="spellbook")]),
                ("town", [action("close"), action("open_panel", "town", panel="hall"),
                          action("swap_garrison", "town"),
                          action("open_capability", "town", panel="marketplace"),
                          action("open_capability", "town", panel="purchases")]),
            ):
                with self.subTest(player=player, context=kind):
                    policy = self.parity(expanded_history(kind, player), actions)
                    if kind != "map":
                        self.assertAlmostEqual(sum(policy.log_prob(a).exp().item() for a in actions), 1., places=6)

    def test_recruitment_counts_remain_exact_in_both_versions_and_cpp(self):
        history = town_history(recruitment())
        actions = [c.Action("back", "town", schema=c.TOWN_ACTION_SCHEMA)]
        actions += [c.Action("select_recruit", "town", selection=ref, schema=c.TOWN_ACTION_SCHEMA)
                    for ref in ("base", "upgrade")]
        actions += [c.Action("recruit", "town", selection="base", destination="upper", count=count,
                             schema=c.TOWN_ACTION_SCHEMA) for count in (1, 2, 3)]
        old = self.parity(history, actions, version=g.REVEALED_VERSION)
        new = self.parity(history, actions)
        for selected in actions:
            self.assert_exact(old.log_prob(selected), new.log_prob(selected))
        self.assertAlmostEqual(sum(new.log_prob(a).exp().item() for a in actions), 1., places=6)

    def test_passive_owned_army_statistics_reach_actual_cpp_exactly(self):
        from tests.test_passive_army_information import informed_observation
        # Each fixture is an independent first map decision. No engine run or
        # fabricated admission is needed to compare the complete tensor path.
        for player in (0, 1):
            base = informed_observation()
            changed = replace(base.own_armies[0], slots=(replace(base.own_armies[0].slots[0],
                                                              attack=12, speed=7, count=21),))
            for variant, obs in (("initial", base), ("changed", replace(base,
                                  own_armies=(changed, *base.own_armies[1:])))):
                with self.subTest(player=player, variant=variant):
                    history = c.PlayerHistory("synthetic-passive-army-native", player, schema=c.EXPANDED_SCHEMA)
                    decide(history, obs)
                    self.parity(history, [action("end_turn"), action("open_hero", "hero"),
                                          action("open_town", "town")])

    def test_native_garrison_query_fixture_keeps_explicit_close(self):
        for owner in ("self", "ally"):
            for locked in (False, True):
                with self.subTest(owner=owner, locked=locked):
                    history = at_garrison(object_kind=33, owner=owner, locked=locked,
                                          stacks={0: (0, 3), 1: (2, 2)})
                    policy = self.parity(history, [action("close")])
                    self.assertEqual(policy._kinds(), ("close",))
                    self.assertEqual(policy.log_prob(action("close")).item(), 0.)

    def test_mandatory_dialog_replies_remain_distinct_policy_choices(self):
        for family in ("neutral_paid_join", "chest"):
            with self.subTest(family=family):
                history = dialog_history(dialog_panel(family))
                actions = [c.Action("choose_dialog", "hero", selection=choice.ref,
                                    schema=c.DIALOGS_ACTION_SCHEMA)
                           for choice in history.decision.offers.dialog_panel.choices]
                policy = self.parity(history, actions)
                self.assertEqual(len(actions), 2)
                self.assertAlmostEqual(sum(policy.log_prob(a).exp().item() for a in actions), 1., places=6)

    def test_forbidden_actions_cannot_evaluate_or_compile_but_historical_version_can(self):
        cases = [(history_at(), c.Action(kind, "town", selection="upper-0", destination="lower-0",
                   schema=c.MANAGEMENT_ACTION_SCHEMA, **({"count": 1} if kind == "split_stack" else {})))
                 for kind in ("split_stack", "merge_stacks", "swap_stacks")]
        cases += [(history_at(), c.Action("open_stack", "town", selection="upper-0", schema=c.MANAGEMENT_ACTION_SCHEMA)),
                  (history_at_armies(), transfer())]
        for history, selected in cases:
            with self.subTest(kind=selected.kind):
                model = self.model(history)
                old_inputs, old = self.reference(history, model, g.REVEALED_VERSION)
                inputs, new = self.reference(history, model, g.RECRUIT_ONLY_VERSION)
                evaluated = old.decision.distribution.evaluate(selected)
                # Old behavior remains replayable by the actual C++ binary.
                self.parity(history, [selected], version=g.REVEALED_VERSION)
                with self.assertRaisesRegex(ValueError, "outside this exact behavior support"):
                    new.decision.distribution.evaluate(selected)
                record = SimpleNamespace(decision_seq=history.decision.seq, action=selected,
                    support=new.decision.distribution.support, conditionals=evaluated.conditionals)
                with self.assertRaisesRegex(ValueError, "outside this exact behavior support"):
                    compile_history(model, inputs, (record,), self.backend)
                record.support = old.decision.distribution.support
                with self.assertRaisesRegex(ValueError, "support differs from recorded behavior"):
                    compile_history(model, inputs, (record,), self.backend)

    def test_stack_upgrade_recipes_are_rejected_while_escape_remains_exact(self):
        for history in (at_stack(), at_confirmation()):
            with self.subTest(panel=history.context.panel):
                model = self.model(history)
                _, old = self.reference(history, model, g.REVEALED_VERSION)
                inputs, new = self.reference(history, model, g.RECRUIT_ONLY_VERSION)
                panel = history.decision.offers.stack_panel
                selected = c.Action("select_upgrade" if panel.selected_upgrade_ref is None else "upgrade_stack",
                    history.context.subject, schema=c.MANAGEMENT_ACTION_SCHEMA,
                    selection=panel.offers[0].ref if panel.selected_upgrade_ref is None else panel.selected_upgrade_ref)
                evaluated = old.decision.distribution.evaluate(selected)
                self.parity(history, [selected], version=g.REVEALED_VERSION)
                with self.assertRaisesRegex(ValueError, "outside this exact behavior support"):
                    new.decision.distribution.evaluate(selected)
                record = SimpleNamespace(decision_seq=history.decision.seq, action=selected,
                    support=new.decision.distribution.support, conditionals=evaluated.conditionals)
                with self.assertRaisesRegex(ValueError, "outside this exact behavior support"):
                    compile_history(model, inputs, (record,), self.backend)
                # Parsing historical frames does not claim they were collected
                # under this guard; admission checks their recorded support.
                back = c.Action("back", history.context.subject, schema=c.MANAGEMENT_ACTION_SCHEMA)
                self.parity(history, [back])


if __name__ == "__main__":
    unittest.main()
