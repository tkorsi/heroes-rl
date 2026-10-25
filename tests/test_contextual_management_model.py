"""Fixed-weight v6 policy checks; no optimization, games or simulated engine proof."""
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch": raise
    raise unittest.SkipTest("Optional torch is not installed") from error

from experiments.learner import contextual as c
from experiments.learner import contextual_model as m
from experiments.learner.checkpoints import save_policy_checkpoint, load_policy_checkpoint
from experiments.learner.contextual_rollout import ContextualCollector
from tests.test_contextual_management import action, army, observation, history_at, at_stack, at_confirmation, pending_exchange, complete_move, decide


def last_output(model, frames):
    state = model.initial_state()
    for encoded in m.encode_history(frames):
        output = model.forward_step(encoded, state)
        state = output.state
    return output


def all_actions(frame):
    offers, context = frame.offers, frame.observation.context
    if context.kind == "stack":
        actions = [action("back", context.subject)]
        if context.panel == "overview":
            actions += [action("select_upgrade", context.subject, selection=choice.ref) for choice in offers.stack_panel.offers]
        elif offers.stack_panel.affordable:
            actions.append(action("upgrade_stack", context.subject, selection=offers.stack_panel.selected_upgrade_ref))
        return actions
    actions = [action("close")]
    if offers.panel:
        actions += [action("open_panel", context.subject, panel=item.panel, selection=item.ref if item.panel == "recruitment" else None) for item in offers.panel.controls]
    for row in offers.armies:
        actions += [action("open_stack", context.subject, selection=slot.ref) for slot in row.slots if slot.count]
    if context.kind == "town" and any(row.role in ("garrison_hero", "visiting_hero") for row in offers.armies):
        actions.append(action("swap_garrison", context.subject))
    for kind in ("split_stack", "merge_stacks", "swap_stacks"):
        for source, destination, maximum in c.army_operation_choices(offers.armies, kind):
            if kind == "split_stack":
                actions += [action(kind, context.subject, selection=source, destination=destination, count=count) for count in range(1, maximum+1)]
            else:
                actions.append(action(kind, context.subject, selection=source, destination=destination))
    return actions


class ManagementModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(131)
        self.model = m.ContextualActorCritic(m.ModelConfig(schema=m.MANAGEMENT_MODEL_SCHEMA, hidden_size=8, feature_size=4))

    def test_versions_are_explicit_and_checkpoints_roundtrip(self):
        self.assertEqual(m.model_schema_for(c.MANAGEMENT_SCHEMA), m.MANAGEMENT_MODEL_SCHEMA)
        self.assertEqual(m.distribution_schema_for(c.MANAGEMENT_ACTION_SCHEMA), m.MANAGEMENT_DISTRIBUTION_SCHEMA)
        for schema in (m.MODEL_SCHEMA, m.ARMIES_MODEL_SCHEMA, m.HEROES_MODEL_SCHEMA):
            old = m.ContextualActorCritic(m.ModelConfig(schema=schema, hidden_size=8, feature_size=4))
            with self.assertRaises(ValueError): last_output(old, history_at().frames)
            with self.assertRaises(ValueError): ContextualCollector((old, self.model), ("old", "new"), learner_player=0,
                opponent_pool="pool", interface=c.MANAGEMENT_ENVIRONMENT)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"policy.pt"
            save_policy_checkpoint(path, self.model, checkpoint="fixed-v6", source_metadata={"purpose":"synthetic"}, interface=c.MANAGEMENT_ENVIRONMENT)
            restored = load_policy_checkpoint(path, interface=c.MANAGEMENT_ENVIRONMENT)
            self.assertEqual(m.weights_sha256(restored.model), m.weights_sha256(self.model))

    def test_every_context_joint_support_normalizes_and_seeded_samples_are_legal(self):
        exchange, seq = pending_exchange(); complete_move(exchange, seq)
        exchange.emit("context_opened", context=observation("exchange").context, revision=exchange.revision+1, reason="own_hero_exchange")
        decide(exchange, observation("exchange"))
        for history in (history_at(), history_at("hero"), at_stack(), at_confirmation(), at_confirmation(gold=10), exchange):
            with self.subTest(context=history.context), torch.no_grad():
                distribution = last_output(self.model, history.frames).decision.distribution
                choices = all_actions(history.decision)
                total = 0
                for selected in choices:
                    evaluation = distribution.evaluate(selected)
                    self.assertAlmostEqual(evaluation.log_prob.item(), sum(item.log_prob.item() for item in evaluation.conditionals), places=5)
                    total += evaluation.log_prob.exp().item()
                self.assertAlmostEqual(total, 1, places=5)
                self.assertEqual(distribution.support.distribution_schema, m.MANAGEMENT_DISTRIBUTION_SCHEMA)
                first, second = torch.Generator().manual_seed(13), torch.Generator().manual_seed(13)
                samples = [distribution.sample(first) for _ in range(20)]
                self.assertEqual(samples, [distribution.sample(second) for _ in range(20)])
                self.assertTrue(all(item in choices for item in samples))

    def test_int32_split_has_bounded_probability_path(self):
        history = history_at("hero", armies=(army("own", "hero", "hero", {0: (0, 2**31-1)}),))
        with torch.no_grad():
            distribution = last_output(self.model, history.frames).decision.distribution
            for count in (1, 2**30, 2**31-1):
                selected = action("split_stack", "hero", selection="own-0", destination="own-6", count=count)
                evaluated = distribution.evaluate(selected)
                self.assertTrue(math.isfinite(evaluated.log_prob.item()))
                self.assertLessEqual(sum(part.name.startswith("count:") for part in evaluated.conditionals), 31)

    def test_opaque_reference_renaming_preserves_exact_values_and_joint_probabilities(self):
        for history, selected in ((history_at(), action("swap_stacks", "town", selection="upper-0", destination="lower-0")),
                                  (at_confirmation(), action("upgrade_stack", "hero", selection="upgrade-1"))):
            def renamed(value, key=None):
                if type(value) is dict: return {k: renamed(v, k) for k, v in value.items()}
                if type(value) is list: return [renamed(v, key) for v in value]
                if type(value) is str and key in ("ref", "handle", "subject", "selection", "destination", "partner", "selected_upgrade_ref",
                                                   "selected_offer_ref", "hero_refs", "town_refs", "movable_hero_refs"):
                    return "opaque-"+value
                return value
            frames = tuple(c.decode_frame(renamed(c.encode_frame(frame))) for frame in history.frames)
            changed_action = c.Action(**renamed(c.encode_action(selected)))
            with torch.no_grad():
                before, after = last_output(self.model, history.frames), last_output(self.model, frames)
                torch.testing.assert_close(before.state, after.state, rtol=0, atol=0)
                torch.testing.assert_close(before.decision.value, after.decision.value, rtol=0, atol=0)
                torch.testing.assert_close(before.decision.distribution.log_prob(selected), after.decision.distribution.log_prob(changed_action), rtol=0, atol=0)

    def test_garrison_flag_changes_movement_support_without_removing_owned_hero(self):
        obs = observation(heroes=(c.Hero("hero", (1,1,0), 100, 100, True),))
        history = c.PlayerHistory("garrison-model", 0, schema=c.MANAGEMENT_SCHEMA); decide(history, obs)
        with torch.no_grad():
            distribution = last_output(self.model, history.frames).decision.distribution
            self.assertNotIn("move_to", distribution._kinds())
            self.assertTrue(math.isfinite(distribution.log_prob(action("open_hero", "hero")).item()))
            with self.assertRaises(ValueError): distribution.log_prob(action("move_to", "hero", x=1, y=1))

    def test_exchange_partner_has_semantic_binding_without_encoding_the_handle(self):
        history, seq = pending_exchange(); complete_move(history, seq)
        history.emit("context_opened", context=observation("exchange").context, revision=history.revision+1, reason="own_hero_exchange")
        encoded = m.encode_history(history.frames)[-1]
        self.assertIn("prior_observation_partner", repr(encoded.semantics))
        self.assertNotIn("other", repr(encoded.semantics))
        self.assertNotIn("own_hero_exchange", repr(encoded.semantics))  # transport reason is metadata


if __name__ == "__main__": unittest.main()
