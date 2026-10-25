"""Fixed-weight v7 policy/value checks. No optimization or native-game claims."""
from dataclasses import replace
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
from tests.test_contextual_dialogs import FAMILIES, panel, action, observation, history_at, begin_answer, finish_answer, open_dialog, decide
from tests.test_contextual_management_model import last_output
from tests.test_contextual_management import history_at as legacy_history


class DialogModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(172)
        self.model = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA,hidden_size=8,feature_size=4))

    def test_explicit_v7_model_and_checkpoint_binding(self):
        self.assertEqual(m.model_schema_for(c.DIALOGS_SCHEMA),m.DIALOGS_MODEL_SCHEMA)
        self.assertEqual(m.distribution_schema_for(c.DIALOGS_ACTION_SCHEMA),m.DIALOGS_DISTRIBUTION_SCHEMA)
        old = m.ContextualActorCritic(m.ModelConfig(schema=m.MANAGEMENT_MODEL_SCHEMA,hidden_size=8,feature_size=4))
        with self.assertRaises(ValueError): last_output(old,history_at().frames)
        with self.assertRaises(ValueError): last_output(self.model,legacy_history().frames)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"fixed.pt"
            save_policy_checkpoint(path,self.model,checkpoint="synthetic-v7",source_metadata={"purpose":"fixed_weight_test"},interface=c.DIALOGS_ENVIRONMENT)
            restored = load_policy_checkpoint(path,interface=c.DIALOGS_ENVIRONMENT)
            self.assertEqual(m.weights_sha256(restored.model),m.weights_sha256(self.model))
            with self.assertRaises(ValueError): load_policy_checkpoint(path,interface=c.MANAGEMENT_ENVIRONMENT)

    def test_every_dialog_joint_distribution_normalizes_and_samples_legal_choices(self):
        identity = m.weights_sha256(self.model)
        for family in FAMILIES:
            with self.subTest(family=family), torch.no_grad():
                history = history_at(panel(family))
                output = last_output(self.model,history.frames)
                distribution = output.decision.distribution
                choices = [action(selection=choice.ref) for choice in history.decision.offers.dialog_panel.choices]
                total = 0
                for selected in choices:
                    result = distribution.evaluate(selected)
                    self.assertEqual([term.name for term in result.conditionals],["kind","dialog_choice"])
                    self.assertAlmostEqual(result.log_prob.item(),sum(term.log_prob.item() for term in result.conditionals),places=6)
                    total += result.log_prob.exp().item()
                self.assertAlmostEqual(total,1,places=6)
                one,two = torch.Generator().manual_seed(14),torch.Generator().manual_seed(14)
                samples = [distribution.sample(one) for _ in range(20)]
                self.assertEqual(samples,[distribution.sample(two) for _ in range(20)])
                self.assertTrue(all(selected in choices for selected in samples))
                self.assertTrue(torch.isfinite(output.decision.value))
                with self.assertRaises(ValueError): distribution.log_prob(action(selection="stale"))
        self.assertEqual(m.weights_sha256(self.model),identity)

    def test_opaque_ids_and_choice_ordinals_are_not_model_features(self):
        history = history_at(panel("school_magic"))
        def rename(value,key=None):
            if type(value) is dict: return {name:rename(item,name) for name,item in value.items()}
            if type(value) is list: return [rename(item,key) for item in value]
            if type(value) is str and key in ("ref","handle","subject","selection","hero_refs","town_refs","movable_hero_refs"):
                return "opaque-renamed-"+value
            return value
        renamed = tuple(c.decode_frame(rename(c.encode_frame(frame))) for frame in history.frames)
        with torch.no_grad():
            before,after = last_output(self.model,history.frames),last_output(self.model,renamed)
            torch.testing.assert_close(before.state,after.state,rtol=0,atol=0)
            torch.testing.assert_close(before.decision.value,after.decision.value,rtol=0,atol=0)
            for choice in history.decision.offers.dialog_panel.choices:
                torch.testing.assert_close(before.decision.distribution.log_prob(action(selection=choice.ref)),
                    after.decision.distribution.log_prob(action(subject="opaque-renamed-hero",selection="opaque-renamed-"+choice.ref)),rtol=0,atol=0)
        encoded = repr(m.encode_history(history.frames))
        # Inspect semantic trees, not attached bookkeeping frames.
        semantic = repr(tuple(item.semantics for item in m.encode_history(history.frames)))
        self.assertNotIn("dialog-1",semantic)
        self.assertNotIn("component_index",semantic)
        self.assertIn("spellpower",semantic)
        with self.assertRaises(ValueError): m._semantic(panel().choices[0])

    def test_actual_displayed_value_changes_state_but_old_absent_fields_do_not(self):
        original = panel()
        altered = replace(original,components=(c.DialogComponent("resource","gold",1500),c.DialogComponent("experience",amount=1000)))
        with torch.no_grad():
            before = last_output(self.model,history_at(original).frames)
            after = last_output(self.model,history_at(altered).frames)
        self.assertFalse(torch.equal(before.state,after.state))
        legacy = legacy_history("hero")
        tree = repr(m._semantic(legacy.decision.observation.window))
        self.assertNotIn("dialog_panel",tree)
        event = c.VisibleEvent("visit","hero",object_kind=1,object_subtype=0,owner="neutral")
        self.assertNotIn("dialog",repr(m._semantic(event)))
        component = c.NoticeComponent("resource",resource="gold",amount=1000)
        self.assertNotIn("skill",repr(m._semantic(component)))

    def test_chained_prefix_reconstruction_matches_incremental_memory(self):
        history = history_at(panel("neutral_free_join"))
        seq = begin_answer(history,1)
        second = panel("neutral_flee","dialog-2")
        history.emit("event",event=c.VisibleEvent("dialog_requested","hero",dialog=second))
        finish_answer(history,seq); open_dialog(history,second)
        codec = m.ContextualEncoder(history.episode,history.player,c.DIALOGS_SCHEMA)
        state = self.model.initial_state()
        with torch.no_grad():
            for frame in history.frames:
                encoded = codec.encode(frame)
                output = self.model.forward_step(encoded,state)
                state = output.state
            folded = last_output(self.model,history.frames)
        torch.testing.assert_close(folded.state,state,rtol=0,atol=0)
        self.assertEqual(codec.history.dialog_refs,{"dialog-1","dialog-1-accept","dialog-1-decline",
                                                   "dialog-2","dialog-2-pursue","dialog-2-let_flee"})
        self.assertEqual(codec.history.digest(),history.digest())
        torch.testing.assert_close(folded.decision.value,output.decision.value,rtol=0,atol=0)


if __name__ == "__main__": unittest.main()
