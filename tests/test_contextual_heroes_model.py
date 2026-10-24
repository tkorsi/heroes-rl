"""Fixed-weight synthetic tavern policy/collector checks; no gameplay learning."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch": raise
    raise unittest.SkipTest("Optional torch dependency is not installed") from error

from experiments.learner import contextual as c
from experiments.learner import contextual_model as m
from experiments.learner.checkpoints import FrozenOpponentPool, save_policy_checkpoint, load_policy_checkpoint
from experiments.learner.contextual_rollout import ContextualSession, ContextualCollector, CollectedEpisode
from tests.test_contextual_heroes import action, decide, observation, history_at_tavern, tavern, navigate
from tests.test_contextual_armies_model import last_output
from tests.test_contextual_rollout import initial, request, resolve, finish_pair


class HeroModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(73)
        self.model = m.ContextualActorCritic(m.ModelConfig(schema=m.HEROES_MODEL_SCHEMA, hidden_size=8, feature_size=4))

    def test_new_profile_model_and_policy_versions_are_required(self):
        history = history_at_tavern()
        self.assertEqual(m.model_schema_for(c.HEROES_SCHEMA), m.HEROES_MODEL_SCHEMA)
        self.assertEqual(m.distribution_schema_for(c.HEROES_ACTION_SCHEMA), m.HEROES_DISTRIBUTION_SCHEMA)
        for schema in (m.MODEL_SCHEMA, m.ARMIES_MODEL_SCHEMA):
            old = m.ContextualActorCritic(m.ModelConfig(schema=schema, hidden_size=8, feature_size=4))
            with self.assertRaises(ValueError): last_output(old, history.frames)
            with self.assertRaises(ValueError): ContextualCollector((old, self.model), ("old", "new"),
                learner_player=0, opponent_pool="frozen", interface=c.HEROES_ENVIRONMENT)

    def test_tavern_actions_normalize_exact_joint_probability_with_seeded_sampling(self):
        for payload in (tavern(), tavern(status="visitor_present"), tavern(empty=True)):
            history = history_at_tavern(payload=payload)
            with torch.no_grad():
                distribution = last_output(self.model, history.frames).decision.distribution
                actions = [action("back", subject="town")]
                actions += [action("select_hero", subject="town", selection=card.ref) for card in payload.offers]
                if payload.recruit_status == "allowed":
                    actions.append(action("hire_hero", subject="town", selection=payload.selected_offer_ref))
                mass = 0
                for selected in actions:
                    evaluation = distribution.evaluate(selected)
                    self.assertEqual(evaluation.log_prob.item(), torch.stack([item.log_prob for item in evaluation.conditionals]).sum().item())
                    mass += evaluation.log_prob.exp().item()
                    if selected.kind == "select_hero": self.assertEqual([item.name for item in evaluation.conditionals], ["kind", "selection"])
                    if selected.kind == "hire_hero": self.assertEqual([item.name for item in evaluation.conditionals], ["kind"])
                self.assertAlmostEqual(mass, 1, places=6)
                a, b = torch.Generator().manual_seed(17), torch.Generator().manual_seed(17)
                sampled = [distribution.sample(a) for _ in range(20)]
                self.assertEqual(sampled, [distribution.sample(b) for _ in range(20)])
                self.assertTrue(all(selected in actions for selected in sampled))

    def test_opaque_rename_and_offer_reordering_do_not_change_values_or_probabilities(self):
        history = history_at_tavern()
        mapping = {key: f"opaque-{index}" for index, key in enumerate(("town", "hero", "offer-a", "offer-b", "hall", "tavern", "upper"))}
        def rename(value, field=None):
            if type(value) is dict: return {key: rename(item, key) for key, item in value.items()}
            if type(value) is list:
                result = [rename(item, field) for item in value]
                return list(reversed(result)) if field == "offers" else result
            if type(value) is str and field in ("ref", "handle", "subject", "selection", "selected_offer_ref", "hero_refs", "town_refs"):
                return mapping.get(value, value)
            return value
        changed = tuple(c.decode_frame(rename(c.encode_frame(frame))) for frame in history.frames)
        self.assertEqual(tuple(item.semantics for item in m.encode_history(history.frames)),
                         tuple(item.semantics for item in m.encode_history(changed)))
        with torch.no_grad():
            before, after = last_output(self.model, history.frames), last_output(self.model, changed)
            torch.testing.assert_close(before.state, after.state, rtol=0, atol=0)
            torch.testing.assert_close(before.decision.value, after.decision.value, rtol=0, atol=0)
            for kind, ref in (("hire_hero", "offer-a"), ("select_hero", "offer-b")):
                selected = action(kind, subject="town", selection=ref)
                renamed = replace(selected, subject=mapping["town"], selection=mapping[ref])
                torch.testing.assert_close(before.decision.distribution.log_prob(selected), after.decision.distribution.log_prob(renamed), rtol=0, atol=0)
        self.assertNotIn("offer-a", repr(m.encode_history(history.frames)[-1].semantics))

    def test_all_owned_heroes_remain_selectable_with_no_artificial_cap(self):
        heroes = tuple(c.Hero(f"own-{n}", (n % 3, n % 2, 0), n, n + 100) for n in range(17))
        history = c.PlayerHistory("synthetic-many-heroes", 0, schema=c.HEROES_SCHEMA)
        decide(history, observation(heroes=heroes))
        with torch.no_grad():
            distribution = last_output(self.model, history.frames).decision.distribution
            choices = [action("end_turn"), action("open_town", subject="town")]
            for hero in heroes:
                choices.append(action("open_hero", subject=hero.handle))
                choices += [action("move_to", subject=hero.handle, x=x, y=y) for y in range(2) for x in range(3)]
            self.assertAlmostEqual(sum(distribution.log_prob(selected).exp().item() for selected in choices), 1, places=6)
            self.assertTrue(all(torch.isfinite(distribution.log_prob(selected)).item() for selected in choices))

    def test_separate_player_recurrent_memories_preserve_explicit_selection_history(self):
        history = history_at_tavern()
        pair = tuple(ContextualSession(self.model, history.episode, player, "fixed", "opponent", schema=c.HEROES_SCHEMA)
                     for player in (0, 1))
        original = history.frames
        pair[0].consume(original)
        original_state = pair[0].state.clone()
        navigate(history, action("select_hero", subject="town", selection="offer-b"),
                 observation(tavern(selected="offer-b")), status="selected")
        pair[0].consume(history.frames[len(original):])
        with torch.no_grad(): expected = last_output(self.model, history.frames)
        torch.testing.assert_close(pair[0].state, expected.state, rtol=0, atol=0)
        torch.testing.assert_close(pair[1].state, self.model.initial_state(), rtol=0, atol=0)
        self.assertFalse(torch.equal(original_state, pair[0].state))
        selected = pair[0].choose(history.decision.seq)
        self.assertEqual(selected.log_prob, expected.decision.distribution.log_prob(selected.action).item())

    def test_diagnostic_collector_roundtrip_records_v5_support_and_remains_ineligible(self):
        collector = ContextualCollector((self.model, self.model), ("fixed-a", "fixed-b"), learner_player=0,
            opponent_pool="fixed-pool", interface=c.HEROES_ENVIRONMENT, purpose="diagnostic", origin="synthetic")
        collector.start("synthetic-collector")
        pair = tuple(initial(player, schema=c.HEROES_SCHEMA) for player in (0, 1))
        for history in pair: resolve(history, collector.act(request(history)))
        artifact = collector.finish(finish_pair(pair, "natural"), histories=tuple(history.frames for history in pair))
        self.assertEqual(CollectedEpisode.from_dict(json.loads(json.dumps(artifact.to_dict()))), artifact)
        self.assertFalse(any(rollout.optimization_eligible for rollout in artifact.rollouts))
        for rollout in artifact.rollouts:
            self.assertEqual(rollout.decisions[0].support.distribution_schema, m.HEROES_DISTRIBUTION_SCHEMA)

    def test_frozen_checkpoint_roundtrip_is_versioned_and_does_not_consume_rng(self):
        pool = FrozenOpponentPool(); pool.add("fixed-opponent", self.model)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "v5.pt"
            save_policy_checkpoint(path, self.model, checkpoint="fixed-policy", source_metadata={"purpose": "synthetic"},
                                   opponent_pool=pool, interface=c.HEROES_ENVIRONMENT)
            state = torch.get_rng_state().clone()
            loaded = load_policy_checkpoint(path, interface=c.HEROES_ENVIRONMENT)
            torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
            self.assertEqual(loaded.weights_sha256, m.weights_sha256(self.model))
            self.assertEqual(loaded.opponent_pool.identities, pool.identities)
            with self.assertRaises(ValueError): load_policy_checkpoint(path, interface=c.ARMIES_ENVIRONMENT)
            malformed = deepcopy(torch.load(path, weights_only=True)); malformed["model_config"]["schema"] = m.ARMIES_MODEL_SCHEMA
            torch.save(malformed, Path(temp) / "wrong.pt")
            with self.assertRaises(ValueError): load_policy_checkpoint(Path(temp) / "wrong.pt", interface=c.HEROES_ENVIRONMENT)


if __name__ == "__main__": unittest.main()
