"""Synthetic transfer policy/collector/optimizer checks, never native training."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Optional torch dependency is not installed") from error

from experiments.learner import contextual as c
from experiments.learner.contextual_model import (
    ARMIES_MODEL_SCHEMA, ARMIES_DISTRIBUTION_SCHEMA, MODEL_SCHEMA,
    ContextualActorCritic, ModelConfig, encode_history, weights_sha256,
)
from experiments.learner.contextual_rollout import (
    BehaviorDecision, ConditionalRecord, ContextualRollout, DecisionEstimate,
    RecurrentPrefix, RolloutSample, ContextualSession, ContextualCollector, CollectedEpisode,
)
from experiments.learner.checkpoints import FrozenOpponentPool, save_policy_checkpoint, load_policy_checkpoint
from experiments.learner.ppo import PPOConfig, ppo_update, _admit
from tests.test_contextual_armies import (
    army, exchange, observation, history_at_armies, selected, begin_transfer, after_transfer,
)
from tests.test_contextual_rollout import initial, request, resolve, finish_pair


def last_output(model, frames):
    state = model.initial_state()
    for encoded in encode_history(frames):
        output = model.forward_step(encoded, state)
        state = output.state
    return output


def transfer_rollout(model, player=0):
    """Two explicit typed transfer examples with synthetic paired terminal proof."""
    pair = tuple(history_at_armies(seat) for seat in (0, 1))
    for history in pair:
        before = history.decision.observation
        seq = begin_transfer(history, selected(2))
        history.emit("view", observation=after_transfer(before, selected(2)))
        history.emit("result", decision_seq=seq, result=c.Result("transferred", True))
        history.emit("terminal", outcome="win" if history.player == 0 else "loss")
    c.finalize_natural(pair, engine_outcomes=("win", "loss"), socket_outcomes=("win", "loss"),
                       process_exit_code=0, evidence_ref="synthetic-not-native")
    history = pair[player]
    chosen = {frame.decision_seq: frame.action for frame in history.frames if frame.kind == "submitted"}
    digest, prefix_hash = weights_sha256(model), hashlib.sha256()
    state, estimates, records = model.initial_state(), [], []
    with torch.no_grad():
        for frame, encoded in zip(history.frames, encode_history(history.frames)):
            prefix_hash.update(json.dumps(c.encode_frame(frame), sort_keys=True, separators=(",", ":")).encode() + b"\n")
            output = model.forward_step(encoded, state); state = output.state
            if frame.kind != "decision": continue
            prefix = RecurrentPrefix(history.episode, player, "synthetic-behavior", digest, frame.seq, prefix_hash.hexdigest())
            value = output.decision.value.item()
            estimates.append(DecisionEstimate(frame.seq, value, prefix))
            evaluated = output.decision.distribution.evaluate(chosen[frame.seq])
            records.append(BehaviorDecision(frame.seq, chosen[frame.seq], output.decision.distribution.support,
                evaluated.log_prob.item(), value, tuple(ConditionalRecord(item.name, item.selected_index,
                item.support_size, item.log_prob.item()) for item in evaluated.conditionals), prefix, "frozen-fixture"))
    values, by_seq = {item.decision_seq: item.value for item in estimates}, {item.decision_seq: item for item in records}
    samples = tuple(RolloutSample(by_seq[item.decision_seq], item.result_seq, item.successor_seq, item.reward,
        values[item.successor_seq] if item.bootstrap_allowed else None, item.eligible) for item in history.transitions())
    return ContextualRollout(history.episode, player, "learner", history.frames, tuple(records), samples, history.boundary,
        history.digest(), "synthetic-not-native", "synthetic-behavior", "frozen-fixture", digest, digest,
        "synthetic-fixed-pool", "synthetic", "synthetic", tuple(estimates))


class ArmyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(37)
        self.model = ContextualActorCritic(ModelConfig(schema=ARMIES_MODEL_SCHEMA, hidden_size=8, feature_size=4))

    def test_explicit_profile_versions_reject_implicit_model_promotion(self):
        frame = history_at_armies().frames[0]
        legacy = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))
        self.assertEqual(ModelConfig().schema, MODEL_SCHEMA)
        with self.assertRaises(ValueError): legacy.forward_step(encode_history((frame,))[0], legacy.initial_state())
        old = initial(0).frames[0]
        with self.assertRaises(ValueError): self.model.forward_step(encode_history((old,))[0], self.model.initial_state())
        with self.assertRaises(ValueError): ContextualCollector((legacy, legacy), ("a", "b"), learner_player=0,
            opponent_pool="pool", interface=c.ARMIES_ENVIRONMENT)

    def test_joint_support_normalizes_and_count_trace_is_exact(self):
        history = history_at_armies()
        with torch.no_grad():
            for parameter in self.model.quantity.parameters(): parameter.zero_()
            distribution = last_output(self.model, history.frames).decision.distribution
            actions = [c.Action("close", schema=c.ARMIES_ACTION_SCHEMA)]
            for source, destination, maximum in history.decision.offers.panel.armies.choices():
                actions.extend(selected(count, source, destination) for count in range(1, maximum + 1))
            masses = []
            for action in actions:
                evaluation = distribution.evaluate(action)
                self.assertAlmostEqual(evaluation.log_prob.item(), sum(item.log_prob.item() for item in evaluation.conditionals), places=6)
                masses.append(evaluation.log_prob.exp().item())
            self.assertAlmostEqual(sum(masses), 1, places=6)
            for destination in ("lower-0", "lower-6"):
                probabilities = [distribution.log_prob(selected(count, destination=destination)).item() for count in (1, 2, 3)]
                self.assertLess(max(probabilities) - min(probabilities), 1e-6)
            self.assertEqual(distribution.support.distribution_schema, ARMIES_DISTRIBUTION_SCHEMA)
            for _ in range(16): self.assertTrue(history.decision.offers.permits(distribution.sample()))

    def test_int32_quantity_support_has_bounded_exact_paths(self):
        source = army("upper", "town_garrison", "town", {0: (0, 2**31 - 1)})
        history = history_at_armies(armies=exchange(source=source))
        with torch.no_grad():
            distribution = last_output(self.model, history.frames).decision.distribution
            for count in (1, 2**30, 2**31 - 1):
                evaluation = distribution.evaluate(selected(count, destination="lower-6"))
                count_path = [item for item in evaluation.conditionals if item.name.startswith("count:")]
                self.assertLessEqual(len(count_path), 31)
                self.assertTrue(math.isfinite(evaluation.log_prob.item()))
            with self.assertRaises(ValueError): distribution.evaluate(selected(1))  # full source+destination sum overflows

    def test_opaque_reference_renaming_preserves_semantics_and_probabilities(self):
        history = history_at_armies()
        mapping = {ref: "opaque-" + str(index) for index, ref in enumerate(
            ("hero", "garrison", "town", "upper", "lower", *(f"upper-{i}" for i in range(7)), *(f"lower-{i}" for i in range(7))))}
        def rename(value, field=None):
            if type(value) is dict: return {key: rename(item, key) for key, item in value.items()}
            if type(value) is list: return [rename(item, field) for item in value]
            if type(value) is str and field in ("handle", "ref", "subject", "selection", "destination", "hero_refs", "town_refs"):
                return mapping.get(value, value)
            return value
        frames = tuple(c.decode_frame(rename(c.encode_frame(frame))) for frame in history.frames)
        left, right = encode_history(history.frames), encode_history(frames)
        self.assertEqual(tuple(item.semantics for item in left), tuple(item.semantics for item in right))
        with torch.no_grad():
            a, b = last_output(self.model, history.frames), last_output(self.model, frames)
            torch.testing.assert_close(a.state, b.state, rtol=0, atol=0)
            action = selected(2)
            renamed = replace(action, subject=mapping[action.subject], selection=mapping[action.selection], destination=mapping[action.destination])
            torch.testing.assert_close(a.decision.distribution.log_prob(action), b.decision.distribution.log_prob(renamed), rtol=0, atol=0)
        self.assertNotIn("upper-0", repr(left[-1].semantics))

    def test_incremental_session_has_exact_full_prefix_and_separate_player_memory(self):
        pair = tuple(history_at_armies(player) for player in (0, 1))
        sessions = tuple(ContextualSession(self.model, pair[0].episode, player, "policy", "opponent", schema=c.ARMIES_SCHEMA)
                         for player in (0, 1))
        for frame in pair[0].frames: sessions[0].consume((frame,))
        torch.testing.assert_close(sessions[1].state, self.model.initial_state(), rtol=0, atol=0)
        with torch.no_grad(): expected = last_output(self.model, pair[0].frames)
        torch.testing.assert_close(sessions[0].state, expected.state, rtol=0, atol=0)
        record = sessions[0].choose(pair[0].decision.seq)
        self.assertEqual(record.value, expected.decision.value.item())
        self.assertEqual(record.log_prob, expected.decision.distribution.log_prob(record.action).item())
        detached = sessions[0].state; detached.add_(1)
        torch.testing.assert_close(sessions[0].state, expected.state, rtol=0, atol=0)

    def test_v4_collector_roundtrip_binds_versions_and_native_diagnostics_stay_excluded(self):
        collector = ContextualCollector((self.model, self.model), ("learner", "opponent"), learner_player=0,
            opponent_pool="synthetic-pool", interface=c.ARMIES_ENVIRONMENT, purpose="synthetic", origin="synthetic")
        collector.start("synthetic-collector")
        pair = tuple(initial(player, schema=c.ARMIES_SCHEMA) for player in (0, 1))
        for history in pair: resolve(history, collector.act(request(history)))
        artifact = collector.finish(finish_pair(pair, "natural"), histories=tuple(history.frames for history in pair))
        self.assertEqual(CollectedEpisode.from_dict(json.loads(json.dumps(artifact.to_dict()))), artifact)
        self.assertTrue(artifact.rollouts[0].optimization_eligible)
        for rollout in artifact.rollouts:
            with self.assertRaises(ValueError): _admit((replace(rollout, purpose="diagnostic", origin="native"),), self.model)
        raw = artifact.to_dict(); raw["model_configs"][0]["schema"] = MODEL_SCHEMA
        with self.assertRaises(ValueError): CollectedEpisode.from_dict(raw)

    def test_checkpoint_roundtrip_and_mixed_encoder_pool_reject(self):
        pool = FrozenOpponentPool(); pool.add("frozen", self.model)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "v4.pt"
            save_policy_checkpoint(path, self.model, checkpoint="initial", source_metadata={"purpose": "synthetic"},
                                   opponent_pool=pool, interface=c.ARMIES_ENVIRONMENT)
            loaded = load_policy_checkpoint(path, interface=c.ARMIES_ENVIRONMENT)
            self.assertEqual(loaded.weights_sha256, weights_sha256(self.model))
            self.assertEqual(loaded.opponent_pool.identities, pool.identities)
            with self.assertRaises(ValueError): load_policy_checkpoint(path)
            saved = torch.load(path, weights_only=True)
            for key, value in (("model_config", None), ("opponent_pool", [None]),
                               ("opponent_pool", [{"model_config": None}])):
                malformed = deepcopy(saved); malformed[key] = value
                broken = Path(temp) / "malformed.pt"; torch.save(malformed, broken)
                with self.assertRaises(ValueError): load_policy_checkpoint(broken, interface=c.ARMIES_ENVIRONMENT)
            legacy = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))
            mixed = FrozenOpponentPool(); mixed.add("old", legacy)
            with self.assertRaises(ValueError): save_policy_checkpoint(Path(temp) / "mixed.pt", self.model, checkpoint="bad",
                source_metadata={}, opponent_pool=mixed, interface=c.ARMIES_ENVIRONMENT)
            with self.assertRaises(ValueError): save_policy_checkpoint(Path(temp) / "promoted.pt", legacy, checkpoint="bad",
                source_metadata={}, interface=c.ARMIES_ENVIRONMENT)

    def test_synthetic_transfer_ppo_updates_policy_value_and_quantity(self):
        rollout = transfer_rollout(self.model)
        self.assertEqual(rollout.samples[-1].decision.action.kind, "transfer_stack")
        self.assertEqual(rollout.samples[-1].reward, 1)
        self.assertTrue(rollout.optimization_eligible)
        self.assertEqual(ContextualRollout.from_dict(rollout.to_dict()), rollout)
        old = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        report = ppo_update(self.model, optimizer, (rollout,), config=PPOConfig(epochs=1, decisions_per_window=1,
            windows_per_minibatch=1, normalize_advantages=False), generator=torch.Generator().manual_seed(4))
        self.assertEqual(report.samples, 2)
        for name in ("policy_query.weight", "value.weight", "quantity.0.weight", "gru.weight_hh"):
            self.assertFalse(torch.equal(old[name], dict(self.model.named_parameters())[name]), name)


if __name__ == "__main__": unittest.main()
