"""Synthetic tensor/optimizer checks only; no engine, training run or fake engine."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None
else:
    from experiments.learner import contextual as c
    from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, encode_history, weights_sha256
    from experiments.learner.contextual_rollout import (
        BehaviorDecision, ConditionalRecord, ContextualRollout, DecisionEstimate, RecurrentPrefix, RolloutSample,
    )
    from experiments.learner.ppo import PPOConfig, clipped_losses, evaluate_window, gae, ppo_update


def synthetic_rollout(model, *, boundary="natural", player=0, episode="synthetic-ppo", unresolved=False, turns=2):
    """Build typed contract examples directly, never masquerading as native play."""
    pair = tuple(c.PlayerHistory(episode, seat, schema=c.TOWN_SCHEMA) for seat in (0, 1))
    for history in pair:
        for turn in range(turns):
            obs = c.Observation((2, 2, 1), (c.Tile((0, 0, 0), 1, 0, False, False),), (),
                                (0, 0, 0, 0, 0, 0, 100 + turn),
                                (c.Hero("own", (0, 0, 0), 100),), (), None)
            history.emit("view", day=turn + 1, observation=obs)
            decision = history.emit("decision", observation=obs, revision=history.revision + 1,
                                    offers=c.Offers.from_observation(obs, schema=c.TOWN_ACTION_SCHEMA))
            action = c.Action("end_turn", schema=c.TOWN_ACTION_SCHEMA)
            history.emit("submitted", decision_seq=decision.seq, action=action)
            history.emit("consumed", decision_seq=decision.seq, consumer="engine")
            if unresolved and turn == 1:
                history.emit("superseded", decision_seq=decision.seq, reason="terminal")
                break
            history.emit("issued", decision_seq=decision.seq, command_index=0)
            history.emit("acknowledged", decision_seq=decision.seq, command_index=0, accepted=True)
            history.emit("result", decision_seq=decision.seq, result=c.Result("turn_ended", True))
        if boundary == "cutoff":
            decision = history.emit("decision", observation=obs, revision=history.revision + 1,
                                    offers=c.Offers.from_observation(obs, schema=c.TOWN_ACTION_SCHEMA))
            history.emit("cutoff", decision_seq=decision.seq, reason="synthetic-horizon")
        elif boundary == "failure":
            history.emit("failure", reason="synthetic-technical-failure")
        else:
            history.emit("terminal", outcome="win" if history.player == 0 else "loss")
    if boundary == "natural":
        c.finalize_natural(pair, engine_outcomes=("win", "loss"), socket_outcomes=("win", "loss"),
                           process_exit_code=0, evidence_ref="synthetic-assertion-not-native")
    elif boundary == "cutoff":
        c.finalize_cutoff(pair, evidence_ref="synthetic-assertion-not-native", histories_complete=(True, True),
                         failure_free=True, shutdown_verified=True, continuation_limit=8, continuation_steps=(1, 0))
    history = pair[player]
    inputs = encode_history(history.frames)
    chosen = {frame.decision_seq: frame.action for frame in history.frames if frame.kind == "submitted"}
    prefix_hash = hashlib.sha256()
    records, estimates = [], []
    state = model.initial_state()
    digest = weights_sha256(model)
    with torch.no_grad():
        for frame, encoded in zip(history.frames, inputs):
            prefix_hash.update(json.dumps(c.encode_frame(frame), sort_keys=True, separators=(",", ":")).encode() + b"\n")
            output = model.forward_step(encoded, state); state = output.state
            if frame.kind == "decision":
                prefix = RecurrentPrefix(episode, player, "synthetic-behavior", digest, frame.seq, prefix_hash.hexdigest())
                value = output.decision.value.item()
                estimates.append(DecisionEstimate(frame.seq, value, prefix))
                if frame.seq in chosen:
                    evaluated = output.decision.distribution.evaluate(chosen[frame.seq])
                    records.append(BehaviorDecision(frame.seq, chosen[frame.seq], output.decision.distribution.support,
                        evaluated.log_prob.item(), value, tuple(ConditionalRecord(item.name, item.selected_index,
                        item.support_size, item.log_prob.item()) for item in evaluated.conditionals), prefix, "frozen-fixture"))
    values = {item.decision_seq: item.value for item in estimates}
    by_seq = {item.decision_seq: item for item in records}
    samples = tuple(RolloutSample(by_seq[item.decision_seq], item.result_seq, item.successor_seq, item.reward,
        values[item.successor_seq] if item.bootstrap_allowed else None, item.eligible) for item in history.transitions())
    return ContextualRollout(episode, player, "learner", history.frames, tuple(records), samples, history.boundary,
        history.digest(), "synthetic-assertion-not-native", "synthetic-behavior", "frozen-fixture", digest, digest,
        "frozen-synthetic-pool-v1", "synthetic", "synthetic", tuple(estimates))


@unittest.skipIf(torch is None, "Optional PyTorch not installed")
class ContextualPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(17)
        self.model = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))

    def test_terminal_and_cutoff_gae_do_not_create_draw_or_failure_targets(self):
        natural = gae((0, 1), (0.2, -0.1), (-0.1, 0))
        self.assertEqual(natural.returns, (1.0, 1.0))
        cutoff = gae((0, 0), (0.2, -0.1), (-0.1, 0.4))
        for result in cutoff.returns: self.assertAlmostEqual(result, 0.4)
        one_step = gae((0, 1), (0.2, -0.1), (-0.1, 0), gae_lambda=0)
        self.assertAlmostEqual(one_step.returns[0], -0.1)
        for invalid in (dict(gamma=0.99), dict(entropy_coefficient=0.01), dict(gae_lambda=1.1)):
            with self.assertRaises(ValueError): PPOConfig(**invalid)

    def test_clipping_uses_joint_ratios_and_old_value_reference(self):
        logp = torch.tensor([2.0, 0.5]).log().requires_grad_()
        values = torch.tensor([2.0, -2.0], requires_grad=True)
        config = PPOConfig(value_clip=0.2)
        total, policy, value, _, fraction = clipped_losses(logp, values, torch.zeros(2), torch.zeros(2),
            torch.tensor([1.0, -1.0]), torch.tensor([1.0, -1.0]), config)
        self.assertAlmostEqual(policy.item(), -0.2, places=6)
        self.assertAlmostEqual(value.item(), 0.5, places=6)
        self.assertEqual(fraction.item(), 1.0)
        total.backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_window_rebuilds_current_prefix_and_detaches_only_before_window(self):
        rollout = synthetic_rollout(self.model)
        inputs = encode_history(rollout.history)
        record = rollout.decisions[1]
        begin = rollout.decisions[0].decision_seq + 1
        old = evaluate_window(self.model, inputs, (record,), start_seq=begin)[0].value.detach().clone()
        with torch.no_grad(): self.model.gru.bias_ih.add_(0.15)
        observed = []
        original = self.model.forward_step
        def watch(item, state):
            output = original(item, state)
            if output.state.requires_grad: output.state.retain_grad()
            observed.append(output.state)
            return output
        self.model.forward_step = watch
        current = evaluate_window(self.model, inputs, (record,), start_seq=begin)[0]
        self.assertTrue(all(not state.requires_grad for state in observed[:begin]))
        self.assertTrue(all(state.requires_grad for state in observed[begin:]))
        current.value.backward()
        self.assertGreater(observed[begin].grad.abs().sum().item(), 0)
        self.assertFalse(torch.equal(old, current.value))
        self.model.forward_step = original
        state = self.model.initial_state()
        with torch.no_grad():
            for item in inputs[:record.decision_seq + 1]:
                output = self.model.forward_step(item, state); state = output.state
        torch.testing.assert_close(current.value, output.decision.value, rtol=0, atol=0)

    def test_synthetic_natural_and_cutoff_updates_change_policy_value_and_gru(self):
        for boundary in ("natural", "cutoff"):
            with self.subTest(boundary=boundary):
                model = deepcopy(self.model)
                rollout = synthetic_rollout(model, boundary=boundary)
                old = {name: value.detach().clone() for name, value in model.named_parameters()}
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                report = ppo_update(model, optimizer, (rollout,), config=PPOConfig(epochs=2, decisions_per_window=1,
                    windows_per_minibatch=1, normalize_advantages=False), generator=torch.Generator().manual_seed(4))
                self.assertEqual(report.optimizer_steps, 4)
                self.assertEqual(report.samples, 2)
                self.assertIsNone(report.entropy)
                for name in ("policy_query.weight", "value.weight", "gru.weight_hh"):
                    self.assertFalse(torch.equal(old[name], dict(model.named_parameters())[name]), name)
                with self.assertRaises(ValueError): ppo_update(model, optimizer, (rollout,), generator=torch.Generator())

    def test_diagnostics_opponents_failure_gaps_and_pool_mixing_reject_before_step(self):
        base = synthetic_rollout(self.model)
        variants = [replace(base, purpose="diagnostic", origin="native"), replace(base, role="frozen_opponent"),
                    synthetic_rollout(self.model, boundary="failure"), synthetic_rollout(self.model, unresolved=True)]
        optimizer = torch.optim.Adam(self.model.parameters())
        old = weights_sha256(self.model)
        for item in variants:
            with self.assertRaises(ValueError): ppo_update(self.model, optimizer, (item,), generator=torch.Generator())
            self.assertEqual(old, weights_sha256(self.model))
            self.assertFalse(optimizer.state)
        with self.assertRaises(ValueError):
            ppo_update(self.model, optimizer, (base, replace(base, opponent_pool="different-mixture")), generator=torch.Generator())

    def test_tampered_conditional_trace_rejects_even_when_joint_probability_matches(self):
        base = synthetic_rollout(self.model)
        first = base.decisions[0]
        changed = replace(first, conditionals=(replace(first.conditionals[0], name="forged-support"), *first.conditionals[1:]))
        malformed = replace(base, decisions=(changed, *base.decisions[1:]),
                            samples=(replace(base.samples[0], decision=changed), *base.samples[1:]))
        optimizer = torch.optim.Adam(self.model.parameters())
        old = weights_sha256(self.model)
        with self.assertRaisesRegex(ValueError, "conditional"):
            ppo_update(self.model, optimizer, (malformed,), generator=torch.Generator())
        self.assertEqual(weights_sha256(self.model), old)
        self.assertFalse(optimizer.state)


if __name__ == "__main__": unittest.main()
