"""V7 synthetic PPO numerics only; no native game, admission or training claim.

The paired synthetic finalization below exercises the existing numerical kernel.
It is not an invented native-opponent trajectory or evidence for mixed-AI admission.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Optional PyTorch is not installed") from error

from experiments.learner import contextual as c
from experiments.learner.contextual_model import (
    ContextualActorCritic, ModelConfig, encode_history, model_schema_for, weights_sha256,
)
from experiments.learner.contextual_rollout import (
    BehaviorDecision, ConditionalRecord, ContextualRollout, DecisionEstimate,
    RecurrentPrefix, RolloutSample, _frame_bytes,
)
from experiments.learner.ppo import PPOConfig, evaluate_window, gae, ppo_update
from tests.test_contextual_dialogs import (
    FAMILIES, begin_answer, decide, finish_answer, history_at, observation, panel,
)


def synthetic_dialog_rollout(model, *, player=0, boundary="natural", family="chest",
                             terminal_closure=False, other_gold=4000, revealed_move=False):
    """Direct typed v7 examples, explicitly synthetic and never engine evidence."""
    if boundary not in ("natural", "cutoff", "failure") or terminal_closure and boundary != "natural":
        raise ValueError("Unsupported synthetic boundary")
    episode = "synthetic-builtin-ppo"
    pair = []
    for seat in (0, 1):
        history = c.PlayerHistory(episode, seat, schema=c.DIALOGS_SCHEMA)
        gold = 1000 if seat == player else other_gold
        for frame in history_at(panel(family)).frames:
            changed = dict(episode=episode, player=seat)
            if frame.observation is not None:
                obs = replace(frame.observation, resources=(0, 0, 0, 0, 0, 0, gold))
                if revealed_move:
                    # The fixture's submitted move targets (3, 2); explicitly
                    # deliver that tile when testing revealed movement support.
                    obs = replace(obs, tiles=(*obs.tiles, c.Tile((3, 2, 0), 1, 0, False, False)))
                changed["observation"] = obs
                if frame.offers is not None:
                    changed["offers"] = c.Offers.from_observation(obs, schema=c.DIALOGS_ACTION_SCHEMA)
            history.append(replace(frame, **changed))
        seq = begin_answer(history)
        if terminal_closure:
            history.emit("superseded", decision_seq=seq, reason="terminal")
        else:
            finish_answer(history, seq)
        if boundary == "cutoff":
            held = decide(history, replace(observation(), resources=(0, 0, 0, 0, 0, 0, gold)))
            history.emit("cutoff", decision_seq=held.seq, reason="synthetic-horizon")
        elif boundary == "failure":
            history.emit("failure", reason="synthetic-technical-failure")
        else:
            history.emit("terminal", outcome="win" if seat == 0 else "loss")
        pair.append(history)
    pair = tuple(pair)
    if boundary == "natural":
        c.finalize_natural(pair, engine_outcomes=("win", "loss"), socket_outcomes=("win", "loss"),
            process_exit_code=0, evidence_ref="synthetic-only-not-native-proof")
    elif boundary == "cutoff":
        c.finalize_cutoff(pair, evidence_ref="synthetic-only-not-native-proof", histories_complete=(True, True),
            failure_free=True, shutdown_verified=True, continuation_limit=8, continuation_steps=(0, 0))
    history = pair[player]
    chosen = {f.decision_seq: f.action for f in history.frames if f.kind == "submitted"}
    weight_digest = weights_sha256(model)
    prefix_hash, records, estimates = hashlib.sha256(), [], []
    state = model.initial_state()
    with torch.no_grad():
        for frame, encoded in zip(history.frames, encode_history(history.frames)):
            prefix_hash.update(_frame_bytes(frame))
            output = model.forward_step(encoded, state)
            state = output.state
            if frame.kind != "decision":
                continue
            prefix = RecurrentPrefix(episode, player, "synthetic-v7-behavior", weight_digest,
                                     frame.seq, prefix_hash.hexdigest())
            value = output.decision.value.item()
            estimates.append(DecisionEstimate(frame.seq, value, prefix))
            if frame.seq in chosen:
                scored = output.decision.distribution.evaluate(chosen[frame.seq])
                records.append(BehaviorDecision(frame.seq, chosen[frame.seq], output.decision.distribution.support,
                    scored.log_prob.item(), value, tuple(ConditionalRecord(item.name, item.selected_index,
                        item.support_size, item.log_prob.item()) for item in scored.conditionals),
                    prefix, "synthetic-fixed-opponent"))
    by_seq = {item.decision_seq: item for item in records}
    values = {item.decision_seq: item.value for item in estimates}
    trajectory = c.TERMINAL_TRAJECTORY_SCHEMA
    samples = tuple(RolloutSample(by_seq[item.decision_seq], item.result_seq, item.successor_seq, item.reward,
        values[item.successor_seq] if item.bootstrap_allowed else None, item.eligible, item.terminal_closure)
        for item in history.transitions(schema=trajectory))
    return ContextualRollout(episode, player, "learner", history.frames, tuple(records), samples, history.boundary,
        history.digest(), "synthetic-only-not-native-proof", "synthetic-v7-behavior", "synthetic-fixed-opponent",
        weight_digest, weight_digest, "synthetic-v7-fixed-pool", "synthetic", "synthetic", tuple(estimates), trajectory)


def fold(model, inputs, state=None):
    state = model.initial_state() if state is None else state
    output = None
    for item in inputs:
        output = model.forward_step(item, state)
        state = output.state
    return output


class BuiltinPPONumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.rng = torch.random.fork_rng(devices=[])
        self.rng.__enter__()
        self.addCleanup(self.rng.__exit__, None, None, None)
        torch.manual_seed(937)
        self.model = ContextualActorCritic(ModelConfig(schema=model_schema_for(c.DIALOGS_SCHEMA),
                                                      hidden_size=8, feature_size=4))

    def test_all_dialog_supports_and_joint_conditionals_reconstruct_exactly(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                rollout = synthetic_dialog_rollout(self.model, family=family)
                self.assertTrue(all(f.schema == c.DIALOGS_SCHEMA for f in rollout.history))
                inputs = encode_history(rollout.history)
                with torch.no_grad():
                    for record in rollout.decisions:
                        output = fold(self.model, inputs[:record.decision_seq + 1]).decision
                        self.assertEqual(output.value.item(), record.value)
                        self.assertEqual(output.distribution.support, record.support)
                        scored = output.distribution.evaluate(record.action)
                        self.assertEqual(scored.log_prob.item(), record.log_prob)
                        self.assertEqual(tuple((v.name, v.selected_index, v.support_size, v.log_prob.item())
                            for v in scored.conditionals), tuple((v.name, v.selected_index, v.support_size, v.log_prob)
                            for v in record.conditionals))
                        torch.testing.assert_close(scored.log_prob, sum(v.log_prob for v in scored.conditionals),
                                                   rtol=0, atol=0)
                    policy = output.distribution
                    choices = rollout.history[rollout.decisions[-1].decision_seq].offers.dialog_panel.choices
                    actions = [c.Action("choose_dialog", "hero", selection=v.ref, schema=c.DIALOGS_ACTION_SCHEMA)
                               for v in choices]
                    self.assertAlmostEqual(sum(policy.log_prob(a).exp().item() for a in actions), 1.0, places=6)
                    for selected in actions:
                        self.assertTrue(policy.support.offers.permits(selected))
                    with self.assertRaises(ValueError):
                        policy.log_prob(replace(actions[0], selection="foreign-choice"))

    def test_natural_targets_are_terminal_only_for_both_seats_and_closures(self):
        for player in (0, 1):
            for closure in (False, True):
                with self.subTest(player=player, closure=closure):
                    rollout = synthetic_dialog_rollout(self.model, player=player, terminal_closure=closure)
                    first, final = rollout.samples
                    reward = 1 if player == 0 else -1
                    self.assertEqual((first.reward, final.reward), (0, reward))
                    self.assertEqual(first.bootstrap_value, final.decision.value)
                    self.assertIsNone(final.successor_seq)
                    self.assertIsNone(final.bootstrap_value)
                    self.assertEqual(final.terminal_closure is not None, closure)
                    target = gae((first.reward, final.reward), (first.decision.value, final.decision.value),
                                 (first.bootstrap_value, 0.0))
                    for value in target.returns:
                        self.assertAlmostEqual(value, reward, places=12)

    def test_cutoff_uses_held_own_value_and_failure_never_becomes_zero_terminal(self):
        for player in (0, 1):
            with self.subTest(player=player):
                rollout = synthetic_dialog_rollout(self.model, boundary="cutoff", player=player)
                held = rollout.estimates[-1]
                self.assertNotIn(held.decision_seq, [r.decision_seq for r in rollout.decisions])
                self.assertEqual(rollout.samples[-1].successor_seq, held.decision_seq)
                self.assertEqual(rollout.samples[-1].bootstrap_value, held.value)
                self.assertNotEqual(held.value, 0)
                targets = gae(tuple(s.reward for s in rollout.samples), tuple(s.decision.value for s in rollout.samples),
                              tuple(s.bootstrap_value for s in rollout.samples))
                self.assertEqual(tuple(s.reward for s in rollout.samples), (0, 0))
                for value in targets.returns:
                    self.assertAlmostEqual(value, held.value, places=12)
                failed = synthetic_dialog_rollout(self.model, boundary="failure", player=player)
                self.assertFalse(failed.optimization_eligible)
                optimizer = torch.optim.Adam(self.model.parameters())
                with self.assertRaises(ValueError):
                    ppo_update(self.model, optimizer, (failed,), generator=torch.Generator())
                self.assertFalse(optimizer.state)

    def test_actor_critic_and_recurrence_receive_distinct_finite_gradients(self):
        rollout = synthetic_dialog_rollout(self.model)
        encoded = encode_history(rollout.history)
        for loss_kind in ("actor", "critic"):
            self.model.zero_grad(set_to_none=True)
            evaluated = evaluate_window(self.model, encoded, rollout.decisions, start_seq=0)
            loss = (-sum(v.log_prob for v in evaluated) if loss_kind == "actor"
                    else sum((v.value - 1).square() for v in evaluated))
            loss.backward()
            for name in (("policy_query.weight", "gru.weight_hh") if loss_kind == "actor"
                         else ("value.weight", "gru.weight_hh")):
                gradient = dict(self.model.named_parameters())[name].grad
                self.assertTrue(torch.isfinite(gradient).all(), (loss_kind, name))
                self.assertGreater(gradient.abs().sum().item(), 0, (loss_kind, name))

    def test_bounded_synthetic_updates_change_both_heads_and_recurrence(self):
        for player, boundary, closure in ((0, "natural", False), (1, "natural", True), (0, "cutoff", False)):
            with self.subTest(player=player, boundary=boundary, closure=closure):
                model = deepcopy(self.model)
                rollout = synthetic_dialog_rollout(model, player=player, boundary=boundary, terminal_closure=closure)
                old = {n: p.detach().clone() for n, p in model.named_parameters()}
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                report = ppo_update(model, optimizer, (rollout,), config=PPOConfig(epochs=2,
                    decisions_per_window=1, windows_per_minibatch=1, normalize_advantages=False),
                    generator=torch.Generator().manual_seed(8))
                self.assertEqual((report.samples, report.optimizer_steps), (2, 4))
                self.assertEqual(report.purpose, "synthetic_optimizer_test")
                for name in ("policy_query.weight", "value.weight", "gru.weight_hh"):
                    self.assertFalse(torch.equal(old[name], dict(model.named_parameters())[name]), name)
                with self.assertRaisesRegex(ValueError, "frozen behavior weights"):
                    ppo_update(model, optimizer, (rollout,), generator=torch.Generator())

    def test_current_weight_prefix_is_rebuilt_without_stale_or_other_player_memory(self):
        rollout = synthetic_dialog_rollout(self.model)
        other = synthetic_dialog_rollout(self.model, player=1)
        unchanged = synthetic_dialog_rollout(self.model, other_gold=90000)
        # Synthetic paired boundary attestations bind both history hashes, but
        # no opposite-player observation enters any own decision prefix/value.
        last = rollout.decisions[-1].decision_seq
        self.assertEqual(rollout.history[:last + 1], unchanged.history[:last + 1])
        self.assertEqual(rollout.decisions, unchanged.decisions)
        self.assertEqual(rollout.estimates, unchanged.estimates)
        encoded = encode_history(rollout.history)
        record = rollout.decisions[-1]
        begin = rollout.decisions[0].decision_seq + 1
        with torch.no_grad():
            stale_prefix = fold(self.model, encoded[:begin]).state
            self.model.gru.bias_ih.add_(0.3)
        observed = []
        original = self.model.forward_step
        def watch(item, state):
            output = original(item, state)
            if output.state.requires_grad:
                output.state.retain_grad()
            observed.append(output.state)
            return output
        self.model.forward_step = watch
        try:
            current = evaluate_window(self.model, encoded, (record,), start_seq=begin)[0]
            current.value.backward()
        finally:
            self.model.forward_step = original
        self.assertTrue(all(not state.requires_grad for state in observed[:begin]))
        self.assertTrue(all(state.requires_grad for state in observed[begin:]))
        self.assertGreater(observed[begin].grad.abs().sum().item(), 0)
        with torch.no_grad():
            independent = fold(self.model, encoded[:record.decision_seq + 1]).decision
            stale = fold(self.model, encoded[begin:record.decision_seq + 1], stale_prefix).decision
        torch.testing.assert_close(current.value, independent.value, rtol=0, atol=0)
        torch.testing.assert_close(current.log_prob, independent.distribution.log_prob(record.action), rtol=0, atol=0)
        self.assertFalse(torch.equal(current.value, stale.value))
        evaluate_window(self.model, encode_history(other.history), (other.decisions[-1],), start_seq=begin)
        repeated = evaluate_window(self.model, encoded, (record,), start_seq=begin)[0]
        torch.testing.assert_close(current.value, repeated.value, rtol=0, atol=0)
        torch.testing.assert_close(current.log_prob, repeated.log_prob, rtol=0, atol=0)

    def test_forged_conditional_and_bootstrap_reject_before_any_optimizer_step(self):
        base = synthetic_dialog_rollout(self.model, boundary="cutoff")
        final = base.decisions[-1]
        forged = replace(final, conditionals=(replace(final.conditionals[0], name="forged-choice-support"),
                                             *final.conditionals[1:]))
        conditional = replace(base, decisions=(*base.decisions[:-1], forged),
            samples=(*base.samples[:-1], replace(base.samples[-1], decision=forged)))
        held = base.estimates[-1]
        bootstrap = replace(base, estimates=(*base.estimates[:-1], replace(held, value=held.value + 0.25)),
            samples=(*base.samples[:-1], replace(base.samples[-1], bootstrap_value=held.value + 0.25)))
        original = weights_sha256(self.model)
        for malformed, message in ((conditional, "conditional"), (bootstrap, "Bootstrap")):
            optimizer = torch.optim.Adam(self.model.parameters())
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                ppo_update(self.model, optimizer, (malformed,), generator=torch.Generator())
            self.assertEqual(weights_sha256(self.model), original)
            self.assertFalse(optimizer.state)
        with self.assertRaises(ValueError):
            replace(base, history=tuple(replace(f, player=1) for f in base.history))
        with self.assertRaises(ValueError):
            replace(base, samples=(replace(base.samples[0], reward=1), *base.samples[1:]))


if __name__ == "__main__":
    unittest.main()
