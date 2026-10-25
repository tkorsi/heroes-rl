"""Synthetic v2 terminal targets and optimizer checks; no gameplay optimization."""
import json
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != 'torch': raise
    raise unittest.SkipTest('Optional torch dependency is not installed') from error

from experiments.learner import contextual as c
from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, encode_history, weights_sha256
from experiments.learner.contextual_rollout import CollectedEpisode, ContextualCollector, ROLLOUT_SCHEMA, TERMINAL_ROLLOUT_SCHEMA
from experiments.learner.ppo import PPOConfig, _admit, ppo_update
from tests.test_contextual_rollout import initial, request, resolve, next_decision, end_record
from tests.test_contextual_model import map_history

torch.set_num_threads(1)


def collect(model, *, trajectory_schema=c.TERMINAL_TRAJECTORY_SCHEMA, learner_player=0, purpose='synthetic',
            previous=True, gap=False, rich=False):
    collector = ContextualCollector((model, model), ('initial-red', 'initial-blue'), learner_player=learner_player,
        opponent_pool='synthetic-terminal-pool', origin='synthetic', purpose=purpose, trajectory_schema=trajectory_schema)
    collector.start('synthetic-collector')
    pair = tuple(map_history(schema=c.TOWN_SCHEMA, episode='synthetic-collector', player=p) if rich else initial(p) for p in (0, 1))
    for h in pair:
        if previous:
            resolve(h, collector.act(request(h))); next_decision(h)
        if gap:
            chosen = collector.act(request(h))
            h.emit('submitted', decision_seq=h.decision.seq, action=chosen)
            h.emit('consumed', decision_seq=h.decision.seq, consumer='engine')
            h.emit('superseded', decision_seq=h.decision.seq, reason='synthetic-stale')
            next_decision(h)
        chosen = collector.act(request(h))
        h.emit('submitted', decision_seq=h.decision.seq, action=chosen)
        h.emit('consumed', decision_seq=h.decision.seq,
               consumer='engine' if chosen.kind in ('end_turn', 'move_to') else 'adapter')
        h.emit('superseded', decision_seq=h.decision.seq, reason='terminal')
    for h in pair:
        h.emit('terminal', outcome='win' if h.player == 0 else 'loss')
        h.emit('view', observation=h.frames[0].observation)
    c.finalize_natural(pair, engine_outcomes=('win', 'loss'), socket_outcomes=('win', 'loss'),
                       process_exit_code=0, evidence_ref='synthetic-not-engine-evidence')
    end = end_record(pair)
    return collector.finish(end, histories=tuple(h.frames for h in pair)), collector, pair


class TerminalRolloutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        self.model = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))

    def test_v2_final_consumed_sample_gets_reward_without_earlier_reassignment(self):
        for learner_player in (0, 1):
            collected, _, _ = collect(self.model, learner_player=learner_player)
            self.assertEqual(collected.artifact_schema, TERMINAL_ROLLOUT_SCHEMA)
            for rollout in collected.rollouts:
                previous, final = rollout.samples
                self.assertEqual(previous.reward, 0)
                self.assertEqual(previous.successor_seq, final.decision.decision_seq)
                self.assertEqual(previous.bootstrap_value, final.decision.value)
                self.assertEqual(final.reward, 1 if rollout.player == 0 else -1)
                self.assertIsNone(final.result_seq)
                self.assertIsNone(final.successor_seq)
                self.assertIsNone(final.bootstrap_value)
                self.assertIsNotNone(final.terminal_closure)
                self.assertTrue(rollout.trajectory_eligible)
                self.assertEqual(rollout.optimization_eligible, rollout.player == learner_player)
            raw = collected.to_dict()
            self.assertEqual(CollectedEpisode.from_dict(json.loads(json.dumps(raw))), collected)

    def test_legacy_roundtrip_preserves_omitted_fields_and_excluded_terminal_action(self):
        old, _, pair = collect(self.model, trajectory_schema=c.TRAJECTORY_SCHEMA)
        self.assertEqual(old.artifact_schema, ROLLOUT_SCHEMA)
        raw = old.to_dict()
        encoded = json.dumps(raw, sort_keys=True, separators=(',', ':'))
        self.assertEqual(encoded, json.dumps(CollectedEpisode.from_dict(json.loads(encoded)).to_dict(), sort_keys=True, separators=(',', ':')))
        for rollout, wire in zip(old.rollouts, raw['rollouts']):
            self.assertNotIn('trajectory_schema', wire)
            self.assertTrue(all('terminal_closure' not in sample for sample in wire['samples']))
            self.assertEqual(len(rollout.samples), 1)
            self.assertEqual(rollout.samples[0].reward, 0)
            self.assertFalse(rollout.trajectory_eligible or rollout.optimization_eligible)
        new, _, other_pair = collect(self.model)
        self.assertEqual([h.frames for h in pair], [h.frames for h in other_pair])
        self.assertEqual([r.decisions for r in old.rollouts], [r.decisions for r in new.rollouts])

    def test_v2_schema_closure_and_pair_proof_tampering_rejects(self):
        collected, _, _ = collect(self.model)
        for mutation in ('version', 'trajectory', 'result', 'consumption', 'supersession', 'terminal', 'finalization', 'pair'):
            raw = collected.to_dict()
            sample = raw['rollouts'][0]['samples'][-1]
            with self.subTest(mutation=mutation):
                if mutation == 'version': raw['schema'] = ROLLOUT_SCHEMA
                elif mutation == 'trajectory': raw['rollouts'][0]['trajectory_schema'] = c.TRAJECTORY_SCHEMA
                elif mutation == 'result': sample['result_seq'] = sample['terminal_closure']['terminal_seq']
                elif mutation == 'pair': raw['rollouts'][1]['history'].pop()
                else:
                    key = {'consumption':'consumed_seq', 'supersession':'superseded_seq',
                           'terminal':'terminal_seq', 'finalization':'finalized_seq'}[mutation]
                    sample['terminal_closure'][key] += 1
                with self.assertRaises(ValueError): CollectedEpisode.from_dict(raw)

    def test_other_unresolved_consumption_remains_a_trajectory_gap(self):
        collected, _, _ = collect(self.model, gap=True)
        for rollout in collected.rollouts:
            self.assertIsNotNone(rollout.samples[-1].terminal_closure)
            self.assertFalse(rollout.trajectory_eligible or rollout.optimization_eligible)
            with self.assertRaises(ValueError): _admit((rollout,), self.model)

    def test_diagnostic_final_closure_is_structural_only_and_ppo_rejects(self):
        collected, _, _ = collect(self.model, purpose='diagnostic')
        for rollout in collected.rollouts:
            self.assertTrue(rollout.trajectory_eligible)
            self.assertFalse(rollout.optimization_eligible)
            with self.assertRaisesRegex(ValueError, 'only eligible synthetic'): _admit((rollout,), self.model)

    def test_full_prefix_replay_reproduces_new_terminal_sample_behavior(self):
        collected, collector, _ = collect(self.model)
        digest = weights_sha256(self.model)
        with torch.no_grad():
            for rollout, session in zip(collected.rollouts, collector.sessions):
                state = self.model.initial_state()
                decisions = {d.decision_seq:d for d in rollout.decisions}
                for item in encode_history(rollout.history):
                    output = self.model.forward_step(item, state); state = output.state
                    if item.frame.seq in decisions:
                        saved = decisions[item.frame.seq]
                        actual = output.decision.distribution.evaluate(saved.action)
                        self.assertEqual(output.decision.value.item(), saved.value)
                        self.assertEqual(actual.log_prob.item(), saved.log_prob)
                        self.assertEqual([t.log_prob.item() for t in actual.conditionals], [t.log_prob for t in saved.conditionals])
                torch.testing.assert_close(state, session.state, rtol=0, atol=0)
        self.assertEqual(weights_sha256(self.model), digest)

    def test_synthetic_ppo_updates_final_action_and_value_without_result(self):
        collected, _, _ = collect(self.model, previous=False, rich=True)
        rollout = collected.rollouts[0]
        self.assertEqual(len(rollout.samples), 1)
        self.assertIsNone(rollout.samples[0].result_seq)
        old_policy = self.model.policy_query.weight.detach().clone()
        old_value = self.model.value.weight.detach().clone()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        report = ppo_update(self.model, optimizer, (rollout,), config=PPOConfig(epochs=1, normalize_advantages=False),
                            generator=torch.Generator().manual_seed(1))
        self.assertEqual(report.samples, 1)
        self.assertEqual(report.optimizer_steps, 1)
        self.assertFalse(torch.equal(old_policy, self.model.policy_query.weight))
        self.assertFalse(torch.equal(old_value, self.model.value.weight))


if __name__ == '__main__': unittest.main()
