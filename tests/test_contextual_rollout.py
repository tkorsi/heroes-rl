"""Synthetic collector evidence checks; no engine or gameplay optimizer runs."""
from dataclasses import FrozenInstanceError
import hashlib
import json
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Optional torch dependency is not installed") from error

from experiments.learner import contextual as c
from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, encode_history, weights_sha256
from experiments.learner.contextual_rollout import CollectedEpisode, ContextualCollector, ContextualSession
from tests.test_contextual_model import recruitment, town_history

torch.set_num_threads(1)


def initial(player, episode="synthetic-collector", schema=c.TOWN_SCHEMA):
    h = c.PlayerHistory(episode, player, schema=schema)
    obs = c.Observation((2, 2, 1), (c.Tile((0, 0, 0), 1, 0, False, False),), (), (0,) * 7, (), ())
    h.emit("decision", observation=obs, revision=0,
           offers=c.Offers.from_observation(obs, schema=h.profile.action_schema))
    return h


def request(history, start=0):
    frames = [c.encode_frame(frame) for frame in history.frames[start:]]
    return SimpleNamespace(done=False, color=("red", "blue")[history.player], player=history.player,
        kind="contextual", past_horizon=False, reward=0, frames=frames, record=frames[-1])


def resolve(history, action):
    seq = history.decision.seq
    history.emit("submitted", decision_seq=seq, action=action)
    history.emit("consumed", decision_seq=seq, consumer="engine")
    history.emit("issued", decision_seq=seq, command_index=0)
    history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
    history.emit("result", decision_seq=seq, result=c.Result("turn_ended", True))


def next_decision(history):
    obs = history.frames[0].observation
    history.emit("decision", day=history.frames[-1].day + 1, observation=obs,
        revision=history.revision + 1, offers=c.Offers.from_observation(obs, schema=history.profile.action_schema))


def finish_pair(pair, boundary):
    for h in pair:
        if boundary == "natural":
            h.emit("terminal", outcome="win" if h.player == 0 else "loss")
        elif boundary == "cutoff":
            next_decision(h)
            h.emit("cutoff", decision_seq=h.decision.seq, reason="synthetic-limit")
        else:
            h.emit("failure", reason="synthetic-reader-failure")
    if boundary == "natural":
        c.finalize_natural(pair, engine_outcomes=("win", "loss"), socket_outcomes=("win", "loss"),
            process_exit_code=0, evidence_ref="synthetic-not-engine-evidence")
    elif boundary == "cutoff":
        c.finalize_cutoff(pair, evidence_ref="synthetic-not-engine-evidence", histories_complete=(True, True),
            failure_free=True, shutdown_verified=True, continuation_limit=8, continuation_steps=(1, 0))
    return end_record(pair)


def end_record(pair):
    boundary = pair[0].boundary
    status = {"natural": "completed", "cutoff": "truncated", "failure": "failed"}[boundary]
    profile = pair[0].profile
    outcomes = {color: h.terminal if status == "completed" else None for color, h in zip(("red", "blue"), pair)}
    rewards = {color: 1 if value == "win" else -1 if value == "loss" else None for color, value in outcomes.items()}
    players, bootstraps = {}, {}
    for color, h in zip(("red", "blue"), pair):
        prefix = h.frames[:-1] if h.frames[-1].kind in ("finalized", "cutoff_finalized") else h.frames
        prefix_hash = hashlib.sha256(b"".join(json.dumps(c.encode_frame(f), sort_keys=True,
            separators=(",", ":")).encode() + b"\n" for f in prefix)).hexdigest()
        players[color] = {profile.history_key: {"canonical_sha256": h.digest(), "prefix_sha256": prefix_hash,
                                             "boundary": boundary, "complete": True}}
        cutoff = next((frame for frame in h.frames if frame.kind == "cutoff"), None)
        bootstraps[color] = c.encode_frame(h.frames[cutoff.decision_seq]) if cutoff else None
    manifest = {"schema_version": 1, "kind": "contextual-headless-self-play", "environment": profile.environment,
        "history_schema": profile.history_schema, "action_schema": profile.action_schema, "episode": pair[0].episode,
        "status": status, "evidence_ref": "synthetic-not-engine-evidence", "outcomes_by_player": outcomes,
        "rewards_by_player": rewards, "returncode": 0 if boundary == "natural" else -15 if boundary == "cutoff" else 1,
        "players": players}
    return SimpleNamespace(done=True, status=status, outcomes=outcomes, rewards=rewards, bootstrap=bootstraps, manifest=manifest)


class ContextualRolloutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.model = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))

    def collector(self, *, purpose="synthetic", interface=c.TOWN_ENVIRONMENT):
        return ContextualCollector((self.model, self.model), ("learner-initial", "frozen-initial"), learner_player=0,
            opponent_pool="synthetic-fixed-pool", purpose=purpose, origin="synthetic", interface=interface)

    def collect(self, boundary="natural", purpose="synthetic", schema=c.TOWN_SCHEMA):
        collector = self.collector(purpose=purpose, interface=c.profile_for(schema).environment)
        collector.start("synthetic-collector")
        pair = tuple(initial(player, schema=schema) for player in (0, 1))
        for h in pair:
            action = collector.act(request(h))
            self.assertEqual(action.kind, "end_turn")
            resolve(h, action)
        end = finish_pair(pair, boundary)
        return collector.finish(end, histories=tuple(h.frames for h in pair)), collector, pair, end

    def test_complete_natural_roundtrip_binds_exact_behavior_and_pair(self):
        digest = weights_sha256(self.model)
        result, collector, pair, _ = self.collect()
        self.assertTrue(result.rollouts[0].optimization_eligible)
        self.assertFalse(result.rollouts[1].optimization_eligible)
        self.assertEqual(tuple(r.samples[0].reward for r in result.rollouts), (1, -1))
        self.assertTrue(all(r.samples[0].bootstrap_value is None for r in result.rollouts))
        raw = json.loads(json.dumps(result.to_dict()))
        self.assertEqual(CollectedEpisode.from_dict(raw), result)
        raw["rollouts"][0]["history"][0]["observation"]["resources"][0] = 999
        self.assertEqual(result.rollouts[0].history[0].observation.resources[0], 0)
        with self.assertRaises(ValueError): CollectedEpisode.from_dict(raw)
        self.assertEqual(weights_sha256(self.model), digest)
        self.assertTrue(all(s.finished and not s.state.requires_grad for s in collector.sessions))
        with self.assertRaises(FrozenInstanceError): result.rollouts[0].player = 1

    def test_cutoff_values_real_tail_decisions_without_sampling_them(self):
        result, collector, _, end = self.collect("cutoff")
        for color, rollout, session in zip(("red", "blue"), result.rollouts, collector.sessions):
            self.assertEqual(len(rollout.decisions), 1)
            self.assertEqual(len(rollout.estimates), 2)
            sample = rollout.samples[0]
            self.assertEqual(sample.reward, 0)
            self.assertEqual(sample.successor_seq, end.bootstrap[color]["seq"])
            self.assertEqual(sample.bootstrap_value, rollout.estimates[-1].value)
            self.assertNotIn(sample.successor_seq, session._choices)
        self.assertTrue(result.rollouts[0].optimization_eligible)

    def test_failures_and_diagnostics_never_optimize(self):
        for boundary, purpose in (("failure", "synthetic"), ("natural", "diagnostic"), ("cutoff", "diagnostic")):
            with self.subTest(boundary=boundary, purpose=purpose):
                result, _, _, _ = self.collect(boundary, purpose)
                self.assertTrue(all(not r.optimization_eligible for r in result.rollouts))
                if boundary == "failure":
                    self.assertTrue(all(s.reward is None and s.bootstrap_value is None and not s.eligible
                                        for r in result.rollouts for s in r.samples))

    def test_incremental_prefix_matches_full_replay_and_never_crosses_players(self):
        collector = self.collector(); collector.start("synthetic-collector")
        red, blue = initial(0), initial(1)
        zero_blue = collector.sessions[1].state
        action = collector.act(request(red)); resolve(red, action); next_decision(red)
        collector.act(request(red, start=1))
        torch.testing.assert_close(collector.sessions[1].state, zero_blue, rtol=0, atol=0)
        state = self.model.initial_state()
        with torch.no_grad():
            for encoded in encode_history(red.frames):
                output = self.model.forward_step(encoded, state); state = output.state
        torch.testing.assert_close(collector.sessions[0].state, state, rtol=0, atol=0)
        self.assertEqual(collector.sessions[0]._values[red.decision.seq], output.decision.value.item())
        before = collector.sessions[0].state
        collector.sessions[0].consume(red.frames)
        torch.testing.assert_close(collector.sessions[0].state, before, rtol=0, atol=0)
        with self.assertRaises(ValueError): collector.sessions[0].consume(blue.frames)
        with self.assertRaisesRegex(ValueError, "Session failed"): collector.sessions[0].consume(red.frames)

    def test_superseded_unsubmitted_choice_is_retained_but_not_a_sample(self):
        collector = self.collector(); collector.start("synthetic-collector")
        pair = initial(0), initial(1)
        first = collector.act(request(pair[0]))
        pair[0].emit("superseded", decision_seq=0, reason="synthetic-visible-update")
        next_decision(pair[0])
        second = collector.act(request(pair[0], start=1))
        self.assertEqual(first, second)
        resolve(pair[0], second)
        resolve(pair[1], collector.act(request(pair[1])))
        end = finish_pair(pair, "natural")
        result = collector.finish(end, histories=tuple(h.frames for h in pair))
        red = result.rollouts[0]
        self.assertEqual(len(red.decisions), 2)
        self.assertEqual(len(red.samples), 1)
        self.assertNotEqual(red.samples[0].decision.decision_seq, 0)
        self.assertTrue(red.optimization_eligible)

    def test_consumed_unresolved_terminal_action_does_not_transfer_reward(self):
        collector = self.collector(); collector.start("synthetic-collector")
        pair = initial(0), initial(1)
        for h in pair:
            resolve(h, collector.act(request(h))); next_decision(h)
            action = collector.act(request(h, start=1))
            seq = h.decision.seq
            h.emit("submitted", decision_seq=seq, action=action)
            h.emit("consumed", decision_seq=seq, consumer="engine")
            h.emit("superseded", decision_seq=seq, reason="terminal")
        end = finish_pair(pair, "natural")
        result = collector.finish(end, histories=tuple(h.frames for h in pair))
        for rollout in result.rollouts:
            self.assertFalse(rollout.optimization_eligible)
            self.assertEqual(len(rollout.samples), 1)
            self.assertEqual(rollout.samples[0].reward, 0)
            self.assertEqual(rollout.samples[0].successor_seq, rollout.decisions[-1].decision_seq)

    def test_weight_mutation_and_state_mutation_cannot_reuse_a_session(self):
        session = ContextualSession(self.model, "synthetic-collector", 0, "initial", "opponent")
        history = initial(0); session.consume(history.frames)
        state = session.state; state.add_(100)
        self.assertFalse(torch.equal(session.state, state))
        with torch.no_grad(): next(self.model.parameters()).add_(0.1)
        with self.assertRaisesRegex(ValueError, "weights changed"): session.choose(0)
        with self.assertRaisesRegex(ValueError, "Session failed"): session.consume(())

    def test_pair_metadata_and_manifest_tampering_rejected(self):
        result, _, _, _ = self.collect("cutoff")
        for mutation in ("opponent", "roles", "pool", "purpose", "prefix", "bootstrap"):
            raw = result.to_dict()
            with self.subTest(mutation=mutation):
                if mutation == "opponent":
                    raw["rollouts"][0]["opponent_checkpoint"] = "unrelated"
                    raw["rollouts"][0]["opponent_weights_sha256"] = "a" * 64
                    for decision in raw["rollouts"][0]["decisions"]: decision["opponent_checkpoint"] = "unrelated"
                elif mutation == "roles": raw["rollouts"][1]["role"] = "learner"
                elif mutation == "pool": raw["rollouts"][0]["opponent_pool"] = "different"
                elif mutation == "purpose": raw["rollouts"][0]["purpose"] = "diagnostic"
                elif mutation == "prefix": raw["manifest"]["players"]["red"]["history_v3"]["prefix_sha256"] = "a" * 64
                else: raw["rollouts"][0]["samples"][0]["bootstrap_value"] += 0.1
                with self.assertRaises(ValueError): CollectedEpisode.from_dict(raw)

    def test_cutoff_wrong_player_bootstrap_and_missing_pair_proof_reject(self):
        collector = self.collector(); collector.start("synthetic-collector")
        pair = initial(0), initial(1)
        for h in pair: resolve(h, collector.act(request(h)))
        end = finish_pair(pair, "cutoff")
        end.bootstrap["red"] = end.bootstrap["blue"]
        with self.assertRaisesRegex(ValueError, "actual own decision"):
            collector.finish(end, histories=tuple(h.frames for h in pair))
        other = self.collector(); other.start("synthetic-collector")
        for h in pair: other.act(request(initial(h.player)))
        with self.assertRaises(ValueError): other.finish(end, histories=(pair[0].frames, pair[1].frames[:-1]))

    def test_quantity_choices_keep_exact_conditionals_and_support(self):
        history = town_history(recruitment(stock=2147483645))
        for seed in range(100):
            session = ContextualSession(self.model, history.episode, 0, "initial", "opponent", seed=seed)
            session.consume(history.frames)
            record = session.choose(history.decision.seq)
            if record.action.kind == "recruit": break
        else: self.fail("No recruit action sampled across deterministic synthetic seeds")
        self.assertTrue(record.support.offers.permits(record.action))
        self.assertGreaterEqual(record.action.count, 1)
        self.assertLessEqual(record.action.count, 2147483645)
        self.assertTrue(any(item.name.startswith("count:") for item in record.conditionals))
        self.assertAlmostEqual(record.log_prob, sum(item.log_prob for item in record.conditionals), places=5)
        self.assertIs(session.choose(history.decision.seq), record)

    def test_v2_is_collected_without_v3_schema_substitution(self):
        result, _, _, _ = self.collect("cutoff", schema=c.SCHEMA)
        self.assertEqual(result.rollouts[0].history[0].schema, c.SCHEMA)
        self.assertEqual(result.rollouts[0].decisions[0].action.schema, c.ACTION_SCHEMA)
        self.assertEqual(CollectedEpisode.from_dict(result.to_dict()), result)


if __name__ == "__main__": unittest.main()
