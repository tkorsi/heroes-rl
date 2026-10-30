"""Synthetic v8 self-play collection controls; no native games or PPO updates."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
import unittest

import torch

from experiments.learner import contextual as c
from experiments.learner import contextual_model as m
from experiments.learner import interaction_guard as guard
from experiments.learner.contextual_rollout import ContextualCollector, ContextualSession, CollectedEpisode
from validation.probes.audit_contextual_histories import _dialog_cutoff_checks
from tests.test_builtin_v8_cutoff import mandatory_history, spell_target_history
from tests.test_contextual_capabilities import observation, decide, begin, action, choice, market_history
from tests.test_contextual_rollout import initial, request, resolve, next_decision, end_record
from tests.test_dialog_collector_cutoff import SyntheticDialogChain


class AlternatingTurns:
    interface = c.EXPANDED_ENVIRONMENT

    def reset(self, *, episode, seed):
        self.pair = tuple(initial(player, str(episode), schema=c.EXPANDED_SCHEMA) for player in (0, 1))
        self.counts, self.player = [0, 0], 0
        return request(self.pair[0])

    def step(self, selected):
        history = self.pair[self.player]
        resolve(history, selected)
        next_decision(history)
        self.counts[self.player] += 1
        self.player = 1 - self.player
        return request(self.pair[self.player])

    def truncate(self, reason):
        for history in self.pair:
            history.emit("cutoff", decision_seq=history.decision.seq, reason=reason)
        c.finalize_cutoff(self.pair, evidence_ref="synthetic-not-engine-evidence", histories_complete=(True, True),
                          failure_free=True, shutdown_verified=True, continuation_limit=8, continuation_steps=(0, 0))
        end = end_record(self.pair)
        end.manifest.update(termination_reason=reason, dialog_cutoff_drain=None)
        return end

    def history(self, color):
        return self.pair[("red", "blue").index(color)].frames


class LevelUpChain(AlternatingTurns):
    """The unchanged neural policy samples the move and the native-query answer."""
    def reset(self, *, episode, seed):
        self.base = replace(observation(), towns=(), heroes=(observation().heroes[0],))
        red = c.PlayerHistory(str(episode), 0, schema=c.EXPANDED_SCHEMA)
        decide(red, self.base)
        self.pair = red, initial(1, str(episode), schema=c.EXPANDED_SCHEMA)
        self.actions, self.drain = [], None
        return request(red)

    def step(self, selected):
        red = self.pair[0]
        seq = begin(red, selected, "engine")
        self.actions.append(selected)
        if len(self.actions) == 1:
            if selected.kind != "move_to":
                raise AssertionError("Fixed synthetic seed no longer reaches the query trigger")
            red.emit("issued", decision_seq=seq, command_index=0)
            panel = c.CapabilityPanel("level-query", "level_up", c.Context(), True, 2, 0, (), (
                choice("choose_level_up", ref="skill-a", subject="hero", skill=1, skill_level=1),
                choice("choose_level_up", ref="skill-b", subject="hero", skill=2, skill_level=1)))
            red.emit("event", event=c.VisibleEvent("capability_requested", "hero", capability=panel))
            red.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
            red.emit("result", decision_seq=seq, result=c.Result("interaction", True, (1, 1, 0), 100,
                     (selected.x, selected.y) == (1, 1)))
            context = c.Context("capability", "hero", "level_up", panel.ref)
            red.emit("context_opened", context=context, revision=red.revision + 1, reason="capability_dialog")
            decide(red, replace(self.base, window=c.Window(context, None, armies=(), capability_controls=(), capability_panel=panel)))
        else:
            if selected.kind != "choose_level_up" or self.drain is None:
                raise AssertionError("Mandatory query must use its real policy choice")
            self.drain["choices"].append({"player": 0, "decision_seq": seq})
            red.emit("context", decision_seq=seq, context=c.Context(), revision=red.revision + 1)
            red.emit("issued", decision_seq=seq, command_index=0)
            red.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
            red.emit("result", decision_seq=seq, result=c.Result("capability_applied", True))
            decide(red, self.base)
        return request(red)

    def defer_dialog_cutoff(self, reason):
        red = self.pair[0]
        if not red.requires_choice:
            raise AssertionError("Only a mandatory choice may defer cutoff")
        self.drain = {"reason": reason, "trigger": {"player": 0, "decision_seq": red.decision.seq,
                      "actions_sent": [len(self.actions), 0]}, "limit": 8, "choices": [], "learner_included": True}

    def truncate(self, reason):
        end = super().truncate(reason)
        end.manifest["dialog_cutoff_drain"] = deepcopy(self.drain)
        return end


class SelfPlayCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self, interface=c.EXPANDED_ENVIRONMENT):
        torch.manual_seed(172)
        return m.ContextualActorCritic(m.ModelConfig(schema=m.model_schema_for(interface), hidden_size=8, feature_size=4))

    def collector(self, model=None, interface=c.EXPANDED_ENVIRONMENT, **kwargs):
        model = model or self.model(interface)
        return ContextualCollector((model, model), ("fixed-learner", "fixed-opponent"), learner_player=0,
            opponent_pool="synthetic-fixed-pool", interface=interface, purpose="diagnostic", origin="synthetic",
            policy_seeds=(1, 2), **kwargs)

    def test_explicit_guard_binds_both_players_and_default_still_tracks_active_version(self):
        model = self.model()
        for explicit in (None, guard.VERSION, guard.REVEALED_VERSION):
            with self.subTest(explicit=explicit), patch.object(guard, "ACTIVE_VERSION", guard.VERSION):
                collector = self.collector(model, guard_version=explicit)
                collector.start("guard-pair")
                for player in (0, 1):
                    history = initial(player, "guard-pair", schema=c.EXPANDED_SCHEMA)
                    collector.act(request(history))
                    support = collector.sessions[player]._choices[0].support.guard
                    self.assertEqual(support.version if support else None, explicit)
        with patch.object(guard, "ACTIVE_VERSION", guard.REVEALED_VERSION):
            collector = self.collector(model)
            collector.start("default-pair")
            collector.act(request(initial(0, "default-pair", schema=c.EXPANDED_SCHEMA)))
            self.assertEqual(collector.sessions[0]._choices[0].support.guard.version, guard.REVEALED_VERSION)
        with self.assertRaises(ValueError):
            self.collector(model, guard_version="unreviewed-guard")

    def test_seat_budget_does_not_halve_both_players_combined_allowance(self):
        env = AlternatingTurns()
        episode = self.collector(guard_version=guard.REVEALED_VERSION).run_episode(
            env, episode=1, seed=17, max_requests_per_player=2, max_dialog_requests_per_player=1)
        self.assertEqual(env.counts, [2, 1])
        self.assertEqual([len(item.samples) for item in episode.rollouts], [2, 1])
        self.assertTrue(all(item.boundary == "cutoff" for item in episode.rollouts))
        self.assertEqual(CollectedEpisode.from_dict(episode.to_dict()), episode)
        combined = AlternatingTurns()
        self.collector().run_episode(combined, episode=1, seed=17, max_requests=2)
        self.assertEqual(combined.counts, [1, 1])

    def test_per_player_arguments_fail_before_reset(self):
        for kwargs in ({"max_requests_per_player": 1}, {"max_dialog_requests_per_player": 1},
                       {"max_requests": 1, "max_requests_per_player": 1, "max_dialog_requests_per_player": 1},
                       {"max_requests_per_player": True, "max_dialog_requests_per_player": 1},
                       {"max_requests_per_player": 1, "max_dialog_requests_per_player": -1}):
            env = AlternatingTurns()
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.collector().run_episode(env, episode=1, seed=17, **kwargs)
            self.assertFalse(hasattr(env, "pair"))

    def test_reserved_answers_are_sampled_but_hard_bound_never_samples_an_extra_choice(self):
        interface = c.DIALOGS_ENVIRONMENT
        env, collector = SyntheticDialogChain(), self.collector(interface=interface)
        with self.assertRaisesRegex(ValueError, "hard collection bound"):
            collector.run_episode(env, episode="hard-bound", seed=17,
                                  max_requests_per_player=1, max_dialog_requests_per_player=0)
        self.assertEqual(len(env.actions), 1)
        self.assertEqual(len(collector.sessions[0]._choices), 1)
        self.assertFalse(collector.sessions[0].finished)
        allowed = SyntheticDialogChain()
        episode = self.collector(interface=interface).run_episode(allowed, episode="reserved", seed=17,
            max_requests_per_player=1, max_dialog_requests_per_player=2)
        self.assertEqual(len(episode.rollouts[0].samples), 3)
        self.assertEqual([item.kind for item in allowed.actions], ["move_to", "choose_dialog", "choose_dialog"])

    def test_v8_mandatory_capability_is_sampled_and_reconstructed_before_cutoff(self):
        for budget in ({"max_requests": 1}, {"max_requests_per_player": 1, "max_dialog_requests_per_player": 1}):
            with self.subTest(budget=budget):
                model, env = self.model(), LevelUpChain()
                collector = self.collector(model)
                episode = collector.run_episode(env, episode="level-up-collector", seed=17, **budget)
                self.assertEqual([item.kind for item in env.actions], ["move_to", "choose_level_up"])
                red = episode.rollouts[0]
                independent = ContextualSession(model, red.episode, 0, "fixed-learner", "fixed-opponent", schema=c.EXPANDED_SCHEMA, seed=1)
                for decision in red.decisions:
                    independent.consume(red.history[:decision.decision_seq + 1])
                    self.assertEqual(independent.choose(decision.decision_seq), decision)
                self.assertEqual(CollectedEpisode.from_dict(episode.to_dict()), episode)

    def test_paired_drain_audit_accepts_only_exact_mandatory_capability_answers(self):
        for build, kind, ref in ((mandatory_history, "choose_level_up", "skill-a"),
                                (spell_target_history, "choose_spell_target", "target"),
                                (spell_target_history, "cancel_spell_target", "cancel")):
            with self.subTest(kind=kind):
                red = build()
                trigger = red.decision.seq
                sent = sum(frame.kind == "submitted" for frame in red.frames)
                seq = begin(red, action(kind, "hero", selection=ref), "engine")
                red.emit("context", decision_seq=seq, context=c.Context(), revision=red.revision + 1)
                red.emit("issued", decision_seq=seq, command_index=0)
                red.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
                red.emit("result", decision_seq=seq, result=c.Result("capability_applied", True))
                decide(red, observation())
                pair = red, initial(1, red.episode, schema=c.EXPANDED_SCHEMA)
                manifest = {"status": "truncated", "termination_reason": "collection_window", "dialog_cutoff_drain": {
                    "reason": "collection_window", "trigger": {"player": 0, "decision_seq": trigger, "actions_sent": [sent, 0]},
                    "limit": 8, "choices": [{"player": 0, "decision_seq": trigger}], "learner_included": True}}
                _dialog_cutoff_checks(manifest, pair)
                for mutate in (lambda d: d["choices"].clear(), lambda d: d.update(limit=160),
                               lambda d: d["trigger"].update(decision_seq=0),
                               lambda d: d["choices"][0].update(player=1),
                               lambda d: d["trigger"]["actions_sent"].__setitem__(1, 1)):
                    altered = deepcopy(manifest)
                    mutate(altered["dialog_cutoff_drain"])
                    with self.assertRaises(ValueError):
                        _dialog_cutoff_checks(altered, pair)

    def test_drain_audit_rejects_optional_capability_and_ordinary_post_query_choice(self):
        red = market_history()
        trigger = red.decision.seq
        manifest = {"status": "truncated", "termination_reason": "collection_window", "dialog_cutoff_drain": {
            "reason": "collection_window", "trigger": {"player": 0, "decision_seq": trigger,
                "actions_sent": [sum(frame.kind == "submitted" for frame in red.frames), 0]},
            "limit": 8, "choices": [], "learner_included": True}}
        with self.assertRaisesRegex(ValueError, "trigger"):
            _dialog_cutoff_checks(manifest, (red, initial(1, red.episode, schema=c.EXPANDED_SCHEMA)))

        red = mandatory_history()
        trigger = red.decision.seq
        seq = begin(red, action("choose_level_up", "hero", selection="skill-a"), "engine")
        red.emit("context", decision_seq=seq, context=c.Context(), revision=red.revision + 1)
        red.emit("issued", decision_seq=seq, command_index=0)
        red.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
        red.emit("result", decision_seq=seq, result=c.Result("capability_applied", True))
        decide(red, observation())
        ordinary = red.decision.seq
        resolve(red, action("end_turn"))
        next_decision(red)
        manifest["dialog_cutoff_drain"].update(trigger={"player": 0, "decision_seq": trigger, "actions_sent": [1, 0]},
            choices=[{"player": 0, "decision_seq": trigger}, {"player": 0, "decision_seq": ordinary}])
        with self.assertRaisesRegex(ValueError, "Post-budget"):
            _dialog_cutoff_checks(manifest, (red, initial(1, red.episode, schema=c.EXPANDED_SCHEMA)))


if __name__ == "__main__":
    unittest.main()
