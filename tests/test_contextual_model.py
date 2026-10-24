"""Synthetic model/contract checks, not native game or learning evidence."""
from dataclasses import replace
import math
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Optional torch dependency is not installed") from error

from experiments.learner import contextual as c
from experiments.learner import contextual_model as m

torch.set_num_threads(1)


def observation(*, size=(3, 2, 1), resources=(3, 0, 1, 0, 0, 0, 100), window=None, hero='hero', town='town'):
    return c.Observation(size, (c.Tile((0, 0, 0), 1, 0, False, False),), (), resources,
                         (c.Hero(hero, (0, 0, 0), 100, 200),), (c.Town(town, (1, 1, 0)),), window)


def map_history(obs=None, schema=c.SCHEMA, episode='episode', player=0):
    history = c.PlayerHistory(episode, player, schema=schema)
    obs = obs or observation()
    history.emit('decision', observation=obs, offers=c.Offers.from_observation(obs, schema=history.profile.action_schema), revision=0)
    return history


def navigate(history, action, obs):
    decision = history.decision.seq
    history.emit('submitted', action=action, decision_seq=decision)
    history.emit('consumed', consumer='adapter', decision_seq=decision)
    history.emit('context', context=obs.context, revision=history.revision + 1, decision_seq=decision)
    history.emit('view', observation=obs)
    history.emit('result', result=c.Result('opened', True), decision_seq=decision)
    history.emit('decision', observation=obs, offers=c.Offers.from_observation(obs, schema=history.profile.action_schema), revision=history.revision + 1)
    return history


def town_history(panel):
    history = map_history(schema=c.TOWN_SCHEMA)
    overview = c.TownPanel('overview', controls=(c.PanelControl('hall', 'hall'),
                                                c.PanelControl('recruitment', 'dwelling'),
                                                c.PanelControl('recruitment', 'dwelling2')))
    context = c.Context('town', 'town', 'overview')
    obs = observation(window=c.Window(context, (), faction=0, town_panel=overview))
    navigate(history, c.Action('open_town', 'town', schema=c.TOWN_ACTION_SCHEMA), obs)
    if panel.kind == 'overview':
        return history
    if panel.kind == 'building':
        choice = c.BuildingChoice(panel.building.ref, panel.building.name_text_id, panel.building.status)
        hall = c.TownPanel('hall', buildings=(choice,))
        context = c.Context('town', 'town', 'hall')
        navigate(history, c.Action('open_panel', 'town', schema=c.TOWN_ACTION_SCHEMA, panel='hall'),
                 observation(window=c.Window(context, None, town_panel=hall)))
        context = c.Context('town', 'town', 'building', panel.building.ref)
        action = c.Action('open_building', 'town', schema=c.TOWN_ACTION_SCHEMA, selection=panel.building.ref)
    else:
        selected = 'dwelling' if panel.kind == 'recruitment' else None
        context = c.Context('town', 'town', panel.kind, selected)
        action = c.Action('open_panel', 'town', schema=c.TOWN_ACTION_SCHEMA, panel=panel.kind, selection=selected)
    navigate(history, action, observation(window=c.Window(context, None, town_panel=panel)))
    return history


def recruitment(stock=3):
    return c.TownPanel('recruitment', recruitment=c.RecruitmentPanel(
        'dwelling', (c.RecruitChoice('base', 0), c.RecruitChoice('upgrade', 1)), 'base',
        stock, (0, 0, 0, 0, 0, 0, 0), min(stock, 2147483645), c.RecruitDestination('upper', 'town_garrison')))


class ContextualModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(47)
        self.model = m.ContextualActorCritic(m.ModelConfig(hidden_size=12, feature_size=8))

    def output(self, history):
        state = self.model.initial_state()
        for frame in m.encode_history(history.frames):
            output = self.model.forward_step(frame, state)
            state = output.state
        return output

    def test_all_map_actions_full_rectangle_normalize(self):
        obs = observation()
        obs = replace(obs, heroes=(*obs.heroes, c.Hero('second', (2, 1, 0), 0)))
        policy = self.output(map_history(obs)).decision.distribution
        actions = [c.Action('end_turn'), c.Action('open_town', 'town')]
        for hero in obs.heroes:
            actions.append(c.Action('open_hero', hero.handle))
            actions.extend(c.Action('move_to', hero.handle, x, y) for x in range(3) for y in range(2))
        probability = sum(policy.log_prob(action).exp() for action in actions)
        self.assertAlmostEqual(float(probability.detach()), 1., places=6)
        # Unknown, blocked and zero-movement destinations stay in support.
        self.assertTrue(torch.isfinite(policy.log_prob(c.Action('move_to', 'second', 2, 1))))
        with self.assertRaises(ValueError):
            policy.log_prob(c.Action('move_to', 'hero', 3, 1))

    def test_sample_reproduces_joint_trace_and_seed(self):
        policy = self.output(map_history()).decision.distribution
        first = torch.Generator().manual_seed(123)
        second = torch.Generator().manual_seed(123)
        actions = [policy.sample(first) for _ in range(20)]
        self.assertEqual(actions, [policy.sample(second) for _ in range(20)])
        for action in actions:
            result = policy.evaluate(action)
            self.assertTrue(policy.support.offers.permits(action))
            self.assertTrue(torch.equal(result.log_prob, sum(x.log_prob for x in result.conditionals)))

    def test_resource_and_coordinate_slots_are_distinct(self):
        base = observation(resources=(1, 0, 0, 0, 0, 0, 10))
        swapped = replace(base, resources=(10, 0, 0, 0, 0, 0, 1))
        self.assertFalse(torch.allclose(self.output(map_history(base)).state, self.output(map_history(swapped)).state))
        a = replace(base, heroes=(c.Hero('hero', (1, 0, 0), 100),))
        b = replace(base, heroes=(c.Hero('hero', (0, 1, 0), 100),))
        self.assertFalse(torch.allclose(self.output(map_history(a)).state, self.output(map_history(b)).state))

    def test_primary_skill_slots_are_distinct(self):
        def history(skills):
            h = map_history()
            window = c.Window(c.Context('hero', 'hero'), (), skills, 10, 1)
            return navigate(h, c.Action('open_hero', 'hero'), observation(window=window))
        self.assertFalse(torch.allclose(self.output(history((1, 2, 3, 4))).state,
                                        self.output(history((4, 2, 3, 1))).state))

    def test_opaque_renaming_and_player_episode_identity_are_invariant(self):
        first = self.output(map_history(observation(), episode='one', player=0)).decision
        second = self.output(map_history(observation(hero='random-999', town='another'), episode='two', player=1)).decision
        self.assertTrue(torch.equal(first.value, second.value))
        self.assertTrue(torch.equal(first.distribution.log_prob(c.Action('move_to', 'hero', 2, 1)),
                                    second.distribution.log_prob(c.Action('move_to', 'random-999', 2, 1))))

    def test_entity_wire_order_does_not_change_semantics(self):
        base = observation()
        base = replace(base, heroes=(*base.heroes, c.Hero('second', (1, 0, 0), 30)))
        reversed_obs = replace(base, heroes=tuple(reversed(base.heroes)))
        first, second = self.output(map_history(base)).decision, self.output(map_history(reversed_obs)).decision
        self.assertTrue(torch.equal(first.value, second.value))
        self.assertTrue(torch.equal(first.distribution.log_prob(c.Action('open_hero', 'hero')),
                                    second.distribution.log_prob(c.Action('open_hero', 'hero'))))

    def test_different_visible_roads_are_encoded(self):
        a = observation()
        b = replace(a, tiles=(replace(a.tiles[0], road=2),))
        self.assertFalse(torch.allclose(self.output(map_history(a)).state, self.output(map_history(b)).state))

    def test_current_grid_drops_old_visibility_while_memory_retains_it(self):
        now = replace(observation(), tiles=(), objects=())
        before = observation()
        h = c.PlayerHistory('e', 0)
        h.emit('view', observation=before)
        h.emit('decision', observation=now, offers=c.Offers.from_observation(now), revision=0)
        frames = m.encode_history(h.frames)
        self.assertEqual(frames[-1].observation.tiles, ())
        self.assertFalse(torch.allclose(self.output(h).state, self.output(map_history(now)).state))
        a = self.model._encode_observation(frames[-1].observation, {})
        b = self.model._encode_observation(now, {})
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))

    def test_notice_order_content_and_relative_owner_enter_memory(self):
        def history(tokens):
            h = c.PlayerHistory('e', 0)
            h.emit('view', observation=observation())
            h.emit('event', event=c.VisibleEvent('notice', notice=c.Notice('info', tokens, ())))
            h.emit('decision', observation=observation(), offers=c.Offers.from_observation(observation()), revision=0)
            return h
        tokens = (c.NoticeToken('append_text', text_id='core.genrltxt.5'), c.NoticeToken('replace_number', number=3))
        baseline = self.output(history(tokens)).state
        self.assertFalse(torch.allclose(baseline, self.output(history(tuple(reversed(tokens)))).state))
        own = (c.NoticeToken('append_text', owner='self'),)
        other = (c.NoticeToken('append_text', owner='opponent'),)
        self.assertFalse(torch.allclose(self.output(history(own)).state, self.output(history(other)).state))

    def test_event_binding_is_explicitly_historical_and_ref_free(self):
        h = c.PlayerHistory('e', 0)
        h.emit('view', observation=observation(hero='secret-handle'))
        h.emit('event', event=c.VisibleEvent('hero_lost', subject='secret-handle'))
        encoded = m.encode_history(h.frames)[-1]
        self.assertIn('prior_observation_subject', repr(encoded.semantics))
        self.assertNotIn('secret-handle', repr(encoded.semantics))
        self.assertIsNone(encoded.observation)

    def test_submitted_action_encodes_exact_coordinates_and_count(self):
        states = []
        for x in (0, 2):
            h = map_history()
            h.emit('submitted', action=c.Action('move_to', 'hero', x, 1), decision_seq=0)
            states.append(self.output(h).state)
        self.assertFalse(torch.allclose(*states))
        states = []
        for count in (1, 3):
            h = town_history(recruitment())
            h.emit('submitted', action=c.Action('recruit', 'town', schema=c.TOWN_ACTION_SCHEMA,
                                               selection='base', count=count, destination='upper'), decision_seq=h.decision.seq)
            states.append(self.output(h).state)
        self.assertFalse(torch.allclose(*states))

    def test_strict_codec_rejects_reordering_and_cross_player_atomically(self):
        frame = map_history().frames[0]
        codec = m.ContextualEncoder('episode', 0)
        with self.assertRaises(ValueError):
            codec.encode(replace(frame, player=1))
        self.assertEqual(codec.history.frames, ())
        codec.encode(frame)
        with self.assertRaises(ValueError):
            codec.encode(frame)
        self.assertEqual(codec.history.frames, (frame,))

    def test_metadata_cutoff_has_no_neural_effect(self):
        history = map_history()
        history.emit('cutoff', reason='arbitrary diagnostic label', decision_seq=0)
        inputs = m.encode_history(history.frames)
        first = self.model.forward_step(inputs[0], self.model.initial_state())
        second = self.model.forward_step(inputs[1], first.state)
        self.assertIs(first.state, second.state)
        self.assertIsNone(second.decision)

    def test_indistinguishable_overview_controls_have_equal_probability(self):
        history = town_history(c.TownPanel('overview'))
        policy = self.output(history).decision.distribution
        a = c.Action('open_panel', 'town', schema=c.TOWN_ACTION_SCHEMA, panel='recruitment', selection='dwelling')
        b = replace(a, selection='dwelling2')
        self.assertTrue(torch.equal(policy.log_prob(a), policy.log_prob(b)))

    def test_hall_and_detail_use_visible_static_text_and_status(self):
        choices = (c.BuildingChoice('a', 'building.core.castle.dwellingLvl1.name', 'allowed'),
                   c.BuildingChoice('b', 'building.core.castle.tavern.name', 'forbidden'))
        policy = self.output(town_history(c.TownPanel('hall', buildings=choices))).decision.distribution
        actions = [c.Action('back', 'town', schema=c.TOWN_ACTION_SCHEMA),
                   *(c.Action('open_building', 'town', schema=c.TOWN_ACTION_SCHEMA, selection=choice.ref) for choice in choices)]
        self.assertAlmostEqual(float(sum(policy.log_prob(a).exp() for a in actions).detach()), 1., places=6)
        detail = c.BuildingDetails('a', choices[0].name_text_id, 'building.core.castle.dwellingLvl1.description', 'allowed', (0, 0, 0, 0, 0, 0, 100))
        policy = self.output(town_history(c.TownPanel('building', building=detail))).decision.distribution
        self.assertTrue(torch.isfinite(policy.log_prob(c.Action('construct', 'town', schema=c.TOWN_ACTION_SCHEMA, selection='a'))))
        forbidden = replace(detail, status='forbidden')
        policy = self.output(town_history(c.TownPanel('building', building=forbidden))).decision.distribution
        with self.assertRaises(ValueError):
            policy.log_prob(c.Action('construct', 'town', schema=c.TOWN_ACTION_SCHEMA, selection='a'))

    def test_small_count_distribution_enumerates_to_one_with_other_actions(self):
        policy = self.output(town_history(recruitment(7))).decision.distribution
        actions = [c.Action('back', 'town', schema=c.TOWN_ACTION_SCHEMA)]
        actions += [c.Action('select_recruit', 'town', schema=c.TOWN_ACTION_SCHEMA, selection=card) for card in ('base', 'upgrade')]
        actions += [c.Action('recruit', 'town', schema=c.TOWN_ACTION_SCHEMA, selection='base', destination='upper', count=count) for count in range(1, 8)]
        self.assertAlmostEqual(float(sum(policy.log_prob(a).exp() for a in actions).detach()), 1., places=6)
        with self.assertRaises(ValueError):
            policy.log_prob(replace(actions[-1], count=8))
        with self.assertRaises(ValueError):
            policy.log_prob(replace(actions[-1], destination='foreign'))
        with self.assertRaises(ValueError):
            policy.log_prob(replace(actions[-1], selection='upgrade'))

    def test_large_count_support_is_logarithmic_and_finite(self):
        policy = self.output(town_history(recruitment(2**31 - 1))).decision.distribution
        for count in (1, 1073741823, 2147483645):
            action = c.Action('recruit', 'town', schema=c.TOWN_ACTION_SCHEMA, selection='base', destination='upper', count=count)
            result = policy.evaluate(action)
            self.assertLessEqual(len(result.conditionals), 32)
            self.assertTrue(torch.isfinite(result.log_prob))
        self.assertFalse(hasattr(policy, 'entropy'))

    def test_policy_value_and_history_encoder_have_gradients(self):
        history = town_history(recruitment(7))
        output = self.output(history)
        action = c.Action('recruit', 'town', schema=c.TOWN_ACTION_SCHEMA, selection='base', destination='upper', count=3)
        loss = -output.decision.distribution.log_prob(action) + output.decision.value.square()
        loss.backward()
        for module in (self.model.gru, self.model.text_gru, self.model.quantity, self.model.value):
            self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in module.parameters()))

    def test_model_config_and_weight_digest_are_stable_and_sensitive(self):
        same = m.ContextualActorCritic(m.ModelConfig.from_dict(self.model.config.to_dict()))
        same.load_state_dict(self.model.state_dict())
        self.assertEqual(m.weights_sha256(self.model), m.weights_sha256(same))
        with torch.no_grad():
            next(same.parameters())[0, 0] += 1
        self.assertNotEqual(m.weights_sha256(self.model), m.weights_sha256(same))
        with self.assertRaises(ValueError):
            m.ModelConfig.from_dict({'hidden_size': 12})
        with self.assertRaises(ValueError):
            m.ModelConfig(schema='future')

    def test_window_closure_removes_private_features_from_current_input(self):
        h = map_history()
        window = c.Window(c.Context('hero', 'hero'), (c.Stack(0, 5, 23),), (1, 2, 3, 4), 20, 2)
        navigate(h, c.Action('open_hero', 'hero'), observation(window=window))
        seq = h.decision.seq
        h.emit('submitted', action=c.Action('close'), decision_seq=seq)
        h.emit('consumed', consumer='adapter', decision_seq=seq)
        h.emit('context', context=c.Context(), revision=h.revision + 1, decision_seq=seq)
        h.emit('view', observation=observation())
        h.emit('result', result=c.Result('closed', True), decision_seq=seq)
        h.emit('decision', observation=observation(), offers=c.Offers.from_observation(observation()), revision=h.revision + 1)
        final = m.encode_history(h.frames)[-1]
        self.assertIsNone(final.observation.window)
        self.assertNotIn('Stack', repr(final.semantics))
        self.assertFalse(torch.allclose(self.output(h).state, self.output(map_history()).state))


if __name__ == '__main__':
    unittest.main()
