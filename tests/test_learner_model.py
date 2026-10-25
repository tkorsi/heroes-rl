"""Offline component tests; fixtures are not gameplay or learning evidence."""
from dataclasses import replace
import io
import unittest

from experiments.learner.contract import Frame, ObjectView, OwnState, Stack, Tile, View
from experiments.learner.actions import ActionArgument, ActionCandidate, CandidateSet

try:
    import torch
except ImportError:
    torch = None
else:
    from experiments.learner.model import PlayerSession, RecurrentActorCritic


MASK = (True, True, False, False, False, False, False, False, True)


def view(enemy=False, town_only=False, player=0):
    own = OwnState(
        hero=None if town_only else (2 + player, 3, 0),
        movement=0 if town_only else 800,
        movement_max=0 if town_only else 1500,
        resources=(3, 2, 1, 0, 0, 0, 500),
        army=() if town_only else (Stack(0, 1, 25), Stack(3, 2, 4)),
        towns=((2, 3, 0), (3, 3, 0)), mines=0,
    )
    tiles = tuple(Tile(p, t, False, True) for p, t in (
        ((2, 3, 0), 1), ((3, 3, 0), 1), ((4, 3, 0), 2), ((5, 3, 0), 2)))
    objects = (ObjectView("local-town", (2, 3, 0), 98, 0, "self"),)
    if enemy:
        objects += (ObjectView("first-sighting", (5, 3, 0), 137, 2, "opponent"),)
    return View(tiles, objects, own)


def history(player=0, episode="episode-a"):
    current = view(player=player)
    return (
        Frame(episode, player, 0, 1, "view", view=view(enemy=True, player=player)),
        Frame(episode, player, 1, 1, "decision", view=current, mask=MASK),
        Frame(episode, player, 2, 1, "action", action=8, decision_seq=1),
        Frame(episode, player, 3, 1, "result", action=8, accepted=True, decision_seq=1),
        Frame(episode, player, 4, 2, "view", view=current),
        Frame(episode, player, 5, 2, "decision", view=current, mask=MASK),
    )


@unittest.skipIf(torch is None, "Optional learner dependency PyTorch is not installed")
class LearnerModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny CPU fixtures should not fan out over all host cores.
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(23)
        self.model = RecurrentActorCritic()

    def assert_decisions_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
            torch.testing.assert_close(a.value, b.value, rtol=0, atol=0)

    def test_attempt_mask_and_end_turn_only_are_valid_distributions(self):
        for town_only, mask in ((False, MASK), (True, (False,) * 8 + (True,))):
            frame = Frame("episode", 0, 0, 1, "decision",
                          view=view(town_only=town_only), mask=mask)
            decision = self.model.forward_step(frame, self.model.initial_state()).decision
            distribution = decision.distribution()
            self.assertTrue(torch.isfinite(distribution.entropy()).item())
            self.assertTrue(torch.isfinite(distribution.log_prob(torch.tensor(8))).item())
            self.assertTrue(torch.isfinite(decision.value).item())
            for action, enabled in enumerate(mask):
                self.assertEqual(distribution.probs[action].item() > 0, enabled)
            if town_only:
                self.assertEqual(distribution.probs[8].item(), 1)
                self.assertEqual(distribution.entropy().item(), 0)
                self.assertEqual(distribution.sample((32,)).tolist(), [8] * 32)

    def test_actor_and_value_both_backpropagate_through_prior_sighting(self):
        first = Frame("episode", 0, 0, 1, "view", view=view(enemy=True))
        final = Frame("episode", 0, 1, 1, "decision", view=view(), mask=MASK)
        for head in ("actor", "value"):
            with self.subTest(head=head):
                self.model.zero_grad(set_to_none=True)
                earlier = self.model.forward_step(first, self.model.initial_state())
                self.assertIsNone(earlier.decision)
                earlier.state.retain_grad()
                decision = self.model.forward_step(final, earlier.state).decision
                loss = (-decision.distribution().log_prob(torch.tensor(0)) if head == "actor"
                        else (decision.value - 0.75).square())
                loss.backward()
                self.assertGreater(earlier.state.grad.abs().sum().item(), 0)
                self.assertGreater(self.model.gru.weight_hh.grad.abs().sum().item(), 0)
                # Kind 137 exists only in the earlier sighting, not the decision.
                self.assertGreater(self.model.object_kind.weight.grad[138].abs().sum().item(), 0)
                active_head = self.model.policy if head == "actor" else self.model.value
                self.assertGreater(active_head.weight.grad.abs().sum().item(), 0)
                if head == "actor":
                    for embedding in (self.model.candidate_type, self.model.argument_slot,
                                      self.model.argument_kind, self.model.argument_category):
                        self.assertGreater(embedding.weight.grad.abs().sum().item(), 0)

    def test_candidate_permutation_renaming_and_argument_order_are_equivariant(self):
        candidates = CandidateSet((
            ActionCandidate("opaque-a", 7, (ActionArgument(0, "category", 12),
                                            ActionArgument(2, "scalar", -500))),
            ActionCandidate("opaque-b", 19, (ActionArgument(3, "position", (3, 8, 0)),)),
            ActionCandidate("opaque-c", 43),
        ))
        permuted = CandidateSet(tuple(replace(candidate, key=f"other-{i}",
                                              arguments=tuple(reversed(candidate.arguments)))
                                      for i, candidate in enumerate(reversed(candidates.candidates))))
        state = self.model.forward_step(history()[0], self.model.initial_state()).state
        expected = self.model.score_candidates(state, candidates)
        actual = self.model.score_candidates(state, permuted)
        torch.testing.assert_close(actual, expected.flip(0), rtol=0, atol=0)

    def test_variable_candidate_counts_and_typed_values_are_contextual(self):
        state = self.model.forward_step(history()[0], self.model.initial_state()).state
        for count in (1, 4, 13):
            candidates = CandidateSet(tuple(
                ActionCandidate(f"choice-{i}", 31, (
                    ActionArgument(1, "category", i),
                    ActionArgument(2, "scalar", (-1) ** i * 1e20),
                    ActionArgument(3, "position", (i, 19 - i, 0)),
                )) for i in range(count)))
            logits = self.model.score_candidates(state, candidates)
            self.assertEqual(logits.shape, (count,))
            self.assertTrue(torch.isfinite(logits).all().item())
            self.assertTrue(torch.isfinite(torch.distributions.Categorical(logits=logits)
                                           .entropy()).item())
            if count > 1:
                other = self.model.score_candidates(torch.zeros_like(state), candidates)
                # Changing history alters relative preferences, not merely a
                # shared additive constant that softmax would discard.
                change = logits - other
                self.assertGreater((change - change.mean()).abs().max().item(), 1e-7)

    def test_candidate_parameter_slots_preserve_value_binding(self):
        a = ActionCandidate("a", 17, (ActionArgument(0, "category", 7),
                                     ActionArgument(1, "category", 11)))
        b = ActionCandidate("b", 17, (ActionArgument(0, "category", 11),
                                     ActionArgument(1, "category", 7)))
        state = self.model.forward_step(history()[0], self.model.initial_state()).state
        logits = self.model.score_candidates(state, CandidateSet((a, b)))
        self.assertGreater(abs((logits[0] - logits[1]).item()), 1e-7)

    def test_transient_sighting_changes_memory_with_identical_final_snapshot(self):
        sessions = [PlayerSession(self.model, "episode", 0) for _ in range(2)]
        final = Frame("episode", 0, 1, 1, "decision", view=view(), mask=MASK)
        outputs = []
        for session, enemy in zip(sessions, (False, True)):
            self.assertIsNone(session.consume(Frame("episode", 0, 0, 1, "view", view=view(enemy))))
            outputs.append(session.consume(final))
        self.assertGreater((sessions[0].state - sessions[1].state).abs().max().item(), 1e-7)
        self.assertGreater((outputs[0].logits[list(MASK)] - outputs[1].logits[list(MASK)])
                           .abs().max().item(), 1e-7)
        # A view encoder has no cache in which to silently keep stale objects.
        expected = self.model.encode_frame(final)
        self.model.encode_frame(Frame("episode", 0, 0, 1, "view", view=view(True)))
        torch.testing.assert_close(self.model.encode_frame(final), expected, rtol=0, atol=0)

    def test_shared_weights_preserve_independent_interleaved_player_memory(self):
        streams = [history(player=side) for side in (0, 1)]
        separate = [PlayerSession(self.model, "episode-a", side) for side in (0, 1)]
        expected = [session.consume_chunk(stream) for session, stream in zip(separate, streams)]
        interleaved = [PlayerSession(self.model, "episode-a", side) for side in (0, 1)]
        actual = [[], []]
        for pair in zip(*streams):
            for side, frame in enumerate(pair):
                output = interleaved[side].consume(frame)
                if output is not None:
                    actual[side].append(output)
        for side in (0, 1):
            self.assert_decisions_equal(actual[side], expected[side])
            torch.testing.assert_close(interleaved[side].state, separate[side].state, rtol=0, atol=0)

    def test_finish_then_fresh_episode_resets_memory(self):
        session = PlayerSession(self.model, "episode-a", 0)
        session.consume_chunk(history())
        with self.assertRaisesRegex(ValueError, "Finish"):
            session.reset("episode-b")
        session.finish()
        with self.assertRaisesRegex(ValueError, "finished"):
            session.consume(history()[0])
        with self.assertRaisesRegex(ValueError, "fresh"):
            session.reset("episode-a")
        session.reset("episode-b")
        fresh = PlayerSession(self.model, "episode-b", 0)
        self.assert_decisions_equal(session.consume_chunk(history(episode="episode-b")),
                                    fresh.consume_chunk(history(episode="episode-b")))

    def test_handles_order_and_transport_metadata_do_not_enter_features(self):
        source = view(True)
        renamed = replace(source,
                          tiles=tuple(reversed(source.tiles)),
                          objects=tuple(replace(obj, handle=f"unrelated-{i}")
                                        for i, obj in enumerate(reversed(source.objects))),
                          own=replace(source.own, army=tuple(reversed(source.own.army)),
                                      towns=tuple(reversed(source.own.towns))))
        a = Frame("episode-a", 0, 0, 2, "decision", view=source, mask=MASK)
        b = Frame("different-episode", 1, 121, 2, "decision", view=renamed, mask=MASK)
        torch.testing.assert_close(self.model.encode_frame(a), self.model.encode_frame(b),
                                   rtol=0, atol=0)
        action = Frame("episode-a", 0, 2, 2, "action", action=1, decision_seq=1)
        other = replace(action, episode="x", player=1, seq=209, decision_seq=207)
        torch.testing.assert_close(self.model.encode_frame(action), self.model.encode_frame(other),
                                   rtol=0, atol=0)

    def test_state_dict_roundtrip_preserves_decisions(self):
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = RecurrentActorCritic()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        a = PlayerSession(self.model, "episode-a", 0).consume_chunk(history())
        b = PlayerSession(restored, "episode-a", 0).consume_chunk(history())
        self.assert_decisions_equal(a, b)

    def test_colocated_objects_preserve_type_owner_binding(self):
        source = view()
        objects = (
            ObjectView("town", (2, 3, 0), 98, 0, "self"),
            ObjectView("hero", (2, 3, 0), 34, 1, "opponent"),
        )
        swapped = (replace(objects[0], owner="opponent"),
                   replace(objects[1], owner="self"))
        a = Frame("episode", 0, 0, 1, "view", view=replace(source, objects=objects))
        b = replace(a, view=replace(source, objects=swapped))
        # The same bag of types and owners must not collapse these two states:
        # our town with an opposing hero differs from their town with our hero.
        encoded = self.model.encode_frame(a)
        self.assertGreater((encoded - self.model.encode_frame(b)).abs().max().item(), 1e-7)
        reordered = replace(a, view=replace(source, objects=tuple(
            replace(obj, handle=f"renamed-{i}") for i, obj in enumerate(reversed(objects)))))
        torch.testing.assert_close(encoded, self.model.encode_frame(reordered), rtol=0, atol=0)

    def test_chunks_and_player_turns_do_not_reset_memory(self):
        stream = history()
        full = PlayerSession(self.model, "episode-a", 0)
        expected = full.consume_chunk(stream)
        chunked = PlayerSession(self.model, "episode-a", 0)
        actual = (chunked.consume_chunk(stream[:2]) + chunked.consume_chunk(stream[2:4])
                  + chunked.consume_chunk(()) + chunked.consume_chunk(stream[4:]))
        self.assert_decisions_equal(actual, expected)
        torch.testing.assert_close(chunked.state, full.state, rtol=0, atol=0)
        self.assertFalse(chunked.state.requires_grad)
        self.assertTrue(all(not d.value.requires_grad for d in actual))

    def test_invalid_identity_or_sequence_does_not_advance_session(self):
        session = PlayerSession(self.model, "episode-a", 0)
        stream = history()
        for invalid in (replace(stream[0], player=1), replace(stream[0], episode="wrong"),
                        replace(stream[0], seq=1)):
            with self.assertRaises(ValueError):
                session.consume(invalid)
            self.assertEqual(session.cursor.next_seq, 0)
        session.consume(stream[0])
        state = session.state
        with self.assertRaises(ValueError):
            session.consume(stream[0])
        self.assertEqual(session.cursor.next_seq, 1)
        torch.testing.assert_close(session.state, state, rtol=0, atol=0)
        session.consume(stream[1])
        with self.assertRaises(ValueError):
            session.consume(replace(stream[2], action=2))  # Masked attempt.
        with self.assertRaises(ValueError):
            session.consume(replace(stream[2], decision_seq=0))
        self.assertEqual(session.cursor.next_seq, 2)
        session.consume_chunk(stream[2:])

    def test_actions_and_results_update_memory_without_policy_output(self):
        session = PlayerSession(self.model, "episode-a", 0)
        stream = history()
        session.consume_chunk(stream[:2])
        before = session.state
        self.assertIsNone(session.consume(stream[2]))
        after_action = session.state
        self.assertIsNone(session.consume(stream[3]))
        self.assertFalse(torch.equal(before, after_action))
        self.assertFalse(torch.equal(after_action, session.state))


if __name__ == "__main__":
    unittest.main()
