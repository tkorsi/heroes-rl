"""Synthetic contract checks only; these do not validate real VCMI episodes."""
from dataclasses import replace
import math
import unittest

from experiments.learner.actions import ACTION_SCHEMA, ActionArgument, CandidateSet, movement_candidates
from experiments.learner.contract import Frame, OwnState, Tile, View
from experiments.learner.trajectory import (
    DecisionCutoff, DecisionRecord, NaturalEnd, PlayerTrajectory,
    TechnicalFailure, TerminalVerification, recurrent_provenance,
)


CHECKPOINT = "actor-critic:fixture-v1"
OPPONENT = "frozen:fixture-v0"


def view(hero=True):
    own = OwnState((1, 1, 0) if hero else None, 10 if hero else 0,
                   10 if hero else 0, (0,) * 7, (), ((2, 2, 0),), 0)
    return View((Tile((1, 1, 0), 0, False, False),), (), own)


def history(player=0, *, pending=False, cutoff=False, hero=True):
    current = view(hero)
    mask = (hero,) * 8 + (True,)
    frames = (
        Frame("episode-1", player, 0, 1, "decision", view=current, mask=mask),
        Frame("episode-1", player, 1, 1, "action", action=8, decision_seq=0),
        Frame("episode-1", player, 2, 1, "result", action=8, accepted=True, decision_seq=0),
        Frame("episode-1", player, 3, 2, "view", view=current),
        Frame("episode-1", player, 4, 2, "decision", view=current, mask=mask),
        Frame("episode-1", player, 5, 2, "action", action=8, decision_seq=4),
    )
    if pending:
        return frames
    frames += (
        Frame("episode-1", player, 6, 2, "result", action=8, accepted=True, decision_seq=4),
        Frame("episode-1", player, 7, 3, "view", view=current),
    )
    if cutoff:
        frames += (Frame("episode-1", player, 8, 3, "decision", view=current, mask=mask),)
    return frames


def records(frames):
    return tuple(DecisionRecord(frames[seq], 8, -math.log(sum(frames[seq].mask)),
                                value, CHECKPOINT, OPPONENT,
                                recurrent_provenance(frames[:seq + 1], CHECKPOINT))
                 for seq, value in ((0, 0.1), (4, -0.2)))


def natural(frames, *, winner=0):
    other = history(1 - frames[0].player, pending=True)
    by_player = sorted((frames, other), key=lambda h: h[0].player)
    outcomes = ("win", "loss") if winner == 0 else ("loss", "win")
    verification = TerminalVerification(
        "episode-1", "fixture:externally-verified-finalization", outcomes, outcomes,
        0, (True, True), tuple(h[-1].seq for h in by_player),
        tuple(recurrent_provenance(h, CHECKPOINT).history_sha256 for h in by_player))
    return NaturalEnd(verification)


def trajectory(frames=None, boundary=None, *, role="learner"):
    frames = history() if frames is None else frames
    boundary = natural(frames) if boundary is None else boundary
    return PlayerTrajectory("episode-1", frames[0].player, CHECKPOINT, OPPONENT,
                            role, frames, records(frames), boundary)


class LearnerTrajectoryTests(unittest.TestCase):
    def test_exact_versioned_candidates_are_stored_with_selected_behavior_log_probability(self):
        record = records(history())[0]
        self.assertEqual(record.action_candidates, movement_candidates(record.frame.mask))
        self.assertEqual(record.action_candidates.schema, ACTION_SCHEMA)
        self.assertEqual(record.selected_candidate_key, "8")
        self.assertIn(record.selected_candidate_key,
                      [candidate.key for candidate in record.action_candidates.candidates])
        self.assertEqual(record.behavior_log_prob, -math.log(9))
        explicit = replace(record, action_candidates=movement_candidates(record.frame.mask))
        self.assertEqual(explicit, record)

    def test_changed_candidate_parameters_or_support_are_rejected(self):
        record = records(history())[0]
        support = record.action_candidates.candidates
        wrong_direction = replace(support[0], arguments=(ActionArgument(0, "category", 9),))
        wrong_type = replace(support[0], type_id=2)
        wrong_key = replace(support[0], key="other-routing-key")
        for candidates in (CandidateSet(support[1:]),
                           CandidateSet((wrong_direction,) + support[1:]),
                           CandidateSet((wrong_type,) + support[1:]),
                           CandidateSet((wrong_key,) + support[1:])):
            with self.subTest(candidates=candidates), self.assertRaisesRegex(ValueError, "support and parameters"):
                replace(record, action_candidates=candidates)
        town_only = records(history(hero=False))[0]
        with self.assertRaisesRegex(ValueError, "support and parameters"):
            replace(town_only, action_candidates=record.action_candidates)

    def test_ordinary_transition_waits_for_same_player_next_decision(self):
        targets = trajectory().transition_targets()
        self.assertEqual(targets[0].next_decision_seq, 4)
        self.assertEqual(targets[0].reward, 0.0)
        self.assertEqual(targets[0].bootstrap_value, -0.2)
        # Its own action_result at seq2 is not the transition boundary.
        self.assertNotEqual(targets[0].next_decision_seq, 2)

    def test_gamma_one_terminal_rewards_and_returns_are_player_relative(self):
        for player, expected in ((0, 1.0), (1, -1.0)):
            with self.subTest(player=player):
                item = trajectory(history(player))
                targets = item.transition_targets()
                self.assertEqual([t.reward for t in targets], [0.0, expected])
                self.assertEqual([t.return_target for t in targets], [expected, expected])
                self.assertEqual(targets[-1].bootstrap_value, 0.0)
                self.assertIsNone(targets[-1].next_decision_seq)
                self.assertTrue(all(t.eligible for t in targets))

    def test_natural_terminal_closes_pending_own_action_without_result(self):
        item = trajectory(history(pending=True))
        self.assertEqual(item.history[-1].kind, "action")
        self.assertEqual(item.transition_targets()[-1].reward, 1.0)
        self.assertTrue(item.eligible)

    def test_natural_terminal_supersedes_unacted_decision_without_fabricated_action(self):
        frames = history(cutoff=True)
        item = trajectory(frames, natural(frames))
        targets = item.transition_targets()
        self.assertEqual(item.history[-1].kind, "decision")
        self.assertEqual(item.history[-1].seq, 8)
        self.assertEqual(sum(f.kind == "action" for f in item.history), 2)
        self.assertEqual([t.decision_seq for t in targets], [0, 4])
        self.assertEqual(targets[-1].reward, 1.0)
        self.assertEqual(targets[-1].bootstrap_value, 0.0)
        self.assertIsNone(targets[-1].next_decision_seq)
        self.assertTrue(item.eligible)

    def test_verified_horizon_uses_next_decision_and_exact_history_prefix(self):
        frames = history(cutoff=True)
        boundary = DecisionCutoff(frames[-1], 0.35, recurrent_provenance(frames, CHECKPOINT))
        item = trajectory(frames, boundary)
        targets = item.transition_targets()
        self.assertEqual([t.reward for t in targets], [0.0, 0.0])
        self.assertEqual([t.return_target for t in targets], [0.35, 0.35])
        self.assertEqual(targets[-1].next_decision_seq, 8)
        self.assertEqual(targets[-1].bootstrap_value, 0.35)

    def test_technical_failure_has_no_rewards_bootstrap_or_targets(self):
        for reason in ("wall_timeout", "engine_failure", "missing_terminal_evidence"):
            with self.subTest(reason=reason):
                item = trajectory(history(pending=True), TechnicalFailure(reason))
                self.assertFalse(item.eligible)
                for target in item.transition_targets():
                    self.assertFalse(target.eligible)
                    self.assertIsNone(target.reward)
                    self.assertIsNone(target.bootstrap_value)
                    self.assertIsNone(target.return_target)

    def test_frozen_opponent_records_never_become_on_policy_samples(self):
        item = trajectory(role="frozen_opponent")
        self.assertFalse(item.eligible)
        self.assertEqual(item.transition_targets()[-1].reward, 1.0)
        for target in item.transition_targets():
            self.assertFalse(target.eligible)
            self.assertIsNone(target.return_target)
            self.assertIsNone(target.bootstrap_value)

    def test_missing_hero_with_town_is_not_a_terminal(self):
        frames = history(cutoff=True, hero=False)
        boundary = DecisionCutoff(frames[-1], -0.1, recurrent_provenance(frames, CHECKPOINT))
        item = trajectory(frames, boundary)
        self.assertIsNone(item.history[-1].view.own.hero)
        self.assertTrue(item.history[-1].view.own.towns)
        self.assertEqual([t.reward for t in item.transition_targets()], [0.0, 0.0])
        self.assertEqual(item.transition_targets()[-1].bootstrap_value, -0.1)

    def test_cross_player_episode_and_history_holes_are_rejected(self):
        original = trajectory()
        mixed = original.history[:3] + (replace(original.history[3], player=1),) + original.history[4:]
        wrong_episode = (replace(original.history[0], episode="episode-2"),) + original.history[1:]
        missing = original.history[:3] + original.history[4:]
        duplicate = original.history[:3] + original.history[2:]
        for frames in (mixed, wrong_episode, missing, duplicate):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                replace(original, history=frames)
        with self.assertRaises(ValueError):
            replace(original, player=1)

    def test_checkpoint_change_and_opponent_change_are_rejected(self):
        original = trajectory()
        second = original.decisions[1]
        prefix = replace(second.recurrent_prefix, checkpoint="new:checkpoint")
        changed = replace(second, behavior_checkpoint="new:checkpoint", recurrent_prefix=prefix)
        opponent_changed = replace(second, opponent_checkpoint="new:opponent")
        for record in (changed, opponent_changed):
            with self.subTest(record=record), self.assertRaises(ValueError):
                replace(original, decisions=(original.decisions[0], record))

    def test_missing_duplicate_reordered_and_mismatched_behavior_records_rejected(self):
        original = trajectory()
        first, second = original.decisions
        wrong_action = replace(first, action=0)
        for decisions in ((first,), (first, first), (second, first), (wrong_action, second)):
            with self.subTest(decisions=decisions), self.assertRaises(ValueError):
                replace(original, decisions=decisions)

    def test_provenance_rejects_altered_intermediate_view(self):
        original = trajectory()
        old = original.history[3]
        changed = replace(old, view=replace(old.view, own=replace(old.view.own, mines=1)))
        frames = original.history[:3] + (changed,) + original.history[4:]
        # Same decision snapshots and sequence numbers cannot hide a changed
        # intervening observation from the recurrent history provenance.
        with self.assertRaisesRegex(ValueError, "exact same-player history"):
            replace(original, history=frames, boundary=natural(frames))

    def test_cutoff_cannot_use_other_player_or_stale_checkpoint_history(self):
        frames = history(cutoff=True)
        correct = DecisionCutoff(frames[-1], 0.2, recurrent_provenance(frames, CHECKPOINT))
        original = trajectory(frames, correct)
        for prefix in (replace(correct.recurrent_prefix, player=1),
                       replace(correct.recurrent_prefix, checkpoint=OPPONENT),
                       original.decisions[-1].recurrent_prefix,
                       replace(correct.recurrent_prefix, history_sha256="0" * 64)):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                replace(original, boundary=replace(correct, recurrent_prefix=prefix))
        with self.assertRaises(ValueError):
            replace(original, boundary=replace(correct, frame=replace(correct.frame, player=1)))

    def test_no_cutoff_from_action_result_or_invented_next_decision(self):
        frames = history()
        with self.assertRaises(ValueError):
            DecisionCutoff(frames[6], 0.0, recurrent_provenance(frames, CHECKPOINT))
        longer = history(cutoff=True)
        invented = DecisionCutoff(longer[-1], 0.0, recurrent_provenance(longer, CHECKPOINT))
        with self.assertRaises(ValueError):
            trajectory(frames, invented)

    def test_terminal_requires_agreeing_complementary_results_and_clean_exit(self):
        original = natural(history()).verification
        for changes in ({"engine_outcomes": ("win", "win")},
                        {"engine_outcomes": ("draw", "draw")},
                        {"socket_outcomes": ("loss", "win")},
                        {"process_exit_code": -9},
                        {"histories_complete": (True, False)},
                        {"evidence_ref": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(original, **changes)

    def test_terminal_finalization_must_cover_exact_current_history(self):
        original = trajectory()
        verification = original.boundary.verification
        for changes in ({"episode": "episode-2"},
                        {"history_last_seq": (6, 5)},
                        {"history_sha256": ("0" * 64, verification.history_sha256[1])}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(original, boundary=NaturalEnd(replace(verification, **changes)))

    def test_invalid_behavior_scalars_masks_and_prefixes_rejected(self):
        record = records(history(hero=False))[0]
        for changes in ({"action": 0}, {"behavior_log_prob": 0.1},
                        {"behavior_log_prob": float("nan")},
                        {"behavior_value": float("inf")},
                        {"recurrent_prefix": replace(record.recurrent_prefix, player=1)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(record, **changes)


if __name__ == "__main__":
    unittest.main()
