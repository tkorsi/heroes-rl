"""Synthetic trajectory-only terminal closure checks; no native engine claims."""
from dataclasses import FrozenInstanceError, replace
import json
import unittest

from experiments.learner import contextual as c


def decision(history):
    obs = c.Observation((2, 2, 1), (c.Tile((0, 0, 0), 1, 0, False, False),), (), (0,) * 7, (), ())
    return history.emit('decision', observation=obs, revision=history.revision + 1,
                        offers=c.Offers.from_observation(obs, schema=history.profile.action_schema))


def action(history, *, consumed=True, commands='pending'):
    seq = history.decision.seq
    history.emit('submitted', decision_seq=seq, action=c.Action('end_turn', schema=history.profile.action_schema))
    if consumed:
        history.emit('consumed', decision_seq=seq, consumer='engine')
        if commands != 'none':
            history.emit('issued', decision_seq=seq, command_index=0)
            if commands != 'pending':
                history.emit('acknowledged', decision_seq=seq, command_index=0, accepted=commands == 'accepted')
    return seq


def completed(history):
    decision(history)
    seq = action(history, commands='accepted')
    history.emit('result', decision_seq=seq, result=c.Result('turn_ended', True))


def terminal(history, outcome, *, reason='terminal', final_view=True):
    if history.decision is not None:
        history.emit('superseded', decision_seq=history.decision.seq, reason=reason)
    history.emit('terminal', outcome=outcome)
    if final_view: history.emit('view', observation=history.frames[0].observation)


def finalize(pair, outcomes=('loss', 'win')):
    c.finalize_natural(pair, engine_outcomes=outcomes, socket_outcomes=outcomes,
                       process_exit_code=0, evidence_ref='synthetic-terminal-proof')


def histories(schema=c.TOWN_SCHEMA):
    return tuple(c.PlayerHistory('synthetic-terminal', p, schema=schema) for p in (0, 1))


class TerminalTrajectoryTests(unittest.TestCase):
    def test_both_colors_closure_keeps_history_and_previous_reward_unchanged(self):
        for schema in (c.SCHEMA, c.TOWN_SCHEMA):
            for outcomes in (('loss', 'win'), ('win', 'loss')):
                pair = histories(schema)
                for h in pair:
                    completed(h); decision(h); action(h)
                    terminal(h, outcomes[h.player])
                finalize(pair, outcomes)
                for h in pair:
                    before = json.dumps([c.encode_frame(f) for f in h.frames], sort_keys=True)
                    legacy = h.transitions()
                    self.assertEqual(len(legacy), 1)
                    self.assertEqual(legacy[0].reward, 0)
                    first, last = h.transitions(schema=c.TERMINAL_TRAJECTORY_SCHEMA)
                    self.assertEqual(first, legacy[0])
                    self.assertEqual(first.successor_seq, last.decision_seq)
                    self.assertIsNone(last.result_seq)
                    self.assertEqual(last.reward, -1 if h.terminal == 'loss' else 1)
                    self.assertTrue(last.eligible)
                    self.assertFalse(last.bootstrap_allowed)
                    self.assertIsNone(last.successor_seq)
                    closure = last.terminal_closure
                    self.assertEqual([h.frames[n].kind for n in vars(closure).values()],
                                     ['consumed', 'superseded', 'terminal', 'finalized'])
                    self.assertEqual(before, json.dumps([c.encode_frame(f) for f in h.frames], sort_keys=True))
                    with self.assertRaises(FrozenInstanceError): closure.terminal_seq = 0

    def test_terminal_closure_does_not_assert_native_issuance_or_ack_success(self):
        for commands in ('none', 'pending', 'accepted', 'rejected'):
            pair = histories()
            for h in pair:
                decision(h); action(h, commands=commands); terminal(h, ('loss', 'win')[h.player])
            finalize(pair)
            for h in pair:
                sample, = h.transitions(schema=c.TERMINAL_TRAJECTORY_SCHEMA)
                self.assertIsNotNone(sample.terminal_closure)
                self.assertIsNone(sample.result_seq)
                self.assertFalse(any(f.kind == 'result' for f in h.frames))
                self.assertEqual(sum(f.kind == 'acknowledged' for f in h.frames), int(commands in ('accepted', 'rejected')))

    def test_player_arrival_order_and_postterminal_views_do_not_change_credit(self):
        for first_player in (0, 1):
            pair = histories()
            for h in pair: decision(h); action(h)
            for player in (first_player, 1-first_player): terminal(pair[player], ('loss', 'win')[player])
            self.assertTrue(all(not h.transitions(schema=c.TERMINAL_TRAJECTORY_SCHEMA) for h in pair))
            finalize(pair)
            self.assertEqual([h.transitions(schema=c.TERMINAL_TRAJECTORY_SCHEMA)[0].reward for h in pair], [-1, 1])
            self.assertTrue(all(h.frames[-2].kind == 'view' for h in pair))

    def test_submitted_only_intentions_never_create_a_closure(self):
        pair = histories()
        for h in pair:
            decision(h); action(h, consumed=False); terminal(h, ('loss', 'win')[h.player])
        finalize(pair)
        self.assertTrue(all(not h.transitions(schema=c.TERMINAL_TRAJECTORY_SCHEMA) for h in pair))

    def test_failure_after_terminal_never_creates_reward_or_closure(self):
        pair = histories()
        for h in pair:
            completed(h); decision(h); action(h); terminal(h, ('loss', 'win')[h.player])
            h.emit('failure', reason='synthetic-late-reader-failure')
            for sample in h.transitions(schema=c.TERMINAL_TRAJECTORY_SCHEMA):
                self.assertFalse(sample.eligible or sample.bootstrap_allowed)
                self.assertIsNone(sample.reward)
                self.assertIsNone(sample.terminal_closure)

    def test_cutoff_never_closes_an_earlier_unresolved_action(self):
        pair = histories()
        for h in pair:
            decision(h); action(h)
            h.emit('superseded', decision_seq=h.decision.seq, reason='synthetic-visible-update')
            d = decision(h); h.emit('cutoff', decision_seq=d.seq, reason='synthetic-limit')
        c.finalize_cutoff(pair, evidence_ref='synthetic-cutoff', histories_complete=(True, True), failure_free=True,
                          shutdown_verified=True, continuation_limit=0, continuation_steps=(0, 0))
        self.assertTrue(all(not h.transitions(schema=c.TERMINAL_TRAJECTORY_SCHEMA) for h in pair))

    def test_nonterminal_supersession_or_later_unconsumed_decision_is_ambiguous(self):
        for later in (False, True):
            pair = histories()
            for h in pair:
                completed(h); decision(h); action(h)
                if later:
                    h.emit('superseded', decision_seq=h.decision.seq, reason='terminal')
                    decision(h)
                    terminal(h, ('loss', 'win')[h.player])
                else: terminal(h, ('loss', 'win')[h.player], reason='stale')
            finalize(pair)
            for h in pair:
                samples = h.transitions(schema=c.TERMINAL_TRAJECTORY_SCHEMA)
                self.assertEqual(len(samples), 1)
                self.assertEqual(samples[0].reward, 0)
                self.assertIsNone(samples[0].terminal_closure)

    def test_completed_terminal_action_keeps_ordinary_result(self):
        pair = histories()
        for h in pair:
            completed(h); terminal(h, ('loss', 'win')[h.player])
        finalize(pair)
        for h in pair:
            self.assertEqual(h.transitions(), h.transitions(schema=c.TERMINAL_TRAJECTORY_SCHEMA))
            self.assertIsNotNone(h.transitions()[0].result_seq)

    def test_closure_references_require_ordered_integer_sequences(self):
        for values in ((1, 1, 3, 4), (1, 2, 4, 3), (True, 2, 3, 4), (-1, 2, 3, 4)):
            with self.assertRaises(ValueError): c.TerminalClosure(*values)
        with self.assertRaises(ValueError): histories()[0].transitions(schema='unknown')


if __name__ == '__main__': unittest.main()
