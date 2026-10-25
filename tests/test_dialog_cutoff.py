"""Synthetic strategic-query horizon checks; no native validation claims."""
from copy import deepcopy
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

from experiments.learner import contextual as c
from experiments.headless.contextual_protocol import ContextualProducer
from validation.probes.audit_contextual_histories import _dialog_cutoff_checks
from tests import test_contextual_adapter as fixtures
from tests.test_contextual_dialogs import history_at, action, panel, observation, decide, begin_answer, finish_answer
from contextual_env import DIALOG_DRAIN_LIMIT
from env import Limits


def producer(history):
    p = ContextualProducer(history.episode, history.player, interface=c.DIALOGS_ENVIRONMENT)
    p.live = p.history = history
    p.ready = True
    p.decision_ids = {f.seq: f.seq for f in history.frames if f.kind == 'decision'}
    p.current_id = history.decision.seq
    return p


class DialogCutoffTests(unittest.TestCase):
    setUp = fixtures.AdapterTests.setUp
    tearDown = fixtures.AdapterTests.tearDown

    def configure(self):
        e = self.env
        e.interface, e.context_profile = c.DIALOGS_ENVIRONMENT, c.DIALOGS_PROFILE
        h = history_at()
        e.sides[0].validator = producer(h)
        e.sides[0].actions_sent = 1
        e.sides[0].unread_frames_from = 0
        e.sides[1].unread_frames_from = 0
        blue = c.PlayerHistory(h.episode, 1, schema=c.DIALOGS_SCHEMA)
        decide(blue, observation())
        e.sides[1].validator = producer(blue)
        e.process.exited = None
        for side in e.sides.values(): side.eof = False
        e._issue(0)
        return e, h

    def test_direct_cutoff_rejects_before_mutation_or_send(self):
        e, h = self.configure(); before = h.digest(); request = e.request
        with self.assertRaisesRegex(ValueError, 'recorded policy answer'): e.truncate()
        self.assertEqual(h.digest(), before)
        self.assertIs(e.request, request)
        self.assertIsNone(e.cutoff_reason)
        self.assertIsNone(e.sides[0].validator.cutoff)
        self.assertFalse(e.sides[0].connection.sent)

    def test_defer_does_not_choose_and_actual_send_remains_learner_experience(self):
        e, h = self.configure(); before = h.digest()
        e.defer_dialog_cutoff('collection_window')
        self.assertEqual(h.digest(), before)
        self.assertFalse(e.request.past_horizon)
        self.assertEqual(e.dialog_cutoff_drain['choices'], [])
        e._send_current(action(selection=panel().choices[1].ref))
        self.assertEqual(h.frames[-1].kind, 'submitted')
        self.assertEqual(h.frames[-1].action.selection, panel().choices[1].ref)
        self.assertEqual(e.dialog_cutoff_drain['choices'], [{'player': 0, 'decision_seq': h.decision.seq}])
        self.assertEqual(e.sides[0].validator.continuation_steps, 0)
        e.sides[0].ledger.flush()
        self.assertIn(b'"outside_learner_prefix":false', (self.root / 'agents/red/bridge.jsonl').read_bytes())

    def test_environment_budgets_defer_active_query_and_freeze_first_safe_boundary(self):
        for limits, reason in ((Limits(max_actions=1), 'agent_action_limit'),
                               (Limits(max_days=1), 'game_day_limit')):
            with self.subTest(reason=reason):
                e, h = self.configure(); e.dialog_cutoff_drain = None; e.cutoff_reason = None
                e.limits = limits; e.request = None
                held = h.decision
                if reason == 'game_day_limit':
                    # Only the native ingest result is stubbed; no history is
                    # mislabeled as a real game or written as validation proof.
                    h.decision = replace(held, day=2)
                with patch.object(e.sides[0].validator, 'ingest', return_value=(h.decision,)), patch.object(e, '_issue', return_value=fixtures.SimpleNamespace(player=0, past_horizon=False)) as issue:
                    def issued(*args, **kwargs):
                        e.request = fixtures.SimpleNamespace(player=0, past_horizon=False)
                        return e.request
                    issue.side_effect = issued
                    result = e._ingest_checked(e.sides[0], {'type':'contextual'}, time.monotonic(), False)
                self.assertFalse(result.past_horizon)
                self.assertEqual(e.dialog_cutoff_drain['reason'], reason)
                self.assertIsNone(e.cutoff_reason)
                h.decision = held
                seq = begin_answer(h); finish_answer(h, seq); decide(h, observation())
                e.request = None
                with patch.object(e.sides[0].validator, 'ingest', return_value=(h.decision,)):
                    request = e._ingest_checked(e.sides[0], {'type':'contextual'}, time.monotonic(), False)
                self.assertTrue(request.past_horizon)
                self.assertEqual(e.cutoff_reason, reason)
                self.assertEqual(e.sides[0].validator.history.boundary, 'cutoff_pending')

    def test_frozen_continuation_never_answers_new_strategic_dialog(self):
        e, h = self.configure(); e.cutoff_reason = 'collection_window'
        with patch.object(e, '_finish', return_value='failed-unscored'):
            self.assertEqual(e._continue_cutoff(), 'failed-unscored')
        self.assertEqual(e.reason, 'cutoff_boundary_unavailable')
        self.assertFalse(e.sides[0].connection.sent)
        self.assertIsNone(e.sides[0].validator.cutoff)

    def test_answer_bound_stops_without_ninth_native_reply(self):
        e, h = self.configure(); e.defer_dialog_cutoff('collection_window')
        e.dialog_cutoff_drain['choices'] = [{'player':0, 'decision_seq': h.decision.seq}] * DIALOG_DRAIN_LIMIT
        e._send_current(action(selection=panel().choices[0].ref))
        self.assertEqual(e.reason, 'cutoff_boundary_unavailable')
        self.assertFalse(e.sides[0].connection.sent)
        self.assertEqual(len(e.dialog_cutoff_drain['choices']), 8)

    def test_independent_audit_derives_extra_choices_and_rejects_tampering(self):
        e, h = self.configure(); e.defer_dialog_cutoff('collection_window')
        held = h.decision.seq
        seq = begin_answer(h); finish_answer(h, seq); decide(h, observation())
        e.dialog_cutoff_drain['choices'].append({'player':0, 'decision_seq': held})
        manifest = {'status':'truncated', 'termination_reason':'collection_window',
                    'dialog_cutoff_drain':deepcopy(e.dialog_cutoff_drain)}
        pair = (h, e.sides[1].validator.history)
        _dialog_cutoff_checks(manifest, pair)
        for mutate in (lambda x: x['choices'].clear(), lambda x: x.update(learner_included=False),
                       lambda x: x.update(limit=9), lambda x: x['trigger']['actions_sent'].__setitem__(0,0),
                       lambda x: x['choices'][0].update(decision_seq=0)):
            bad = deepcopy(manifest); mutate(bad['dialog_cutoff_drain'])
            with self.assertRaises(ValueError): _dialog_cutoff_checks(bad, pair)


if __name__ == '__main__': unittest.main()
