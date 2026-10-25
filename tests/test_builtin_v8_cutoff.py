"""Mandatory capability horizons remain actual policy decisions, never forced replies."""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

from experiments.learner import contextual as c
from experiments.headless import builtin_evidence as evidence
from experiments.headless.builtin_env import BuiltinHeadlessEnv
from tests.test_contextual_capabilities import observation, decide, begin, action, choice, navigate
from tests.test_builtin_env import manifest_for


def mandatory_history():
    h = c.PlayerHistory('mandatory-cutoff', 0, schema=c.EXPANDED_SCHEMA)
    decide(h, observation())
    seq = begin(h, action('move_to', 'hero', x=2, y=1), 'engine')
    h.emit('issued', decision_seq=seq, command_index=0)
    panel = c.CapabilityPanel('exact-level-panel', 'level_up', c.Context(), True, 2, 0, (), (
        choice('choose_level_up', ref='skill-a', subject='hero', skill=1, skill_level=1),
        choice('choose_level_up', ref='skill-b', subject='hero', skill=2, skill_level=1)))
    h.emit('event', event=c.VisibleEvent('capability_requested', 'hero', capability=panel))
    h.emit('acknowledged', decision_seq=seq, command_index=0, accepted=True)
    h.emit('result', decision_seq=seq, result=c.Result('interaction', True, (1,1,0), 100, False))
    context = c.Context('capability', 'hero', 'level_up', panel.ref)
    h.emit('context_opened', context=context, revision=h.revision+1, reason='capability_dialog')
    decide(h, replace(observation(), window=c.Window(context, None, armies=(), capability_controls=(), capability_panel=panel)))
    return h


def spell_target_history():
    h = c.PlayerHistory('spell-target-cutoff', 0, schema=c.EXPANDED_SCHEMA)
    decide(h, observation())
    navigate(h, action('open_hero', 'hero'), observation('hero'))
    book = c.CapabilityPanel('book', 'spellbook', c.Context('hero','hero'), False, None, None, (),
        (choice('cast_adventure_spell', ref='cast', subject='hero', spell=8),))
    obs = replace(observation(), window=c.Window(c.Context('capability','hero','spellbook'), None,
        armies=(), capability_controls=(), capability_panel=book, mana=100))
    navigate(h, action('open_capability','hero',panel='spellbook'), obs)
    seq = begin(h, action('cast_adventure_spell','hero',selection='cast'), 'engine')
    h.emit('context', decision_seq=seq, context=c.Context(), revision=h.revision+1)
    h.emit('issued', decision_seq=seq, command_index=0)
    panel = c.CapabilityPanel('target-query','spell_target',c.Context(),True,None,None,(),(
        choice('choose_spell_target',ref='target',subject='hero',spell=8,destination='town'),
        choice('cancel_spell_target',ref='cancel',subject='hero',spell=8)))
    h.emit('event', event=c.VisibleEvent('capability_requested','hero',capability=panel))
    h.emit('acknowledged',decision_seq=seq,command_index=0,accepted=True)
    h.emit('result',decision_seq=seq,result=c.Result('capability_applied',True))
    context = c.Context('capability','hero','spell_target',panel.ref)
    h.emit('context_opened',context=context,revision=h.revision+1,reason='capability_dialog')
    decide(h,replace(observation(),window=c.Window(context,None,armies=(),capability_controls=(),capability_panel=panel)))
    return h


class MandatoryCutoffTests(unittest.TestCase):
    def test_defer_binds_exact_query_and_never_sends_an_answer(self):
        h = mandatory_history()
        env = BuiltinHeadlessEnv(object(), Path('/unused-synthetic-v8'), learner_player=0, interface=c.EXPANDED_ENVIRONMENT)
        side = SimpleNamespace(validator=SimpleNamespace(live=h, history=h, cutoff=None), actions_sent=1)
        env.sides = {0: side}; env.cutoff_reason = None; env.dialog_cutoff_drain = None
        env.request = SimpleNamespace(player=0, past_horizon=False)
        before = h.digest()
        with self.assertRaises(ValueError): env.truncate()
        env.defer_dialog_cutoff('collection_window')
        self.assertEqual(h.digest(), before)
        self.assertEqual(env.dialog_cutoff_drain['trigger']['decision_seq'], h.decision.seq)
        with self.assertRaises(ValueError): env._continue_cutoff()
        with self.assertRaises(ValueError):
            h.emit('submitted', decision_seq=h.decision.seq, action=action('choose_level_up', 'hero', selection='stale-skill'))
        trigger = h.decision.seq
        seq = begin(h, action('choose_level_up', 'hero', selection='skill-a'), 'engine')
        h.emit('context', decision_seq=seq, context=c.Context(), revision=h.revision+1)
        h.emit('issued', decision_seq=seq, command_index=0)
        with self.assertRaises(ValueError):
            h.emit('result', decision_seq=seq, result=c.Result('capability_applied', True))
        h.emit('acknowledged', decision_seq=seq, command_index=0, accepted=True)
        h.emit('result', decision_seq=seq, result=c.Result('capability_applied', True))
        decide(h, observation())
        h.emit('cutoff', decision_seq=h.decision.seq, reason='collection_window')
        manifest = manifest_for(h)
        manifest.update(environment=c.EXPANDED_ENVIRONMENT, history_schema=c.EXPANDED_SCHEMA, action_schema=c.EXPANDED_ACTION_SCHEMA)
        manifest['players']['red']['history_v8'] = manifest['players']['red'].pop('history_v7')
        drain = deepcopy(env.dialog_cutoff_drain)
        drain['choices'] = [{'player': 0, 'decision_seq': trigger}]
        manifest['dialog_cutoff_drain'] = drain
        self.assertTrue(evidence.validate_manifest_history(h, manifest)['consistent'])
        for mutation in (lambda d: d['choices'].clear(), lambda d: d['trigger'].update(decision_seq=0)):
            changed = deepcopy(manifest); mutation(changed['dialog_cutoff_drain'])
            with self.assertRaises(ValueError): evidence.validate_manifest_history(h, changed)


    def test_spell_target_and_cancel_are_real_bounded_policy_choices(self):
        for kind, ref in (('choose_spell_target','target'),('cancel_spell_target','cancel')):
            h = spell_target_history()
            self.assertTrue(h.requires_choice)
            trigger = h.decision.seq
            sent = sum(f.kind == 'submitted' for f in h.frames)
            with self.assertRaises(ValueError):
                h.emit('cutoff', decision_seq=trigger, reason='collection_window')
            seq = begin(h,action(kind,'hero',selection=ref),'engine')
            h.emit('context',decision_seq=seq,context=c.Context(),revision=h.revision+1)
            h.emit('issued',decision_seq=seq,command_index=0)
            h.emit('acknowledged',decision_seq=seq,command_index=0,accepted=True)
            h.emit('result',decision_seq=seq,result=c.Result('capability_applied',True))
            decide(h,observation())
            h.emit('cutoff',decision_seq=h.decision.seq,reason='collection_window')
            manifest=manifest_for(h)
            manifest.update(environment=c.EXPANDED_ENVIRONMENT,history_schema=c.EXPANDED_SCHEMA,action_schema=c.EXPANDED_ACTION_SCHEMA)
            manifest['players']['red']['history_v8']=manifest['players']['red'].pop('history_v7')
            manifest['dialog_cutoff_drain']={'reason':'collection_window','trigger':{'player':0,'decision_seq':trigger,'actions_sent':sent},
                'limit':8,'choices':[{'player':0,'decision_seq':trigger}],'learner_included':True}
            self.assertTrue(evidence.validate_manifest_history(h,manifest)['consistent'])


if __name__ == '__main__': unittest.main()
