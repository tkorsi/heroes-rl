"""Continuation admission and failure boundaries, with no native game launch."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.learner import builtin_training as runner, builtin_gameplay as gameplay
from tests import test_builtin_gameplay as fixtures


class ContinuationTests(unittest.TestCase):
    def test_training_seeds_never_enter_fixed_heldout_set(self):
        heldout = runner.schedule(1, True)
        seen = set()
        for update in range(1, 101):
            self.assertEqual(runner.schedule(update, True), heldout)
            rows = runner.schedule(update)
            self.assertEqual([r['learner_player'] for r in rows], [0, 1, 0, 1])
            for row in rows:
                key = (row['engine_seed'], row['learner_player'], row['policy_seed'])
                self.assertNotIn(key, seen); seen.add(key)
                self.assertNotIn(row['engine_seed'], {r['engine_seed'] for r in heldout})
                self.assertNotIn(row['policy_seed'], {r['policy_seed'] for r in heldout})

    def test_only_explicit_training_schema_and_training_slot_admit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            path, plan = fixtures.write_plan(root)
            model = fixtures.m.ContextualActorCritic(fixtures.m.ModelConfig(
                schema=fixtures.m.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
            plan['schema'] = runner.SCHEMA+'-plan'; path.write_text(json.dumps(plan))
            fixtures.binding_for(root, model, path, plan, 0)
            plan['schema'] = runner.EVALUATION+'-plan'; path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'Invalid training plan'):
                fixtures.binding_for(root, model, path, plan, 0)
            plan['schema'] = runner.SCHEMA+'-plan'; plan['schedule'][0]['phase']='evaluation'
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'Only a planned training slot'):
                fixtures.binding_for(root, model, path, plan, 0)

    def test_failed_capture_stops_without_update_or_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d); rows=runner.schedule(1)
            plan={'configuration':runner.configuration(),'execution_output':d,'parent':{},'mode':'training','schedule':rows}
            report={'attempts':[runner.pilot.empty_attempt(r) for r in rows], 'update_status':'not_started',
                    'optimizer_updates':0,'optimizer_steps':0,'training_performed':False}
            (output/'report.json').write_text(json.dumps(report))
            args=SimpleNamespace(plan=output/'plan.json',sha256='a'*64,started=time.monotonic(),wall_started=time.time())
            state=SimpleNamespace(model=object(),checkpoint='fixture')
            failed={**report['attempts'][0], 'status':'failed'}
            def attempt(*args):
                args[-1](failed)
                return failed,None
            with patch.object(runner,'read_plan',return_value=plan), patch.object(runner,'parent_state',return_value=state), \
                 patch.object(runner.pilot,'check_identity'), patch.object(runner.pilot,'run_attempt',side_effect=attempt) as collect, \
                 patch.object(gameplay,'native_update') as update:
                result=runner.worker(args)
            self.assertEqual(result['status'],'stopped');self.assertEqual(collect.call_count,1)
            update.assert_not_called();self.assertEqual(result['optimizer_updates'],0)
            self.assertEqual([r['status'] for r in result['attempts']],['failed']+['not_started']*3)
