"""Local authorization, deadlines and narrow source transition rejection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.learner import builtin_local365 as local,builtin_training as runner,builtin_preservation as preservation


class LocalTests(unittest.TestCase):
    def test_365_is_applied_without_changing_ppo_or_cloud(self):
        cfg=local.configuration({})
        self.assertEqual(cfg['max_days'],365)
        self.assertEqual(runner.configuration()['max_days'],30)
        self.assertEqual(cfg['ppo'],runner.configuration()['ppo'])
        self.assertEqual(cfg['adam'],runner.configuration()['adam'])
        self.assertEqual((cfg['ordinary_choice_limit'],cfg['reserved_dialog_choices'],cfg['max_total_choices']),(4960,160,5120))
        self.assertFalse(cfg['allow_verified_training_prefix'])

    def test_generic_migration_still_rejects_model_changes(self):
        from experiments.learner.builtin_portable import Migration
        instance=object.__new__(Migration);instance.value={'source_changes':{'experiments/learner/contextual_model.py':{'from':'x','to':'y'}}}
        with self.assertRaisesRegex(ValueError,'cannot change policy'):instance.validate_source_changes()

    def test_local_performance_exception_rejects_arbitrary_hashes(self):
        instance=object.__new__(local.LocalMigration)
        instance.value={'scope_id':local.LOCAL_SCOPE,'parent':{'policy_sha256':local.PARENT_POLICY,'resume_sha256':local.PARENT_RESUME,
            'progress':{'updates':31,'rollouts':124,'decisions':6776},'optimizer_steps':253},'source_changes':local.performance_changes()}
        instance.value['source_changes']['experiments/learner/contextual_model.py']['to']='f'*64
        with self.assertRaisesRegex(ValueError,'Unreviewed performance'):instance.validate_source_changes()

    def test_one_campaign_identity_and_exact_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'local365/campaign-session.json'
            with patch.object(preservation,'DURABLE_ROOT',root),patch.object(preservation,'durable_path',side_effect=Path), \
                 patch.object(local,'LocalMigration') as migration,patch.object(local.time,'time',return_value=1000):
                with patch.object(local,'binding',side_effect=lambda p:{'path':str(p),'sha256':'a','bytes':1}):
                    with self.assertRaisesRegex(ValueError,'Only one authorized'):local.create_session(root/'other.json','campaign',root/'migration')
                    value=local.create_session(path,'campaign',root/'migration');migration.return_value.training_gate.assert_called_once()
                    self.assertEqual(value['absolute_deadline_unix'],8200);self.assertEqual(value['compute_deadline_unix'],7900)
                    with self.assertRaisesRegex(ValueError,'Never overwrite'):local.create_session(path,'campaign',root/'migration')
                    local.session_for(path)
                    changed={**value,'absolute_deadline_unix':8201};path.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError,'expired or changed'):local.session_for(path)
                    path.write_text(json.dumps(value))
                    with patch.object(local.time,'time',return_value=7900):
                        with self.assertRaisesRegex(ValueError,'expired or changed'):local.session_for(path)

    def test_past_cloud_deadline_is_not_extended_by_local_module(self):
        from types import SimpleNamespace
        with patch.object(runner.time,'time',return_value=runner.COMPUTE_DEADLINE+1):
            with self.assertRaisesRegex(ValueError,'insufficient execution'):runner.prepare(SimpleNamespace(evaluate=False))

if __name__=='__main__':unittest.main()
