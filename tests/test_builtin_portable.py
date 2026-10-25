"""Synthetic cross-platform restore and rejection checks; no gameplay claims."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest

import torch

from experiments.learner import builtin_portable as portable
from experiments.learner import builtin_training_resume as resume
from experiments.learner import builtin_training as runner
from tests import test_builtin_training_resume as fixtures
write = fixtures.write


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads(); torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)
        self.fixture = fixtures.ParentResumeTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        f = self.fixture
        self.relocated = f.root/'relocated'
        self.relocated.mkdir()
        shutil.copytree(f.root/'runs', self.relocated/'runs')
        shutil.copyfile(f.game_map, self.relocated/'map.vmap')
        shutil.rmtree(f.root/'runs')
        f.plan_path = self.relocated/'runs/pilot/plan.json'
        f.report_path = f.plan_path.parent/'execution/report.json'
        self.config = f.root/'configs/portable-test.json'; self.config.parent.mkdir()
        self.value = {'schema': portable.SCHEMA, 'historical_training_admission': False,
            'historical_artifact_root': str(f.root), 'artifact_root': str(self.relocated),
            'from_source': f.source, 'to_source': f.current, 'source_changes': {},
            'runtime_transition': {'from_runtime': f.runtime, 'from_native': f.native,
                                  'to_runtime': f.runtime, 'to_native': f.native},
            'parent': {'plan_sha256': resume.binding(f.plan_path)['sha256'],
                       'report_sha256': resume.binding(f.report_path)['sha256'],
                       'resume_sha256': f.report['after_checkpoint']['resume']['sha256'],
                       'policy_sha256': f.report['after_checkpoint']['policy']['sha256'],
                       'progress': vars(f.progress), 'optimizer_steps': 1}}
        self.value['to_source']['python'] = '3.12.14 (different platform build)'
        self.commit_manifest()

    def commit_manifest(self):
        write(self.config, self.value)
        self.fixture.commit('commit explicit synthetic migration')
        self.fixture.current['git_commit'] = self.fixture.git('rev-parse', 'HEAD')
        return portable.Migration(self.config, self.fixture.root)

    def load(self, migration=None):
        return self.fixture.load(migration=migration or portable.Migration(self.config, self.fixture.root))

    def test_restore_without_original_absolute_paths_keeps_full_state(self):
        state = self.load()
        f = self.fixture
        resume._exact(state.optimizer.state_dict(), f.raw['resume']['optimizer_state'])
        resume._exact(dict(state.model.state_dict()), f.raw['model_state'])
        resume._exact(state.pool.state_dict(), f.pool.state_dict())
        resume._exact(torch.get_rng_state(), f.raw['rng']['torch_cpu'])
        self.assertEqual(state.metadata, f.metadata)
        self.assertEqual(state.progress, f.progress)
        self.assertTrue(Path(state.bindings['policy']['path']).is_relative_to(self.relocated))
        self.assertEqual(json.loads(f.plan_path.read_text()), f.plan)

    def test_unpinned_parent_or_runtime_and_corruption_reject(self):
        for key in ('policy_sha256', 'resume_sha256', 'report_sha256'):
            original = deepcopy(self.value)
            self.value['parent'][key] = 'f'*64
            migration = self.commit_manifest()
            with self.assertRaisesRegex(ValueError, 'parent state differs'): self.load(migration)
            self.value = original
        migration = self.commit_manifest()
        changed = {**self.fixture.runtime, 'libvcmi': 'changed'}
        with self.assertRaisesRegex(ValueError, 'Unpinned native'):
            self.fixture.load(migration=migration, current_runtime=changed)
        checkpoint = Path(self.fixture.report['after_checkpoint']['resume']['path'])
        migration.resolve(checkpoint).write_bytes(b'corruption')
        with self.assertRaisesRegex(ValueError, 'artifact bytes changed'): self.load(migration)

    def test_model_source_allowance_and_path_escape_reject(self):
        migration = self.commit_manifest()
        with self.assertRaisesRegex(ValueError, 'Unsafe'): migration.resolve('/old/../escape')
        self.value['source_changes']['experiments/learner/contextual_model.py'] = {'from':'a', 'to':'b'}
        with self.assertRaisesRegex(ValueError, 'cannot change policy'): self.commit_manifest()

    def test_missing_linux_gate_blocks_training(self):
        migration = self.commit_manifest()
        with self.assertRaisesRegex(ValueError, 'Fresh Linux'): migration.training_gate(self.fixture.runtime)

    def test_linux_gate_binds_client_and_the_exact_parent_policy(self):
        f = self.fixture
        plan_path = self.relocated/'linux-evaluation-plan.json'
        report_path = self.relocated/'linux-evaluation-report.json'
        plan = {'schema': runner.EVALUATION+'-plan', 'runtime':f.runtime, 'native':f.native,
                'mode':'evaluation', 'parent':{'plan':{'sha256':self.value['parent']['plan_sha256']}},
                'checkpoint':{'sha256':self.value['parent']['policy_sha256']},
                'schedule':runner.schedule(21,True)}
        write(plan_path,plan)
        rows = [{**row,'status':'passed','training_samples':0,'completion_receipt':True,
                 'cleanup_verified':True,'identity_verified':True,'replay_verified':True}
                for row in plan['schedule']]
        report = {'schema':runner.EVALUATION+'-result','plan_sha256':resume.binding(plan_path)['sha256'],
                  'status':'finished','worker_complete':True,'supervisor_cleanup_verified':True,'within_shared_budget':True,
                  'optimizer_updates':0,'optimizer_steps':0,'training_performed':False,'attempts':rows}
        write(report_path,report)
        self.value['linux_gate']={'plan':resume.binding(plan_path),'report':resume.binding(report_path)}
        migration=self.commit_manifest();migration.training_gate(f.runtime)
        report['within_shared_budget']=False;write(report_path,report)
        self.value['linux_gate']['report']=resume.binding(report_path)
        migration=self.commit_manifest()
        with self.assertRaisesRegex(ValueError,'clean current-runtime'): migration.training_gate(f.runtime)
        report['within_shared_budget']=True
        plan['native']={**f.native,'binary_sha256':'changed-client'};write(plan_path,plan)
        report['plan_sha256']=resume.binding(plan_path)['sha256'];write(report_path,report)
        self.value['linux_gate']={'plan':resume.binding(plan_path),'report':resume.binding(report_path)}
        migration=self.commit_manifest()
        with self.assertRaisesRegex(ValueError,'clean current-runtime'):
            migration.training_gate(f.runtime)

    def test_configuration_preserves_math_and_requires_complete_20x_batch(self):
        cfg = runner.configuration()
        self.assertEqual((cfg['ordinary_choice_limit'], cfg['reserved_dialog_choices'], cfg['max_total_choices']),
                         (4960,160,5120))
        self.assertEqual(cfg['max_optimizer_steps'], 640)
        self.assertEqual(cfg['ppo'], runner.pilot.config()['ppo'])
        self.assertEqual(cfg['adam'], runner.pilot.config()['adam'])
        self.assertFalse(cfg['allow_verified_training_prefix'])


class CpuWheelTests(unittest.TestCase):
    def test_only_explicit_same_release_cpu_variant_loads(self):
        import tempfile
        from experiments.learner import checkpoints as cp, contextual_model as models
        from experiments.learner.ppo import PPOConfig
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'policy.pt'
            model = models.ContextualActorCritic(models.ModelConfig(
                schema=models.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
            historical = str(torch.__version__).removesuffix('+cpu')
            metadata = {'torch': historical}
            with patch.object(torch, '__version__', historical):
                cp.save_policy_checkpoint(path, model, checkpoint='synthetic-wheel',
                    source_metadata=metadata, interface=resume.INTERFACE)
            with patch.object(torch, '__version__', historical+'+cpu'):
                with self.assertRaisesRegex(ValueError, 'version or provenance'):
                    cp.load_policy_checkpoint(path, interface=resume.INTERFACE)
                loaded = cp.load_policy_checkpoint(path, expected_source_metadata=metadata,
                    interface=resume.INTERFACE, historical_torch_version=historical)
                resume._exact(dict(loaded.model.state_dict()), dict(model.state_dict()))
                for version in ('0.0.1', historical+'+cu128'):
                    with self.assertRaisesRegex(ValueError, 'same public CPU'):
                        cp.load_policy_checkpoint(path, expected_source_metadata=metadata,
                            interface=resume.INTERFACE, historical_torch_version=version)


class ExpandedBatchTests(unittest.TestCase):
    def test_all_20480_decisions_fit_640_steps_and_overflow_never_updates_prefix(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import tempfile
        from experiments.learner import builtin_gameplay as gameplay
        from experiments.learner.ppo import PPOConfig, PPOReport
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'plan.json'
            cfg = runner.configuration()
            plan = {'schema': runner.SCHEMA+'-plan', 'configuration': cfg, 'schedule': runner.schedule(22)}
            write(path,plan)
            receipts = tuple(SimpleNamespace(binding=SimpleNamespace(plan_path=str(path))) for _ in range(4))
            rollouts = tuple(SimpleNamespace(samples=range(5120)) for _ in range(4))
            model = torch.nn.Linear(1,1); optimizer = torch.optim.Adam(model.parameters(),lr=0.0003)
            result = PPOReport(20480,640,0,0,0,0)
            options = dict(plan_sha256='synthetic',config=PPOConfig(**cfg['ppo']), generator=torch.Generator(),
                           remaining=lambda:10,max_optimizer_steps=640)
            with patch.object(gameplay,'_admit_builtin',return_value=rollouts), \
                 patch.object(gameplay,'transactional_update',return_value=result) as update:
                report, complete = gameplay.native_update(model,optimizer,receipts,**options)
                self.assertEqual(report.samples,20480)
                self.assertEqual(len(complete),4)
                self.assertEqual(update.call_args.kwargs['max_optimizer_steps'],640)
            oversized = (*rollouts[:3], SimpleNamespace(samples=range(5121)))
            with patch.object(gameplay,'_admit_builtin',return_value=oversized), \
                 patch.object(gameplay,'transactional_update') as update:
                with self.assertRaisesRegex(ValueError,'Complete update exceeds'):
                    gameplay.native_update(model,optimizer,receipts,**options)
                update.assert_not_called()


class HeldoutTests(unittest.TestCase):
    def test_every_fifth_update_requires_its_own_verified_pair(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = SimpleNamespace(progress=SimpleNamespace(updates=25),bindings={'plan':{'sha256':'parent'}})
            runtime = {'vcmireplay':'synthetic'}
            with patch.object(runner,'DURABLE_ROOT',root):
                with self.assertRaisesRegex(ValueError,'held-out pair'): runner.require_heldout(state,runtime,{})
                run = root/'runs/eval'; run.mkdir(parents=True)
                plan = {'schema':runner.EVALUATION+'-plan','parent':state.bindings,'runtime':runtime,'native':{},
                        'schedule':runner.schedule(25,True),'execution_output':str(run/'execution')}
                write(run/'plan.json',plan); (run/'execution').mkdir()
                rows = [{**row,'status':'passed','training_samples':0,'completion_receipt':True,
                         'cleanup_verified':True,'identity_verified':True,'replay_verified':True}
                        for row in plan['schedule']]
                report = {'plan_sha256':runner.pilot.digest(run/'plan.json'),'status':'finished',
                    'worker_complete':True,'supervisor_cleanup_verified':True,'within_shared_budget':True,'training_performed':False,
                    'optimizer_updates':0,'optimizer_steps':0,'attempts':rows}
                write(run/'execution/report.json',report)
                runner.require_heldout(state,runtime,{})
                report['attempts'][1]['replay_verified']=False
                write(run/'execution/report.json',report)
                with self.assertRaisesRegex(ValueError,'held-out pair'): runner.require_heldout(state,runtime,{})
