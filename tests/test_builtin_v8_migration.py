"""Asset-free conversion invariants; no optimizer step or gameplay."""
from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
from experiments.learner import checkpoints as cp, contextual as c, contextual_model as m
from experiments.learner import builtin_v8_migration as migration
from experiments.learner import builtin_v8_session as scope
from experiments.learner.builtin_training_resume import _exact


class ConversionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); self.root = Path(tmp.name)
        model = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA, hidden_size=8, feature_size=4))
        pool = cp.FrozenOpponentPool(); pool.add('synthetic-original', model)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
        # Deliberately populated named moments without a training step.
        for i, parameter in enumerate(model.parameters()):
            optimizer.state[parameter] = {'step': torch.tensor(783.), 'exp_avg': torch.full_like(parameter, .01*(i+1)),
                                          'exp_avg_sq': torch.full_like(parameter, .02*(i+1))}
        metadata = {'plan_sha256': 'a'*64, 'initial_checkpoint': {'path': '/synthetic/parent.pt', 'sha256': 'b'*64, 'bytes': 1},
            'native_opponent': {'kind':'builtin','module':'Nullkiller','difficulty':0,'binary_sha256':'c'*64},
            'archived_neural_pool_executed': False}
        self.metadata = metadata
        options = dict(checkpoint='synthetic-update92', provenance=cp.BUILTIN_GAMEPLAY_PROVENANCE,
            source_metadata=metadata, opponent_pool=pool, generators={'minibatch': torch.Generator().manual_seed(17)},
            interface=c.DIALOGS_ENVIRONMENT)
        cp.save_checkpoint(self.root/'old-resume.pt', model, optimizer, ppo_config={}, progress=cp.TrainingProgress(92,368,21062),
            sampler_state={'optimizer_steps':783}, at_rollout_boundary=True, **options)
        cp.save_policy_checkpoint(self.root/'old-policy.pt', model, **options)
        self.old = [torch.load(self.root/('old-'+name+'.pt'), weights_only=True) for name in ('policy','resume')]

    def convert(self):
        return migration.convert_payloads(*self.old, metadata={**self.metadata, 'explicit_transition': True}, checkpoint='synthetic-derived-v8')

    def test_exact_named_tensors_adam_rng_progress_and_pool_with_new_identities(self):
        before = (torch.get_rng_state().clone(), random.getstate())
        policy, saved, mapping = self.convert()
        _exact(before, (torch.get_rng_state(), random.getstate()))
        self.assertTrue(migration.verify_conversion(*self.old, policy, saved, mapping)['adam_rng_progress_sampler_unchanged'])
        self.assertNotEqual(saved['weights_sha256'], self.old[1]['weights_sha256'])
        _exact(saved['resume'], self.old[1]['resume'])
        for kind, raw in (('policy', policy), ('resume', saved)): torch.save(raw, self.root/(kind+'.pt'))
        loaded = cp.load_policy_checkpoint(self.root/'policy.pt', expected_source_metadata=saved['source_metadata'],
            allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,), interface=c.EXPANDED_ENVIRONMENT)
        model = loaded.model.requires_grad_(True); optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
        generator = torch.Generator()
        resumed = cp.load_checkpoint(self.root/'resume.pt', model, optimizer, expected_ppo_config={},
            expected_source_metadata=saved['source_metadata'], allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,),
            generators={'minibatch':generator}, interface=c.EXPANDED_ENVIRONMENT, at_rollout_boundary=True)
        _exact(optimizer.state_dict(), self.old[1]['resume']['optimizer_state'])
        _exact(cp._rng_state({'minibatch':generator}, {}), self.old[1]['rng'])
        self.assertEqual(resumed.progress.updates, 92)

    def test_tampered_moment_rng_progress_pool_and_configuration_reject(self):
        policy, saved, mapping = self.convert()
        for field in ('adam', 'rng', 'progress', 'pool', 'config'):
            bad = deepcopy(saved)
            if field == 'adam': next(iter(bad['resume']['optimizer_state']['state'].values()))['step'] += 1
            elif field == 'rng': bad['rng']['torch_cpu'][0] ^= 1
            elif field == 'progress': bad['resume']['progress']['updates'] += 1
            elif field == 'pool': next(iter(bad['opponent_pool'][0]['model_state'].values())).view(-1)[0] += 1
            else: bad['model_config']['hidden_size'] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                migration.verify_conversion(*self.old, policy, bad, mapping)

    def test_source_pair_disagreement_and_shape_changes_fail(self):
        self.old[0]['checkpoint'] = 'other'
        with self.assertRaises(ValueError): self.convert()


class SessionScopeTests(unittest.TestCase):
    def test_configuration_and_release_fail_closed(self):
        cfg = scope.configuration()
        self.assertEqual((cfg['max_days'],cfg['max_total_choices']), (365,5120))
        self.assertFalse(cfg['allow_verified_training_prefix'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'release.json'; path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Explicit coordinator release'):
                scope.validate_release(path)


    def test_single_use_session_exact_wall_and_monotonic_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'campaign-session.json'; release_path = root/'release.json'
            release_path.write_text('{}')
            release = {'source': {'synthetic': True}}
            with patch.object(scope, 'SESSION', path), patch.object(scope, 'durable_path', side_effect=Path), \
                 patch.object(scope, 'validate_release', return_value=release), patch.object(scope, 'checked', return_value={}), \
                 patch.object(scope.time, 'time', return_value=1000), patch.object(scope.time, 'monotonic', return_value=100):
                value = scope.create_session(release_path)
                self.assertEqual(value['absolute_deadline_unix'], 8200)
                self.assertEqual(value['compute_deadline_unix'], 7900)
                scope.session_for(path)
                with self.assertRaises(ValueError): scope.create_session(release_path)
                with patch.object(scope.time, 'time', return_value=7900):
                    with self.assertRaises(ValueError): scope.session_for(path)
                with patch.object(scope.time, 'monotonic', return_value=7000):
                    with self.assertRaises(ValueError): scope.session_for(path)

    def test_controller_deadline_covers_worker_and_cleanup(self):
        from types import SimpleNamespace
        from tools import v8_teacher_campaign as controller
        from experiments.headless import process_deadline
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); session = root/'campaign-session.json'; session.write_text('{}')
            args = SimpleNamespace(private_repo=root/'private', release=root/'release.json')
            with patch.object(scope, 'SESSION', session), patch.object(scope, 'validate_release', return_value={'migration_receipt': {},'private_migration_receipt': {'path':'/synthetic-private-receipt'}}), \
                 patch.object(scope, 'checked', return_value={'plan': {'path': str(root/'migration.json')}}), \
                 patch.object(scope, 'create_session', return_value={'absolute_deadline_unix':8200, 'started_monotonic':100}), \
                 patch.object(migration, 'load_parent'), patch.object(migration, 'verify_private'), patch.object(controller, 'preflight_private'), patch.object(controller, 'DURABLE_ROOT', root), \
                 patch.object(controller.time, 'time', return_value=1000), patch.object(controller.time, 'monotonic', return_value=100), \
                 patch.object(process_deadline, 'guarded_popen') as popen, patch.object(process_deadline, 'close_owned_process') as close:
                popen.return_value.wait.return_value = 0
                controller.supervise(args)
                self.assertEqual(popen.call_args.kwargs['deadline_wall'], 8180)
                self.assertEqual(popen.call_args.kwargs['deadline_monotonic'], 7280)
                self.assertIn('campaign-worker', popen.call_args.args[0])
                close.assert_called_once()
                self.assertEqual(close.call_args.kwargs['deadline_wall'], 8200)
                self.assertEqual(close.call_args.kwargs['deadline_monotonic'], 7300)

    def test_v2_map_is_explicit_and_rejects_v1(self):
        root = Path(__file__).resolve().parents[1]
        identity = migration.map_identity(root/'experiments/explorer/maps/hero-management-v2.vmap')
        self.assertEqual(identity['scenario'], 'hero-management-v2')
        with self.assertRaises(ValueError):
            migration.map_identity(root/'experiments/explorer/maps/hero-management-v1.vmap')


    def test_failed_prepare_is_journaled_before_call_and_never_replaced(self):
        from types import SimpleNamespace
        from tools import v8_teacher_campaign as controller
        import os, json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); session=root/'campaign-session.json'; session.write_text('{}')
            parent=root/'parent.json'; parent.write_text('{}')
            args=SimpleNamespace(private_repo=root/'private')
            value={'controller_pid':os.getppid(),'release':{'path':'/synthetic-release'},
                   'compute_deadline_unix':7900,'absolute_deadline_unix':8200}
            def fail(*args):
                report=json.loads((root/'campaign-report.json').read_text())
                self.assertEqual(len(report['batches']),1)
                self.assertEqual(report['batches'][0]['status'],'preparing')
                self.assertEqual(len(report['batches'][0]['schedule']),2)
                self.assertTrue(report['pending_plan'].endswith('/plan.json'))
                raise RuntimeError('synthetic preparation failure')
            with patch.object(scope,'SESSION',session), patch.object(scope,'session_for',return_value=value), \
                 patch.object(scope,'validate_release',return_value={'migration_receipt':{}}), \
                 patch.object(scope,'checked',return_value={'plan':{'path':str(parent)}}), \
                 patch.object(controller,'DURABLE_ROOT',root), patch.object(controller.time,'time',return_value=1000), \
                 patch.object(controller,'run_batch',side_effect=fail) as batch, \
                 patch.object(controller,'publish_failure',side_effect=OSError('synthetic private failure')) as preserve:
                with self.assertRaisesRegex(RuntimeError,'preparation failure'): controller.campaign(args)
                batch.assert_called_once(); preserve.assert_called_once()
                report=json.loads((root/'campaign-report.json').read_text())
                self.assertEqual(report['status'],'stopped_failure')
                self.assertIsNotNone(report['pending_plan'])
                self.assertEqual(report['latest_parent']['path'],str(parent.resolve()))
                self.assertIn('failure_preservation_error',report['batches'][0])

    def test_failure_archive_retains_unstarted_schedule_without_a_plan(self):
        from tools import v8_teacher_campaign as controller
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan=root/'runs/v8-update92-heldout-092/plan.json'; repo=root/'private'
            rows=[{'episode':1,'phase':'evaluation','learner_player':0}]
            with patch.object(scope,'SESSION',root/'campaign-session.json'), \
                 patch.object(controller,'preflight_private'), patch.object(controller.subprocess,'check_output',return_value='commit'):
                controller.publish_failure(plan,repo,controller.time.time()+60,'synthetic failure',rows)
            failure=json.loads((plan.parent/'controller-failure.json').read_text())
            self.assertFalse(failure['checkpoint_admitted'])
            self.assertEqual(failure['attempts'][0]['status'],'not_started')
            self.assertTrue((repo/'results/local-v8-update92/v8-update92-heldout-092-failed.tar.gz').is_file())


    def test_private_verifier_reads_committed_blobs_and_rejects_hash_tampering(self):
        import json, subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); repo=root/'private'; repo.mkdir(); folder=root/'canonical'; folder.mkdir()
            subprocess.run(['git','init','-q',str(repo)],check=True)
            subprocess.run(['git','remote','add','origin','https://github.com/tkorsi/heroes-rl-training-state.git'],cwd=repo,check=True)
            target=repo/'snapshot'; target.mkdir()
            (folder/'plan.json').write_text('{}')
            for name in ('after-policy.pt','after-resume.pt'): (folder/name).write_bytes(name.encode())
            report={'files':{k:migration.binding(folder/('after-'+k+'.pt')) for k in ('policy','resume')}}
            (folder/'report.json').write_text(json.dumps(report))
            proof={'synthetic_exact_restore':True}
            canonical={'plan':migration.binding(folder/'plan.json'),'report':migration.binding(folder/'report.json'),'restore':proof}
            (folder/'migration-receipt.json').write_text(json.dumps(canonical))
            names=('plan.json','report.json','migration-receipt.json','after-policy.pt','after-resume.pt')
            for name in names: (target/name).write_bytes((folder/name).read_bytes())
            subprocess.run(['git','add','.'],cwd=repo,check=True)
            subprocess.run(['git','-c','user.name=Synthetic Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],cwd=repo,check=True)
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
            receipt={'schema':migration.SCHEMA+'-private-receipt','pushed':True,'repository':str(repo),'commit':commit,
                'migration_receipt':migration.binding(folder/'migration-receipt.json'),'restore':proof,
                'files':{name:{'git_path':'snapshot/'+name,**{k:v for k,v in migration.binding(folder/name).items() if k!='path'}} for name in names}}
            path=root/'private-receipt.json'; path.write_text(json.dumps(receipt))
            (target/'after-policy.pt').write_bytes(b'changed working tree is not the committed evidence')
            with patch.object(migration,'restore_pair',return_value=(None,proof)) as restore:
                self.assertEqual(migration.verify_private(path),proof)
                restore.assert_called_once()
                receipt['files']['after-policy.pt']['sha256']='0'*64; path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError,'Git blob differs'): migration.verify_private(path)

    def test_cumulative_cleanup_deadline_and_session_owned_paths(self):
        from types import SimpleNamespace
        from tools import v8_teacher_campaign as controller
        from experiments.headless import process_deadline
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); session=root/'campaign-session.json'; session.write_text('{}')
            owned=root/'runs/v8-update92-heldout-092'; (owned/'execution').mkdir(parents=True)
            for name in ('native-deadline-1.jsonl','native-deadline-2.jsonl'): (owned/'execution'/name).write_text('{}')
            other=root/'runs/v8-update92-historical/execution'; other.mkdir(parents=True); (other/'native-deadline-1.jsonl').write_text('{}')
            (root/'campaign-report.json').write_text(json.dumps({'session':migration.binding(session),
                'batches':[{'intended_plan':str(owned/'plan.json')}]}))
            current=[1000]
            def cleaned(*args,**kwargs): current[0]=8200
            args=SimpleNamespace(private_repo=root/'private',release=root/'release.json')
            with patch.object(scope,'SESSION',session), patch.object(scope,'validate_release',return_value={
                    'migration_receipt':{},'private_migration_receipt':{'path':'/synthetic'}}), \
                 patch.object(scope,'checked',return_value={'plan':{'path':str(root/'migration.json')}}), \
                 patch.object(scope,'create_session',return_value={'absolute_deadline_unix':8200,'started_monotonic':100}), \
                 patch.object(migration,'load_parent'),patch.object(migration,'verify_private'),patch.object(controller,'preflight_private'), \
                 patch.object(controller,'DURABLE_ROOT',root),patch.object(controller.time,'time',side_effect=lambda:current[0]), \
                 patch.object(controller.time,'monotonic',return_value=100),patch.object(process_deadline,'guarded_popen') as popen, \
                 patch.object(process_deadline,'close_owned_process'),patch.object(process_deadline,'cleanup_from_evidence',side_effect=cleaned) as cleanup:
                popen.return_value.wait.return_value=0
                with self.assertRaisesRegex(ValueError, 'cleanup incomplete'): controller.supervise(args)
                cleanup.assert_called_once()
                self.assertTrue(cleanup.call_args.args[0].is_relative_to(owned))
                self.assertEqual(cleanup.call_args.kwargs['deadline_wall'],8200)
                self.assertEqual(cleanup.call_args.kwargs['deadline_monotonic'],7300)
                result=json.loads((root/'controller-result.json').read_text())
                self.assertFalse(result['controller_cleanup_verified'])
                self.assertEqual(len(result['cleanup_incomplete']),1)


if __name__ == '__main__': unittest.main()
