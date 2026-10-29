"""Immutable runtime-refresh outputs and exact frozen-checkpoint bookkeeping."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import refresh_parallel_runtime_acceptance as refresh


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class RuntimeRefreshTests(unittest.TestCase):
    def options(self, root):
        return SimpleNamespace(checkpoint_update=1335,release_id='terminal-replay-cloud1335-v1',
            native_output=root/'native',regression_output=root/'regression',candidate_output=root/'candidate.json',
            policy_output=root/'policy1335',acceptance_output=root/'acceptance.json',
            policy_acceptance_output=root/'policy-acceptance.json')

    def test_historical_defaults_remain_exact_and_new_outputs_are_distinct(self):
        old=refresh.checked_options()
        self.assertEqual((old.checkpoint_update,old.candidate_output,old.policy_output,old.acceptance_output),
                         (1095,refresh.CANDIDATE,refresh.POLICY,refresh.ACCEPTANCE))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            with patch.object(refresh,'ARTIFACTS',root):
                options=self.options(root)
                self.assertEqual(refresh.checked_options(options).checkpoint_update,1335)
                for name,value in [('release_id','terminal-replay-parallel-v1'),('checkpoint_update',1229),
                                   ('candidate_output',root/'local-script-training/latest-checkpoint.json'),
                                   ('policy_output',root/'candidate.json')]:
                    changed=deepcopy(options);setattr(changed,name,value)
                    with self.subTest(name=name),self.assertRaises(ValueError):
                        refresh.checked_options(changed)

    def test_publication_never_overwrites_previous_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'new'/'receipt.json'
            refresh.write_new(path,{'first':True});before=path.read_bytes()
            with self.assertRaises(FileExistsError):refresh.write_new(path,{'second':True})
            self.assertEqual(path.read_bytes(),before)

    def test1392_requires_its_explicit_private_diagnostic_and_reviewed_release(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            options = self.options(root)
            options.checkpoint_update = 1392
            options.release_id = 'terminal-replay-1392-v1'
            diagnostic = root/'run-investigations/2026-10-29-batch1032/replay/diagnostic.json'
            with patch.object(refresh, 'ARTIFACTS', root):
                with self.assertRaisesRegex(ValueError, 'explicit excluded replay proof'):
                    refresh.checked_options(deepcopy(options))
                options.terminal_diagnostic = diagnostic
                checked = refresh.checked_options(deepcopy(options))
                self.assertEqual(checked.terminal_diagnostic, diagnostic)
                for name, value in (('terminal_diagnostic', None), ('terminal_diagnostic', root/'diagnostic.json'),
                                    ('terminal_diagnostic', root/'local-script-training/diagnostic.json'),
                                    ('terminal_diagnostic', diagnostic.parent/'../../../outside.json'),
                                    ('checkpoint_update', 1335), ('release_id', 'terminal-replay-cloud1335-v1')):
                    changed = deepcopy(options); setattr(changed, name, value)
                    with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                        refresh.checked_options(changed)

    def test1392_candidate_binds_new_freeze_replay_and_native_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            options = self.options(root)
            options.checkpoint_update = 1392
            options.release_id = 'terminal-replay-1392-v1'
            options.terminal_diagnostic = root/'run-investigations/2026-10-29-batch1032/replay/diagnostic.json'
            old_runtime, new_runtime = root/'runtime/old', root/'runtime/new'
            old = old_runtime/'accepted-runtime.json'
            write(old, dict(policy_reconstruction={'historical': True}, runtime_bin=str(old_runtime/'bin'),
                runtime_files={'vcmiclient': 'old-client', 'vcmireplay': 'old-replay'},
                freeze={'old': True}, preflight={'old': True}, terminal_replay={'old': True},
                engine_recipe={'old': 'recipe'}, change_classification='replay-verifier-support-extension',
                native_engine_correction='Verifier-only ExchangeArtifacts accounting.'))
            write(new_runtime/'freeze-receipt.json', dict(artifacts=[
                {'path': 'vcmiclient', 'frozen_sha256': 'new-client'},
                {'path': 'vcmireplay', 'frozen_sha256': 'new-replay'},
                {'path': 'vcmiserver', 'frozen_sha256': 'new-server'}]))
            write(new_runtime/'preflight.json', {'new': 'preflight'})
            write(options.terminal_diagnostic, {'new': 'excluded terminal replay'})
            for output in (options.native_output, options.regression_output):
                for name in ('execution/plan.json', 'execution/report.json', 'shutdown.json'):
                    write(output/name, {'new': str(output/name)})
            real_binding = refresh.binding
            def bound(path):
                result = real_binding(path)
                if Path(path) == old:
                    result['sha256'] = 'fffce8b1ee74c010f235b97a88547adc6ada3dce65251ac959a4b6dbe6e032b8'
                return result
            with patch.object(refresh, 'ARTIFACTS', root), patch.object(refresh, 'RUNTIME', old_runtime), \
                    patch.object(refresh, 'TOWN_SWAP_RUNTIME', new_runtime), patch.object(refresh, 'binding', side_effect=bound), \
                    patch.object(refresh.subprocess, 'check_output', return_value='c'*40+'\n'), \
                    patch('tools.prepare_local_training.current_engine_recipe', return_value={'new': 'recipe'}), \
                    patch.object(refresh, 'candidate_runtime_origin') as native_gate, \
                    patch.object(refresh, 'terminal_replay_acceptance') as replay_gate, redirect_stdout(io.StringIO()):
                refresh.candidate(options)
            candidate = refresh.read(options.candidate_output)
            self.assertEqual(candidate['runtime_bin'], str(new_runtime/'bin'))
            self.assertEqual(candidate['runtime_files'], {'vcmiclient': 'new-client', 'vcmireplay': 'new-replay'})
            self.assertEqual(candidate['engine_recipe'], {'new': 'recipe'})
            self.assertEqual(candidate['freeze'], real_binding(new_runtime/'freeze-receipt.json'))
            self.assertEqual(candidate['preflight'], real_binding(new_runtime/'preflight.json'))
            self.assertEqual(candidate['terminal_replay'], real_binding(options.terminal_diagnostic))
            self.assertEqual(candidate['native'], refresh.proof(options.native_output))
            self.assertEqual(candidate['regression'], refresh.proof(options.regression_output))
            self.assertEqual(candidate['native_engine_correction'], refresh.TOWN_SWAP_CORRECTION)
            self.assertEqual(refresh.read(old)['native_engine_correction'], 'Verifier-only ExchangeArtifacts accounting.')
            self.assertNotIn('policy_reconstruction', candidate)
            native_gate.assert_called_once_with(new_runtime/'bin', {'vcmireplay': 'new-replay'}, candidate)
            replay_gate.assert_called_once_with(candidate, {'vcmireplay': 'new-replay'})

    def fixture(self,root,checkpoint_update=1335):
        options=self.options(root)
        options.checkpoint_update=checkpoint_update
        if checkpoint_update == 1392:
            options.release_id='terminal-replay-1392-v1'
            options.terminal_diagnostic=root/'run-investigations/2026-10-29-batch1032/replay/diagnostic.json'
        candidate=dict(schema='heroes-local-training-runtime-candidate-v1',status='candidate_native_validated',
            release_id=options.release_id,checkpoint_update=checkpoint_update,runtime_bin=str(root/'runtime'),
            runtime_files={'vcmiclient':'client','vcmireplay':'replay'},native={'report':{'sha256':'n'*64}},
            native_engine_correction='Verifier-only ExchangeArtifacts accounting.')
        write(options.candidate_output,candidate)
        schedule=[dict(episode=i+1,phase='evaluation',learner_player=i//4,
                       engine_seed=1740000000+i,policy_seed=1750000000+i) for i in range(8)]
        plan=dict(runtime_candidate=refresh.binding(options.candidate_output),mode='evaluation',parent_update=checkpoint_update,
            runtime={'vcmireplay':'replay'},source={'tracked_worktree_dirty':False,'git_commit':'c'*40},
            absolute_deadline_unix=1000,schedule=schedule,parent_descriptor={'copied':'full-state fixture'})
        write(options.policy_output/'plan.json',plan)
        attempts=[]
        for row in schedule:
            attempts.append(dict(**row,status='passed',training_samples=0,sampled_choices=1,
                reconstruction={'passed':True,'maximum_absolute_difference':0,'actions_resampled':1},
                guard_invariants_verified=True,identity_verified=True,cleanup_verified=True,completion_receipt=True,
                replay_verified=True,weights_unchanged=True,
                outcomes={'red':'loss' if row['learner_player']==0 else 'win',
                          'blue':'loss' if row['learner_player']==1 else 'win'}))
        report=dict(status='passed',plan_sha256=refresh.binding(options.policy_output/'plan.json')['sha256'],
            training_performed=False,optimizer_steps=0,training_samples=0,full_parent_unchanged=True,
            full_parent_restore=True,cleanup_verified=True,parent_progress={'updates':checkpoint_update},attempts=attempts)
        write(options.policy_output/'execution/report.json',report)
        write(options.policy_output/'shutdown.json',dict(returncode=0,owned_cleanup_verified=True,
            hard_deadline_exhausted=False,errors=[],finished_utc='1970-01-01T00:16:00+00:00'))
        return options,report

    def test1335_acceptance_retains_strict_eight_game_no_learning_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();options,report=self.fixture(root)
            with patch.object(refresh,'ARTIFACTS',root),patch.object(refresh,'candidate_runtime_origin'), \
                    patch.object(refresh,'accepted_runtime_origin') as gate, \
                    patch.object(refresh,'implementation_identity',return_value={'native':'unchanged'}), \
                    redirect_stdout(io.StringIO()):
                refresh.acceptance(options)
            gate.assert_called_once()
            accepted=refresh.read(options.acceptance_output)
            policy=refresh.read(options.policy_acceptance_output)
            self.assertEqual((accepted['checkpoint_update'],accepted['release_id']),
                             (1335,'terminal-replay-cloud1335-v1'))
            self.assertEqual((policy['decisions_checked'],policy['optimizer_steps'],policy['training_performed']),(8,0,False))
            self.assertEqual(policy['outcomes'],{'win':0,'loss':8,'cutoff':0})
            self.assertEqual(accepted['native_engine_correction'], 'Verifier-only ExchangeArtifacts accounting.')

    def test1392_acceptance_corrects_inherited_text_without_rewriting_captured_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();options,_=self.fixture(root,1392)
            before=options.candidate_output.read_bytes()
            candidate_binding=refresh.binding(options.candidate_output)
            with patch.object(refresh,'ARTIFACTS',root),patch.object(refresh,'candidate_runtime_origin'), \
                    patch.object(refresh,'accepted_runtime_origin') as gate, \
                    patch.object(refresh,'implementation_identity',return_value={'native':'unchanged'}), \
                    redirect_stdout(io.StringIO()):
                refresh.acceptance(options)
            accepted=refresh.read(options.acceptance_output)
            self.assertEqual(accepted['native_engine_correction'], refresh.TOWN_SWAP_CORRECTION)
            self.assertEqual(options.candidate_output.read_bytes(), before)
            self.assertEqual(refresh.binding(options.candidate_output), candidate_binding)
            self.assertEqual(refresh.read(options.policy_output/'plan.json')['runtime_candidate'], candidate_binding)
            self.assertEqual(gate.call_args.args[2]['native_engine_correction'], refresh.TOWN_SWAP_CORRECTION)

    def test_incomplete_policy_never_creates_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();options,report=self.fixture(root)
            report['attempts'][0]['reconstruction']['maximum_absolute_difference']=0.01
            write(options.policy_output/'execution/report.json',report)
            with patch.object(refresh,'ARTIFACTS',root),patch.object(refresh,'candidate_runtime_origin'), \
                    self.assertRaisesRegex(ValueError,'exact replay/reconstruction'):
                refresh.acceptance(options)
            self.assertFalse(options.acceptance_output.exists())
            self.assertFalse(options.policy_acceptance_output.exists())


if __name__=='__main__':unittest.main()
