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

    def fixture(self,root):
        options=self.options(root)
        candidate=dict(schema='heroes-local-training-runtime-candidate-v1',status='candidate_native_validated',
            release_id=options.release_id,checkpoint_update=1335,runtime_bin=str(root/'runtime'),
            runtime_files={'vcmiclient':'client','vcmireplay':'replay'},native={'report':{'sha256':'n'*64}})
        write(options.candidate_output,candidate)
        schedule=[dict(episode=i+1,phase='evaluation',learner_player=i//4,
                       engine_seed=1740000000+i,policy_seed=1750000000+i) for i in range(8)]
        plan=dict(runtime_candidate=refresh.binding(options.candidate_output),mode='evaluation',parent_update=1335,
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
            full_parent_restore=True,cleanup_verified=True,parent_progress={'updates':1335},attempts=attempts)
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
