"""Non-gameplay fake-marker controls for the one-time autonomous launch chain."""
import copy
import fcntl
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import MagicMock,patch

from tools import overnight_handoff as h,overnight_autocontinue as chain


def marker():
    return dict(schema=h.SCHEMA,ownership_transfer='local -> cloud',local_training_finished=True,
        local_auto_resume=False,no_local_training_after_readiness=True,
        repository='tkorsi/heroes-rl-training-state',branch='codex/local365-teacher-results',
        public_repository='tkorsi/heroes-rl',public_branch='codex/teacher-curriculum-v1',
        executable_source_commit=h.SOURCE,source={'git_commit':h.SOURCE},
        clock=dict(started_unix=h.LOCAL_START,compute_deadline_unix=h.LOCAL_START+6900,
            absolute_deadline_unix=h.LOCAL_START+7200,started_monotonic=999999999),
        ready_utc='2026-10-26T03:10:59+00:00',local_stop='2026-10-26T03:05:50+00:00',
        local_finalization='2026-10-26T03:10:55+00:00',cloud_hard_deadline_utc='2026-10-26T10:00:00+00:00',
        final_update=200,progress={'updates':200,'rollouts':800,'decisions':50000},
        local_ppo_updates=54,admitted_local_games=216,
        configuration=dict(interface='adventure-capabilities-v1',max_days=365,ordinary_choice_limit=4960,
            reserved_dialog_choices=160,max_total_choices=5120,allow_verified_training_prefix=False,
            max_optimizer_steps=640,ppo={'epochs':1},torch_threads=1),maps_by_seat={'0':{},'1':{}})


class HandoffControls(unittest.TestCase):
    def test_complete_fake_bundle_materializes_without_rewriting_and_rejects_tamper(self):
        """Exercise the actual handoff reader, with mocked Git blobs, no tensors/gameplay."""
        def encoded(value):return (json.dumps(value,indent=2)+'\n').encode()
        def bound(path,data):return dict(path=str(path),sha256=h.digest(data),bytes=len(data))
        old=h.OLD/'local-curriculum-update146/campaign-v3'
        folder=old/'batches/train-0053-seat1'
        previous=old/'batches/train-0052-seat0'
        original={}
        def add(name,path,data):original[name]=(str(path),data)
        m=marker();m.update(archive_commit='c'*40,public_commit='a'*40,
            initial_checkpoint_dependency=bound(previous/'execution/after-policy.pt',b'old-policy'),
            cumulative_adam_steps=1900,next_unused_training_sequence=54)
        from experiments.learner.curriculum_training import schedule
        m['next_training_schedule']=schedule(54,0)
        m['heldout_schedule_by_seat']={str(s):schedule(200,s,True) for s in (0,1)}
        source=m['source'];source['source_sha256']={'experiments/learner/example.py':h.digest(b'unchanged source')}
        m['runtime']={'test':'Mac historical bytes'}
        saved=dict(policy=bound(folder/'execution/after-policy.pt',b'final-policy'),
            resume=bound(folder/'execution/after-resume.pt',b'final-resume'),weights_sha256='f'*64)
        plan=dict(source=source,runtime=m['runtime'],configuration=m['configuration'],
            checkpoint_metadata={'initial_checkpoint':m['initial_checkpoint_dependency']},
            parent=dict(plan=bound(previous/'plan.json',b'previous-plan'),report=bound(previous/'execution/report.json',b'previous-report'),resume=bound(previous/'execution/after-resume.pt',b'previous-resume')))
        add('plan.json',folder/'plan.json',encoded(plan))
        plan_binding=bound(folder/'plan.json',encoded(plan))
        report=dict(plan_sha256=plan_binding['sha256'],after_checkpoint=saved)
        add('report.json',folder/'execution/report.json',encoded(report))
        add('after-policy.pt',folder/'execution/after-policy.pt',b'final-policy')
        add('after-resume.pt',folder/'execution/after-resume.pt',b'final-resume')
        add('dependencies/initial-policy.pt',previous/'execution/after-policy.pt',b'old-policy')
        add('dependencies/parent-plan.json',previous/'plan.json',b'previous-plan')
        add('dependencies/parent-report.json',previous/'execution/report.json',b'previous-report')
        add('dependencies/parent-resume.pt',previous/'execution/after-resume.pt',b'previous-resume')
        session=dict(m['clock']);add('session.json',old/'session.json',encoded(session))
        add('window.json',old/'window.json',encoded(m['clock']))
        controller=dict(owned_cleanup_verified=True,stopped_utc=m['local_stop'],finalized_utc=m['local_finalization'],
            latest_restore={'policy':saved['policy']})
        add('controller-result.json',old/'controller-result.json',encoded(controller))
        raw_files={folder.name+'/plan.json':encoded(plan),folder.name+'/execution/report.json':encoded(report)}
        def archive(files):
            stream=io.BytesIO()
            with tarfile.open(fileobj=stream,mode='w:gz') as t:
                for name,data in files.items():
                    info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
            return stream.getvalue()
        raw=archive(raw_files);raw_path='results/local-curriculum-update146/last/evidence.tar.gz'
        preserved=dict(commit='b'*40,archive=bound(h.OLD/'private-local365-state'/raw_path,raw))
        journal=dict(status='finished_budget_reserve',latest_progress=m['progress'],latest_verified_parent=plan_binding,
            new_updates=m['local_ppo_updates'],batches=[dict(mode='training',sequence=53,plan=plan_binding,private_preservation=preserved)])
        add('campaign-report.json',old/'campaign-report.json',encoded(journal))
        for seat in (0,1):add(f'maps/curriculum-3-{seat}.vmap',h.OLD/f'local-curriculum-update146/maps/curriculum-3-{seat}.vmap',bytes([seat]))
        inventory={name:dict(original=bound(path,data),committed_path=h.PREFIX+'bundle/'+name,sha256=h.digest(data),bytes=len(data)) for name,(path,data) in original.items()}
        files={'bundle/'+name:data for name,(path,data) in original.items()}
        files['bundle/shutdown-proof.json']=encoded(dict(local_auto_resume=False,verified_utc='2026-10-26T03:10:56+00:00',receipts=[{'closed_groups':[123]}],controllers=[{'owned_command_absent':True,'session':inventory['session.json']['original']}]))
        restore=dict(exact_model_adam_rng_progress_pool=True,progress=m['progress'])
        files['bundle/restore-receipt.json']=encoded(dict(canonical=restore,copied=restore))
        final_raw=archive({'final-evidence/session.json':encoded(session)})
        final_path='results/local-curriculum-update146/final/evidence.tar.gz'
        m['final_controller_archive']=dict(commit='e'*40,archive=bound(h.OLD/'private-local365-state'/final_path,final_raw))
        bundle=archive(files)
        m.update(files=inventory,checkpoint=saved,local_session=inventory['session.json']['original'],
            archive=dict(path=h.PREFIX+'bundle.tar.gz',sha256=h.digest(bundle),bytes=len(bundle)),
            maps_by_seat={str(s):inventory[f'maps/curriculum-3-{s}.vmap'] for s in (0,1)},prior_batch_archives=['b'*40])
        for key,name in [('shutdown_proof','shutdown-proof.json'),('restore_evidence','restore-receipt.json')]:
            m[key]=dict(commit='c'*40,path=h.PREFIX+'bundle/'+name,sha256=h.digest(files['bundle/'+name]))
        blobs={'d'*40+':'+h.PREFIX+'ready.json':encoded(m),'c'*40+':'+h.PREFIX+'bundle.tar.gz':bundle,
            'c'*40+':'+h.PREFIX+'inventory.json':encoded(inventory),'b'*40+':'+raw_path:raw,
            h.SOURCE+':experiments/learner/example.py':b'unchanged source','e'*40+':'+final_path:final_raw}
        blobs.update({'c'*40+':'+h.PREFIX+name:data for name,data in files.items()})
        def git(repo,*args):
            if args[0]=='merge-base':return b''
            if args[0]=='cat-file':return b'commit\n'
            self.assertEqual(args[0],'show');return blobs[args[1]]
        with tempfile.TemporaryDirectory() as td,patch.object(h,'git',side_effect=git),patch.object(h.time,'time',return_value=h.LOCAL_START+7500):
            root=Path(td)/'valid';receipt=h.materialize('private','d'*40,root,'public')
            self.assertTrue(receipt['ownership_transfer_verified'])
            proof=h.audit_materialized(root/'handoff/verification.json','private')
            self.assertTrue(json.loads(Path(proof['path']).read_text())['passed'])
            self.assertEqual(Path(receipt['parent']['plan']['path']).read_bytes(),encoded(plan))
            self.assertEqual(receipt['marker']['clock']['started_monotonic'],999999999)
            self.assertFalse(receipt['historical_metadata_rewritten'])
            blobs['c'*40+':'+h.PREFIX+'bundle.tar.gz']=bundle+b'changed'
            with self.assertRaisesRegex(ValueError,'hash/size'):h.materialize('private','d'*40,Path(td)/'invalid','public')

    def test_fresh_final_marker_and_fail_closed_variants(self):
        now=h.LOCAL_START+7500
        self.assertEqual(h.validate_marker(marker(),now)['final_update'],200)
        changes=[('schema','intermediate'),('ownership_transfer','local'),('local_auto_resume',True),
            ('local_training_finished',False),('executable_source_commit','stale'),('final_update',146),
            ('ready_utc','2026-10-25T17:00:00Z'),('cloud_hard_deadline_utc','2026-10-26T12:00:00Z')]
        for key,value in changes:
            with self.subTest(key=key):
                m=marker();m[key]=value
                with self.assertRaises(ValueError):h.validate_marker(m,now)
        with self.assertRaises(ValueError):h.validate_marker(marker(),h.COMPUTE-100)

    def archive(self,names):
        output=io.BytesIO()
        with tarfile.open(fileobj=output,mode='w:gz') as t:
            for name in names:
                m=tarfile.TarInfo(name);m.size=1;t.addfile(m,io.BytesIO(b'x'))
        return output.getvalue()

    def test_archive_escape_and_duplicate_fail_closed(self):
        for names in (['../escape'],['/absolute'],['bundle/a','bundle/a']):
            with self.assertRaises(ValueError):h.safe_files(self.archive(names))
        self.assertEqual(h.safe_files(self.archive(['bundle/a']))[0],{'bundle/a':b'x'})

    def test_acceptance_failure_never_reaches_ppo(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);(base/'study4').mkdir()
            calls=[]
            def command(args,label,*a,**kw):
                calls.append(label)
                if label=='linux-acceptance':raise RuntimeError('fake rejected replay')
            with patch.object(chain,'BASE',base),patch.object(chain,'run',side_effect=command),\
                patch.object(chain,'event'),patch.object(chain,'status'),patch.object(chain,'public_push'),\
                patch.object(chain,'private_publish',return_value={'retained':True}) as preserve,\
                patch.object(chain,'guarded_popen') as launch:
                with self.assertRaisesRegex(RuntimeError,'fake rejected replay'):chain.campaign(base)
                launch.assert_not_called();preserve.assert_called_once()
                self.assertEqual(calls,['freeze-final-release','linux-acceptance'])

    def test_good_offline_chain_has_one_launch_only_after_all_gates(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);calls=[];proc=MagicMock();proc.poll.return_value=0;proc.returncode=0;proc.pid=123
            def command(args,label,*a,**kw):calls.append(label)
            def launch(*a,**kw):
                self.assertEqual(calls,['freeze-final-release','linux-acceptance','prepare-training-release'])
                self.assertEqual(kw['deadline_wall'],h.HARD-90)
                return proc
            with patch.object(chain,'BASE',base),patch.object(chain,'run',side_effect=command),\
                patch.object(chain,'event'),patch.object(chain,'status'),patch.object(chain,'public_push'),\
                patch.object(chain,'private_publish',return_value={'retained':True}),\
                patch.object(chain,'guarded_popen',side_effect=launch) as start,\
                patch.object(chain,'close_owned_process'),patch.object(chain,'cleanup_from_evidence'):
                chain.campaign(base);start.assert_called_once()
                self.assertEqual(json.loads((base/'automatic-campaign-command.json').read_text())['compute_deadline_unix'],h.COMPUTE)

    def test_duplicate_controller_exclusion_precedes_any_work(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            with (base/'autocontinue.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch.object(chain,'BASE',base),patch.object(chain,'wait_for_handoff') as wait:
                    with self.assertRaises(BlockingIOError):chain.main()
                    wait.assert_not_called()


if __name__=='__main__':unittest.main()
