"""Offline admission and supervision regressions; no native games or PPO launches."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import subprocess
from experiments.learner import curriculum_training as c
from tools import curriculum_campaign as ctl
from validation.probes.validate_curriculum_policy import schedule


class CampaignTests(unittest.TestCase):
    def test_seed_cohorts_disjoint(self):
        cohorts=[]
        for study in (1,2,3,4):
            cohorts.append({(r['engine_seed'],r['policy_seed']) for b in range(8) for r in schedule(0,3,b,study)})
        cohorts.append({(r['engine_seed'],r['policy_seed']) for r in c.schedule(146,0,True)})
        cohorts.append({(r['engine_seed'],r['policy_seed']) for b in range(500) for r in c.schedule(b,b%2)})
        for i,a in enumerate(cohorts):
            for b in cohorts[i+1:]:self.assertFalse(a & b)

    def fixture(self,root):
        (root/'campaign-v3').mkdir();c.write_new(root/'campaign-v3/frozen.json',{})
        frozen={'capability_implementation':{'test':'fixed'},'runtime':{'fixed':True},'parent':{'policy':{'sha256':'fixed'}}}
        gate={'status':'finished','within_20_minutes':True,'owned_cleanup_verified':True,
              'frozen':c.binding(root/'campaign-v3/frozen.json'),'attempts':[]}
        def add(kind,seat,index,report,**kw):
            p=root/f'{kind}-{seat}-{index}.json';c.write_new(p,report)
            gate['attempts'].append(dict(kind=kind,seat=seat,status='finished',owned_cleanup_verified=True,report=c.binding(p),**kw))
        for mode in ('capture','wait'):
            for seat in (0,1):add('control',seat,mode,dict(replay={'verified':True},owned_processes_closed=True,
                termination_reason='natural_terminal',outcomes={('red','blue')[seat]:'win' if mode=='capture' else 'loss'}),mode=mode)
        for block in range(8):
            for seat in (0,1):
                p=root/f'plan-{block}-{seat}.json';c.write_new(p,dict(study=4,schedule=schedule(seat,3,block,4),
                    implementation=frozen['capability_implementation'],runtime=frozen['runtime'],checkpoint=frozen['parent']['policy']))
                a=[dict(status='passed',replay_verified=True,weights_unchanged=True,finished=True,
                    reconstruction={'passed':True,'weights_unchanged':True},termination_reason='natural_terminal',
                    learner_player=seat,outcomes={('red','blue')[seat]:'win' if i==0 else 'loss'}) for i in range(4)]
                add('policy',seat,block,dict(plan=c.binding(p),status='passed',training_samples=0,optimizer_steps=0,attempts=a),block=block)
        p=root/'gate.json';c.write_new(p,gate);return p,gate,frozen

    def test_complete_gate_and_fail_closed_variants(self):
        for failure in (None,'missing_block','failed_row','source_drift','altered_report','admitted_diagnostic','no_wins'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as td:
                root=Path(td);p,g,f=self.fixture(root)
                if failure=='missing_block':g['attempts'].pop()
                elif failure=='failed_row':g['attempts'][-1]['status']='failed'
                elif failure=='source_drift':f['runtime']={'other':True}
                elif failure in ('altered_report','admitted_diagnostic','no_wins'):
                    rows=g['attempts'][4:] if failure=='no_wins' else [g['attempts'][-1]]
                    for row in rows:
                        q=Path(row['report']['path']);r=json.loads(q.read_text())
                        if failure=='admitted_diagnostic':r['training_samples']=1
                        else:
                            for a in r['attempts']:a['outcomes'][('red','blue')[row['seat']]]='loss'
                        q.write_text(json.dumps(r))
                        if failure!='altered_report':row['report']=c.binding(q)
                p.write_text(json.dumps(g))
                with patch.object(c,'BASE',root),patch.object(c,'SESSION',root/'campaign-v3/session.json'),patch.object(c,'check_frozen',return_value=f):
                    if failure is None:self.assertEqual(c.check_gate(p),{0:{'win':8,'loss':24},1:{'win':8,'loss':24}})
                    else:
                        with self.assertRaises(ValueError):c.check_gate(p)

    def test_monotonic_deadline_dominates_wall_rollback(self):
        s={'absolute_deadline_unix':8200,'compute_deadline_unix':7900,'started_monotonic':10}
        with patch.object(ctl.time,'time',return_value=0),patch.object(ctl.time,'monotonic',return_value=7211):
            self.assertLess(ctl.remaining(s),0);self.assertFalse(ctl.enough_for_batch(s,[10]))

    def test_complete_receipts_required(self):
        r={'status':'finished','worker_complete':True,'supervisor_cleanup_verified':True,'within_shared_budget':True,
           'attempts':[dict(status='passed',completion_receipt=True,cleanup_verified=True,identity_verified=True,replay_verified=True) for _ in range(4)]}
        self.assertTrue(ctl.verified_report(r,0));self.assertFalse(ctl.verified_report(r,1))
        r['attempts'][3]['replay_verified']=False;self.assertFalse(ctl.verified_report(r,0))
        r['attempts'].pop();self.assertFalse(ctl.verified_report(r,0))

    def test_timeout_preserves_partial_batch_and_does_not_advance(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'plan.json';p.write_text('{}')
            fake=MagicMock();fake.wait.side_effect=subprocess.TimeoutExpired('worker',1)
            v={'batches':[],'new_updates':0,'latest_private_parent':{'old':True}}
            with patch.object(c,'prepare',return_value=p),patch.object(c,'session',return_value={'absolute_deadline_unix':9999999999,'compute_deadline_unix':9999999699,'started_monotonic':9999999999}),patch.object(ctl,'save'),patch.object(ctl,'event'),patch.object(ctl,'guarded_popen',return_value=fake),patch.object(ctl,'close_owned_process') as close,patch.object(ctl,'command_timeout',return_value=2),patch.object(ctl,'publish',return_value={'commit':'saved'}) as publish:
                with self.assertRaises(RuntimeError):ctl.run(v,p,0,0,'training')
                publish.assert_called_once_with(root,'campaign-v3-'+c.SESSION.parent.name+'-'+root.name,checkpoint=False)
                close.assert_called_once();self.assertEqual(v['new_updates'],0);self.assertEqual(v['batches'][0]['status'],'failed')

    def test_no_partial_admission_after_verification_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'plan.json';p.write_text('{}');(root/'execution').mkdir()
            (root/'execution/report.json').write_text(json.dumps({'attempts':[], 'optimizer_steps':0,'training_samples':0,'status':'stopped'}))
            fake=MagicMock();fake.wait.return_value=0
            v={'batches':[],'new_updates':0}
            with patch.object(c,'prepare',return_value=p),patch.object(c,'session',return_value={'absolute_deadline_unix':9999999999,'started_monotonic':9999999999}),patch.object(ctl,'save'),patch.object(ctl,'event'),patch.object(ctl,'guarded_popen',return_value=fake),patch.object(ctl,'close_owned_process'),patch.object(ctl,'command_timeout',return_value=2),patch.object(ctl,'publish',return_value={}) as publish:
                with self.assertRaises(RuntimeError):ctl.run(v,p,0,0,'training')
                self.assertFalse(publish.call_args.kwargs['checkpoint']);self.assertEqual(v['new_updates'],0)

    @unittest.skipUnless(c.PARENT.exists(), 'Local exact update146 fixture unavailable')
    def test_exact_parent_and_real_plan_admission_without_gameplay(self):
        import shutil
        from experiments.learner.builtin_gameplay import TrainingBinding
        from experiments.learner.inference import source_identity
        state=c.restore(c.PARENT)
        self.assertEqual(vars(state.progress),dict(updates=146,rollouts=584,decisions=40641))
        self.assertEqual(state.optimizer_steps,1475)
        original=json.loads(c.PARENT.read_text())
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).resolve();(base/'maps').mkdir()
            for seat in (0,1):shutil.copy2(c.BASE/f'maps/curriculum-3-{seat}.vmap',base/f'maps/curriculum-3-{seat}.vmap')
            source=source_identity(c.INTERFACE);source['tracked_worktree_dirty']=False
            release=dict(runtime_bin=str(Path(original['binary']).parent),data=original['data'],runtime=original['runtime'],authorization='Offline plan validation; never executable')
            rp=base/'release.json';c.write_new(rp,release)
            sp=base/'session.json';c.write_new(sp,{})
            session=dict(source=source,release=c.binding(rp),absolute_deadline_unix=9999999999,compute_deadline_unix=9999999699)
            with patch.object(c,'BASE',base),patch.object(c,'SESSION',sp),patch.object(c,'session',return_value=session):
                for seat in (0,1):
                    p=c.prepare(c.PARENT,seat,seat);plan=c.read_plan(p,c.binding(p)['sha256'])
                    bound=TrainingBinding.create(plan_path=p,plan_sha256=c.binding(p)['sha256'],schedule_index=0,
                        folder=p.parent/'execution/episode-000001',checkpoint=state.checkpoint,
                        checkpoint_file=plan['checkpoint']['path'],weights_sha256=state.weights_sha256,
                        opponent_identity=c.pilot.native_opponent(plan['runtime'],seat),source_identity=source,
                        native_identity=plan['native'],max_days=365,nominal_choices=4960,max_choices=5120)
                    bound.recheck()
                    ep=c.prepare(c.PARENT,146,seat,evaluate=True)
                    self.assertEqual(c.read_plan(ep,c.binding(ep)['sha256'])['mode'],'evaluation')


    def test_freeze_rejects_missing_engine_before_artifact_or_game_work(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'dependencies').mkdir()
            (root/'dependencies/vcmi.json').write_text(json.dumps({'directory':'vcmi','commit':'pinned'}))
            with patch.object(ctl,'ROOT',root),patch.object(ctl,'private_preflight') as private:
                with self.assertRaisesRegex(ValueError,'Generated VCMI dependency missing'):ctl.freeze()
                private.assert_not_called()

    def test_engine_preflight_rejects_wrong_revision_without_preparation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'dependencies').mkdir();(root/'vcmi').mkdir()
            (root/'dependencies/vcmi.json').write_text(json.dumps({'directory':'vcmi','commit':'pinned'}))
            with patch.object(ctl,'ROOT',root),patch.object(ctl.subprocess,'check_output',return_value='other'),patch.object(ctl.subprocess,'run') as run:
                with self.assertRaisesRegex(ValueError,'HEAD differs'):ctl.check_engine_checkout()
                run.assert_not_called()

    def test_recovery_session_cannot_reset_or_extend_original_window(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).resolve();root=base/'campaign-v3';root.mkdir()
            window=root/'window.json';clock=dict(started_unix=0,started_monotonic=0,compute_deadline_unix=6900,absolute_deadline_unix=7200)
            c.write_new(window,clock);sp=root/'recoveries/r1/session.json';release=root/'release.json'
            source={'frozen':'source'};c.write_new(release,{'source':source,'configuration':{}})
            v=dict(clock,scope=c.SCOPE,source=source,window=c.binding(window),release=c.binding(release))
            c.write_new(sp,v)
            with patch.object(c,'BASE',base),patch.object(c,'SESSION',sp),patch.object(c,'WINDOW',window),patch.object(c,'check_frozen'),patch.object(c,'source_identity',return_value=source),patch.object(c,'configuration',return_value={}),patch.object(c.time,'time',return_value=100),patch.object(c.time,'monotonic',return_value=100):
                self.assertEqual(c.session()['absolute_deadline_unix'],7200)
                for name in ('started_unix','started_monotonic','compute_deadline_unix','absolute_deadline_unix'):
                    wrong=dict(v);wrong[name]+=60;sp.write_text(json.dumps(wrong))
                    with self.assertRaisesRegex(ValueError,'cannot reset or extend'):c.session()
                sp.write_text(json.dumps(v));window.write_text(json.dumps(dict(clock,absolute_deadline_unix=8000)))
                with self.assertRaisesRegex(ValueError,'clock binding changed'):c.session()
