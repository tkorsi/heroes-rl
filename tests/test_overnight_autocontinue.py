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
