"""Scientific counter controls: valid cutoffs stay separate from failed prefixes."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import overnight_curriculum_campaign as campaign


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    return campaign.binding(path)


class AnalysisControls(unittest.TestCase):
    def test_failed_prefix_never_supplies_learning_or_behavior_totals(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            report=write(root/'failed-report.json',dict(attempts=[{'status':'passed'},{'status':'failed'},
                {'status':'not_started'},{'status':'not_started'}]))
            v=dict(status='stopped_failure',new_updates=0,latest_progress={'updates':200},latest_verified_parent={'old':True},
                batches=[dict(mode='training',seat=0,status='failed',report=report,completed_games=1,
                    training_samples=999,optimizer_steps=999,outcomes=['win'])])
            result=campaign.analysis(v)
            self.assertEqual(result['training_games'],0);self.assertEqual(result['admitted_decisions'],0)
            self.assertEqual(result['new_adam_steps'],0);self.assertEqual(result['failed_game_attempts'],1)
            self.assertEqual(result['unstarted_games'],2);self.assertEqual(result['behavior_totals'],{})
            self.assertFalse(result['valid_final_paired_comparison_available'])

    def test_complete_games_separate_natural_outcomes_cutoffs_and_observed_effects(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);attempts=[]
            for i in range(4):
                folder=root/f'episode-{i}';folder.mkdir()
                history=(json.dumps({'kind':'event','event':{'kind':'battle_start','side':'attacker' if i<2 else 'defender'}})+'\n').encode()
                (folder/'own.jsonl').write_bytes(history)
                write(folder/'manifest.json',{'players':{'red':{'history':{'path':'own.jsonl','sha256':hashlib.sha256(history).hexdigest()}}}})
                aggregate=dict(recruitment={'completed':1},army_operations={'across_armies':2,
                    'observed_cross_army_net_count_redistribution':3},movement={'attempts':4,'endpoint_changes':2,
                    'targets_visible_opponent_town':1},town_capture_evidence=1)
                behavior=write(folder/'behavior.json',dict(status='complete',coverage={'complete':True},aggregate=aggregate))
                attempts.append(dict(status='passed',replay_verified=True,identity_verified=True,
                    behavior=behavior,episode_directory=str(folder),learner_player=0))
            report=write(root/'report.json',dict(attempts=attempts,execution_wall_seconds=20.))
            parent={'sha256':'final'}
            v=dict(status='finished_budget_reserve',new_updates=1,latest_progress={'updates':201},latest_verified_parent=parent,
                batches=[dict(mode='training',seat=0,status='passed',checkpoint_verified=True,report=report,
                    completed_games=4,training_samples=8,optimizer_steps=1,outcomes=['win','loss',None,'loss'])])
            with patch('experiments.headless.builtin_evidence.manifest_profile',return_value=SimpleNamespace(history_key='history')):
                result=campaign.analysis(v)
            self.assertEqual(result['training_outcomes_by_seat']['red'],{'natural_wins':1,'natural_losses':2,'valid_cutoffs':1})
            self.assertEqual(result['admitted_decisions_per_reported_batch_second'],.4)
            self.assertEqual(result['behavior_totals']['recruitment_completed_actions'],4)
            self.assertEqual(result['behavior_totals']['visible_net_army_redistribution'],12)
            self.assertEqual(result['battle_start_roles'],{'attacker':2,'defender':2})
            self.assertFalse(result['valid_final_paired_comparison_available'])


if __name__=='__main__':unittest.main()
