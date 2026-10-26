"""Offline strict-equivalence, integrity and ownership controls; no games."""
import copy
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch
from tools import run_transfer_eval as controller
from tools import transfer_eval_watchdog as guard
from tools import transfer_eval_worker as worker


def gate_rows():
    cells=[]
    for backend in ('cold-reference','accelerated-embedded'):
        for seat in (0,1):
            rows=[dict(schedule=dict(seat=seat,engine_seed=710001+i,policy_seed=720001+i),status='passed',history_sha256='h',decisions_sha256='d',estimates_sha256='e',outcome='win',termination_reason='natural_terminal') for i in range(2)]
            cells.append(dict(label=f'gate-{backend}-seat{seat}',rows=rows))
    return cells


class TransferControls(unittest.TestCase):
    def test_exact_both_seat_gate_passes(self):
        self.assertTrue(controller.gate_comparison(gate_rows())['passed'])

    def test_numeric_evidence_difference_has_no_tolerance_bypass(self):
        rows=gate_rows();rows[-1]['rows'][-1]['estimates_sha256']='different even if numerically close'
        self.assertFalse(controller.gate_comparison(rows)['passed'])

    def test_missing_or_failed_optimized_row_cannot_pass(self):
        rows=gate_rows();rows[-1]['rows'][-1]['status']='failed'
        self.assertFalse(controller.gate_comparison(rows)['passed'])
        self.assertFalse(controller.gate_comparison(gate_rows()[:-1])['passed'])

    def test_reused_pid_never_signalled(self):
        identity=dict(pid=42,start_ticks=100,command=['python','owned'])
        with patch.object(guard,'process',return_value=dict(identity,start_ticks=101)),patch.object(guard.os,'kill') as kill:
            self.assertFalse(guard.stop(identity,signal.SIGTERM));kill.assert_not_called()

    def test_only_run_workers_are_compute_not_restore_or_publication(self):
        self.assertTrue(guard.compute(['python','worker','--mode','run']))
        for command in (['python','worker','--mode','restore'],['git','push'],['python','--mode']):
            self.assertFalse(guard.compute(command))

    def test_changed_checkpoint_rejected_before_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'after-policy.pt';path.write_bytes(b'changed')
            spec=dict(files={'after-policy.pt':dict(path=str(path),sha256='wrong',bytes=7)})
            with patch('experiments.learner.builtin_preservation.verify_pair') as restore:
                with self.assertRaisesRegex(ValueError,'Frozen input changed'):worker.load_frozen(spec)
                restore.assert_not_called()


if __name__=='__main__':unittest.main()
