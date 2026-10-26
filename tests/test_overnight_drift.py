"""Cross-platform sampling is descriptive; probability/support drift still blocks."""
from types import SimpleNamespace as N
import unittest
from unittest.mock import MagicMock,patch
import torch
from tools import measure_overnight_platform_drift as drift


class DriftControls(unittest.TestCase):
    def test_platform_samples_do_not_override_numeric_or_support_gate(self):
        for difference,support,passed in [(0.,('same',),True),(.01,('same',),False),(0.,('changed',),None)]:
            with self.subTest(difference=difference,support=support):
                distribution=MagicMock();distribution.support=support
                distribution.sample.return_value='platform_sample'
                distribution.evaluate.return_value=N(log_prob=torch.tensor(-.5+difference),conditionals=())
                model=MagicMock();model.forward_step.return_value=N(state=None,decision=N(value=torch.tensor(0.),distribution=distribution))
                episode=N(weights_sha256='fixed',player=0,policy_seed=1,
                    history=[N(episode='offline',player=0,schema='adventure-capabilities-v1',seq=0)],
                    estimates=[N(decision_seq=0,value=0.)],
                    decisions=[N(decision_seq=0,support=('same',),action='historical_sample',log_prob=-.5,conditionals=())])
                with patch.object(drift,'weights_sha256',return_value='fixed'),patch.object(drift,'ContextualEncoder'):
                    if passed is None:
                        with self.assertRaisesRegex(ValueError,'support drift'):drift.measure(episode,model)
                    else:
                        result=drift.measure(episode,model)
                        self.assertEqual(result['passed'],passed);self.assertEqual(result['action_mismatches'],1)
                        self.assertTrue(result['same_linux_reconstruction_requires_exact_actions'])


if __name__=='__main__':unittest.main()
