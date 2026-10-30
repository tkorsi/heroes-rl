"""An untrained width experiment cannot silently change its learning contract."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import prepare_tiny_policy as tiny
from tools import run_selfplay_training as campaign


class TinyPreparationTests(unittest.TestCase):
    def test_separate_untrained_model_retains_interface_support_and_dimensions(self):
        spec, config = tiny.specification()
        self.assertEqual((config.hidden_size, config.feature_size), (8, 4))
        self.assertEqual(spec['expected_parameters'],15460)
        self.assertIsNone(spec['parent_checkpoint'])
        self.assertFalse(spec['training_launcher_accepted'])
        self.assertEqual(spec['policy_support'],campaign.GUARD)

    def test_changed_parent_rewards_support_or_size_are_rejected(self):
        original = campaign.read(tiny.CONFIG)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'spec.json'
            for mutate in (lambda s:s.update(parent_checkpoint=1753),
                           lambda s:s.update(initialization='distillation'),
                           lambda s:s['rewards'].update(ordinary_transition=0.1),
                           lambda s:s.update(policy_support='none'),
                           lambda s:s['model'].update(hidden_size=64),
                           lambda s:s.update(training_launcher_accepted=True)):
                spec=deepcopy(original);mutate(spec);path.write_text(json.dumps(spec))
                with self.subTest(spec=spec),self.assertRaises(ValueError):tiny.specification(path)

    def test_existing_candidate_and_nonprivate_output_reject_before_lineage_or_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);existing=root/'tiny-policy-experiments'/'existing';existing.mkdir(parents=True)
            marker=existing/'marker';marker.write_text('preserve')
            with patch.object(campaign,'ARTIFACTS',root),patch.object(campaign,'inspect_lineage') as inspect:
                for path in (existing,root/'official-training'):
                    with self.assertRaisesRegex(ValueError,'fresh durable'):tiny.prepare(path)
                inspect.assert_not_called()
            self.assertEqual(marker.read_text(),'preserve')
            self.assertFalse((root/'official-training').exists())


if __name__=='__main__':unittest.main()
