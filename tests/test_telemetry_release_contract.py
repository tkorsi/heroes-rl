"""The telemetry removal release preserves explicit checkpoint and support gates."""
import unittest

from tools import accept_cpu_next_launch as acceptance
from tools import benchmark_pipeline_collection as diagnostic


class TelemetryReleaseContractTests(unittest.TestCase):
    def test_completed_1728_is_required_without_reinterpreting_1585(self):
        descriptor = dict(release_id=acceptance.TELEMETRY_FREE_RELEASE,
                          checkpoint_update=1392, policy_checkpoint_update=1728,
                          policy_support='visible-interaction-budget-revealed-v1',
                          python_environment=acceptance.NEXT_LAUNCH_ENVIRONMENT)
        schema, support, update = acceptance.policy_contract(descriptor)
        self.assertEqual((schema, support, update),
                         (acceptance.REVEALED_SCHEMA, descriptor['policy_support'], 1728))
        self.assertEqual(diagnostic.study_contract(descriptor['release_id'])['policy_checkpoint_update'], 1728)
        for wrong in (1392, 1585, 1727, 1729):
            with self.subTest(update=wrong), self.assertRaises(ValueError):
                acceptance.policy_contract(dict(descriptor, policy_checkpoint_update=wrong))
        with self.assertRaises(ValueError):
            acceptance.policy_contract(dict(descriptor, policy_support='visible-interaction-budget-v1'))
        with self.assertRaises(ValueError):
            acceptance.policy_contract(dict(descriptor, checkpoint_update=1728))
        old = dict(descriptor, release_id=acceptance.REVEALED_RELEASE, policy_checkpoint_update=1585)
        self.assertEqual(acceptance.policy_contract(old)[2], 1585)


if __name__ == '__main__':
    unittest.main()
