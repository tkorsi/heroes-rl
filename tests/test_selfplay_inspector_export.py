"""Presentation comparisons retain every replay fact except elapsed timings."""
import unittest
from tools.export_selfplay_inspector import replay_facts


class ReplayFactTests(unittest.TestCase):
    def test_nondeterministic_clocks_do_not_change_replay_facts(self):
        a = {'server': {'checked': 11, 'timing_seconds': {'input': 1}}, 'boundary': 'cutoff'}
        b = {'server': {'checked': 11, 'timing_seconds': {'input': 2}}, 'boundary': 'cutoff'}
        self.assertEqual(replay_facts(a), replay_facts(b))
        for changed in ({'server': {'checked': 10}, 'boundary': 'cutoff'},
                        {'server': {'checked': 11}, 'boundary': 'natural'},
                        {'server': {'checked': 11, 'unchecked': 1}, 'boundary': 'cutoff'}):
            self.assertNotEqual(replay_facts(a), replay_facts(changed))


if __name__ == '__main__':
    unittest.main()
