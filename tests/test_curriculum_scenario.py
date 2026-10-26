"""Initial-state differences never alter native rules or policy actions."""
import tempfile
from pathlib import Path
import unittest
from experiments.explorer.create_curriculum_scenario import payloads,create,verify
from experiments.explorer.create_capability_scenarios import ordinary_payloads

class CurriculumTests(unittest.TestCase):
    def test_rules_unchanged_and_town_capture_required(self):
        original=ordinary_payloads()
        for stage in (0,1):
            for seat in (0,1):
                value=payloads(stage,seat)
                self.assertEqual(value['header.json']['triggeredEvents'],original['header.json']['triggeredEvents'])
                self.assertEqual(value['surface_terrain.json'],original['surface_terrain.json'])
                towns=[x for x in value['objects.json'] if x['type']=='town']
                self.assertEqual(len(towns),1)
                self.assertEqual(towns[0]['options']['owner'],('blue','red')[seat])
                self.assertEqual(towns[0]['options']['buildings']['noneOf'],[])
                self.assertEqual(value['header.json']['heroLevelLimit'],0)
        self.assertEqual(payloads(2,0),original)
    def test_exact_deterministic_map_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'a.vmap';b=Path(t)/'b.vmap'
            create(a,0,0);create(b,0,0);verify(a,0,0)
            self.assertEqual(a.read_bytes(),b.read_bytes())
            with self.assertRaises(FileExistsError):create(a,0,0)
            with self.assertRaises(ValueError):verify(a,0,1)
