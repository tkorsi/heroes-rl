"""Run the actual Swift map projection against disappearance and ownership cases."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('swiftc'), 'Swift compiler required')
class MapProjectionTests(unittest.TestCase):
    def test_visibility_removal_and_mine_ownership(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p / 'main.swift').write_text('''
let redTown = [0,4,7,0,98,0,0]
let gold = [8,4,10,0,79,6,-2]
let enemy = [3,15,7,0,34,2,1]
let objects = [redTown,gold,enemy]
precondition(ReplayMapProjection.visibleObjects(objects,ids:[0,8]) == [redTown,gold])
precondition(ReplayMapProjection.visibleObjects(objects,ids:[3]) == [enemy])
// Old knowledge IDs cannot display resources or heroes removed from native state.
precondition(ReplayMapProjection.visibleObjects([redTown],ids:[0,8,3]) == [redTown])
precondition(ReplayMapProjection.mineOwner([4,5,16,0,53,0,-1]) == nil)
precondition(ReplayMapProjection.mineOwner([4,5,16,0,53,0,0]) == 0)
precondition(ReplayMapProjection.mineOwner([4,5,16,0,53,0,1]) == 1)
precondition(ReplayMapProjection.mineOwner(redTown) == nil)
''')
            subprocess.run(['swiftc',str(ROOT/'apps/replay-inspector/Sources/HeroesReplay/ReplayMapProjection.swift'),
                            str(p/'main.swift'),'-o',str(p/'check')],check=True,capture_output=True,timeout=60)
            subprocess.run([str(p/'check')],check=True,timeout=10)


if __name__ == '__main__':
    unittest.main()
