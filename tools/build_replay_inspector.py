#!/usr/bin/env python3
"""Build a local macOS app, optionally bundling an exported verified replay."""
import argparse
from pathlib import Path
import plistlib
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--match', type=Path)
p.add_argument('--output', type=Path, default=ROOT / 'runs/replay-inspector/Heroes Replay.app')
a = p.parse_args()
package = ROOT / 'apps/replay-inspector'
subprocess.run(['swift', 'build', '--package-path', str(package), '-c', 'release'], check=True)
bin_dir = subprocess.check_output(['swift', 'build', '--package-path', str(package), '-c', 'release', '--show-bin-path'], text=True).strip()
contents = a.output.resolve() / 'Contents'
(contents / 'MacOS').mkdir(parents=True, exist_ok=True)
(contents / 'Resources').mkdir(exist_ok=True)
shutil.copy2(Path(bin_dir) / 'HeroesReplay', contents / 'MacOS/HeroesReplay')
if a.match:
    shutil.copy2(a.match, contents / 'Resources/match.json')
elif (contents / 'Resources/match.json').exists():
    (contents / 'Resources/match.json').unlink()
with (contents / 'Info.plist').open('wb') as f:
    plistlib.dump(dict(CFBundleExecutable='HeroesReplay', CFBundleIdentifier='local.heroes.replay-inspector',
                      CFBundleName='Heroes Replay', CFBundleDisplayName='Heroes Replay',
                      CFBundlePackageType='APPL', CFBundleVersion='1', CFBundleShortVersionString='0.1.0',
                      NSHighResolutionCapable=True, LSMinimumSystemVersion='14.0'), f)
subprocess.run(['codesign', '--force', '--deep', '--sign', '-', str(a.output.resolve())], check=True)
print(a.output.resolve())
