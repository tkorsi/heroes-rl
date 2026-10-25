#!/usr/bin/env python3
"""Record Linux CPU runtime and native CLI/library checks without launching games.

Use the locked environment activation first. CLI checks get a disposable XDG
profile because a cloud user's home may be read-only. Output is exclusive;
failed checks remain recorded and cause a nonzero exit.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ('vcmiclient', 'vcmiserver', 'vcmireplay', 'libvcmi.so',
             'AI/libExplorerRL.so', 'AI/libBattleAI.so', 'AI/libNullkiller.so')


def binding(path):
    with path.open('rb') as stream:
        return {'sha256': hashlib.file_digest(stream, 'sha256').hexdigest(),
                'bytes': os.fstat(stream.fileno()).st_size}


def command(arguments, **kwargs):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=120, **kwargs)
    return {'command': arguments, 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


def inspect(binary_directory):
    import torch
    binary_directory = Path(binary_directory).resolve()
    started = time.monotonic()
    cgroup = {}
    for name in ('cpu.max', 'memory.max', 'memory.current', 'memory.swap.max'):
        path = Path('/sys/fs/cgroup') / name
        cgroup[name] = path.read_text().strip() if path.is_file() else None
    report = {'schema': 'heroes-cloud-runtime-check-v1',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'platform': platform.platform(), 'machine': platform.machine(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'cgroup': cgroup,
        'disk': shutil.disk_usage(ROOT)._asdict(), 'python': sys.version,
        'torch': str(torch.__version__), 'torch_threads': torch.get_num_threads(),
        'cuda': torch.version.cuda, 'hip': torch.version.hip,
        'binary_directory': str(binary_directory), 'artifacts': {}, 'checks': [],
        'gameplay_performed': False, 'game_replay_performed': False, 'optimizer_updates': 0,
        'scope': 'Native build and CLI/library startup only; no claim of usable private Data or gameplay.'}
    report['cpu_runtime_matches_lock'] = (platform.system() == 'Linux' and platform.machine() == 'x86_64'
        and sys.version_info[:2] == (3, 12) and str(torch.__version__) == '2.9.1+cpu'
        and torch.get_num_threads() == 1 and torch.version.cuda is None and torch.version.hip is None)
    report['tools'] = {}
    for tool in ('c++', 'cmake', 'ninja', 'ffmpeg'):
        observed = command([tool, '-version' if tool == 'ffmpeg' else '--version'])
        report['tools'][tool] = {'returncode': observed['returncode'],
            'version': (observed['stdout'] + observed['stderr']).splitlines()[0]}
    report['recipe'] = command([sys.executable, str(ROOT / 'tools/prepare_vcmi.py'), '--check'], cwd=ROOT)
    for name in ARTIFACTS:
        report['artifacts'][name] = binding(binary_directory / name)
    report['locks'] = {name: binding(ROOT / 'dependencies' / name) for name in
        ('vcmi.json', 'cloud-debian13-amd64.json', 'learner-linux-x86_64-py312.lock')}
    with tempfile.TemporaryDirectory(prefix='heroes-cloud-cli-') as temporary:
        env = dict(os.environ)
        for key, folder in (('XDG_DATA_HOME', 'data'), ('XDG_CONFIG_HOME', 'config'), ('XDG_CACHE_HOME', 'cache')):
            env[key] = str(Path(temporary) / folder)
        for executable, args, expected_code, expected_text in (
            ('vcmiclient', ['--version'], 0, 'VCMI 1.7.5'),
            ('vcmiclient', ['--help'], 0, '--headless'),
            ('vcmiserver', ['--version'], 0, 'VCMI 1.7.5'),
            ('vcmireplay', [], 2, 'Usage:')):
            result = command([str(binary_directory / executable), *args], cwd=binary_directory, env=env)
            result.update(expected_returncode=expected_code,
                passed=result['returncode'] == expected_code and expected_text in result['stdout'] + result['stderr'])
            report['checks'].append(result)
    for name in ARTIFACTS[3:]:
        result = {'library': name, 'passed': False}
        try:
            ctypes.CDLL(str(binary_directory / name))
            result['passed'] = True
        except OSError as error:
            result['error'] = str(error)
        report['checks'].append(result)
    report['passed'] = (report['cpu_runtime_matches_lock'] and report['recipe']['returncode'] == 0
        and all(value['returncode'] == 0 for value in report['tools'].values())
        and all(value['passed'] for value in report['checks']))
    report['wall_seconds'] = time.monotonic() - started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary-directory', type=Path, default=ROOT / 'vcmi/build/bin')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Reserve the report before running checks, so an existing record is never
    # replaced and even an unexpected inventory exception remains visible.
    with args.output.open('x') as stream:
        try:
            report = inspect(args.binary_directory)
        except Exception as error:
            report = {'schema': 'heroes-cloud-runtime-check-v1', 'passed': False,
                      'error': f'{type(error).__name__}: {error}', 'gameplay_performed': False}
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'passed': report['passed'], 'report': str(args.output.resolve())}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
