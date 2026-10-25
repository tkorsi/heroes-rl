#!/usr/bin/env python3
"""Hash user-supplied private inputs without copying assets or claiming game readiness."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.port_fixed_policy_linux import ORIGINAL_BYTES, ORIGINAL_SHA256


def file_binding(path):
    if not Path(path).is_file():
        raise ValueError('Expected a regular input file')
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Input changed while hashing')
    return {'bytes': after.st_size, 'sha256': digest}


def inventory_data(data):
    """The selected Data root may be a mount; its descendants must be regular files."""
    data = (Path(data) / 'Data').resolve(strict=True)
    if not data.is_dir():
        raise ValueError('Expected the parent of the original Data directory')
    records = []
    def fail(error):
        raise error
    for directory, dirs, files in os.walk(data, followlinks=False, onerror=fail):
        for name in dirs + files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError('Private input contains a symlink or nonregular entry')
        for name in files:
            path = Path(directory) / name
            records.append({'path': path.relative_to(data).as_posix(), **file_binding(path)})
    if not records:
        raise ValueError('Data directory is empty')
    return sorted(records, key=lambda record: record['path'])


def inventory(data, checkpoint, output, provenance='User-supplied original game data and fixed checkpoint'):
    output = Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('Store the private manifest outside the Git checkout')
    if output.exists():
        raise FileExistsError('Never overwrite a private input manifest')
    checkpoint_binding = file_binding(checkpoint)
    if checkpoint_binding != {'bytes': ORIGINAL_BYTES, 'sha256': ORIGINAL_SHA256}:
        raise ValueError('Original checkpoint size or SHA-256 differs from the cloud handoff')
    records = inventory_data(data)
    manifest = {'schema': 'heroes-cloud-private-inputs-v1', 'provenance': provenance,
                'data_files': records, 'checkpoint': {'filename': 'fixed-policy.pt', **checkpoint_binding},
                'game_data_validation': 'byte_inventory_only', 'gameplay_performed': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return {'manifest': file_binding(output), 'data_files': len(records),
            'data_bytes': sum(record['bytes'] for record in records),
            'checkpoint_identity_verified': True, 'game_data_validation': 'byte_inventory_only'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Parent of user-supplied Data/')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New private manifest outside the checkout')
    parser.add_argument('--provenance', default='User-supplied original game data and fixed checkpoint')
    args = parser.parse_args()
    print(json.dumps(inventory(args.data, args.checkpoint, args.output, args.provenance), indent=2))
