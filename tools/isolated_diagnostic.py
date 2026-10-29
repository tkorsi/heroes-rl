"""Explicit leases for private native diagnostics alongside an unchanged run.

The default still excludes official training. Isolated mode is only for
bounded diagnostic supervisors whose outputs cannot be official training state.
It serializes native diagnostics without opening any official ownership lock.
The caller must retain its existing deadline guardians and owned cleanup.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
import subprocess

ARTIFACTS = Path.home() / 'Projects/Heroes-artifacts'
DEFAULT_OFFICIAL_LOCK = ARTIFACTS / 'local-script-training/owner.lock'


def private_output(output, *, artifacts=ARTIFACTS):
    output = Path(output).expanduser().absolute()
    root = Path(artifacts).absolute() / 'run-investigations'
    if '..' in output.parts or not output.is_relative_to(root) or output == root:
        raise ValueError('Isolated diagnostics require a fresh private investigation output')
    for item in (output, *output.parents):
        if item.is_symlink():
            raise ValueError('Diagnostic output or parent is a symlink')
    if output.exists():
        raise ValueError('Preserve previous diagnostic attempts; output already exists')
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    probe = subprocess.run(['git', '-C', str(ancestor), 'rev-parse', '--show-toplevel'],
                           text=True, capture_output=True, timeout=10)
    if probe.returncode == 0:
        raise ValueError('Private diagnostic output cannot be inside a Git checkout')
    return output


@contextmanager
def diagnostic_lease(output, *, isolated=False, official_lock=DEFAULT_OFFICIAL_LOCK,
                     artifacts=ARTIFACTS):
    if type(isolated) is not bool:
        raise ValueError('Diagnostic isolation choice must be boolean')
    if not isolated:
        with Path(official_lock).expanduser().resolve().open('r') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield {'mode': 'official-training-excluded', 'path': str(Path(official_lock).resolve())}
        return
    if Path(official_lock).expanduser().absolute() != DEFAULT_OFFICIAL_LOCK:
        raise ValueError('Isolated diagnostics cannot override an official lock')
    private_output(output, artifacts=artifacts)
    root = Path(artifacts).absolute() / 'run-investigations'
    root.mkdir(parents=True, exist_ok=True)
    lease = root / '.isolated-native-diagnostics.lock'
    descriptor = os.open(lease, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'r+') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError('Diagnostic lease must be a private regular file')
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield {'mode': 'isolated-diagnostics-only', 'path': str(lease),
               'official_lock_opened': False, 'official_state_writes': False}
