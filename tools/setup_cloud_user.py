#!/usr/bin/env python3
"""Install hash-bound Debian build packages into a writable local prefix.

Fallback for the dedicated Debian 13 amd64 cloud host without root/sudo.
APT authenticates repository metadata; every downloaded archive also has a
committed SHA-256. Existing host libraries remain dependencies, not copied OS
images. No maintainer scripts run and no system directories are modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'dependencies/cloud-debian13-amd64.json'


def run(*command, **kwargs):
    subprocess.run(command, check=True, **kwargs)


def setup(prefix):
    prefix = Path(prefix).resolve()
    os_info = dict(line.split('=', 1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
    if os_info.get('ID', '').strip('"') != 'debian' or os_info.get('VERSION_ID', '').strip('"') != '13' or platform.machine() != 'x86_64':
        raise ValueError('This local package lock requires Debian 13 amd64')
    apt, sysroot = prefix / 'apt', prefix / 'sysroot'
    for path in (apt / 'lists/partial', apt / 'archives/partial', apt / 'config-empty', sysroot):
        path.mkdir(parents=True, exist_ok=True)
    (apt / 'sources.list').write_text(
        'deb [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] https://deb.debian.org/debian trixie main\n'
        'deb [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] https://deb.debian.org/debian-security trixie-security main\n')
    config = apt / 'apt.conf'
    config.write_text('\n'.join([
        f'Dir::Etc::parts "{apt / "config-empty"}";',
        f'Dir::Etc::main "{apt / "config-empty/no-main"}";',
        f'Dir::Etc::sourcelist "{apt / "sources.list"}";',
        f'Dir::Etc::sourceparts "{apt / "config-empty"}";',
        f'Dir::State::lists "{apt / "lists"}";',
        f'Dir::Cache::archives "{apt / "archives"}";',
        f'APT::Sandbox::User "{__import__("pwd").getpwuid(os.getuid()).pw_name}";',
    ]) + '\n')
    env = {**os.environ, 'APT_CONFIG': str(config)}
    packages = json.loads(LOCK.read_text())['packages']
    missing = [p for p in packages if not (apt / 'archives' / p['filename']).exists()]
    if missing:
        run('/usr/bin/apt-get', 'update', env=env)
        run('/usr/bin/apt-get', 'download', *(p['name'] + '=' + p['version'] for p in missing),
            cwd=apt / 'archives', env=env)
    # Verify everything before extracting anything, including retained downloads.
    for package in packages:
        path = apt / 'archives' / package['filename']
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != package['sha256']:
            raise ValueError('Package checksum differs: ' + package['filename'])
    for package in packages:
        run('dpkg-deb', '-x', str(apt / 'archives' / package['filename']), str(sysroot))
    # Relocated CMake configs refer to sibling runtime libraries. Reuse libraries
    # supplied by this Debian host when APT did not need to download them.
    for base in ('/usr/lib/x86_64-linux-gnu', '/usr/include', '/lib/x86_64-linux-gnu'):
        for directory, _, files in os.walk(base):
            dest = sysroot / Path(directory).relative_to('/')
            dest.mkdir(parents=True, exist_ok=True)
            for name in files:
                target = dest / name
                if not target.exists() and not target.is_symlink():
                    target.symlink_to(Path(directory) / name)
    q = shlex.quote
    activation = '\n'.join([
        '# Source this file before building, testing or invoking native tools.',
        f'export PATH={q(str(ROOT / ".venv/bin"))}:{q(str(sysroot / "usr/bin"))}:$PATH',
        f'export LD_LIBRARY_PATH={q(str(sysroot / "usr/lib/x86_64-linux-gnu"))}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}',
        f'export CMAKE_PREFIX_PATH={q(str(sysroot / "usr"))}',
        f'export CPATH={q(str(sysroot / "usr/include"))}:{q(str(sysroot / "usr/include/x86_64-linux-gnu"))}',
        f'export LIBRARY_PATH={q(str(sysroot / "usr/lib/x86_64-linux-gnu"))}',
        f'export PKG_CONFIG_PATH={q(str(sysroot / "usr/lib/x86_64-linux-gnu/pkgconfig"))}:{q(str(sysroot / "usr/share/pkgconfig"))}',
        f'export CCACHE_DIR={q(str(prefix / "ccache"))}',
        'export OMP_NUM_THREADS=1', 'export MKL_NUM_THREADS=1', 'export HEROES_CLOUD_BUILD_JOBS=2',
    ]) + '\n'
    (prefix / 'activate.sh').write_text(activation)
    with (prefix / 'base-system-packages.tsv').open('w') as stream:
        run('dpkg-query', '-W', '-f=${binary:Package}\t${Version}\n', stdout=stream)
    print(f'Verified {len(packages)} Debian archives; activation: {prefix / "activate.sh"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, default=Path('/workspace/cloud-setup'))
    setup(parser.parse_args().prefix)
