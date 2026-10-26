#!/usr/bin/env python3
"""Compress the raw episode folders of finished benchmark campaigns to fit the host's disk allowance.

    compress_episodes.py RUN_ROOT [RUN_ROOT ...]

Only for benchmark campaigns that have ended (no live owner lock) and have been summarized. Each
``batches/*/execution/episode-*`` folder becomes ``episode-*.tar.gz`` beside it (private, local);
its manifest lists every member's size and SHA-256, the archive is re-read and every member checked
against the manifest before the folder is removed. Plans, reports, logs, research logs and
checkpoints are never touched, so the runner's own records and every admitted state remain.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ended(run_root):
    lock = Path(run_root) / 'owner.lock'
    if not lock.exists():
        return True
    with lock.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream, fcntl.LOCK_UN)
    return True


def compress(episode):
    files = sorted(p for p in episode.rglob('*') if p.is_file() and not p.is_symlink())
    manifest = {p.relative_to(episode).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in files}
    archive = episode.with_name(episode.name + '.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
        tar.add(episode, arcname=episode.name)
    seen = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if member.isfile():
                data = tar.extractfile(member).read()
                seen[member.name.split('/', 1)[1]] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    if seen != manifest:
        archive.unlink()
        raise RuntimeError(f'archive of {episode} differs from its folder; folder kept')
    record = dict(episode=episode.name, archive=archive.name, archive_sha256=sha(archive),
                  archive_bytes=archive.stat().st_size, files=manifest)
    episode.with_name(episode.name + '.manifest.json').write_text(json.dumps(record, indent=1) + '\n')
    shutil.rmtree(episode)
    return sum(item['bytes'] for item in manifest.values()), record['archive_bytes']


def main():
    total = packed = 0
    for root in map(Path, sys.argv[1:]):
        if not ended(root):
            print('campaign still owns its lock, skipped:', root); continue
        for episode in sorted(root.glob('run-*/batches/*/execution/episode-*')):
            if episode.is_dir():
                raw, archive = compress(episode)
                total += raw; packed += archive
    print(json.dumps(dict(raw_bytes=total, archive_bytes=packed)))


if __name__ == '__main__':
    main()
