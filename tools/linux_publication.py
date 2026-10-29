"""Private Git publication with atomic refs, fresh readback and bounded commands.

Temporary filtered clones are discarded after readback so raw Git objects do not
accumulate on the training disk. No game Data symlink is followed into archives.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time

REMOTE = 'https://github.com/tkorsi/heroes-rl-training-state.git'


def require(condition, message):
    if not condition: raise ValueError(message)


def binding(path):
    path = Path(path).resolve()
    with path.open('rb') as stream: sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=sha, bytes=path.stat().st_size)


def command(arguments, deadline, cwd=None):
    left = deadline-time.time()-10
    require(left > 0, 'Private publication deadline exhausted')
    result = subprocess.run(arguments, cwd=cwd, capture_output=True, timeout=min(120, left), check=False)
    require(result.returncode == 0, 'Private Git operation failed: '+result.stderr.decode(errors='replace')[-1800:])
    return result.stdout


def git(arguments, deadline, cwd=None):
    return command(['git', *arguments], deadline, cwd).decode().strip()


def ref_sha(ref, deadline):
    result = git(['ls-remote', REMOTE, ref], deadline)
    require(not result or len(result.splitlines()) == 1, 'Ambiguous private ref')
    return result.split()[0] if result else None


def clone(ref, target, deadline):
    git(['clone', '--filter=blob:none', '--depth', '1', '--no-checkout', '--single-branch',
         '--branch', ref.removeprefix('refs/heads/'), REMOTE, str(target)], deadline)


def publish_transaction(*, base_ref, destination_ref, prefix, files, deadline, temporary_root, raw_root=None, base_sha=None):
    """Commit once with a compare-and-swap ref, then read every new blob remotely."""
    require(destination_ref.startswith('refs/heads/codex/') and '..' not in destination_ref,
            'Explicit private Codex branch required')
    prefix = Path(prefix)
    require(not prefix.is_absolute() and '..' not in prefix.parts, 'Unsafe publication path')
    previous = ref_sha(destination_ref, deadline)
    Path(temporary_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='publish-', dir=temporary_root) as temporary:
        workspace = Path(temporary); repo = workspace/'repo'
        clone(destination_ref if previous else base_ref, repo, deadline)
        git(['sparse-checkout', 'init', '--cone'], deadline, repo)
        git(['sparse-checkout', 'set', str(prefix)], deadline, repo)
        git(['checkout', '--detach'], deadline, repo)
        base = git(['rev-parse', 'HEAD'], deadline, repo)
        require(previous is None or previous == base, 'Publication ref changed during fetch')
        require(previous is not None or base_sha is None or base == base_sha, 'Publication base commit changed')
        output = repo/prefix
        require(not output.exists(), 'Publication label already exists; no overwrite or duplicate update')
        output.mkdir(parents=True)
        entries = {}
        for name, source in files.items():
            require(Path(name).name == name and not Path(source).is_symlink(), 'Unsafe checkpoint publication source')
            shutil.copyfile(source, output/name)
            original, copied = binding(source), binding(output/name)
            require(all(original[k] == copied[k] for k in ('bytes', 'sha256')), 'Private copy differs')
            entries[name] = {k:copied[k] for k in ('bytes', 'sha256')}
        raw_inventory = {}
        if raw_root is not None:
            raw_root = Path(raw_root).resolve()
            for p in sorted(raw_root.rglob('*')):
                if p.is_file() and not p.is_symlink():
                    raw_inventory[str(p.relative_to(raw_root))] = binding(p)
            total = sum(x['bytes'] for x in raw_inventory.values())
            require(shutil.disk_usage(temporary_root).free > total*1.05+2*1024**3,
                    'Insufficient private archive headroom; verified candidate remains pending')
            archive = workspace/'raw.tar.gz'
            with tarfile.open(archive, 'w:gz', dereference=False) as stream:
                stream.add(raw_root, arcname='batch')
            for relative, expected in raw_inventory.items():
                require(binding(raw_root/relative) == expected, 'Raw evidence changed while archiving')
            with archive.open('rb') as stream:
                index = 0
                while block := stream.read(16*1024**2):
                    name = f'raw.part{index:05d}'; index += 1
                    (output/name).write_bytes(block)
                    entries[name] = dict(bytes=len(block), sha256=hashlib.sha256(block).hexdigest())
        manifest = dict(schema='heroes-linux-private-publication-v1', entries=entries,
                        raw_inventory=raw_inventory, game_data_symlinks_followed=False)
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        manifest_hash = binding(output/'manifest.json')['sha256']
        git(['add', '--sparse', '--', str(prefix)], deadline, repo)
        git(['commit', '-m', 'Preserve verified Linux continuation evidence '+prefix.name], deadline, repo)
        commit = git(['rev-parse', 'HEAD'], deadline, repo)
        git(['push', '--force-with-lease='+destination_ref+':'+(previous or ''),
             'origin', 'HEAD:'+destination_ref], deadline, repo)
        fresh = workspace/'readback'
        clone(destination_ref, fresh, deadline)
        require(git(['rev-parse', 'HEAD'], deadline, fresh) == commit, 'Remote publication identity differs')
        read_manifest = command(['git', 'show', 'HEAD:'+str(prefix/'manifest.json')], deadline, fresh)
        require(hashlib.sha256(read_manifest).hexdigest() == manifest_hash, 'Remote manifest differs')
        restored = workspace/'restored'; restored.mkdir()
        for name, expected in entries.items():
            data = command(['git', 'show', 'HEAD:'+str(prefix/name)], deadline, fresh)
            require(len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'],
                    'Remote artifact bytes differ: '+name)
            if name in files: (restored/name).write_bytes(data)
        if {'plan.json', 'report.json', 'after-policy.pt', 'after-resume.pt'} <= files.keys():
            from experiments.learner.builtin_preservation import verify_pair
            verify_pair(restored/'after-resume.pt', restored/'after-policy.pt',
                        json.loads((restored/'plan.json').read_text()), json.loads((restored/'report.json').read_text()))
        return dict(commit=commit, ref=destination_ref, prefix=str(prefix), manifest_sha256=manifest_hash,
                    fresh_readback_verified=True, checkpoint_roundtrip_verified='after-resume.pt' in files,
                    raw_inventory=raw_inventory)


def check_claim(session):
    claim = session['linux_cloud']['claim']
    require(ref_sha(claim['ref'], session['absolute_deadline_unix']) == claim['commit'],
            'Exclusive cloud handoff claim is absent or changed')


def publish_checkpoint(session, parent, progress, backup, plan_path):
    """No latest pointer advances until the complete candidate is read back."""
    from tools.run_local_training import atomic
    check_claim(session)
    root = Path(session['run_root'])
    require(not (root/'pending-checkpoint.json').exists(), 'An uncommitted verified checkpoint is already pending')
    candidate = dict(parent=parent, progress=progress, backup=str(backup), run_dir=session['run_dir'])
    atomic(root/'pending-checkpoint.json', candidate)
    files = {'plan.json': plan_path, 'report.json': parent['report']['path'],
             'after-policy.pt': parent['policy']['path'], 'after-resume.pt': parent['resume']['path'],
             'initial-checkpoint.pt': parent['initial_checkpoint']['path'],
             'preservation-receipt.json': backup/'preservation-receipt.json',
             'sequence-ledger.json': root/'sequence-ledger.json', 'candidate.json': root/'pending-checkpoint.json'}
    result = publish_transaction(base_ref=session['linux_cloud']['claim']['ref'],
        destination_ref=session['linux_cloud']['results_ref'],
        prefix='cloud/'+session['linux_cloud']['handoff_id']+'/batches/'+Path(plan_path).parent.name,
        files=files, deadline=session['compute_deadline_unix'],
        temporary_root=Path(session['artifact_root'])/'publication-tmp', raw_root=Path(plan_path).parent)
    require(result['fresh_readback_verified'] is True and result['checkpoint_roundtrip_verified'] is True,
            'Private checkpoint readback and full restoration were not verified')
    atomic(Path(plan_path).parent/'private-publication.json', result)
    # Only large raw files whose exact bytes were archived/read back are evicted.
    # Every checkpoint and deadline receipt remains local and immutable.
    for relative, expected in result['raw_inventory'].items():
        p = Path(plan_path).parent/relative
        if (expected['bytes'] > 64*1024 and p.suffix != '.pt'
                and p.name not in ('plan.json', 'report.json') and 'deadline' not in p.name):
            require(binding(p) == expected, 'Raw evidence changed before private-cache eviction')
            p.unlink()
    return {k:v for k,v in result.items() if k != 'raw_inventory'}
