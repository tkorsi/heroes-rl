"""Private Git publication with atomic refs, fresh readback and bounded commands.

Temporary filtered clones are discarded after readback so raw Git objects do not
accumulate on the training disk. No game Data symlink is followed into archives.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time

REMOTE = 'https://github.com/tkorsi/heroes-rl-training-state.git'
TRANSFER_SECONDS = 300
PREFETCH_BYTES = 32 * 1024**2
PREFETCH_OBJECTS = 128


def require(condition, message):
    if not condition: raise ValueError(message)


def binding(path):
    path = Path(path).resolve()
    with path.open('rb') as stream: sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=sha, bytes=path.stat().st_size)


def command(arguments, deadline, cwd=None, *, input_data=None, output=None, transfer=False, local_only=False):
    """Bound the owned Git process group, including any transport children.

    Explicit transfers have a separate finite allowance; the caller's original
    deadline always wins. Local verification forbids every transport protocol,
    so an unexpectedly absent blob cannot silently become another network fetch.
    """
    left = deadline-time.time()-10
    require(left > 0, 'Private publication deadline exhausted')
    environment = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_LITERAL_PATHSPECS': '1',
                   'GIT_NO_REPLACE_OBJECTS': '1'}
    if local_only:
        environment.update(GIT_NO_LAZY_FETCH='1', GIT_ALLOW_PROTOCOL='')
    process = subprocess.Popen(arguments, cwd=cwd, env=environment, start_new_session=True,
                               stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
                               stdout=output if output is not None else subprocess.PIPE,
                               stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate(input_data, timeout=min(TRANSFER_SECONDS if transfer else 120, left))
    except BaseException:
        # This process group was created solely for this command. Git transports
        # must not outlive a timed-out verification or interrupted publication.
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.communicate(timeout=2)
        raise
    require(process.returncode == 0, 'Private Git operation failed: '+stderr.decode(errors='replace')[-1800:])
    return stdout


def git(arguments, deadline, cwd=None):
    return command(['git', *arguments], deadline, cwd).decode().strip()


def ref_sha(ref, deadline):
    result = git(['ls-remote', REMOTE, ref], deadline)
    require(not result or len(result.splitlines()) == 1, 'Ambiguous private ref')
    return result.split()[0] if result else None


def clone(ref, target, deadline):
    git(['clone', '--filter=blob:none', '--depth', '1', '--no-checkout', '--single-branch',
         '--branch', ref.removeprefix('refs/heads/'), REMOTE, str(target)], deadline)


def publication_path(prefix):
    prefix = Path(prefix)
    require(not prefix.is_absolute() and prefix.parts and '..' not in prefix.parts,
            'Unsafe publication path')
    return prefix


def source_entries(files):
    entries = {}
    for name, source in files.items():
        require(name not in ('', '.', '..', 'manifest.json') and Path(name).name == name
                and not Path(source).is_symlink(), 'Unsafe checkpoint publication source')
        entries[name] = {k: value for k, value in binding(source).items() if k in ('bytes', 'sha256')}
    return entries


def prefetch_blobs(repo, blobs, deadline):
    """Fetch only these exact OIDs, in finite groups, before any local readback."""
    groups, group, total, seen = [], [], 0, set()
    for oid, size in blobs:
        require(re.fullmatch('[0-9a-f]{40}', oid) and type(size) is int and size >= 0,
                'Malformed private blob request')
        if oid in seen: continue
        if group and (total + size > PREFETCH_BYTES or len(group) >= PREFETCH_OBJECTS):
            groups.append(group); group, total = [], 0
        group.append(oid); total += size; seen.add(oid)
    if group: groups.append(group)
    for group in groups:
        command(['git', '-c', 'fetch.negotiationAlgorithm=noop', 'fetch', '--no-tags',
                 '--no-write-fetch-head', '--no-recurse-submodules', '--filter=blob:none',
                 '--stdin', 'origin'], deadline, repo,
                input_data=('\n'.join(group)+'\n').encode(), transfer=True)
    return len(groups)


def readback_transaction(*, ref, commit, prefix, manifest_sha256, files, deadline,
                         temporary_root, destination=None, migration=None):
    """Verify an exact existing transaction without a new commit or state advance.

    ``files`` binds the caller's preserved originals to the committed manifest.
    A supplied destination must be new and receives all verified entries only
    after full checkpoint restoration succeeds. This operation admits no update.
    """
    require(ref.startswith('refs/heads/codex/') and '..' not in ref, 'Explicit private Codex branch required')
    require(re.fullmatch('[0-9a-f]{40}', commit) and re.fullmatch('[0-9a-f]{64}', manifest_sha256),
            'Exact publication commit and manifest SHA256 are required')
    prefix = publication_path(prefix)
    originals = source_entries(files)
    destination = None if destination is None else Path(destination).absolute()
    require(destination is None or (not destination.exists() and not destination.is_symlink()),
            'Readback destination already exists; no overwrite')
    Path(temporary_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='readback-', dir=temporary_root) as temporary:
        workspace = Path(temporary); fresh = workspace/'repo'
        clone(ref, fresh, deadline)
        require(git(['rev-parse', 'HEAD'], deadline, fresh) == commit, 'Remote publication identity differs')
        listing = command(['git', 'ls-tree', '-r', '-z', commit, '--', str(prefix)], deadline,
                          fresh, local_only=True)
        blobs = {}
        for entry in listing.split(b'\0'):
            if not entry: continue
            meta, name = entry.split(b'\t', 1)
            mode, kind, oid = meta.decode().split()
            path = Path(name.decode())
            require(kind == 'blob' and mode in ('100644', '100755') and path.parent == prefix,
                    'Unexpected non-file or nested publication entry')
            require(path.name not in blobs, 'Duplicate publication entry')
            blobs[path.name] = oid
        require('manifest.json' in blobs, 'Remote publication manifest is missing')
        batches = prefetch_blobs(fresh, [(blobs['manifest.json'], 0)], deadline)
        raw_manifest = command(['git', 'cat-file', 'blob', blobs['manifest.json']], deadline, fresh, local_only=True)
        require(hashlib.sha256(raw_manifest).hexdigest() == manifest_sha256, 'Remote manifest differs')
        manifest = json.loads(raw_manifest)
        require(manifest.get('schema') == 'heroes-linux-private-publication-v1'
                and manifest.get('game_data_symlinks_followed') is False
                and type(manifest.get('entries')) is dict and type(manifest.get('raw_inventory')) is dict,
                'Malformed private publication manifest')
        entries = manifest['entries']
        require(set(blobs) == set(entries) | {'manifest.json'}, 'Remote publication entry set differs')
        for name, expected in entries.items():
            require(name not in ('', '.', '..', 'manifest.json') and Path(name).name == name
                    and type(expected) is dict and set(expected) == {'bytes', 'sha256'}
                    and type(expected['bytes']) is int and expected['bytes'] >= 0
                    and isinstance(expected['sha256'], str) and re.fullmatch('[0-9a-f]{64}', expected['sha256']),
                    'Malformed remote artifact binding')
        require(all(entries.get(name) == expected for name, expected in originals.items()),
                'Existing publication differs from preserved source bindings')
        batches += prefetch_blobs(fresh, [(blobs[name], expected['bytes']) for name, expected in entries.items()], deadline)
        restored = workspace/'restored'; restored.mkdir()
        (restored/'manifest.json').write_bytes(raw_manifest)
        for name, expected in entries.items():
            with (restored/name).open('xb') as stream:
                command(['git', 'cat-file', 'blob', blobs[name]], deadline, fresh, output=stream, local_only=True)
            actual = binding(restored/name)
            require(all(actual[k] == expected[k] for k in ('bytes', 'sha256')), 'Remote artifact bytes differ: '+name)
        pair = {'plan.json', 'report.json', 'after-policy.pt', 'after-resume.pt'}
        verified_pair = pair <= entries.keys()
        if verified_pair:
            from experiments.learner.builtin_preservation import verify_pair
            verify_pair(restored/'after-resume.pt', restored/'after-policy.pt',
                        json.loads((restored/'plan.json').read_text()), json.loads((restored/'report.json').read_text()),
                        **({'migration': migration} if migration is not None else {}))
        require(originals == source_entries(files), 'Preserved publication sources changed during readback')
        require(time.time() < deadline, 'Private publication deadline exhausted during restoration')
        if destination is not None:
            # Exclusively claim a new directory only after verification. An
            # interrupted install is retained and refused on retry; it is never
            # returned as verified. Install the completion manifest last.
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.readback-', dir=destination.parent) as staged:
                staged = Path(staged)
                for name in ('manifest.json', *entries):
                    shutil.copyfile(restored/name, staged/name)
                    require(binding(restored/name)['sha256'] == binding(staged/name)['sha256'], 'Readback staging differs')
                    with (staged/name).open('rb') as stream: os.fsync(stream.fileno())
                require(time.time() < deadline, 'Private publication deadline exhausted during staging')
                destination.mkdir()
                for name in (*entries, 'manifest.json'):
                    os.link(staged/name, destination/name)
                for path in (destination, destination.parent):
                    descriptor = os.open(path, os.O_RDONLY)
                    try: os.fsync(descriptor)
                    finally: os.close(descriptor)
        return dict(commit=commit, ref=ref, prefix=str(prefix), manifest_sha256=manifest_sha256,
                    fresh_readback_verified=True, checkpoint_roundtrip_verified=verified_pair,
                    raw_inventory=manifest['raw_inventory'], explicit_blob_fetch_batches=batches,
                    lazy_fetch_during_verification=False)


def publish_transaction(*, base_ref, destination_ref, prefix, files, deadline, temporary_root, raw_root=None, base_sha=None):
    """Commit once with a compare-and-swap ref, then read every new blob remotely."""
    require(destination_ref.startswith('refs/heads/codex/') and '..' not in destination_ref,
            'Explicit private Codex branch required')
    prefix = publication_path(prefix)
    originals = source_entries(files)
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
        require(originals == source_entries(files), 'Publication sources changed before commit')
        git(['add', '--sparse', '--', str(prefix)], deadline, repo)
        git(['commit', '-m', 'Preserve verified Linux continuation evidence '+prefix.name], deadline, repo)
        commit = git(['rev-parse', 'HEAD'], deadline, repo)
        git(['push', '--force-with-lease='+destination_ref+':'+(previous or ''),
             'origin', 'HEAD:'+destination_ref], deadline, repo)
        require(originals == source_entries(files), 'Publication sources changed before remote verification')
        return readback_transaction(ref=destination_ref, commit=commit, prefix=prefix,
            manifest_sha256=manifest_hash, files=files, deadline=deadline, temporary_root=temporary_root)


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
