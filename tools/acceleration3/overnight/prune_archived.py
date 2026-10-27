"""Remove raw episode directories only for batches whose evidence archive is preserved and pushed.

Keeps every plan, report, journal, log and checkpoint; writes a manifest binding each removed
directory to the archive (sha256) and remote commit that hold its bytes."""
import hashlib, json, subprocess, shutil, sys
from pathlib import Path
bench = Path('/home/claude/bench2'); repo = bench/'durable/repo'
skip = set(sys.argv[1:])
refs = dict(reversed(l.split('\t')) for l in subprocess.check_output(['git', '-C', str(repo), 'ls-remote', 'origin'], text=True).splitlines())
manifest = []
for campaign in sorted((repo/'results').iterdir()):
    if campaign.name in skip: continue
    for batch in sorted((bench/'campaigns'/campaign.name/'batches').glob('*')):
        archive_dir = campaign/f'{campaign.name}-{batch.name}'
        receipt = archive_dir/'receipt.json'
        episodes = sorted(batch.glob('execution/episode-*'))
        if not episodes: continue
        if not receipt.exists(): print('not archived, kept:', batch); continue
        r = json.loads(receipt.read_text()); tar = archive_dir/'evidence.tar.gz'
        sha = hashlib.sha256(tar.read_bytes()).hexdigest()
        if sha != r['archive']['sha256']: print('archive mismatch, kept:', batch); continue
        rel = str(tar.relative_to(repo))
        commit = subprocess.check_output(['git', '-C', str(repo), 'log', '-1', '--format=%H', '--', rel], text=True).strip()
        head = refs.get(f'refs/heads/bench/{campaign.name}')
        reachable = head is not None and subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', commit, head]).returncode == 0
        blob = hashlib.sha256(subprocess.check_output(['git', '-C', str(repo), 'show', f'{commit}:{rel}'])).hexdigest()
        if not reachable or blob != sha: print('not pushed, kept:', batch); continue
        size = sum(f.stat().st_size for e in episodes for f in e.rglob('*') if f.is_file())
        for e in episodes: shutil.rmtree(e)
        manifest.append(dict(batch=str(batch.relative_to(bench)), removed=[e.name for e in episodes], bytes=size,
                             archive=rel, archive_sha256=sha, commit=commit, verified_files=r['verified_files']))
out = Path('/home/claude/overnight/pruned-manifest.json')
old = json.loads(out.read_text()) if out.exists() else []
out.write_text(json.dumps(old + manifest, indent=1) + '\n')
print(len(manifest), 'batches pruned,', sum(m['bytes'] for m in manifest) / 1e9, 'GB')
