"""Validate the immutable final local handoff and materialize explicit relocation.

No gameplay or optimizer is available in this module. Original JSON and tensors
remain unchanged. Only ordinary archive files are copied; links are inventoried.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time

SCHEMA='heroes-local-to-cloud-ready-v1'
PREFIX='handoffs/2026-10-26-local-to-cloud/'
OLD=Path('/Users/orsi/Projects/Heroes-artifacts')
SOURCE='cb3d324cb09919383dac7e97a97bc0696dcd588d'
LOCAL_START=datetime.fromisoformat('2026-10-26T01:10:53.815159+00:00').timestamp()
HARD=datetime(2026,10,26,10,tzinfo=timezone.utc).timestamp()
COMPUTE=HARD-300
DATA_COMMIT='c7257a047373a55901e04bd9fa024dbb6df964d3'
DATA_MANIFEST_SHA256='66c1291c5ff79c1967d64f304c52527c155ce5daadf586566424d195cc105130'


def require(value,message):
    if not value:raise ValueError(message)


def digest(data):return hashlib.sha256(data).hexdigest()


def binding(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=digest(path.read_bytes()),bytes=path.stat().st_size)


def git(repository,*args):
    return subprocess.check_output(['git','-C',str(repository),*args],timeout=min(90,max(1,HARD-time.time())))


def stamp(value):
    require(isinstance(value,str),'Missing UTC timestamp')
    v=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(v.tzinfo is not None,'Unzoned timestamp')
    return v.timestamp()


def validate_marker(m,now):
    require(m.get('schema')==SCHEMA,'Not a final local handoff')
    require(m.get('ownership_transfer')=='local -> cloud' and m.get('local_training_finished') is True
        and m.get('local_auto_resume') is False and m.get('no_local_training_after_readiness') is True,
        'Exclusive final ownership transfer required')
    require(m['repository']=='tkorsi/heroes-rl-training-state' and m['branch']=='codex/local365-teacher-results'
        and m['public_repository']=='tkorsi/heroes-rl' and m['public_branch']=='codex/teacher-curriculum-v1',
        'Wrong repository or local campaign branch')
    require(m['executable_source_commit']==SOURCE and m['source']['git_commit']==SOURCE,'Unexpected local executable source')
    clock=m['clock']
    require(abs(clock['started_unix']-LOCAL_START)<.001 and clock['compute_deadline_unix']==clock['started_unix']+6900
        and clock['absolute_deadline_unix']==clock['started_unix']+7200,'Wrong original local window')
    ready=stamp(m['ready_utc']);stop=stamp(m['local_stop']);final=stamp(m['local_finalization'])
    require(LOCAL_START<=stop<=final<=ready<=now+5 and ready>=LOCAL_START+6600,'Stale/premature local readiness')
    require(stamp(m['cloud_hard_deadline_utc'])==HARD and now<COMPUTE-600,'No safe cloud window or altered deadline')
    require(m['final_update']==m['progress']['updates']>146 and m['local_ppo_updates']>0,'Old checkpoint is not a final handoff')
    require(m['progress']['updates']==146+m['local_ppo_updates']
        and m['progress']['rollouts']==584+4*m['local_ppo_updates']
        and m['admitted_local_games']==4*m['local_ppo_updates'],'Local update/rollout accounting mismatch')
    cfg=m['configuration']
    require(cfg['interface']=='adventure-capabilities-v1' and cfg['max_days']==365
        and cfg['ordinary_choice_limit']==4960 and cfg['reserved_dialog_choices']==160
        and cfg['max_total_choices']==5120 and cfg['allow_verified_training_prefix'] is False
        and cfg['max_optimizer_steps']==640 and cfg['ppo']['epochs']==1 and cfg['torch_threads']==1,
        'Wrong curriculum/learner configuration')
    require(set(m['maps_by_seat'])=={'0','1'},'Both seat maps required')
    return m


def safe_files(raw):
    files={};links=[]
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive.getmembers():
            name=PurePosixPath(member.name)
            require(not name.is_absolute() and '..' not in name.parts,'Unsafe archive member')
            if member.isdir():continue
            if member.issym() or member.islnk():
                links.append(dict(path=str(name),target=member.linkname));continue
            require(member.isfile() and str(name) not in files,'Unsupported or duplicate archive member')
            files[str(name)]=archive.extractfile(member).read()
    return files,links


def put(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.is_file() and not path.is_symlink() and path.read_bytes()==data,'Existing relocation bytes differ')
    else:
        with path.open('xb') as f:f.write(data)


def materialize(repository,ready_commit,root,public_repository):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    raw=git(repository,'show',ready_commit+':'+PREFIX+'ready.json')
    m=validate_marker(json.loads(raw),time.time())
    require(git(repository,'merge-base','--is-ancestor',m['archive_commit'],ready_commit)==b'','Archive is not an ancestor of ready')
    archive=m['archive'];require(archive['path']==PREFIX+'bundle.tar.gz','Unexpected archive path')
    data=git(repository,'show',m['archive_commit']+':'+archive['path'])
    require(digest(data)==archive['sha256'] and len(data)==archive['bytes'],'Final private archive hash/size differs')
    files,links=safe_files(data);require(not links,'Final handoff bundle must contain ordinary files only')
    inventory=json.loads(git(repository,'show',m['archive_commit']+':'+PREFIX+'inventory.json'))
    require(inventory==m['files'],'Ready inventory differs from immutable archive inventory')
    relocation=root/'historical'
    def relocate(original):
        p=Path(original);require(p.is_absolute() and '..' not in p.parts and p.is_relative_to(OLD),'Unregistered original artifact root')
        target=relocation/p.relative_to(OLD)
        require(target.resolve().is_relative_to(relocation.resolve()),'Relocation escapes via an existing link')
        return target
    for name,item in inventory.items():
        require(item['committed_path']==PREFIX+'bundle/'+name,'Unexpected committed dependency path')
        original=item['original'];payload=files['bundle/'+name]
        require(digest(payload)==item['sha256']==original['sha256'] and len(payload)==item['bytes']==original['bytes'],'Bundle dependency mismatch')
        require(git(repository,'show',m['archive_commit']+':'+item['committed_path'])==payload,'Archive and private committed dependency differ')
        put(relocate(original['path']),payload)
    for name,payload in files.items():put(root/'handoff'/name,payload)
    put(root/'handoff/ready.json',raw)
    shutdown=json.loads(files['bundle/shutdown-proof.json']);restores=json.loads(files['bundle/restore-receipt.json'])
    for key,name in [('shutdown_proof','shutdown-proof.json'),('restore_evidence','restore-receipt.json')]:
        require(m[key]['commit']==m['archive_commit'] and m[key]['path']==PREFIX+'bundle/'+name
            and m[key]['sha256']==digest(files['bundle/'+name]),'Final proof binding differs')
    require(shutdown['local_auto_resume'] is False and shutdown['controllers']
        and all(v['owned_command_absent'] is True for v in shutdown['controllers']),'Local controller remains owned/active')
    for proof in restores.values():require(proof['exact_model_adam_rng_progress_pool'] is True and proof['progress']==m['progress'],'Local exact restoration proof differs')
    plan=json.loads(files['bundle/plan.json']);report=json.loads(files['bundle/report.json'])
    session=json.loads(files['bundle/session.json']);controller=json.loads(files['bundle/controller-result.json'])
    journal=json.loads(files['bundle/campaign-report.json']);window=json.loads(files['bundle/window.json'])
    require(plan['source']==m['source'] and plan['runtime']==m['runtime'] and plan['configuration']==m['configuration'],'Final source/runtime/config differ')
    require(report['plan_sha256']==digest(files['bundle/plan.json']) and report['after_checkpoint']==m['checkpoint'],'Final checkpoint report differs')
    require(plan['checkpoint_metadata']['initial_checkpoint']==m['initial_checkpoint_dependency']==inventory['dependencies/initial-policy.pt']['original'],'Initial-policy metadata dependency differs')
    require(all(session[k]==window[k]==m['clock'][k] for k in m['clock']),'Local clock/session/window differ')
    require(inventory['session.json']['original']==m['local_session'] and controller['owned_cleanup_verified'] is True
        and controller['stopped_utc']==m['local_stop'] and controller['finalized_utc']==m['local_finalization'],'Final local shutdown/session mismatch')
    require(journal['status'] in ('finished_budget_reserve','stopped_failure') and journal['latest_progress']==m['progress']
        and journal['latest_verified_parent']==inventory['plan.json']['original']
        and journal['new_updates']==m['local_ppo_updates'],'Campaign final state mismatch')
    require(controller['latest_restore']['policy']['sha256']==m['checkpoint']['policy']['sha256'],'Final controller restored a different policy')
    from experiments.learner.curriculum_training import schedule
    seq=1+max(b['sequence'] for b in journal['batches'] if b['mode']=='training')
    require(m['next_unused_training_sequence']==seq and m['next_training_schedule']==schedule(seq,seq%2),'Next unused training seeds differ')
    require(m['heldout_schedule_by_seat']=={str(seat):schedule(m['final_update'],seat,True) for seat in (0,1)},'Heldout schedule changed')
    for seat in (0,1):
        name=f'maps/curriculum-3-{seat}.vmap'
        require(m['maps_by_seat'][str(seat)]==inventory[name],'Seat map declaration mismatch')
        put(root/name,files['bundle/'+name])
    # Validate original committed implementation, without rewriting provenance.
    require(git(public_repository,'cat-file','-t',m['public_commit']).strip()==b'commit','Final public source commit unavailable')
    for name,h in m['source']['source_sha256'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe source name')
        require(digest(git(public_repository,'show',SOURCE+':'+name))==h,'Historical source hash is not committed')
    # Preserve the raw last complete batch for offline cross-platform comparison.
    last=next(b for b in journal['batches'] if b.get('plan')==inventory['plan.json']['original'])
    prior=last['private_preservation'];require(prior['commit'] in m['prior_batch_archives'],'Final raw batch not included in immutable provenance')
    archive_path=Path(prior['archive']['path']).relative_to(OLD/'private-local365-state').as_posix()
    raw_batch=git(repository,'show',prior['commit']+':'+archive_path)
    require(digest(raw_batch)==prior['archive']['sha256'] and len(raw_batch)==prior['archive']['bytes'],'Final raw batch archive differs')
    batch_files,batch_links=safe_files(raw_batch)
    original_parent=Path(inventory['plan.json']['original']['path']).parent
    for name,payload in batch_files.items():
        parts=PurePosixPath(name).parts;require(parts[0]==original_parent.name,'Unexpected batch archive root')
        put(relocate(original_parent.parent/name),payload)
    receipt=dict(schema='heroes-cloud-final-handoff-verified-v1',ready_commit=ready_commit,ready_sha256=digest(raw),
        archive_commit=m['archive_commit'],archive_sha256=archive['sha256'],public_commit=m['public_commit'],
        ownership_transfer_verified=True,local_shutdown_verified=True,historical_metadata_rewritten=False,
        original_root=str(OLD),relocated_root=str(relocation),preserved_link_metadata=batch_links,
        parent={kind:binding(relocate(inventory[name]['original']['path'])) for kind,name in [('plan','plan.json'),('report','report.json'),('policy','after-policy.pt'),('resume','after-resume.pt')]},
        historical_campaign_root=str(relocate(OLD/'local-curriculum-update146/campaign-v3')),
        raw_batch_root=str(relocate(original_parent)),marker=m)
    put(root/'handoff/verification.json',(json.dumps(receipt,indent=2)+'\n').encode())
    return receipt


def audit_materialized(receipt_path,repository):
    """Independent release gate for dependency closure and all local archives.

The preparation bootstrap may copy evidence, but this gate runs from the final
committed release before native acceptance. Mac process numbers/clocks are
provenance only; never query or signal them on Linux.
"""
    receipt_path=Path(receipt_path);receipt=json.loads(receipt_path.read_text())
    m=receipt['marker'];bundle=receipt_path.parent/'bundle';index=m['files']
    plan=json.loads((bundle/'plan.json').read_text())
    for kind,name in [('plan','parent-plan.json'),('report','parent-report.json'),('resume','parent-resume.pt')]:
        require(index['dependencies/'+name]['original']==plan['parent'][kind],
            'Final metadata parent dependency differs: '+kind)
    for kind,name in [('policy','after-policy.pt'),('resume','after-resume.pt')]:
        require(index[name]['original']==m['checkpoint'][kind],'Final checkpoint inventory differs')
    restores=json.loads((bundle/'restore-receipt.json').read_text())
    require(set(restores)=={'canonical','copied'},'Both original local restore receipts required')
    for proof in restores.values():
        require(proof['exact_model_adam_rng_progress_pool'] is True and proof['progress']==m['progress'],
            'Local restore receipt is incomplete')
    shutdown=json.loads((bundle/'shutdown-proof.json').read_text())
    require(shutdown['local_auto_resume'] is False and shutdown['receipts']
        and all(isinstance(row['closed_groups'],list) for row in shutdown['receipts'])
        and all(row['owned_command_absent'] is True for row in shutdown['controllers'])
        and any(row['session']==m['local_session'] for row in shutdown['controllers']),
        'Current local controller/shutdown evidence absent')
    require(stamp(m['local_stop'])<=stamp(shutdown['verified_utc'])<=stamp(m['ready_utc']),
        'Shutdown proof is not final')
    for key,name in [('shutdown_proof','shutdown-proof.json'),('restore_evidence','restore-receipt.json')]:
        payload=git(repository,'show',m[key]['commit']+':'+m[key]['path'])
        require(payload==(bundle/name).read_bytes() and digest(payload)==m[key]['sha256'],
            'Private proof blob differs from handed-off archive')
    journal=json.loads((bundle/'campaign-report.json').read_text())
    preserved=[row['private_preservation'] for row in journal['batches'] if row.get('private_preservation')]
    require([row['commit'] for row in preserved]==m['prior_batch_archives'],'Private historical batch sequence differs')
    final=m['final_controller_archive']
    require(isinstance(final,dict) and set(('commit','archive'))<=set(final),'Final controller private preservation absent')
    verified=[]
    for item in [*preserved,final]:
        archive=item['archive'];relative=Path(archive['path']).relative_to(OLD/'private-local365-state').as_posix()
        require(git(repository,'merge-base','--is-ancestor',item['commit'],m['archive_commit'])==b'',
            'Historical archive is not immutable handoff ancestry')
        payload=git(repository,'show',item['commit']+':'+relative)
        require(digest(payload)==archive['sha256'] and len(payload)==archive['bytes'],
            'Historical/final controller private archive differs')
        # Check path safety without applying historical symlinks or editing JSON.
        contents,links=safe_files(payload)
        verified.append(dict(commit=item['commit'],path=relative,sha256=digest(payload),
            bytes=len(payload),regular_files=len(contents),preserved_link_metadata=links))
    result=dict(schema='heroes-cloud-local-dependency-closure-audit-v1',passed=True,
        ready_commit=receipt['ready_commit'],final_checkpoint=m['checkpoint'],
        parent_metadata_dependencies_verified=True,both_local_restore_receipts_verified=True,
        current_local_controller_shutdown_verified=True,local_auto_resume=False,
        private_archives=verified,mac_monotonic_clocks_imported_as_live_clocks=False,
        historical_training_samples_admitted=0)
    target=receipt_path.parent/'dependency-closure-audit.json'
    put(target,(json.dumps(result,indent=2)+'\n').encode())
    return binding(target)


def verify_assets(repository,artifact_root,output):
    """Rehash the actual eight original files against immutable private Git."""
    raw=git(repository,'show',DATA_COMMIT+':upload-manifest.json')
    require(digest(raw)==DATA_MANIFEST_SHA256,'Original private transfer manifest differs')
    manifest=json.loads(raw);root=Path(artifact_root).resolve();rows=[]
    for item in manifest['files']:
        relative=Path(item['output_relative_path'])
        if not relative.is_relative_to('game-data'):continue
        require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe original Data path')
        path=root/relative
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),'Original asset is not an ordinary private file')
        before=path.stat()
        with path.open('rb') as stream:observed=hashlib.file_digest(stream,'sha256').hexdigest()
        after=path.stat()
        signature=lambda s:dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
        require(signature(before)==signature(after) and observed==item['sha256']
            and after.st_size==item['size_bytes'],'Actual original asset hash/size differs')
        rows.append(dict(path=str(path),sha256=observed,**signature(after)))
    require(len(rows)==8,'Expected all eight original Data files')
    result=dict(schema='heroes-overnight-original-assets-verification-v1',passed=True,
        transfer_commit=DATA_COMMIT,manifest_sha256=DATA_MANIFEST_SHA256,files=rows,
        original_assets_copied_to_public_git=False)
    put(output,(json.dumps(result,indent=2)+'\n').encode())
    return binding(output)
