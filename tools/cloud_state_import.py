"""Read-only validation of the reviewed checkpoint1335 cloud-to-Mac boundary.

Historical cloud JSON and checkpoint bytes remain immutable. A separately
published recovery certificate can admit the completed publication-pending
update exactly once. This module never trains, writes official state, changes
ownership, publishes files or signals processes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import random
import re
import subprocess
import time

from tools.training_environment import NEXT_LAUNCH_ENVIRONMENT, verify_environment

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'configs/cloud1335-import.json'
SCHEMA = 'heroes-cloud1335-mac-import-v1'
HANDOFF = 'checkpoint1229-cloud-a08f55ff0c7548f58ff9d50ef5af8ba7'
CLOUD_COMMIT = 'c6ebc8327e8dfe65dfa384a3fe77da709343a4dd'
CLOSURE_COMMIT = '89550dba6c481653ffc38cd78997438e846947f2'
RESULTS_REF = 'refs/heads/codex/results-' + HANDOFF
CLAIM_REF = 'refs/heads/codex/claim-' + HANDOFF
CLAIM_COMMIT = '85e58378474334ee34209f78a60c3187c360fbff'
CLOUD_THREAD = '01a1854c-f989-7642-aec2-785e3f0ec054'
FROZEN_SUPERVISOR_SHA256 = 'a14aa51bf2b5553233aea2f99aa0a3010fdc37b50d17d6810f36d0003f48a413'
DELETED_BUILD_ERROR = ('FailedPrecondition: The cloud build for this thread has been deleted and cannot be resumed. '
                       'View saved history. If this is a setup thread, report the unavailable environment.')
MAP_SHA256 = 'ccfc8467847067a75091d68626e5b241839036ab729b5520ce4445a75a45b7b3'
DESCRIPTOR_FILES = {'plan': 'plan.json', 'report': 'report.json', 'policy': 'after-policy.pt',
                    'resume': 'after-resume.pt', 'initial_checkpoint': 'initial-checkpoint.pt'}
RECEIPT_KEY = 'cloud_import/receipt'
# A later local runtime repair must not rewrite the machine used to prove the
# cloud handoff. Each reviewed transition names its exact permitted fields;
# the complete current runtime acceptance gate still runs before selection.
RUNTIME_UPGRADES = {
    'cpu-next-launch-v1': {
        'historical_machine_sha256': '51776f9d27f6b2879a2ddf5b6ef442a9820ea76737ea85afcc09248a7b42e27c',
        'descriptor_sha256': None,
        'python_environment': NEXT_LAUNCH_ENVIRONMENT,
    },
    'terminal-replay-1392-v1': {
        'historical_machine_sha256': '51776f9d27f6b2879a2ddf5b6ef442a9820ea76737ea85afcc09248a7b42e27c',
        'descriptor_sha256': '9f9292bc8b5e959d7f7ed1152dc50b9d3404b5ed67b73ec0a11f2ec8331d313d',
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(),
            'Cloud import requires an existing absolute regular file: ' + str(path))
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def bound(value):
    require(type(value) is dict and set(value) == {'path', 'sha256', 'bytes'}
            and type(value['path']) is str and type(value['bytes']) is int
            and type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']),
            'Malformed cloud import file binding')
    require(binding(value['path']) == value, 'Cloud import evidence changed: ' + value['path'])
    return Path(value['path'])


def read(path):
    from experiments.headless.contextual_evidence import strict_json
    return strict_json(Path(path).read_bytes())


def same_bytes(left, right):
    return (type(left) is dict and type(right) is dict and set(left) == set(right) == set(DESCRIPTOR_FILES)
            and all(all(left[key].get(name) == right[key].get(name) for name in ('sha256', 'bytes'))
                    for key in DESCRIPTOR_FILES))


def reviewed_profile():
    require(PROFILE.is_file() and not PROFILE.is_symlink(), 'Cloud import lacks a reviewed profile')
    committed = subprocess.check_output(['git', 'show', 'HEAD:' + PROFILE.relative_to(ROOT).as_posix()],
                                        cwd=ROOT, timeout=10)
    require(committed == PROFILE.read_bytes(), 'Cloud import profile must be exactly committed')
    value = read(PROFILE)
    require(set(value) == {'schema', 'handoff_id', 'cloud_source_commit', 'closure_commit', 'receipt'}
            and value['schema'] == SCHEMA + '-profile' and value['handoff_id'] == HANDOFF
            and value['cloud_source_commit'] == CLOUD_COMMIT and value['closure_commit'] == CLOSURE_COMMIT,
            'Unsupported cloud import profile')
    return value


def import_required(config):
    """A committed import cannot be bypassed by deleting its official marker."""
    present = PROFILE.exists() or PROFILE.is_symlink()
    if not present:
        present = subprocess.run(['git', 'cat-file', '-e', 'HEAD:' + PROFILE.relative_to(ROOT).as_posix()],
                                 cwd=ROOT, timeout=10, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL).returncode == 0
    if not present:
        return False
    profile = reviewed_profile()
    return Path(profile['receipt']['path']).parent == Path(config['run_root']).resolve()


def ledger(value, label):
    require(type(value) is dict and set(value) == {'next_sequence', 'consumed'}
            and type(value['next_sequence']) is int and type(value['consumed']) is list,
            'Malformed cloud-import ledger: ' + label)
    sequences = []
    for row in value['consumed']:
        require(type(row) is dict and 'sequence' in row and set(row) <= {'sequence', 'reserved_utc'}
                and type(row['sequence']) is int and row['sequence'] >= 0
                and ('reserved_utc' not in row or type(row['reserved_utc']) is str),
                'Malformed reservation in ' + label)
        sequences.append(row['sequence'])
    require(sequences == sorted(set(sequences)) and sequences
            and value['next_sequence'] == sequences[-1] + 1,
            'Inconsistent cloud-import ledger: ' + label)
    return value


def manifest_files(root, relative, expected):
    """Recheck exact metadata/state bytes; raw readback has a separate pinned audit."""
    folder = root / relative
    require(bound(expected) == folder / 'manifest.json', 'Publication manifest location changed')
    value = read(bound(expected))
    require(value.get('schema') == 'heroes-linux-private-publication-v1'
            and value.get('game_data_symlinks_followed') is False
            and type(value.get('entries')) is dict and type(value.get('raw_inventory')) is dict,
            'Invalid private publication manifest')
    for name, item in value['entries'].items():
        relative_path = PurePosixPath(name)
        require(name == str(relative_path) and not relative_path.is_absolute()
                and '..' not in relative_path.parts and type(item) is dict
                and set(item) == {'bytes', 'sha256'}, 'Unsafe publication member')
        if re.fullmatch(r'raw\.part[0-9]{5}', name):
            continue
        actual = binding(folder / name)
        require(all(actual[key] == item[key] for key in ('bytes', 'sha256')),
                'Published metadata/state changed: ' + name)
    return value


def descriptor(folder):
    return {key: binding(folder / filename) for key, filename in DESCRIPTOR_FILES.items()}


def schedule(plan, sequence):
    require(type(plan.get('schedule')) is list and len(plan['schedule']) == 4,
            'Cloud import requires complete four-game schedules')
    keys = []
    for index, item in enumerate(plan['schedule']):
        key = (item.get('engine_seed'), item.get('policy_seed'), item.get('learner_player'))
        require(key == (40_000_000 + 4*sequence + index, 50_000_000 + 4*sequence + index, sequence % 2),
                'Cloud import schedule changed or reused a seed')
        keys.append(key)
    return keys


def seed_namespace(config):
    require(config.get('seed_namespace') == {'engine_base': 40_000_000, 'policy_base': 50_000_000},
            'The original Mac seed namespace changed')


def verify_state(parent, expected):
    import torch
    from experiments.learner import local_script_training as local
    from tools.recover_interrupted_training import fingerprint
    torch_before, python_before = torch.get_rng_state().clone(), random.getstate()
    try:
        state = local.restore_parent(parent)
        values = dict(model=dict(state.model.state_dict()), optimizer=state.optimizer.state_dict(),
                      generator=state.generator.get_state(), pool=state.pool.state_dict(), progress=vars(state.progress),
                      sampler=state.sampler_state, torch_rng=torch.get_rng_state(), python_rng=random.getstate())
        require(set(expected) == {'full_state_sha256', 'component_sha256', 'optimizer_steps'}
                and fingerprint(values) == expected['full_state_sha256']
                and {key: fingerprint(value) for key, value in values.items()} == expected['component_sha256']
                and state.optimizer_steps == expected['optimizer_steps'],
                'Imported model/Adam/RNG/progress/sampler/pool differs')
        return dict(values['progress'])
    finally:
        torch.set_rng_state(torch_before)
        random.setstate(python_before)


def check_cloud_closure(closure, cloud):
    """Require either observed closure or this exact source-proven terminal boundary.

    The deleted-environment case does not claim a fresh process inventory, a
    changed cloud lock, or absence of the outer publication supervisor. Its
    reviewed source can only publish evidence after the training controller has
    returned. The immutable consumed claim prevents the same handoff relaunch.
    """
    require(closure.get('schema') == 'heroes-cloud1335-process-closure-v1'
            and closure.get('handoff_id') == HANDOFF and closure.get('closure_commit') == CLOSURE_COMMIT
            and closure.get('automatic_restart') is False and closure.get('training_launched') is False
            and type(closure.get('observed_utc')) is str and bool(closure['observed_utc']),
            'Cloud ownership has not been verifiably closed and fenced')
    if closure.get('method') == 'live_process_snapshot':
        require(closure.get('live_owned_processes') == [] and closure.get('supervisor_absent') is True
                and closure.get('cloud_launch_fenced') is True,
                'Cloud ownership has not been verifiably closed and fenced')
        return
    require(closure.get('method') == 'published_terminal_training_closure_deleted_environment'
            and all(closure.get(key) is False for key in ('live_process_snapshot_obtained',
                'outer_supervisor_absence_verified', 'cloud_lock_mode_changed'))
            and all(closure.get(key) is True for key in ('training_worker_closed',
                'cloud_launch_claim_consumed', 'cloud_training_excluded'))
            and 'supervisor_absent' not in closure and 'live_owned_processes' not in closure,
            'Unsupported or overstated deleted-environment closure')
    for key, name in (('cloud_shutdown', 'shutdown.json'), ('cloud_campaign', 'campaign-report.json'),
                      ('cloud_session', 'session.json')):
        require(bound(closure[key]) == cloud / name, 'Terminal closure evidence was substituted')
    app = read(bound(closure['app_unavailable']))
    require(app.get('schema') == 'heroes-cloud-runtime-access-observation-v1'
            and app.get('thread_id') == CLOUD_THREAD and app.get('host_id') == 'durable'
            and app.get('is_error') is True and app.get('error') == DELETED_BUILD_ERROR
            and app.get('live_process_snapshot_obtained') is False and app.get('training_launched') is False,
            'The exact cloud environment deletion has not been observed')
    for key, ref, commit in (('remote_claim_ref', CLAIM_REF, CLAIM_COMMIT),
                             ('remote_results_ref', RESULTS_REF, CLOSURE_COMMIT)):
        remote = read(bound(closure[key]))
        require(remote.get('ref') == ref and remote.get('object', {}).get('sha') == commit
                and remote['object'].get('type') == 'commit', 'Original cloud ownership or closure ref changed')
    claim = read(bound(closure['remote_claim']))
    session = read(cloud / 'session.json')
    require(claim.get('handoff_id') == HANDOFF and claim.get('pid') == claim.get('process_group') == 95785
            and claim.get('started_unix') == session['started_unix'] == 1793275142.1684587
            and claim.get('hard_deadline_unix') == session['absolute_deadline_unix'] == 1793311142.1684587
            and session['supervisor_pid'] == 95785 and claim.get('source') == session['source']
            and claim['source'].get('git_commit') == CLOUD_COMMIT
            and claim['source'].get('tracked_worktree_dirty') is False
            and session.get('linux_cloud') == dict(handoff_id=HANDOFF,
                claim=dict(ref=CLAIM_REF, commit=CLAIM_COMMIT), results_ref=RESULTS_REF),
            'The closed training worker does not match the consumed cloud claim')
    bound(closure['frozen_supervisor'])
    require(closure['frozen_supervisor']['sha256'] == FROZEN_SUPERVISOR_SHA256,
            'The reviewed post-training supervisor control flow changed')


def historical_machine(config, base_receipt):
    """Keep the1229 machine immutable across acceptance refresh or reviewed repair."""
    audit = read(bound(base_receipt['verification']['immutable_evidence']['checkpoint_audit']))
    machine = audit['source_runtime']['machine']
    previous = read(bound(machine))
    acceptance_fields = {'created_utc', 'runtime_origin', 'runtime_release'}
    release = config.get('runtime_release', {})
    upgrade = RUNTIME_UPGRADES.get(release.get('id')) if type(release) is dict else None
    descriptor = release.get('descriptor', {}) if type(release) is dict else {}
    added_fields = set()
    environment = None
    if (upgrade is not None and machine['sha256'] == upgrade['historical_machine_sha256']
            and type(upgrade['descriptor_sha256']) is str and len(upgrade['descriptor_sha256']) == 64
            and type(descriptor) is dict and descriptor.get('sha256') == upgrade['descriptor_sha256']):
        acceptance_fields |= {'runtime_bin', 'runtime_files', 'runtime'}
        if 'python_environment' in upgrade:
            require(config.get('python_environment') == upgrade['python_environment'],
                    'Cloud continuation requires its reviewed Python environment binding')
            environment = upgrade['python_environment']
            acceptance_fields.add('python')
            added_fields.add('python_environment')
    require(set(config) == set(previous) | added_fields
            and all(config[key] == previous[key] for key in set(previous) - acceptance_fields),
            'Cloud continuation requires the identical historical machine outside reviewed runtime changes')
    if environment is not None:
        verify_environment(environment, previous_python=previous['python'],
                           current_python=config['python'], probe_current=False)
    if config != previous:
        # The fresh current-source native and policy evidence must pass every
        # normal gate. Historical proof still uses its original machine bytes.
        from tools.prepare_local_training import validate_config
        validate_config(config)
    return previous


def audit_import(receipt_path, config, anchor, *, local_auditor, preview=False):
    """Recompute this exact handoff's immutable lineage and selected complete state."""
    from experiments.learner import local_script_training as local
    from tools.run_local_training import verified_batch
    seed_namespace(config)
    started = time.monotonic()
    profile = reviewed_profile()
    receipt_path = Path(receipt_path).resolve()
    actual_receipt = binding(receipt_path)
    require(type(preview) is bool
            and profile['receipt']['path'] == str(Path(config['run_root']).resolve() / 'cloud-import.json')
            and all(actual_receipt[key] == profile['receipt'][key] for key in ('sha256', 'bytes'))
            and (preview or actual_receipt == profile['receipt']),
            'Cloud import marker is not the reviewed official selection')
    receipt = read(receipt_path)
    require(receipt.get('schema') == SCHEMA and receipt.get('handoff_id') == HANDOFF
            and receipt.get('cloud_source_commit') == CLOUD_COMMIT and receipt.get('closure_commit') == CLOSURE_COMMIT
            and receipt.get('run_root') == str(Path(config['run_root']).resolve())
            and receipt.get('selected_update') in (1334, 1335), 'Unsupported cloud import receipt')
    evidence_root = Path(receipt['evidence_root'])
    require(evidence_root.is_absolute() and evidence_root.resolve() == evidence_root
            and evidence_root.is_dir() and not evidence_root.is_relative_to(ROOT),
            'Imported evidence must remain in a durable private directory')
    expected_folders = {'start', 'closure'} | {f'batches/train-{n:06d}-seat{n%2}' for n in range(869, 975)}
    require(set(receipt['manifests']) == {folder + '/manifest.json' for folder in expected_folders},
            'Cloud import omitted or added a publication manifest')
    manifests = {}
    for folder in sorted(expected_folders):
        require(time.monotonic() - started < 180, 'Cloud import verification exceeded its deadline')
        manifests[folder] = manifest_files(evidence_root, folder, receipt['manifests'][folder + '/manifest.json'])

    base_receipt = read(bound(receipt['base_recovery_receipt']))
    base_config = historical_machine(config, base_receipt)
    base_parent, remaining, base = local_auditor(anchor, base_config, receipt['base_recovery_receipt']['path'],
        moving_snapshots=base_receipt['verification']['original_moving_snapshots'])
    require(remaining == 2600 and base['admitted_before_session'] == 2400 and base['next_sequence'] == 869,
            'Imported cloud state does not follow the exact local1229 lineage')
    cloud = evidence_root / 'closure'
    session, campaign, shutdown, latest, pending, cloud_ledger = [read(cloud / name) for name in
        ('session.json', 'campaign-report.json', 'shutdown.json', 'latest-checkpoint.json',
         'pending-checkpoint.json', 'sequence-ledger.json')]
    require(read(evidence_root / 'start/session.json') == session
            and read(evidence_root / 'start/sequence-ledger.json') == base['seed_ledger_snapshot'],
            'Cloud start does not retain the exact Mac handoff')
    require(session['source']['git_commit'] == CLOUD_COMMIT and session['source']['tracked_worktree_dirty'] is False
            and session['profile'] == 'original' and session['scope'] == local.SCOPE
            and session['policy_support'] == 'visible-interaction-budget-v1'
            and session['training_engine_seed_base'] == 40_000_000
            and session['training_policy_seed_base'] == 50_000_000
            and same_bytes(session['initial_parent'], base_parent)
            and session['original_continuation']['origin_run'] == str(Path(anchor).resolve())
            and session['target_games'] == 2600 and session['next_sequence'] == 869,
            'Cloud session lineage, source or learning contract changed')
    require(campaign['status'] == 'stopped_failure' and campaign['initial_state_unchanged'] is True
            and campaign['owned_cleanup_verified'] is True and campaign['full_final_restore']['exact'] is True
            and campaign['latest_progress']['updates'] == 1334 and campaign['latest_parent'] == latest['parent']
            and campaign['latest_progress'] == latest['progress'] == campaign['full_final_restore']['progress']
            and shutdown['owned_cleanup_verified'] is True and shutdown['worker_returncode'] == 1
            and shutdown['within_hard_deadline'] is True and shutdown.get('supervisor_error') is None,
            'The original cloud failure/closure evidence changed')
    closure = read(bound(receipt['cloud_closure']))
    check_cloud_closure(closure, cloud)
    rows = [row for row in campaign['batches'] if row.get('mode') == 'training']
    require([row.get('sequence') for row in rows] == list(range(869, 975)), 'Cloud attempt ledger is incomplete')
    cloud_ledger = ledger(cloud_ledger, 'cloud closure')
    prefix = base['seed_ledger_snapshot']['consumed']
    require(cloud_ledger['consumed'][:len(prefix)] == prefix
            and [row['sequence'] for row in cloud_ledger['consumed'][len(prefix):]] == list(range(869, 975))
            and cloud_ledger['next_sequence'] == 975, 'Cloud reservations were dropped, changed or reused')
    previous = session['initial_parent']
    selected = None
    for row in rows:
        require(time.monotonic() - started < 180, 'Cloud import verification exceeded its deadline')
        sequence = row['sequence']
        folder_name = f'batches/train-{sequence:06d}-seat{sequence%2}'
        folder = evidence_root / folder_name
        actual = descriptor(folder)
        plan, report = local.read_descriptor(actual)
        require(plan['source']['git_commit'] == CLOUD_COMMIT and plan['profile'] == 'original'
                and plan['mode'] == 'training' and plan['scope'] == local.SCOPE
                and plan['batch_sequence'] == sequence and plan['parent_descriptor'] == previous
                and plan['configuration'] == session['configuration']
                and plan['execution'] == session['execution'] and plan['policy_support'] == session['policy_support']
                and plan['map']['sha256'] == MAP_SHA256 and plan['map']['scenario'] == 'hero-management-v2'
                and plan['map']['profile'] == 'original' and plan['map']['fixture'] is False
                and plan['local_session']['sha256'] == binding(cloud / 'session.json')['sha256']
                and verified_batch(report, 0) and report['optimizer_updates'] == 1
                and report['update_status'] == 'checkpoint_verified',
                'Cloud checkpoint is not a complete unchanged original-map update')
        require(len(plan['native_opponents']) == 2
                and {opponent.get('player') for opponent in plan['native_opponents']} == {0, 1}
                and all(opponent.get('module') == 'Nullkiller' and opponent.get('difficulty') == 0
                    and opponent.get('kind') == 'builtin'
                    for opponent in plan['native_opponents']), 'Cloud opponent changed')
        schedule(plan, sequence)
        candidate = read(folder / 'candidate.json')
        require(same_bytes(candidate['parent'], actual) and row['parent'] == plan['parent_descriptor']
                and all(row[name]['sha256'] == actual[name]['sha256'] for name in ('plan', 'report')),
                'Cloud candidate/attempt bindings differ')
        require(candidate['progress']['updates'] == sequence + 361
                and candidate['progress']['rollouts'] == 4 * (sequence + 361)
                and candidate['progress'] == report['cumulative_progress'], 'Cloud progress is not contiguous')
        if sequence < 974:
            publication = row.get('private_publication') or {}
            require(row.get('status') == 'passed' and row.get('checkpoint_verified') is True
                    and publication.get('ref') == RESULTS_REF
                    and publication.get('prefix') == 'cloud/' + HANDOFF + '/' + folder_name
                    and publication.get('manifest_sha256') == receipt['manifests'][folder_name + '/manifest.json']['sha256']
                    and publication.get('fresh_readback_verified') is True
                    and publication.get('checkpoint_roundtrip_verified') is True,
                    'Cloud checkpoint was not privately admitted')
            if sequence == 973:
                require(candidate['parent'] == latest['parent'] and candidate['progress'] == latest['progress'],
                        'The admitted checkpoint1334 pointer differs from its final verified batch')
        else:
            require(row.get('status') == 'failed' and not row.get('checkpoint_verified')
                    and row.get('owned_cleanup_verified') is True and row.get('completed_games') == 4
                    and 'TimeoutExpired' in row.get('error', '') and 'raw.part00000' in row.get('error', '')
                    and '120 seconds' in row.get('error', '') and candidate == pending,
                    'The pending checkpoint is not the reviewed publication-only failure')
        previous = candidate['parent']
        if candidate['progress']['updates'] == receipt['selected_update']:
            selected = (actual, candidate['progress'])
    require(selected is not None, 'Selected cloud checkpoint is absent')
    if receipt['selected_update'] == 1335:
        recovery = read(bound(receipt['publication_recovery']))
        require(recovery.get('schema') == 'heroes-cloud1335-publication-recovery-v1'
                and recovery.get('handoff_id') == HANDOFF and recovery.get('closure_commit') == CLOSURE_COMMIT
                and recovery.get('sequence') == 974 and recovery.get('progress') == selected[1]
                and all(recovery.get(key) is True for key in ('fresh_readback_verified', 'raw_inventory_verified',
                    'checkpoint_roundtrip_verified', 'original_failure_preserved'))
                and recovery.get('update_reapplied') is False and recovery.get('games_replayed') == 0
                and recovery.get('training_launched') is False
                and recovery.get('manifest_sha256') == receipt['manifests']['batches/train-000974-seat0/manifest.json']['sha256']
                and recovery.get('full_state_sha256') == receipt['state']['full_state_sha256']
                and type(recovery.get('commit')) is str and re.fullmatch('[0-9a-f]{40}', recovery['commit'])
                and type(recovery.get('ref')) is str and recovery['ref'].startswith('refs/heads/codex/'),
                'Checkpoint1335 lacks exact private publication recovery')
    progress = verify_state(selected[0], receipt['state'])
    require(progress == selected[1], 'Restored cloud progress changed')
    require(time.monotonic() - started < 180, 'Cloud import verification exceeded its deadline')
    require(binding(receipt_path) == actual_receipt, 'Import receipt changed during verification')
    return dict(receipt=profile['receipt'], parent=selected[0], progress=progress, ledger=cloud_ledger,
                base=base, selected_update=receipt['selected_update'],
                import_run=str(Path(config['run_root']).resolve() / ('cloud-import-' + str(receipt['selected_update']))))


def moving_pointers(audit):
    """Return the explicit import markers; the caller owns atomic application."""
    return {
        'latest-run.json': {'run_dir': audit['import_run'], 'cloud_import': audit['receipt']},
        'latest-checkpoint.json': {'run_dir': audit['import_run'], 'parent': audit['parent'],
                                   'progress': audit['progress'], 'cloud_import': audit['receipt']},
        'sequence-ledger.json': audit['ledger'],
    }


def continuation(anchor, config, receipt_path, recovery_receipt=None, *, local_auditor):
    """Cross the certified import boundary and audit any subsequent local updates."""
    from experiments.learner import local_script_training as local
    from tools.run_local_training import verified_batch
    seed_namespace(config)
    audit = audit_import(receipt_path, config, anchor, local_auditor=local_auditor)
    if recovery_receipt is not None:
        require(binding(recovery_receipt) == read(receipt_path)['base_recovery_receipt'],
                'Cloud continuation cannot replace its pinned local recovery receipt')
    root = Path(config['run_root']).resolve()
    expected = moving_pointers(audit)
    latest, newest, current = [read(root / name) for name in
                             ('latest-checkpoint.json', 'latest-run.json', 'sequence-ledger.json')]
    current = ledger(current, 'current Mac')
    require(current['consumed'][:len(audit['ledger']['consumed'])] == audit['ledger']['consumed'],
            'Current Mac ledger dropped or changed a consumed cloud reservation')
    reservations = set()
    evidence = {**audit['base']['evidence'], RECEIPT_KEY: audit['receipt']}

    def inherit(session):
        proof = session.get('original_continuation') or {}
        require(session['run_root'] == str(root) and session['profile'] == 'original'
                and session['training_engine_seed_base'] == 40_000_000
                and session['training_policy_seed_base'] == 50_000_000
                and proof.get('origin_run') == str(Path(anchor).resolve())
                and proof.get('evidence', {}).get(RECEIPT_KEY) == audit['receipt'],
                'Local descendant omitted or changed the cloud import boundary')
        snapshot = ledger(proof.get('seed_ledger_snapshot'), 'Mac ancestor session')
        require(snapshot['consumed'][:len(audit['ledger']['consumed'])] == audit['ledger']['consumed']
                and current['consumed'][:len(snapshot['consumed'])] == snapshot['consumed'],
                'Current Mac ledger is not an exact append of an ancestor reservation snapshot')

    def closed(run):
        run = Path(run).resolve()
        require(run.parent == root, 'Foreign run cannot descend from the cloud import')
        session, report, shutdown = [read(run / name) for name in ('session.json', 'campaign-report.json', 'shutdown.json')]
        inherit(session)
        require(session['run_dir'] == str(run), 'Local descendant run directory changed')
        require(report.get('status') in ('finished', 'stopped_requested', 'stopped_budget', 'stopped_failure')
                and report.get('owned_cleanup_verified') is True and report.get('initial_state_unchanged') is True
                and (report.get('full_final_restore') or {}).get('exact') is True
                and shutdown.get('owned_cleanup_verified') is True and shutdown.get('within_hard_deadline') is True
                and shutdown.get('worker_returncode') in (0, 1), 'Local descendant is not verifiably closed')
        require(report['latest_progress'] == report['full_final_restore']['progress'],
                'Local descendant final restoration differs from its latest progress')
        for row in report['batches']:
            if row.get('mode') == 'training':
                require(type(row.get('sequence')) is int and row['sequence'] >= 975,
                        'Local descendant reused a consumed cloud sequence')
                reservations.add(row['sequence'])
        for name in ('session.json', 'campaign-report.json', 'shutdown.json'):
            evidence[str(run / name)] = binding(run / name)
        return session, report

    if newest.get('run_dir') == audit['import_run']:
        require(newest == expected['latest-run.json'], 'Initial import run pointer changed')
        require(latest == expected['latest-checkpoint.json'],
                'Import-only newest pointer cannot hide a later Mac attempt')
    else:
        require(binding(newest['session']['path']) == newest['session']
                and Path(newest['session']['path']) == Path(newest['run_dir']) / 'session.json',
                'Newest Mac attempt session binding changed')
        _, newest_report = closed(newest['run_dir'])
        require(newest_report['latest_parent'] == latest['parent']
                and newest_report['latest_progress'] == latest['progress'],
                'Latest pointer cannot discard the newest Mac attempt state')
    if latest.get('run_dir') == audit['import_run']:
        require(latest == expected['latest-checkpoint.json'], 'Initial import checkpoint pointer changed')
    else:
        _, report = closed(latest['run_dir'])
        require(latest['parent'] == report['latest_parent']
                and latest['progress'] == report['latest_progress'] == report['full_final_restore']['progress'],
                'Latest Mac checkpoint differs from closed restored state')
    cursor, seen, count, scheduled = latest['parent'], set(), 0, set()
    available_updates = (5000 - (2400 + 4*(audit['selected_update']-1229))) // 4
    while cursor != audit['parent']:
        require(cursor['plan']['sha256'] not in seen and count < available_updates,
                'Post-import ancestry loops or exceeds target')
        seen.add(cursor['plan']['sha256'])
        plan, report = local.read_descriptor(cursor)
        require(plan['profile'] == 'original' and plan['mode'] == 'training' and plan['scope'] == local.SCOPE
                and verified_batch(report, 0) and report['optimizer_updates'] == 1
                and report['update_status'] == 'checkpoint_verified', 'Invalid post-import complete checkpoint')
        require(binding(plan['local_session']['path']) == plan['local_session'], 'Post-import session changed')
        session = read(plan['local_session']['path'])
        inherit(session)
        require(Path(plan['local_session']['path']) == Path(session['run_dir']) / 'session.json'
                and Path(session['run_dir']).parent == root,
                'Foreign checkpoint crossed the cloud import boundary')
        sequence = plan['batch_sequence']
        require(type(sequence) is int and sequence >= 975, 'A cloud seed was replayed after import')
        reservations.add(sequence)
        for key in schedule(plan, sequence):
            require(key not in scheduled, 'A post-import training seed was reused')
            scheduled.add(key)
        cursor = plan['parent_descriptor']
        count += 1
    require(reservations <= {row['sequence'] for row in current['consumed']},
            'Current ledger omitted a completed or failed Mac reservation')
    require(latest['progress']['updates'] == audit['progress']['updates'] + count
            and latest['progress']['rollouts'] == audit['progress']['rollouts'] + 4*count,
            'Post-import progress differs from complete ancestry')
    admitted = 2400 + 4 * (audit['selected_update'] - 1229 + count)
    remaining = 5000 - admitted
    require(remaining > 0, 'The original5000-game target is already complete')
    return latest['parent'], remaining, dict(schema='heroes-original5000-continuation-v1',
        origin_run=str(Path(anchor).resolve()), origin_checkpoint=699, origin_admitted_games=280,
        target_games=5000, admitted_before_session=admitted, remaining_before_session=remaining,
        next_sequence=current['next_sequence'], seed_ledger_snapshot=current, evidence=evidence,
        rule='Fresh manual session after verified cloud import; no repeated update or consumed reservation')
