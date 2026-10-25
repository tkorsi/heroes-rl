"""Single-use v8 scope and explicit coordinator release; no implicit launch."""
import json
import os
from pathlib import Path
import time

from .builtin_training_resume import binding, require
from .builtin_preservation import DURABLE_ROOT, durable_path
from .builtin_v8_migration import INTERFACE, SCHEMA as MIGRATION_SCHEMA, PINS
from .inference import source_identity

ROOT = Path(__file__).resolve().parents[2]
SCOPE = 'heroes-local-v8-update92-v1'
SESSION = DURABLE_ROOT/'local-v8-update92/campaign-session.json'
FEATURES = ('marketplace', 'purchases', 'artifact_equipment', 'artifact_assembly', 'artifact_transfer',
    'level_up', 'adventure_spells', 'visions', 'flight_movement', 'water_walk_movement',
    'eagle_eye', 'stables', 'neutral_join')
AUTHORIZATION = {'scope': SCOPE, 'source_thread': '01a16d9e-5f8d-7cb0-a939-3ff005a741e4',
    'instruction': 'After all v8 features pass and coordinator releases, continue exact update92 locally on hero-management-v2 for at most two hours; 365 days, 5120 choices, full four-game batches, Nullkiller Easy/BattleAI and strict native admission.'}


def configuration():
    from .builtin_first_run import config
    value = config()
    value.update(interface=INTERFACE, max_days=365, ordinary_choice_limit=4960,
        reserved_dialog_choices=160, max_total_choices=5120, max_optimizer_steps=640,
        allow_verified_training_prefix=False, shared_wall_seconds=3600, attempt_wall_seconds=600,
        phase_deadlines_seconds={'training': 2400, 'update': 3540, 'evaluation': 3540})
    return value


def checked(bound):
    require(type(bound) is dict and binding(bound['path']) == bound, 'Bound release artifact changed')
    return json.loads(Path(bound['path']).read_text())


def validate_release(path):
    value = json.loads(Path(path).read_text())
    require(value.get('schema') == SCOPE+'-release' and value.get('coordinator_released') is True
            and type(value.get('release_message')) is str and bool(value['release_message'].strip()),
            'Explicit coordinator release with message evidence is required')
    require(value.get('authorization') == AUTHORIZATION and value.get('configuration') == configuration(), 'Release scope differs')
    require(value.get('source') == source_identity(INTERFACE) and not value['source']['tracked_worktree_dirty'],
            'Release source must be exact and clean')
    receipt = checked(value['migration_receipt'])
    require(receipt['schema'] == MIGRATION_SCHEMA+'-receipt', 'Wrong migration receipt')
    from .builtin_v8_migration import verify_private
    private = checked(value['private_migration_receipt'])
    require(private['migration_receipt'] == value['migration_receipt'], 'Private copy belongs to another migration')
    verify_private(value['private_migration_receipt']['path'])
    plan = checked(receipt['plan'])
    checked(receipt['report'])
    require(plan['source'] == value['source'] and plan['parent']['policy']['sha256'] == PINS['policy']
            and plan['parent']['resume']['sha256'] == PINS['resume'], 'Release does not bind exact derived update92')
    require(value.get('runtime') == plan['runtime'] and value.get('map') == plan['map'], 'Release runtime/map differs')
    from experiments.headless.contextual_evidence import runtime_identity
    require(runtime_identity(Path(plan['binary']).parent) == plan['runtime'], 'Released runtime bytes changed')
    require(binding(plan['game_map'])['sha256'] == plan['native']['map']['sha256'], 'Released map changed')
    cases = value.get('feature_cases')
    require(type(cases) is list and len(cases) == 2*len(FEATURES), 'Every feature requires both-seat native evidence')
    require({(case['feature'], case['player']) for case in cases} == {(f, p) for f in FEATURES for p in (0, 1)},
            'Feature coverage missing or duplicated')
    for case in cases:
        require(type(case['player']) is int and case.get('passed') is True and case.get('replay_verified') is True,
                'Feature case has no native pass/replay proof')
        report = checked(case['evidence'])
        probe_plan = checked(case['plan'])
        names = {'artifact_equipment': 'artifacts', 'artifact_assembly': 'artifacts',
                 'artifact_transfer': 'artifact-transfer', 'level_up': 'level-up',
                 'adventure_spells': 'adventure-spells', 'flight_movement': 'fly-movement',
                 'water_walk_movement': 'water-walk-movement', 'neutral_join': 'neutral-overflow', 'eagle_eye': 'eagle-eye'}
        name = names.get(case['feature'], case['feature'])
        require(report.get('schema') == 'heroes-capability-native-validation-v1'
                and report.get('plan_sha256') == case['plan']['sha256']
                and probe_plan.get('native_runtime') == plan['runtime']
                and probe_plan.get('interface') == INTERFACE,
                'Feature evidence does not bind the released native runtime')
        matching = [row for row in report.get('cases', []) if row.get('case') == name
                    and row.get('player') == ('red', 'blue')[case['player']]]
        require(any(row.get('status') == 'passed' and row.get('replay_verified') is True
                    for row in matching), 'Required native feature/seat has no actual passed case')
    for proof in value.get('additional_evidence', []):
        checked(proof)
    return value


def create_session(release_path):
    """Called only by the explicit launch command, never by preparation."""
    from .builtin_v8_migration import write_new
    release_path = durable_path(release_path)
    release = validate_release(release_path)
    SESSION.parent.mkdir(parents=True, exist_ok=True)
    require(not SESSION.exists(), 'Consumed v8 session cannot restart')
    with SESSION.with_suffix('.lock').open('x') as stream:
        stream.write(str(os.getpid())+'\n')
    now = time.time()
    value = {'schema': SCOPE, 'mode': 'campaign', 'started_unix': now, 'started_monotonic': time.monotonic(),
        'absolute_deadline_unix': now+7200, 'compute_deadline_unix': now+6900,
        'release': binding(release_path), 'configuration': configuration(), 'source': release['source'],
        'authorization': AUTHORIZATION, 'controller_pid': os.getpid(), 'controller': binding(ROOT/'tools/v8_teacher_campaign.py')}
    write_new(SESSION, value)
    return value


def session_for(path):
    path = durable_path(path)
    require(path == SESSION, 'Only one authorized v8 session path')
    value = json.loads(path.read_text())
    require(value.get('schema') == SCOPE and value.get('mode') == 'campaign'
            and value['absolute_deadline_unix'] == value['started_unix']+7200
            and value['compute_deadline_unix'] == value['started_unix']+6900
            and value['started_unix'] <= time.time() < value['compute_deadline_unix']
            and 0 <= time.monotonic()-value['started_monotonic'] < 6900, 'V8 session expired or changed')
    require(value['configuration'] == configuration() and value['authorization'] == AUTHORIZATION
            and value['controller'] == binding(ROOT/'tools/v8_teacher_campaign.py'), 'V8 session code/scope drift')
    checked(value['release'])
    release = validate_release(value['release']['path'])
    require(value['source'] == release['source'], 'Session release/source differs')
    return value
