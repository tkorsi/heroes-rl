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
ORIGINAL_SESSION = DURABLE_ROOT/'local-v8-update92/campaign-session.json'
REPAIR = os.environ.get('HEROES_V8_REPAIR', '')
require(REPAIR in ('', 'attempt2'), 'Only explicit repair attempt2 is supported')
SESSION = (DURABLE_ROOT/'local-v8-update92-repair2/campaign-session.json') if REPAIR else ORIGINAL_SESSION
RUN_PREFIX = 'v8-update92-repair2' if REPAIR else 'v8-update92'
ORIGINAL_SESSION_SHA = 'fc9fa9d5de659a2d833918fe9b373343074dfd3729b62e752dd7564b9dcc376f'
ORIGINAL_FAILURE_SHA = '4a62c384c566f1b8d03fdcbf63702147bce9e4674adc3881a1fa68b01df2cd21'


def repair_origin(release):
    repair = release.get('repair')
    if not REPAIR:
        require(repair is None, 'Repair release requires explicit attempt2 mode')
        return None
    require(type(repair) is dict and repair.get('attempt') == 2, 'Explicit repair attempt2 release required')
    require(repair['original_session'] == binding(ORIGINAL_SESSION)
            and repair['original_session']['sha256'] == ORIGINAL_SESSION_SHA,
            'Repair must bind the exact consumed original session')
    original = checked(repair['original_session'])
    failure_path = DURABLE_ROOT/'runs/v8-update92-heldout-092/execution/report.json'
    require(repair['failure'] == binding(failure_path)
            and repair['failure']['sha256'] == ORIGINAL_FAILURE_SHA, 'Repair must retain exact failed baseline')
    failed = checked(repair['failure'])
    require(failed['training_samples'] == failed['optimizer_steps'] == 0
            and failed['attempts'][0]['sampled_choices'] == 0
            and failed['attempts'][1]['status'] == 'not_started', 'Only zero-choice startup repair is authorized')
    require(repair['controller_result'] == binding(ORIGINAL_SESSION.parent/'controller-result.json'),
            'Repair must bind original cleanup')
    cleanup = checked(repair['controller_result'])
    require(cleanup['controller_cleanup_verified'] is True and cleanup['cleanup_incomplete'] == [],
            'Original controller cleanup is incomplete')
    require(repair.get('absolute_deadline_unix') == original['absolute_deadline_unix']
            and original['absolute_deadline_unix'] == original['started_unix']+7200
            and original['compute_deadline_unix'] == original['started_unix']+6900,
            'Repair cannot reset or extend original wall budget')
    require(time.time() < original['compute_deadline_unix']
            and 0 <= time.monotonic()-original['started_monotonic'] < 6900,
            'Original repair window expired')
    return original

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


def native_codec_identity():
    """Bind the accepted engine/interface, independently of campaign tooling."""
    from validation.probes.validate_capabilities import capability_implementation_identity
    return capability_implementation_identity()


def validate_native_acceptance(value, migration_plan):
    """Recheck retained final mechanics evidence; never execute a game."""
    from validation.probes.summarize_capabilities import compact_report, summarize
    summary = checked(value['native_acceptance'])
    require(summary.get('schema') == 'heroes-capability-native-summary-v1'
            and summary.get('interface') == INTERFACE, 'Wrong final native acceptance summary')
    saved_reports = summary.get('reports')
    require(type(saved_reports) is list and bool(saved_reports), 'Final acceptance has no reports')
    reports = [compact_report(row['report_path']) for row in saved_reports]
    derived = summarize(reports)
    require(all(summary.get(key) == actual for key, actual in derived.items() if key != 'reports'),
            'Final native acceptance summary differs from its raw reports')
    require(derived['native_coverage_complete'] is True
            and derived['one_frozen_implementation_and_runtime'] is True
            and derived['all_declared_attempts_completed_without_failure'] is True
            and derived['started_attempts'] == 28
            and derived['unverified_attempts'] == derived['failed_attempts'] == 0,
            'Final 28-case native acceptance is incomplete')
    current = native_codec_identity()
    accepted = set()
    for saved, actual in zip(saved_reports, reports):
        # The committed summary may add replay counters to its compact rows.
        require(all(saved.get(key) == content for key, content in actual.items() if key != 'rows')
                and len(saved.get('rows', [])) == len(actual['rows'])
                and all(all(recorded.get(key) == content for key, content in row.items())
                        for recorded, row in zip(saved['rows'], actual['rows'])),
                'Final native report binding or mechanics changed')
        probe_path = Path(actual['report_path']).parent/'plan.json'
        probe = json.loads(probe_path.read_text())
        require(probe.get('native_runtime') == migration_plan['runtime']
                and probe.get('binary_sha256') == binding(migration_plan['binary'])['sha256']
                and probe.get('replay_binary_sha256') == migration_plan['runtime'].get('vcmireplay'),
                'Final native acceptance runtime differs from the released runtime')
        recorded = probe.get('implementation', {})
        # The probe and release share one scope, including added/removed sources.
        require(recorded == current,
                'Final native acceptance engine or contextual codec source changed')
        accepted.add((actual['report_sha256'], actual['plan_sha256']))
    return accepted


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
    accepted_native = validate_native_acceptance(value, plan)
    cases = value.get('feature_cases')
    require(type(cases) is list and len(cases) == 2*len(FEATURES), 'Every feature requires both-seat native evidence')
    require({(case['feature'], case['player']) for case in cases} == {(f, p) for f in FEATURES for p in (0, 1)},
            'Feature coverage missing or duplicated')
    for case in cases:
        require(type(case['player']) is int and case.get('passed') is True and case.get('replay_verified') is True,
                'Feature case has no native pass/replay proof')
        report = checked(case['evidence'])
        probe_plan = checked(case['plan'])
        require((case['evidence']['sha256'], case['plan']['sha256']) in accepted_native,
                'Feature case is outside the final native acceptance')
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
    repair_origin(value)
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
    origin = repair_origin(release)
    now = origin['started_unix'] if origin else time.time()
    monotonic = origin['started_monotonic'] if origin else time.monotonic()
    value = {'schema': SCOPE, 'mode': 'campaign', 'started_unix': now, 'started_monotonic': monotonic,
        'absolute_deadline_unix': now+7200, 'compute_deadline_unix': now+6900,
        'release': binding(release_path), 'configuration': configuration(), 'source': release['source'],
        'authorization': AUTHORIZATION, 'controller_pid': os.getpid(), 'controller': binding(ROOT/'tools/v8_teacher_campaign.py')}
    if origin:
        value.update(repair=release['repair'], attempt_started_unix=time.time(), attempt=2)
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
    origin = repair_origin(release)
    if origin:
        require(value.get('repair') == release['repair'] and value.get('attempt') == 2
                and all(value[key] == origin[key] for key in ('started_unix', 'started_monotonic',
                    'absolute_deadline_unix', 'compute_deadline_unix')), 'Repair session changed original budget')
    require(value['source'] == release['source'], 'Session release/source differs')
    return value
