"""Explicit update-31 macOS migration and bounded local-session contract.

The dated cloud runner remains expired. A local session is a new immutable
artifact, not a rewrite of any historical plan or checkpoint metadata.
"""
import json
from pathlib import Path
import subprocess
import time

from .builtin_training_resume import binding, require
from .builtin_portable import Migration, GLUE

ROOT = Path(__file__).resolve().parents[2]
LOCAL_SCOPE = 'heroes-local365-update31-v1'
PARENT_COMMIT = '23556aa0308b5862b8f8d1dab2e596cf8daa8637'
BASE_COMMIT = '8451791dcff90df4ea60b9250699dc64837a5f0b'
PERFORMANCE = ('contextual.py', 'contextual_model.py', 'contextual_rollout.py')
PARENT_POLICY = '84b0cc3c884023b92c68bc7b98c4499a97d65d39492ceaade4fe1e4c06dd45a3'
AUTHORIZATION = {'scope':LOCAL_SCOPE,'instruction':'Continue cloud update 31 locally for at most two hours after fresh both-seat gates; max_days=365, 5120 choices, complete four-game batches, Nullkiller Easy and BattleAI; preserve rewards and strict admission; cloud remains stopped.'}
PARENT_RESUME = '63ddf8d19fbd3a963a6886f9bfa4df6944e7a8d68b53f8b34ddfe39d6af86bf4'


def performance_changes():
    import hashlib
    return {'experiments/learner/'+name: {
        label: hashlib.sha256(subprocess.check_output(['git','show',commit+':experiments/learner/'+name],cwd=ROOT)).hexdigest()
        for label,commit in [('from',PARENT_COMMIT),('to',BASE_COMMIT)]} for name in PERFORMANCE}


class LocalMigration(Migration):
    def validate_source_changes(self):
        require(self.value.get('scope_id') == LOCAL_SCOPE, 'Unrecognized local transition')
        parent = self.value['parent']
        require(parent['policy_sha256'] == PARENT_POLICY and parent['resume_sha256'] == PARENT_RESUME
                and parent['progress'] == {'updates':31,'rollouts':124,'decisions':6776}
                and parent['optimizer_steps'] == 253, 'Local migration must retain exact cloud update 31')
        changes = self.value['source_changes']
        expected = performance_changes()
        require(all(changes.get(k) == v for k,v in expected.items()), 'Unreviewed performance source transition')
        for name in changes:
            require(name in expected or (name.startswith('experiments/learner/')
                    and Path(name).name in GLUE | {'builtin_portable.py'}), 'Migration cannot change policy, features or PPO math')
        proof = self.value['performance_parity']
        require(binding(proof['path']) == proof, 'Parity evidence changed')
        path = Path(proof['path']).resolve()
        require(path.is_relative_to(ROOT/'validation'), 'Parity evidence must be committed')
        require(subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT) == path.read_bytes(),
                'Parity evidence is not exactly committed')
        evidence = json.loads(path.read_text())
        require(evidence['checker_sha256'] == binding(ROOT/'tools/check_local365_parity.py')['sha256'], 'Parity checker evidence drift')
        require(evidence['source_changes'] == expected and evidence['policy_sha256'] == PARENT_POLICY
                and evidence['status'] == 'passed' and evidence['exact_trace_and_gradients'] is True,
                'Missing exact trained-policy parity evidence')


    def training_gate(self, runtime):
        gate = super().training_gate(runtime)
        plan = json.loads(Path(gate['plan']['path']).read_text())
        require(plan['configuration']['max_days'] == 365 and plan.get('authorization') == AUTHORIZATION
                and plan.get('local_session'), 'Fresh local365 native gate required')
        return gate


def migration_class(path):
    value = json.loads(Path(path).read_text())
    return LocalMigration if value.get('scope_id') == LOCAL_SCOPE else Migration


def configuration(session):
    from .builtin_first_run import config
    cfg = config()
    path = ROOT/'configs/builtin-local365-v1.json'
    expected = {'schema':LOCAL_SCOPE,'max_days':365,'ordinary_choice_limit':4960,
        'reserved_dialog_choices':160,'max_total_choices':5120,'max_optimizer_steps':640,
        'training_games':4,'heldout_every_updates':5,'allow_verified_training_prefix':False,
        'shared_wall_seconds':3600,'attempt_wall_seconds':600,
        'phase_deadlines_seconds':{'training':2400,'update':3540,'evaluation':3540},
        'campaign_seconds':7200,'cleanup_publication_reserve_seconds':300,
        'native_backend':'cold','experimental_workers':False}
    require(json.loads(path.read_text()) == expected, 'Local continuation configuration changed')
    require(subprocess.check_output(['git','show','HEAD:configs/builtin-local365-v1.json'],cwd=ROOT) == path.read_bytes(),
            'Local config must be exactly committed')
    cfg.update({k:v for k,v in expected.items() if k in cfg})
    # Explicit assignment is intentional: the cloud runner inherited the pilot's 30-day limit.
    cfg['max_days'] = 365
    return cfg


def session_for(path):
    if path is None:
        return None
    from .builtin_preservation import durable_path
    path = durable_path(path)
    value = json.loads(path.read_text())
    require(value.get('schema') == LOCAL_SCOPE and value['mode'] in ('gate','campaign'), 'Invalid local session')
    duration = 1800 if value['mode'] == 'gate' else 7200
    require(value['absolute_deadline_unix'] == value['started_unix']+duration
            and value['compute_deadline_unix'] == value['absolute_deadline_unix']-300
            and value['started_unix'] <= time.time() < value['compute_deadline_unix'], 'Local session expired or changed')
    require(value['parent_policy_sha256'] == PARENT_POLICY, 'Local session must originate from update 31')
    require(value['config'] == binding(ROOT/'configs/builtin-local365-v1.json'), 'Local session config drift')
    require(value['controller'] == binding(ROOT/'tools/local_teacher_campaign.py'), 'Local controller drift')
    require(value['parity_checker'] == binding(ROOT/'tools/check_local365_parity.py'), 'Parity checker drift')
    require(value['authorization'] == AUTHORIZATION, 'Local authorization drift')
    if value['mode'] == 'campaign':
        from .builtin_preservation import DURABLE_ROOT
        require(path == DURABLE_ROOT/'local365/campaign-session.json', 'Only one authorized campaign session path')
        require(value['gate_verified'] is True, 'Local campaign needs fresh both-seat gates')
        migration_bound = value['migration']
        require(binding(migration_bound['path']) == migration_bound, 'Campaign migration drift')
        migration = LocalMigration(migration_bound['path'],ROOT)
        migration.training_gate(migration.value['runtime_transition']['to_runtime'])
    return value


def create_session(path, mode, migration_path):
    from .builtin_preservation import durable_path, DURABLE_ROOT
    path = durable_path(path)
    if mode == 'campaign':
        require(path == DURABLE_ROOT/'local365/campaign-session.json', 'Only one authorized campaign session path')
    require(not path.exists(), 'Never overwrite or restart a local session')
    migration = LocalMigration(migration_path,ROOT)
    if mode == 'campaign': migration.training_gate(migration.value['runtime_transition']['to_runtime'])
    require(mode in ('gate','campaign'), 'Invalid session mode')
    now = time.time(); duration = 1800 if mode == 'gate' else 7200
    value = {'schema':LOCAL_SCOPE,'mode':mode,'started_unix':now,
        'absolute_deadline_unix':now+duration,'compute_deadline_unix':now+duration-300,
        'parent_policy_sha256':PARENT_POLICY,'gate_verified':mode=='campaign',
        'migration':binding(migration_path),'config':binding(ROOT/'configs/builtin-local365-v1.json'),
        'controller':binding(ROOT/'tools/local_teacher_campaign.py'),
        'parity_checker':binding(ROOT/'tools/check_local365_parity.py'),'authorization':AUTHORIZATION}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream: json.dump(value,stream,indent=2); stream.write('\n')
    return value
