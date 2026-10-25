"""Explicit update92 schema conversion; no gameplay, PPO or session creation.

Original v7 files remain immutable. Derived v8 files preserve every named tensor,
Adam/RNG/progress field and archived tensor, with new contract-bound identities.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random

import torch

from . import checkpoints as cp, contextual as c, contextual_model as m
from . import builtin_training_resume as resume
from .builtin_preservation import DURABLE_ROOT, durable_path, verify_pair
from .inference import source_identity
from experiments.headless.builtin_evidence import profile_for

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'heroes-update92-v8-migration-v1'
INTERFACE = 'adventure-capabilities-v1'
PINS = {'policy': '1d6f9a68b61fd60d6c6e49155b2c8b479c0324b2d3c5bd89ddda3e97fbaab344',
        'resume': '1c21b70deaf35de410501cf1b6e1b37768dfc054cbcd23bfe86df6c05f483ce4',
        'plan': 'aa65d37019dc62252fee9d6269719c31d48b9df011fc9886abf59c314835517e',
        'report': 'a76c8e1ec41e7ce4196a1f54c4bb343ee1a1a92a25e2913bf6b94402e05c353a'}
require, binding, exact = resume.require, resume.binding, resume._exact


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _retag(entry, identity):
    require(entry['model_config']['schema'] == m.DIALOGS_MODEL_SCHEMA, 'Expected original v7 model')
    old = cp._model_from_payload(entry)
    target = m.ContextualActorCritic(replace(old.config, schema=m.model_schema_for(INTERFACE)))
    require([(n, p.shape, p.dtype) for n, p in old.named_parameters()] ==
            [(n, p.shape, p.dtype) for n, p in target.named_parameters()], 'Target parameter layout differs')
    cp._validate_weights(target, entry['model_state'])
    target.load_state_dict(entry['model_state'], strict=True)
    return {**deepcopy(entry), 'checkpoint': identity, 'model_config': target.config.to_dict(),
            'weights_sha256': m.weights_sha256(target)}


def convert_payloads(policy, saved, *, metadata, checkpoint):
    """Pure deterministic payload conversion; caller validates source files first."""
    before_torch, before_python = torch.get_rng_state(), random.getstate()
    try:
        profile_for(INTERFACE)
        require(policy['kind'] == 'policy' and policy['resume'] is None and saved['kind'] == 'optimizer',
                'Expected a policy/resume pair')
        require(saved['contracts'] == cp._contracts(c.DIALOGS_ENVIRONMENT), 'Expected exact v7 parent contract')
        for key in set(policy) - {'kind', 'resume'}:
            exact(policy[key], saved[key])
        result = _retag(saved, checkpoint)
        result['contracts'] = cp._contracts(INTERFACE)
        result['source_metadata'] = deepcopy(metadata)
        pool, mappings = [], []
        for old in saved['opponent_pool']:
            identity = old['checkpoint'] + ':derived-v8:' + old['weights_sha256'][:16]
            new = _retag(old, identity)
            pool.append(new)
            mappings.append({'from_checkpoint': old['checkpoint'], 'from_weights_sha256': old['weights_sha256'],
                             'to_checkpoint': identity, 'to_weights_sha256': new['weights_sha256']})
        result['opponent_pool'] = pool
        policy_result = {**deepcopy(result), 'kind': 'policy', 'resume': None}
        return policy_result, result, mappings
    finally:
        torch.set_rng_state(before_torch)
        random.setstate(before_python)


def verify_conversion(old_policy, old_resume, new_policy, new_resume, mappings):
    """Independent field-by-field invariant check, without rerunning conversion."""
    allowed = {'checkpoint', 'contracts', 'model_config', 'weights_sha256', 'source_metadata', 'opponent_pool'}
    for old, new in ((old_policy, new_policy), (old_resume, new_resume)):
        require(set(old) == set(new), 'Checkpoint envelope changed')
        for key in set(old) - allowed:
            exact(old[key], new[key])
        require(new['contracts'] == cp._contracts(INTERFACE), 'Wrong target contracts')
        expected = {**old['model_config'], 'schema': m.model_schema_for(INTERFACE)}
        require(new['model_config'] == expected, 'Model configuration changed beyond its schema')
        require(new['checkpoint'] != old['checkpoint'] and new['weights_sha256'] != old['weights_sha256'],
                'Migration must have distinct contract-bound identities')
        cp._model_from_payload(new)
        require(len(old['opponent_pool']) == len(new['opponent_pool']) == len(mappings), 'Pool membership changed')
        for parent, derived, link in zip(old['opponent_pool'], new['opponent_pool'], mappings, strict=True):
            exact(parent['model_state'], derived['model_state'])
            require(derived['model_config'] == {**parent['model_config'], 'schema': m.model_schema_for(INTERFACE)},
                    'Archived model configuration changed')
            require(link == {'from_checkpoint': parent['checkpoint'], 'from_weights_sha256': parent['weights_sha256'],
                             'to_checkpoint': derived['checkpoint'], 'to_weights_sha256': derived['weights_sha256']}
                    and parent['checkpoint'] != derived['checkpoint'], 'Archived identity mapping differs')
            cp._model_from_payload(derived)
    for key in set(new_policy) - {'kind', 'resume'}:
        exact(new_policy[key], new_resume[key])
    return {'named_tensors_unchanged': True, 'adam_rng_progress_sampler_unchanged': True,
            'archived_pool_tensors_unchanged': True, 'probability_parity_claimed': False,
            'native_opponent': 'Nullkiller Easy', 'archived_neural_pool_executed': False}


def parent_files(plan_path):
    plan_path = durable_path(plan_path)
    plan = json.loads(plan_path.read_text())
    folder = Path(plan['execution_output'])
    paths = {'plan': plan_path, 'report': folder/'report.json',
             'policy': folder/'after-policy.pt', 'resume': folder/'after-resume.pt'}
    for key, path in paths.items():
        require(binding(path)['sha256'] == PINS[key], 'Wrong exact update92 ' + key)
    report = json.loads(paths['report'].read_text())
    verify_pair(paths['resume'], paths['policy'], plan, report)
    return paths, plan, report


def metadata(plan, sha):
    return {**plan['source'], 'plan_sha256': sha, 'initial_checkpoint': plan['parent']['policy'],
        'native_opponent': {k: v for k, v in plan['native_opponents'][0].items() if k != 'player'},
        'archived_neural_pool_executed': False,
        'interface_migration': {'schema': SCHEMA, 'parent': plan['parent'],
                                'from': c.DIALOGS_ENVIRONMENT, 'to': INTERFACE}}


def restore_pair(folder, plan, report):
    """Load the derived pair through normal strict checkpoint APIs, then compare old state."""
    folder = Path(folder)
    original_torch, original_python = torch.get_rng_state(), random.getstate()
    try:
        old_paths, old_plan, _ = parent_files(plan['parent']['plan']['path'])
        require({k: binding(p) for k, p in old_paths.items()} == plan['parent'], 'Parent identities changed')
        new = {kind: torch.load(folder/('after-'+kind+'.pt'), map_location='cpu', weights_only=True)
               for kind in ('policy', 'resume')}
        old = {kind: torch.load(old_paths[kind], map_location='cpu', weights_only=True) for kind in new}
        proof = verify_conversion(old['policy'], old['resume'], new['policy'], new['resume'], report['pool_mapping'])
        expected = metadata(plan, report['plan_sha256'])
        policy = cp.load_policy_checkpoint(folder/'after-policy.pt', expected_source_metadata=expected,
            allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,), interface=INTERFACE)
        model = policy.model.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), **old_plan['configuration']['adam'])
        generator = torch.Generator()
        state = cp.load_checkpoint(folder/'after-resume.pt', model, optimizer,
            expected_ppo_config=old_plan['configuration']['ppo'], expected_source_metadata=expected,
            generators={'minibatch': generator}, allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,),
            interface=INTERFACE, at_rollout_boundary=True)
        exact(dict(model.state_dict()), old['resume']['model_state'])
        exact(optimizer.state_dict(), old['resume']['resume']['optimizer_state'])
        exact(cp._rng_state({'minibatch': generator}, {}), old['resume']['rng'])
        require(vars(state.progress) == {'updates': 92, 'rollouts': 368, 'decisions': 21062}
                and state.sampler_state['optimizer_steps'] == 783, 'Update92 counters changed')
        model.eval().requires_grad_(False)
        links = {'plan': binding(Path(plan['output'])/'plan.json'),
            'report': binding(Path(plan['output'])/'report.json'),
            'policy': binding(folder/'after-policy.pt'), 'resume': binding(folder/'after-resume.pt'),
            'progress': vars(state.progress), 'optimizer_steps': 783,
            'source_proof': {'explicit_schema_migration': SCHEMA}, 'runtime_proof': plan['runtime_transition']}
        return resume.ParentState(model, optimizer, generator, state.opponent_pool, state.progress,
            state.sampler_state, expected, state.checkpoint, policy.weights_sha256, links, 783), proof
    finally:
        torch.set_rng_state(original_torch)
        random.setstate(original_python)


def map_identity(path):
    """The ordinary v2 map has exactly the authored cap1-to0 transition."""
    from experiments.explorer.create_capability_scenarios import verify_scenario
    path = Path(path).resolve()
    verify_scenario(path, fixture=None)
    sha = binding(path)['sha256']
    require(sha == 'ccfc8467847067a75091d68626e5b241839036ab729b5520ce4445a75a45b7b3',
            'Expected exact hero-management-v2 map bytes')
    return {'scenario': 'hero-management-v2', 'path': str(path), 'sha256': sha,
            'layout_seed': 0, 'semantics_verified': True, 'fixture': False}


def prepare(args):
    """Create derived artifacts only; no game engine process is launched."""
    from experiments.headless.env import VcmiLauncher
    from experiments.headless.contextual_evidence import runtime_identity
    from validation.probes.validate_hero_policy import hero_map_identity
    from .builtin_first_run import native_opponent
    from .builtin_v8_session import configuration
    output = durable_path(args.output)
    require(not output.exists(), 'Never overwrite a migration')
    paths, parent, _ = parent_files(args.parent)
    source = source_identity(INTERFACE)
    resume._committed_source(source)
    require(not source['tracked_worktree_dirty'], 'Migration source must be committed')
    launcher = VcmiLauncher(Path(args.runtime_bin)/'vcmiclient', Path(args.data), Path(args.game_map))
    launcher.preflight()
    runtime, native = runtime_identity(Path(args.runtime_bin)), launcher.identity()
    plan = {'schema': SCHEMA, 'output': str(output), 'parent': {k: binding(p) for k, p in paths.items()},
        'source': source, 'runtime': runtime, 'native': native, 'binary': native['binary'],
        'data': str(Path(args.data).resolve()), 'game_map': str(Path(args.game_map).resolve()),
        'map': map_identity(args.game_map), 'replay_binary': binding(Path(args.runtime_bin)/'vcmireplay'),
        'native_opponents': [native_opponent(runtime, seat) for seat in (0, 1)],
        'configuration': configuration(), 'runtime_transition': {'from': parent['runtime'], 'to': runtime},
        'gameplay_started': False, 'probability_parity_claimed': False}
    plan['map_transition'] = {'from': parent['map'], 'to': plan['map'], 'native_level_cap': {'from': 1, 'to': 0}}
    old = {k: torch.load(paths[k], map_location='cpu', weights_only=True) for k in ('policy', 'resume')}
    require(plan['configuration']['ppo'] == parent['configuration']['ppo']
            and plan['configuration']['adam'] == parent['configuration']['adam'], 'Optimizer configuration drift')
    output.mkdir(parents=True)
    write_new(output/'plan.json', plan)
    sha = binding(output/'plan.json')['sha256']
    converted = convert_payloads(old['policy'], old['resume'], metadata=metadata(plan, sha),
                                 checkpoint='update92-derived-v8-'+sha[:16])
    for kind, payload in zip(('policy', 'resume'), converted[:2]):
        with (output/('after-'+kind+'.pt')).open('xb') as stream:
            torch.save(payload, stream)
    report = {'schema': SCHEMA+'-result', 'plan_sha256': sha, 'pool_mapping': converted[2],
        'files': {k: binding(output/('after-'+k+'.pt')) for k in ('policy', 'resume')},
        'gameplay_started': False, 'optimizer_run': False}
    write_new(output/'report.json', report)
    _, proof = restore_pair(output, plan, report)
    import shutil
    backup = output/'independent-copy'
    backup.mkdir()
    for kind in ('policy', 'resume'):
        shutil.copyfile(output/('after-'+kind+'.pt'), backup/('after-'+kind+'.pt'))
        require(binding(backup/('after-'+kind+'.pt'))['sha256'] == report['files'][kind]['sha256'], 'Backup bytes differ')
    _, copied_proof = restore_pair(backup, plan, report)
    require(proof == copied_proof, 'Independent restore differs')
    receipt = {'schema': SCHEMA+'-receipt', 'plan': binding(output/'plan.json'), 'report': binding(output/'report.json'),
        'restore': proof, 'backup_restore': copied_proof, 'backup': {k: binding(backup/('after-'+k+'.pt')) for k in ('policy','resume')},
        'launch_ready': False}
    write_new(output/'migration-receipt.json', receipt)
    return receipt


def load_parent(path):
    path = durable_path(path)
    plan = json.loads(path.read_text())
    require(plan['schema'] == SCHEMA and path == Path(plan['output'])/'plan.json', 'Wrong migration plan')
    require(source_identity(INTERFACE) == plan['source'], 'Migrated target source drift')
    receipt = json.loads((path.parent/'migration-receipt.json').read_text())
    report_path = path.parent/'report.json'
    require(binding(path) == receipt['plan'] and binding(report_path) == receipt['report'], 'Migration receipt drift')
    report = json.loads(report_path.read_text())
    for value in [*report['files'].values(), *receipt['backup'].values()]:
        require(binding(value['path']) == value, 'Derived checkpoint bytes changed')
    state, proof = restore_pair(path.parent, plan, report)
    require(proof == receipt['restore'] == receipt['backup_restore'], 'Migration proof drift')
    # restore_pair is read-only; a training caller explicitly takes ownership of saved RNGs.
    raw = torch.load(report['files']['resume']['path'], map_location='cpu', weights_only=True)
    cp._restore_rng(raw['rng'], {'minibatch': state.generator}, {})
    return state


def verify_private(receipt_path, *, check_remote=False):
    """Restore the immutable private Git blobs, not merely a working-tree copy."""
    import hashlib
    import subprocess
    import tempfile
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    require(receipt['schema'] == SCHEMA+'-private-receipt' and receipt['pushed'] is True, 'Unpublished migration')
    repo = Path(receipt['repository'])
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo, timeout=60)
    require(git('remote','get-url','origin').decode().strip() == 'https://github.com/tkorsi/heroes-rl-training-state.git',
            'Wrong private migration repository')
    commit = receipt['commit']
    require(type(commit) is str and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'Invalid private commit')
    if check_remote:
        remote = git('ls-remote','origin','refs/heads/codex/local365-teacher-results').decode().split()
        require(remote and remote[0] == commit, 'Private migration commit is not the published branch head')
    require(binding(receipt['migration_receipt']['path']) == receipt['migration_receipt'], 'Canonical migration receipt changed')
    with tempfile.TemporaryDirectory(prefix='heroes-v8-private-restore-') as tmp:
        folder = Path(tmp)
        require(set(receipt['files']) == {'plan.json','report.json','migration-receipt.json','after-policy.pt','after-resume.pt'},
                'Private migration snapshot is incomplete')
        for name, value in receipt['files'].items():
            require(value['git_path'].endswith('/'+name) and not Path(value['git_path']).is_absolute()
                    and '..' not in Path(value['git_path']).parts, 'Unsafe private Git path')
            data = git('cat-file','blob',commit+':'+value['git_path'])
            require(hashlib.sha256(data).hexdigest() == value['sha256'] and len(data) == value['bytes'], 'Private Git blob differs')
            (folder/name).write_bytes(data)
        plan = json.loads((folder/'plan.json').read_text())
        report = json.loads((folder/'report.json').read_text())
        canonical = json.loads(Path(receipt['migration_receipt']['path']).read_text())
        require(binding(folder/'plan.json')['sha256'] == canonical['plan']['sha256']
                and binding(folder/'report.json')['sha256'] == canonical['report']['sha256']
                and binding(folder/'migration-receipt.json')['sha256'] == receipt['migration_receipt']['sha256'],
                'Private snapshot does not bind the reviewed canonical migration')
        for kind in ('policy','resume'):
            require(binding(folder/('after-'+kind+'.pt'))['sha256'] == report['files'][kind]['sha256'], 'Private pair differs')
        _, proof = restore_pair(folder, plan, report)
        require(proof == canonical['restore'] == receipt['restore'], 'Private Git-blob restore differs')
        return proof


def preserve_private(plan_path, repo):
    """Publish the derived baseline before any gameplay, then independently restore Git blobs."""
    import shutil
    import subprocess
    from tools.v8_teacher_campaign import preflight_private
    repo = Path(repo).resolve(); plan_path = durable_path(plan_path)
    preflight_private(repo)
    load_parent(plan_path)
    folder = plan_path.parent
    receipt_path = folder/'private-preservation-receipt.json'
    require(not receipt_path.exists(), 'Never replace private migration receipt')
    canonical = json.loads((folder/'migration-receipt.json').read_text())
    target = repo/'checkpoints/local-v8-update92'/('migration-'+binding(plan_path)['sha256'][:16])
    require(not target.exists(), 'Never overwrite private migrated baseline')
    target.mkdir(parents=True)
    names = ('plan.json','report.json','migration-receipt.json','after-policy.pt','after-resume.pt')
    for name in names:
        shutil.copyfile(folder/name, target/name)
        require(binding(folder/name)['sha256'] == binding(target/name)['sha256'], 'Private copy bytes differ')
    plan, report = [json.loads((folder/name).read_text()) for name in ('plan.json','report.json')]
    _, proof = restore_pair(target, plan, report)
    require(proof == canonical['restore'], 'Private copy restore differs')
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo, text=True, timeout=90).strip()
    git('add','--',str(target.relative_to(repo)))
    git('commit','-m','Preserve exact update92 derived v8 baseline before campaign')
    commit = git('rev-parse','HEAD')
    git('push','origin','HEAD:refs/heads/codex/local365-teacher-results')
    receipt = {'schema': SCHEMA+'-private-receipt', 'repository': str(repo), 'commit': commit, 'pushed': True,
        'migration_receipt': binding(folder/'migration-receipt.json'), 'restore': proof,
        'files': {name: {'git_path': str((target/name).relative_to(repo)),
            **{k:v for k,v in binding(target/name).items() if k != 'path'}} for name in names}}
    write_new(receipt_path, receipt)
    verify_private(receipt_path, check_remote=True)
    return receipt
