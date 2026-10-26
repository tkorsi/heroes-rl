"""Benchmark-only continuation of disposable checkpoint copies; unchanged PPO/admission worker.

Mirrors ``experiments/learner/overnight_training.py`` of the Teacher overnight campaign
(origin/codex/teacher-overnight-20261026): the same frozen-release checks, exact parent
restore, plan freezing, seed-consumption scan, per-call identity checks and worker. Its
fixed /workspace paths are replaced by an explicit private benchmark campaign directory
named by ``HEROES_BENCH_CAMPAIGN``. Benchmark plans, games, updates and checkpoints are
disposable measurements; they never enter or advance the official training lineage.

The first parent of every benchmark campaign is a relocated copy of an archived official
checkpoint (``tools/acceleration2/relocate_checkpoint.py``); its restore is compared
field by field with the original archived bytes.
"""
import argparse
from copy import deepcopy
import json, os, random, subprocess, sys, time
from pathlib import Path
import torch
from . import builtin_first_run as pilot, builtin_training as batches, builtin_training_resume as resume, checkpoints as cp
from .builtin_preservation import binding, preserve
from .inference import source_identity
from experiments.headless import digest_scope
from experiments.headless.contextual_evidence import runtime_identity
from experiments.headless.env import VcmiLauncher

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(os.environ.get('HEROES_BENCH_CAMPAIGN', '/nonexistent-benchmark-campaign')).resolve()
CONTEXT_PATH = BASE/'context.json'
WINDOW = BASE/'window.json'
SESSION = BASE/'session.json'
INTERFACE = 'adventure-capabilities-v1'
SCOPE = 'heroes-benchmark-phase2-20261026'
PERIODIC_ORIGIN = 146
PERIODIC_EVERY = 5
# Separate numerical experiments a benchmark campaign may select; never defaults (see ppo.EXPERIMENTS).
EXPERIMENTS = ('shared_graph_encodings',)
CORE = ('contextual.py', 'contextual_model.py', 'contextual_rollout.py', 'checkpoints.py', 'ppo.py',
        'builtin_gameplay.py', 'builtin_rollout.py', 'trajectory.py', 'contract.py')
require = resume.require


def context():
    return json.loads(CONTEXT_PATH.read_text())


def latest_periodic_update(update):
    require(type(update) is int and update >= PERIODIC_ORIGIN, 'Invalid curriculum evaluation progress')
    return PERIODIC_ORIGIN+((update-PERIODIC_ORIGIN)//PERIODIC_EVERY)*PERIODIC_EVERY


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, indent=2); f.write('\n')


def experiments():
    chosen = context().get('experiments', [])
    require(type(chosen) is list and chosen == sorted(set(chosen)) and set(chosen) <= set(EXPERIMENTS), 'Unknown numerical experiment')
    return chosen


def prefix_helpers():
    # Exact execution choice (ppo.PREFIX_HELPERS): bound in the plan, not a numerical experiment.
    helpers = context().get('prefix_helpers', 0)
    require(type(helpers) is int and 0 <= helpers <= 15, 'Invalid prefix helper count')
    return helpers


def historical_plan():
    return Path(context()['historical']['plan']['path'])


def configuration():
    return deepcopy(json.loads(historical_plan().read_text())['configuration'])


def map_identity(path, seat):
    path = Path(path).resolve(); ctx = context()
    require(seat in (0, 1) and path == Path(ctx['maps'][str(seat)]).resolve(), 'Only the preregistered benchmark map')
    if ctx['workload'] == 'easy':
        from experiments.explorer.create_curriculum_scenario import verify
        verify(path, 3, seat)
        return dict(binding(path), scenario='curriculum-v1-open8x8', stage=3, seat=seat)
    require(ctx['workload'] == 'original', 'Unknown benchmark workload')
    from .builtin_v8_migration import map_identity as original
    return dict(original(path), seat=seat)


def schedule(sequence, seat):
    require(type(sequence) is int and sequence >= 0 and seat in (0, 1), 'Invalid schedule')
    origin = context()['seed_origin']
    return [dict(episode=i+1, phase='training', learner_player=seat,
                 engine_seed=origin['engine']+4*sequence+i, policy_seed=origin['policy']+4*sequence+i) for i in range(4)]


def dependency_identity():
    import hashlib
    repo = ROOT/'vcmi'
    require(repo.is_dir() and not repo.is_symlink(), 'Independent generated engine required')
    def git(*args): return subprocess.check_output(['git', '-C', str(repo), *args])
    head = git('rev-parse', 'HEAD').decode().strip()
    require(head == json.loads((ROOT/'dependencies/vcmi.json').read_text())['commit'], 'Pinned engine HEAD changed')
    untracked = [p for p in git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0') if p]
    return dict(head=head, diff_sha256=hashlib.sha256(git('diff', 'HEAD', '--binary')).hexdigest(),
                untracked={p: binding(repo/p)['sha256'] for p in sorted(untracked)})


def runtime_files(directory):
    directory = Path(directory).resolve()
    files = {}
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink(), 'Frozen Linux runtime cannot retain mutable external links')
        if path.is_file(): files[str(path.relative_to(directory))] = binding(path)
    require(all(name in files for name in ('vcmiclient', 'vcmiserver', 'vcmireplay', 'libvcmi.so', 'AI/libExplorerRL.so',
                                           'AI/libNullkiller.so', 'AI/libBattleAI.so')), 'Incomplete executable runtime')
    return files


def frozen_inputs(frozen):
    """Files one identity check reads, for concurrent prefetch; the check itself still compares every value."""
    runtime = Path(frozen['runtime_bin'])
    return [*runtime.rglob('*'), *(ROOT/name for name in frozen['implementation']), *(ROOT/name for name in frozen['source']['source_sha256']),
            *(Path(m['path']) for m in frozen['maps']), *(Path(v['path']) for v in frozen['parent'].values() if type(v) is dict and 'path' in v)]


def check_frozen():
    with digest_scope.scope():
        return _check_frozen()


def _check_frozen():
    frozen = json.loads((BASE/'frozen.json').read_text())
    digest_scope.prefetch(frozen_inputs(frozen))
    require(binding(CONTEXT_PATH) == frozen['context'], 'Benchmark context drift')
    require(dependency_identity() == frozen['dependency'], 'Generated dependency drift')
    asset_proof = frozen['assets_verification']; require(binding(asset_proof['path']) == asset_proof, 'Private asset verification receipt drift')
    for asset in json.loads(Path(asset_proof['path']).read_text())['files']:
        path = Path(asset['path']); stat = path.stat()
        require(not path.is_symlink() and {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}
                == {k: asset[k] for k in ('bytes', 'mtime_ns', 'ctime_ns')}, 'Original private asset changed since full hash verification')
    require(str(ROOT) == frozen['checkout'], 'Run only from the frozen checkout')
    require(source_identity(INTERFACE) == frozen['source'] and not frozen['source']['tracked_worktree_dirty'], 'Frozen source drift')
    require(pilot.implementation() == frozen['implementation'], 'Frozen implementation drift')
    require(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip() == '', 'Frozen checkout must remain clean')
    for name, value in frozen['bound_files'].items(): require(binding(ROOT/name) == value, 'Frozen tool/config drift')
    require(configuration() == frozen['configuration'], 'Frozen configuration drift')
    require(runtime_identity(Path(frozen['runtime_bin'])) == frozen['runtime'], 'Frozen runtime drift')
    require(runtime_files(frozen['runtime_bin']) == frozen['runtime_files'], 'Frozen client/server/module/resource tree drift')
    require(binding(frozen['runtime_report']['path']) == frozen['runtime_report'], 'Native build/check receipt drift')
    require([map_identity(context()['maps'][str(seat)], seat) for seat in (0, 1)] == frozen['maps'], 'Frozen map drift')
    for key in ('plan', 'report', 'policy', 'resume'): require(binding(frozen['parent'][key]['path']) == frozen['parent'][key], 'Frozen initial checkpoint drift')
    return frozen


def session():
    with digest_scope.scope():
        return _session()


def _session():
    check_frozen()
    v = json.loads(SESSION.read_text()); clock = json.loads(WINDOW.read_text())
    require(binding(WINDOW) == v['window'], 'Benchmark clock binding changed')
    require(all(v[k] == clock[k] for k in ('started_unix', 'started_monotonic', 'compute_deadline_unix', 'absolute_deadline_unix')), 'Recovery cannot reset benchmark clock')
    require(v['scope'] == SCOPE and v['source'] == source_identity(INTERFACE), 'Campaign source drift')
    require(time.time() < v['compute_deadline_unix'] and 0 <= time.monotonic()-v['started_monotonic'] < v['compute_deadline_unix']-v['started_unix'], 'Benchmark compute expired')
    require(binding(v['release']['path']) == v['release'], 'Release drift')
    release = json.loads(Path(v['release']['path']).read_text())
    require(release['source'] == v['source'] and release['configuration'] == configuration(), 'Release source/config drift')
    return v


def restore(path):
    """Exact learned-state restore; the first parent is a relocated archived checkpoint."""
    path = Path(path).resolve(); plan = json.loads(path.read_text()); ctx = context()
    historical = path == historical_plan()
    if historical:
        expected = ctx['historical']
        report_path = Path(expected['report']['path'])
        require(binding(path) == expected['plan'] and binding(report_path) == expected['report'], 'Historical parent differs')
        report = json.loads(report_path.read_text()); require(report['plan_sha256'] == binding(path)['sha256'], 'Parent report mismatch')
        resume._committed_source(plan['source'])
        current = source_identity(INTERFACE)
        changes = {name: {'from': digest, 'to': current['source_sha256'].get(name)}
                   for name, digest in plan['source']['source_sha256'].items() if current['source_sha256'].get(name) != digest}
        require(changes == ctx['reviewed_source_changes'], 'Unreviewed learner source change for this benchmark')
        require(plan['source']['interface'] == INTERFACE, 'Learned schema changed')
        relocation = expected['relocation']; require(binding(relocation['path']) == relocation, 'Relocation receipt changed')
        receipt = json.loads(Path(relocation['path']).read_text())
        for kind in ('policy', 'resume'):
            original = binding(receipt['original'][kind]['path'])
            require(original == receipt['original'][kind] and
                    {k: original[k] for k in ('sha256', 'bytes')} == {k: report['after_checkpoint'][kind][k] for k in ('sha256', 'bytes')},
                    'Archived checkpoint differs from its completion report')
        saved = {kind: binding(receipt['relocated'][kind]['path']) for kind in ('policy', 'resume')}
        for kind in saved: require(saved[kind] == receipt['relocated'][kind], 'Relocated benchmark copy changed')
        raw = torch.load(receipt['original']['resume']['path'], map_location='cpu', weights_only=True)
    else:
        report_path = Path(plan['execution_output'])/'report.json'
        report = json.loads(report_path.read_text()); require(report['plan_sha256'] == binding(path)['sha256'], 'Parent report mismatch')
        require(path.is_relative_to(BASE/'batches') and plan['scope'] == SCOPE, 'Foreign benchmark parent')
        require(plan['source']['torch'] == str(torch.__version__) and plan['source']['python'] == sys.version, 'Unpinned benchmark runtime change')
        saved = {kind: report['after_checkpoint'][kind] for kind in ('policy', 'resume')}
        for value in saved.values(): require(binding(value['path']) == value, 'Benchmark checkpoint changed')
        for name in CORE:
            require(binding(ROOT/'experiments/learner'/name)['sha256'] == plan['source']['source_sha256']['experiments/learner/'+name], 'Policy/PPO implementation changed')
        raw = None
    metadata = resume._metadata(plan, report['plan_sha256']); progress, steps, sampler = resume._completion(plan, report)
    policy = cp.load_policy_checkpoint(saved['policy']['path'], expected_source_metadata=metadata,
        allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,), interface=INTERFACE)
    model = policy.model.requires_grad_(True); optimizer = torch.optim.Adam(model.parameters(), **plan['configuration']['adam']); generator = torch.Generator()
    state = cp.load_checkpoint(saved['resume']['path'], model, optimizer, expected_ppo_config=plan['configuration']['ppo'],
        expected_source_metadata=metadata, allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,), interface=INTERFACE,
        generators={'minibatch': generator}, at_rollout_boundary=True)
    raw = raw if raw is not None else torch.load(saved['resume']['path'], map_location='cpu', weights_only=True)
    for a, b in [(dict(model.state_dict()), raw['model_state']), (optimizer.state_dict(), raw['resume']['optimizer_state']),
                 (torch.get_rng_state(), raw['rng']['torch_cpu']), (random.getstate(), raw['rng']['python']),
                 (generator.get_state(), raw['rng']['torch_generators']['minibatch']),
                 (state.opponent_pool.state_dict(), policy.opponent_pool.state_dict())]: resume._exact(a, b)
    require(state.progress == progress and state.sampler_state == sampler, 'Parent progress drift')
    require(cp.model_weight_digest(model) == policy.weights_sha256 == report['after_checkpoint']['weights_sha256'], 'Learned weights differ')
    if historical: require(vars(progress) == ctx['progress'] and steps == ctx['optimizer_steps'], 'Historical progress differs')
    links = dict(plan=binding(path), report=binding(report_path), policy=saved['policy'], resume=saved['resume'],
                 progress=vars(progress), optimizer_steps=steps,
                 source_proof={'explicit_benchmark_transition': historical, 'policy_ppo_unchanged': not historical},
                 runtime_proof={'benchmark_relocation': ctx['historical']['relocation']} if historical else {'runtime_unchanged': True})
    model.eval().requires_grad_(False)
    return resume.ParentState(model, optimizer, generator, state.opponent_pool, progress, sampler, metadata, policy.checkpoint, policy.weights_sha256, links, steps)


def check_identity(plan):
    # One scope: each unchanged file is hashed once per complete check, never across checks.
    with digest_scope.scope():
        _check_identity(plan)


def _check_identity(plan):
    v = session(); require(plan['scope'] == SCOPE and plan['source'] == v['source'] and not v['source']['tracked_worktree_dirty'], 'Unreleased source')
    require(plan['implementation'] == pilot.implementation() and plan['prerequisites'] == pilot.prerequisites(), 'Implementation drift')
    require(plan['configuration'] == configuration(), 'PPO/configuration drift')
    seat = plan['curriculum']['seat']; require(plan['map'] == map_identity(plan['game_map'], seat), 'Map drift')
    require(plan['schedule'] == schedule(plan['batch_sequence'], seat), 'Schedule drift')
    require(plan['local_session'] == binding(SESSION), 'Wrong session')
    for k in ('checkpoint', 'replay_binary'): require(binding(plan[k]['path']) == plan[k], 'Artifact changed')
    launcher = VcmiLauncher(Path(plan['binary']), Path(plan['data']), Path(plan['game_map']))
    require(launcher.identity() == plan['native'] and runtime_identity(Path(plan['binary']).parent) == plan['runtime'], 'Native drift')
    release = json.loads(Path(v['release']['path']).read_text()); require(plan['runtime'] == release['runtime'], 'Unreleased runtime')
    require(plan['schema'] == batches.SCHEMA+'-plan', 'Schema drift')
    require(plan['native_opponents'] == [pilot.native_opponent(release['runtime'], s) for s in (0, 1)], 'Opponent drift')
    for key in ('plan', 'report', 'policy', 'resume'): require(binding(plan['parent'][key]['path']) == plan['parent'][key], 'Parent artifact drift')
    require(plan['absolute_deadline_unix'] == v['absolute_deadline_unix'], 'Extended batch deadline')
    require(plan['compute_deadline_unix'] == v['compute_deadline_unix'], 'Changed compute budget')
    require(plan.get('experiments', []) == experiments() and plan['checkpoint_metadata'].get('numerical_experiments', []) == experiments(),
            'Numerical experiment selection changed')
    require(plan.get('prefix_helpers', 0) == prefix_helpers(), 'Prefix helper selection changed')


def prepare(parent, sequence, seat):
    v = session()
    # The canonical prepared-checkout check runs while the plan is built; no plan is written unless it passed.
    canonical = subprocess.Popen([sys.executable, str(ROOT/'tools/prepare_vcmi.py'), '--check'], cwd=ROOT,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        plan = _prepare(v, parent, sequence, seat)
        out, err = canonical.communicate(timeout=60)
        if canonical.returncode:
            raise subprocess.CalledProcessError(canonical.returncode, canonical.args, out, err)
    finally:
        if canonical.poll() is None:
            canonical.kill(); canonical.wait()
    output = BASE/'batches'/f'train-{sequence:04d}-seat{seat}'
    write_new(output/'plan.json', plan); return output/'plan.json'


def _prepare(v, parent, sequence, seat):
    state = restore(parent); release = json.loads(Path(v['release']['path']).read_text())
    output = BASE/'batches'/f'train-{sequence:04d}-seat{seat}'
    require(not output.exists(), 'No retry or overwrite')
    rows = schedule(sequence, seat)
    keys = {(r['engine_seed'], r['policy_seed'], r['learner_player']) for r in rows}
    for f in [*(BASE/'batches').rglob('plan.json'), *Path(context()['historical_plans_root']).rglob('plan.json')]:
        old = json.loads(f.read_text()); require(not keys & {(r['engine_seed'], r['policy_seed'], r['learner_player']) for r in old['schedule'] if r['phase'] == 'training'}, 'Consumed seeds')
    game_map = Path(context()['maps'][str(seat)]); binary = Path(release['runtime_bin'])/'vcmiclient'
    native = VcmiLauncher(binary, Path(release['data']), game_map).identity()
    plan = dict(schema=batches.SCHEMA+'-plan', scope=SCOPE, mode='training',
        source=v['source'], runtime=release['runtime'], native=native, map=map_identity(game_map, seat), binary=str(binary), data=release['data'],
        game_map=str(game_map), replay_binary=binding(binary.parent/'vcmireplay'), native_opponents=[pilot.native_opponent(release['runtime'], s) for s in (0, 1)],
        configuration=configuration(), schedule=rows, batch_sequence=sequence, execution_output=str(output/'execution'),
        parent=state.bindings, checkpoint=state.bindings['policy'], local_session=binding(SESSION), authorization=release['authorization'],
        absolute_deadline_unix=v['absolute_deadline_unix'], compute_deadline_unix=v['compute_deadline_unix'],
        implementation=pilot.implementation(), prerequisites=pilot.prerequisites(),
        wall_scope='Benchmark campaign window; disposable measurement, never official progress',
        comparison_scope='Benchmark '+context()['workload']+' workload; paired seeds across code variants',
        curriculum={'stage': 3, 'seat': seat, 'map_change_explicit': True, 'workload': context()['workload']}, final_evaluation=False)
    plan['checkpoint_metadata'] = {**v['source'], 'parent_plan_sha256': state.bindings['plan']['sha256'],
        'parent_report_sha256': state.bindings['report']['sha256'], 'initial_checkpoint': state.bindings['policy'],
        'native_opponent': {k: x for k, x in plan['native_opponents'][0].items() if k != 'player'}, 'archived_neural_pool_executed': False}
    if experiments():
        # The plan, and through its metadata every checkpoint it produces, names the experiment.
        plan['experiments'] = plan['checkpoint_metadata']['numerical_experiments'] = experiments()
    if prefix_helpers():
        plan['prefix_helpers'] = prefix_helpers()
    check_identity(plan); return plan


def read_plan(path, sha, *, check=True):
    path = Path(path); require(path.is_relative_to(BASE/'batches') and binding(path)['sha256'] == sha, 'Plan changed')
    plan = json.loads(path.read_text()); require(plan['execution_output'] == str(path.parent/'execution'), 'Output drift')
    if check: check_identity(plan)
    return plan


def heldout(state, runtime, native):
    # Benchmark chains stop before the official fifth-update held-out rule would require an evaluation.
    session()
    require(latest_periodic_update(state.progress.updates) <= context()['progress']['updates'],
            'A due held-out evaluation would be required before more PPO; benchmark chain too long')


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['execute', 'worker']); p.add_argument('--plan', type=Path, required=True); p.add_argument('--sha256', required=True)
    p.add_argument('--started', type=float); p.add_argument('--wall-started', type=float); a = p.parse_args()
    if a.command == 'worker':
        def reader(path, sha):
            # builtin_training.worker checks the identity first thing, before any use of the plan.
            plan = read_plan(path, sha, check=False)
            from . import ppo
            ppo.EXPERIMENTS = frozenset(plan.get('experiments', ()))
            ppo.PREFIX_HELPERS = plan.get('prefix_helpers', 0)
            return plan
        result = batches.worker(a, plan_reader=reader, state_reader=lambda plan: restore(plan['parent']['plan']['path']), heldout_checker=heldout)
    else:
        result = pilot.execute(a, plan_reader=read_plan, worker_module=__spec__.name)
        if result['status'] == 'finished' and result['training_performed']:
            preserve(a.plan, a.plan.parent/'execution/report.json', BASE/'checkpoints'/a.plan.parent.name)
    print(json.dumps({k: result.get(k) for k in ('status', 'stop_reason', 'optimizer_updates', 'optimizer_steps', 'training_samples')}))


if __name__ == '__main__': main()
