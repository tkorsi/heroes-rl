"""Benchmark campaign controller for disposable checkpoint copies; ordinary verified PPO.

Mirrors ``tools/overnight_curriculum_campaign.py`` of the Teacher overnight campaign
(origin/codex/teacher-overnight-20261026): per batch it prepares and freezes a plan,
runs the deadline-guarded supervisor and worker, verifies the preserved checkpoint copy,
archives and hash-checks the batch evidence, commits it to a private durability
repository, pushes it and confirms the pushed blob. Its fixed paths are replaced by a
private benchmark root, the private GitHub repository by a local bare repository, and
the acceptance gate and periodic held-out evaluations are omitted (benchmark chains stop
before an evaluation is due). The controller records a monotonic timeline of every batch.

    bench_campaign.py init --campaign DIR --variant NAME --workload easy|original ...
    bench_campaign.py freeze --campaign DIR
    bench_campaign.py release --campaign DIR
    bench_campaign.py launch --campaign DIR --batches N --first-sequence S --seats 0,1,0,1

Nothing here touches the official lineage, the private training-state repository or Teacher.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CAMPAIGN = Path(os.environ['HEROES_BENCH_CAMPAIGN']).resolve() if os.environ.get('HEROES_BENCH_CAMPAIGN') else None


def utc(value=None):
    return datetime.fromtimestamp(time.time() if value is None else value, timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); os.replace(tmp, path)


def _modules():
    from experiments.learner import bench_training as c
    from experiments.learner.builtin_training_resume import binding, require
    return c, binding, require


def remaining(session, reserve=0):
    return min(session['absolute_deadline_unix'] - time.time(),
               session['started_monotonic'] + (session['absolute_deadline_unix']-session['started_unix']) - time.monotonic()) - reserve


def command_timeout(maximum=60):
    c, _, _ = _modules()
    if not c.WINDOW.exists(): return maximum
    value = remaining(json.loads(c.WINDOW.read_text()), 3)
    if value <= 0: raise TimeoutError('Campaign deadline reached')
    return min(maximum, value)


def repo():
    c, _, _ = _modules()
    return Path(c.context()['durable_repo'])


def git(*args, binary=False):
    output = subprocess.check_output(['git', '-C', str(repo()), *args], timeout=command_timeout())
    return output if binary else output.decode().strip()


def private_preflight():
    c, _, require = _modules()
    ctx = c.context()
    require(git('remote', 'get-url', 'origin') == ctx['durable_remote']
            and git('branch', '--show-current') == ctx['durable_branch']
            and git('status', '--porcelain') == '', 'Durability repository not ready')


def event(kind, **values):
    c, _, _ = _modules()
    row = dict(timestamp_utc=utc(), monotonic=time.monotonic(), event=kind, **values)
    for path in (c.BASE/'research-log.jsonl', c.BASE/'validation-log.jsonl'):
        with path.open('a') as f: f.write(json.dumps(row) + '\n'); f.flush(); os.fsync(f.fileno())


def save(v):
    c, _, _ = _modules()
    atomic(c.BASE/'campaign-report.json', v)
    s = json.loads(c.SESSION.read_text())
    admitted = [b for b in v['batches'] if b.get('checkpoint_verified')]
    status = dict(timestamp_utc=utc(), phase=v['status'], source=s['source'], runtime=s['runtime'],
                  admitted_training_games=sum(b['completed_games'] for b in admitted), ppo_updates=v['new_updates'],
                  latest_verified_checkpoint=v['latest_verified_parent'], latest_preserved_checkpoint=v['latest_private_parent'],
                  latest_complete_progress=v['latest_progress'], failed_batches=sum(b.get('status') == 'failed' for b in v['batches']),
                  current_activity={k: v['batches'][-1].get(k) for k in ('sequence', 'seat', 'status')} if v['batches'] else {'mode': 'initializing'})
    atomic(c.BASE/'campaign-status.json', status)


def publish(folder, label, checkpoint=False):
    c, binding, require = _modules()
    private_preflight(); dest = repo()/'results'/c.BASE.name/label
    require(not dest.exists(), 'Private attempt already exists'); dest.mkdir(parents=True)
    archive = dest/'evidence.tar.gz'
    with tarfile.open(archive, 'x:gz', dereference=False) as t: t.add(folder, arcname=folder.name)
    files = links = 0
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            original = folder.parent/m.name
            if m.isfile():
                require(hashlib.sha256(t.extractfile(m).read()).hexdigest() == binding(original)['sha256'], 'Archive file mismatch'); files += 1
            elif m.issym(): require(m.linkname == os.readlink(original), 'Archive symlink mismatch'); links += 1
    receipt = dict(archive=binding(archive), verified_files=files, verified_symlinks=links, external_data_dereferenced=False)
    if checkpoint:
        from experiments.learner.builtin_preservation import verify_pair
        plan = json.loads((folder/'plan.json').read_text()); report = json.loads((folder/'execution/report.json').read_text())
        backup = c.BASE/'checkpoints'/folder.name
        target = repo()/'checkpoints'/c.BASE.name/folder.name
        require(not target.exists(), 'Private checkpoint exists'); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(backup, target)
        receipt['checkpoint_restore'] = verify_pair(target/'after-resume.pt', target/'after-policy.pt', plan, report)
        git('add', '--', str(target.relative_to(repo())))
    c.write_new(dest/'receipt.json', receipt)
    git('add', '--', str(dest.relative_to(repo()))); git('commit', '-q', '-m', 'Preserve benchmark '+label)
    delay = float(os.environ.get('HEROES_BENCH_PUSH_DELAY', '0'))
    if delay > 0: time.sleep(delay)  # Optional emulated network latency, recorded with the batch.
    git('push', '-q', 'origin', 'HEAD:refs/heads/'+c.context()['durable_branch'])
    commit = git('rev-parse', 'HEAD')
    require(git('ls-remote', 'origin', 'refs/heads/'+c.context()['durable_branch']).split()[0] == commit, 'Remote did not acknowledge')
    require(hashlib.sha256(git('show', commit+':'+str(archive.relative_to(repo())), binary=True)).hexdigest() == receipt['archive']['sha256'], 'Private Git blob differs')
    return dict(commit=commit, emulated_push_delay_seconds=delay, **receipt)


def verified_report(report, returncode):
    return (returncode == 0 and report.get('status') == 'finished'
        and all(report.get(k) is True for k in ('worker_complete', 'supervisor_cleanup_verified', 'within_shared_budget'))
        and len(report.get('attempts', [])) == 4
        and all(a.get('status') == 'passed' and all(a.get(k) is True for k in
            ('completion_receipt', 'cleanup_verified', 'identity_verified', 'replay_verified')) for a in report['attempts']))


def run(v, parent, seq, seat):
    c, binding, require = _modules()
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    from experiments.learner.builtin_preservation import verify_pair
    timeline = dict(intent=time.monotonic())
    row = dict(sequence=seq, seat=seat, mode='training', status='preparing', parent=binding(parent), started_utc=utc(), timeline=timeline)
    v['batches'].append(row); save(v); plan_path = None; proc = None; error = None
    event('campaign_batch_intent', sequence=seq, seat=seat, schedule=c.schedule(seq, seat))
    try:
        timeline['prepare_start'] = time.monotonic()
        plan_path = c.prepare(parent, seq, seat); row.update(plan=binding(plan_path), status='running'); save(v)
        timeline['prepared'] = time.monotonic()
        s = c.session(); wall = s['compute_deadline_unix']; mono = s['compute_deadline_monotonic']
        with (plan_path.parent/'controller.log').open('x') as log:
            timeline['supervisor_launch'] = time.monotonic()
            proc = guarded_popen([sys.executable, '-m', 'experiments.learner.bench_training', 'execute',
                '--plan', str(plan_path), '--sha256', row['plan']['sha256']],
                deadline_monotonic=mono, deadline_wall=wall, evidence_path=plan_path.parent/'supervisor-deadline.jsonl',
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            rc = proc.wait(timeout=max(.1, min(wall-time.time(), mono-time.monotonic())))
            timeline['supervisor_exit'] = time.monotonic()
        report_path = plan_path.parent/'execution/report.json'; report = json.loads(report_path.read_text())
        row.update(report=binding(report_path), completed_games=sum(a['status'] == 'passed' for a in report['attempts']),
            outcomes=[a['outcomes'][('red', 'blue')[seat]] for a in report['attempts']],
            optimizer_steps=report['optimizer_steps'], training_samples=report['training_samples'])
        require(verified_report(report, rc), 'Material batch failure: '+str(report.get('stop_reason') or report.get('supervisor_error')))
        require(report['optimizer_updates'] == 1 and report['training_performed'] is True, 'Missing complete PPO update')
        backup = c.BASE/'checkpoints'/plan_path.parent.name; plan = json.loads(plan_path.read_text())
        proof = verify_pair(backup/'after-resume.pt', backup/'after-policy.pt', plan, report)
        row['checkpoint_verified'] = True
        v.update(latest_verified_parent=binding(plan_path), latest_progress=proof['progress'], new_updates=v['new_updates']+1)
        row['status'] = 'verified'; save(v)
        timeline['verified'] = time.monotonic()
    except Exception as exc:
        row.update(status='failed', error=f'{type(exc).__name__}: {exc}'); error = exc
    finally:
        try:
            if proc is not None: close_owned_process(proc, timeout=min(2, command_timeout()))
            if plan_path:
                for p in plan_path.parent.rglob('*deadline*.jsonl'): cleanup_from_evidence(p, timeout=min(2, command_timeout()))
            row['owned_cleanup_verified'] = True
        except Exception as exc: row['cleanup_error'] = str(exc); error = error or exc
        save(v)
        timeline['cleanup_verified'] = time.monotonic()
        if plan_path:
            try:
                row['private_preservation'] = publish(plan_path.parent, c.BASE.name+'-'+plan_path.parent.name,
                    checkpoint=row.get('checkpoint_verified', False))
                if row.get('checkpoint_verified'): v['latest_private_parent'] = binding(plan_path)
            except Exception as exc: row['preservation_error'] = str(exc); error = error or exc
        timeline['acknowledged'] = time.monotonic()
        row['status'] = 'passed' if error is None else 'failed'; row['finished_utc'] = utc(); save(v)
        event('campaign_batch_result', sequence=seq, seat=seat, status=row['status'], timeline=timeline,
              error=row.get('error') or row.get('preservation_error') or row.get('cleanup_error'))
    if error: raise RuntimeError(row.get('error') or row.get('preservation_error') or row.get('cleanup_error')) from error
    return plan_path


def worker(args):
    c, binding, require = _modules()
    s = c.session(); require(os.getppid() == s['controller_pid'], 'Wrong supervisor')
    with (c.BASE/'worker.lock').open('x') as f: f.write(str(os.getpid()))
    parent = c.historical_plan()
    v = dict(status='running', batches=[], new_updates=0, latest_verified_parent=binding(parent),
             latest_private_parent=binding(parent), latest_progress=c.context()['progress'])
    save(v)
    try:
        for index, seat in enumerate(args.seats):
            parent = run(v, parent, args.first_sequence+index, seat)
        v['status'] = 'finished'
    except Exception as exc: v.update(status='stopped_failure', error=f'{type(exc).__name__}: {exc}')
    finally: v['stopped_utc'] = utc(); save(v)


def runtime_report(runtime_bin, output):
    c, binding, require = _modules()
    checked = subprocess.run([sys.executable, str(ROOT/'tools/prepare_vcmi.py'), '--check'], cwd=ROOT,
                             capture_output=True, text=True, timeout=120, check=True)
    files = c.runtime_files(runtime_bin)
    report = dict(passed=True, prepare_check=checked.stdout.strip(),
                  artifacts={name: {k: files[name][k] for k in ('sha256', 'bytes')} for name in
                             ('vcmiclient', 'vcmiserver', 'vcmireplay', 'libvcmi.so', 'AI/libExplorerRL.so', 'AI/libNullkiller.so', 'AI/libBattleAI.so')})
    c.write_new(output, report)
    return binding(output)


def verify_assets(data, output):
    c, binding, require = _modules()
    rows = []
    for path in sorted(Path(data).resolve().rglob('*')):
        require(not path.is_symlink(), 'Game data must not contain links')
        if path.is_file():
            stat = path.stat(); bound = binding(path)
            rows.append(dict(path=bound['path'], sha256=bound['sha256'], bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns))
    c.write_new(output, dict(schema='heroes-benchmark-assets-v1', root=str(Path(data).resolve()), files=rows))
    return binding(output)


def init(args):
    """Write the campaign context; computed once from the frozen inputs and this checkout."""
    from experiments.learner.builtin_training_resume import binding, require
    from experiments.learner.inference import source_identity
    campaign = args.campaign.resolve()
    require(not campaign.exists(), 'Campaign directory must be new')
    receipt_path = args.relocated.resolve()/'relocation-receipt.json'
    receipt = json.loads(receipt_path.read_text())
    plan = json.loads(Path(receipt['plan']['path']).read_text())
    current = source_identity('adventure-capabilities-v1')
    changes = {name: {'from': digest, 'to': current['source_sha256'].get(name)}
               for name, digest in plan['source']['source_sha256'].items() if current['source_sha256'].get(name) != digest}
    maps = {'easy': {'0': args.maps/'curriculum-3-0.vmap', '1': args.maps/'curriculum-3-1.vmap'},
            'original': {'0': args.maps/'hero-management-v2.vmap', '1': args.maps/'hero-management-v2.vmap'}}[args.workload]
    durable = args.durable.resolve()
    branch = 'bench/'+campaign.name
    campaign.mkdir(parents=True)
    ctx = dict(schema='heroes-benchmark-context-v1', campaign=campaign.name, variant=args.variant, workload=args.workload,
               checkout=str(ROOT), runtime_bin=str(args.runtime_bin.resolve()), data=str(args.data.resolve()),
               maps={k: str(Path(p).resolve()) for k, p in maps.items()},
               historical=dict(plan=receipt['plan'], report=receipt['report'], relocation=binding(receipt_path)),
               progress=receipt['progress'], optimizer_steps=receipt['optimizer_steps'],
               reviewed_source_changes=changes, historical_plans_root=str(args.historical_plans.resolve()),
               seed_origin=dict(engine=5100001, policy=5200001), durable_repo=str(durable/'repo'),
               durable_remote=str(durable/'remote.git'), durable_branch=branch,
               hours=args.hours, created_utc=utc(), **({'experiments': sorted(set(args.experiment))} if args.experiment else {}),
               **({'prefix_helpers': args.prefix_helpers} if args.prefix_helpers else {}),
               note='Benchmark-only disposable lineage; never official progress.')
    if not (durable/'remote.git').exists():
        subprocess.run(['git', 'init', '-q', '--bare', str(durable/'remote.git')], check=True)
    if not (durable/'repo').exists():
        subprocess.run(['git', 'init', '-q', str(durable/'repo')], check=True)
        subprocess.run(['git', '-C', str(durable/'repo'), 'remote', 'add', 'origin', str(durable/'remote.git')], check=True)
        for key, value in (('user.name', 'Benchmark'), ('user.email', 'benchmark@localhost')):
            subprocess.run(['git', '-C', str(durable/'repo'), 'config', key, value], check=True)
        (durable/'repo'/'README.md').write_text('Disposable benchmark durability repository.\n')
        subprocess.run(['git', '-C', str(durable/'repo'), 'add', 'README.md'], check=True)
        subprocess.run(['git', '-C', str(durable/'repo'), 'commit', '-q', '-m', 'Initialize'], check=True)
    subprocess.run(['git', '-C', str(durable/'repo'), 'checkout', '-q', '-b', branch], check=True)
    c_path = campaign/'context.json'
    with c_path.open('x') as f: json.dump(ctx, f, indent=2); f.write('\n')
    print(json.dumps(binding(c_path)))


def freeze(args):
    c, binding, require = _modules()
    from experiments.learner.inference import source_identity
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(), 'Freeze only a clean checkout')
    private_preflight()
    ctx = c.context()
    assets = verify_assets(ctx['data'], c.BASE/'assets-verification.json')
    state = c.restore(c.historical_plan())
    require(vars(state.progress) == ctx['progress'] and state.optimizer_steps == ctx['optimizer_steps'], 'Wrong historical state')
    from experiments.headless.contextual_evidence import runtime_identity
    runtime = runtime_identity(Path(ctx['runtime_bin']))
    report = runtime_report(ctx['runtime_bin'], c.BASE/'runtime-check.json')
    for seat in (0, 1):
        c.VcmiLauncher(Path(ctx['runtime_bin'])/'vcmiclient', Path(ctx['data']), Path(ctx['maps'][str(seat)])).preflight()
    names = ['tools/acceleration2/bench_campaign.py', 'tools/acceleration2/relocate_checkpoint.py', 'experiments/explorer/create_curriculum_scenario.py']
    frozen = dict(context=binding(c.CONTEXT_PATH), assets_verification=assets, checkout=str(ROOT), dependency=c.dependency_identity(),
                  source=source_identity(c.INTERFACE), implementation=c.pilot.implementation(), bound_files={n: binding(ROOT/n) for n in names},
                  parent=state.bindings, configuration=c.configuration(), runtime_bin=ctx['runtime_bin'], runtime=runtime,
                  runtime_files=c.runtime_files(ctx['runtime_bin']), runtime_report=report, data=ctx['data'],
                  maps=[c.map_identity(ctx['maps'][str(seat)], seat) for seat in (0, 1)], created_utc=utc())
    path = c.BASE/'frozen.json'; c.write_new(path, frozen)
    event('benchmark_frozen', frozen=binding(path))
    print(json.dumps(binding(path)))


def release(args):
    c, binding, require = _modules()
    from experiments.learner.builtin_preservation import verify_pair
    private_preflight(); frozen = c.check_frozen(); require(not c.SESSION.exists(), 'Consumed campaign cannot restart')
    state = c.restore(c.historical_plan())
    require(state.bindings == frozen['parent'], 'Parent drift')
    plan = json.loads(c.historical_plan().read_text()); report = json.loads(Path(state.bindings['report']['path']).read_text())
    relocated = json.loads(Path(c.context()['historical']['relocation']['path']).read_text())['relocated']
    proof = verify_pair(relocated['resume']['path'], relocated['policy']['path'], plan, report)
    value = dict(frozen, scope=c.SCOPE, authorization='Benchmark-only disposable lineage on the existing Linux allocation; never official progress.',
                 independent_parent_restores=[proof], diagnostic_admission=False)
    path = c.BASE/'release.json'; c.write_new(path, value)
    event('benchmark_release_ready', release=binding(path))
    print(json.dumps(binding(path)))


def launch(args):
    c, binding, require = _modules()
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    private_preflight(); path = c.BASE/'release.json'; r = json.loads(path.read_text()); frozen = c.check_frozen()
    require(r['source'] == frozen['source'] and r['parent'] == c.restore(c.historical_plan()).bindings, 'Release source/parent drift')
    require(not c.SESSION.exists(), 'One campaign only')
    now = time.time(); mono = time.monotonic(); hours = c.context()['hours']
    clock = dict(started_unix=now, started_monotonic=mono, compute_deadline_unix=now+hours*3600-300, absolute_deadline_unix=now+hours*3600)
    c.write_new(c.WINDOW, clock)
    s = dict(scope=c.SCOPE, source=r['source'], runtime=r['runtime'], release=binding(path), window=binding(c.WINDOW), **clock,
             compute_deadline_monotonic=mono+hours*3600-300, hard_deadline_monotonic=mono+hours*3600, controller_pid=os.getpid(),
             attempt_started_utc=utc())
    c.write_new(c.SESSION, s); event('campaign_launched', session=binding(c.SESSION))
    proc = None; result = dict(session=binding(c.SESSION))
    seats = ','.join(str(x) for x in args.seats)
    try:
        with (c.BASE/'controller.log').open('x') as log:
            proc = guarded_popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--seats', seats,
                                  '--first-sequence', str(args.first_sequence)],
                deadline_monotonic=s['compute_deadline_monotonic'], deadline_wall=s['compute_deadline_unix'],
                evidence_path=c.BASE/'controller-deadline.jsonl', cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            result['returncode'] = proc.wait(timeout=max(.1, remaining(s, 90)))
    except Exception as exc: result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        errors = []
        try:
            if proc is not None: close_owned_process(proc, timeout=2)
            for p in (c.BASE/'batches').rglob('*deadline*.jsonl'): cleanup_from_evidence(p, timeout=1)
        except Exception as exc: errors.append(str(exc))
        result.update(cleanup_errors=errors, owned_cleanup_verified=not errors, stopped_utc=utc())
        atomic(c.BASE/'controller-result.json', result)
        event('campaign_stopped', result=result)
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='mode', required=True)
    i = sub.add_parser('init')
    for name in ('campaign', 'relocated', 'runtime-bin', 'data', 'maps', 'historical-plans', 'durable'):
        i.add_argument('--'+name, type=Path, required=True)
    i.add_argument('--variant', required=True); i.add_argument('--workload', choices=['easy', 'original'], required=True)
    i.add_argument('--hours', type=float, default=3.0)
    i.add_argument('--experiment', action='append', default=[], choices=['shared_graph_encodings'],
                   help='select a separate numerical experiment for this campaign (never a default)')
    i.add_argument('--prefix-helpers', type=int, default=0, help='exact forked prefix-encoding helpers bound in every plan')
    for name in ('freeze', 'release'): sub.add_parser(name)
    for name in ('launch', 'worker'):
        x = sub.add_parser(name); x.add_argument('--seats', type=lambda s: [int(v) for v in s.split(',')], required=True)
        x.add_argument('--first-sequence', type=int, required=True)
    a = p.parse_args()
    {'init': init, 'freeze': freeze, 'release': release, 'launch': launch, 'worker': worker}[a.mode](a)


if __name__ == '__main__':
    main()
