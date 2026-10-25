#!/usr/bin/env python3
"""Explicitly released, single-use two-hour v8 controller. Preparation never launches."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.learner import builtin_v8_session as scope, builtin_v8_migration as migration
from experiments.learner.builtin_preservation import DURABLE_ROOT, verify_pair
from experiments.learner.builtin_training_resume import binding, require


def invoke(command, log, deadline):
    with log.open('x') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=max(1, deadline-time.time()))
    require(result.returncode == 0, 'Command failed; inspect '+str(log))


def preflight_private(repo):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo, text=True, timeout=30).strip()
    require(git('remote','get-url','origin') == 'https://github.com/tkorsi/heroes-rl-training-state.git', 'Wrong private repository')
    require(git('branch','--show-current') == 'codex/local365-teacher-results' and not git('status','--porcelain'),
            'Private preservation checkout must be clean and on the authorized result branch')


def publish_run(plan_path, repo, deadline):
    """Retain every attempt, including failed and evaluation runs, in the private repository."""
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo, text=True,
            timeout=max(1, min(60, deadline-time.time()))).strip()
    require(git('remote','get-url','origin') == 'https://github.com/tkorsi/heroes-rl-training-state.git', 'Wrong private repository')
    require(git('branch','--show-current') == 'codex/local365-teacher-results' and not git('status','--porcelain'),
            'Private result branch must be clean')
    plan = json.loads(plan_path.read_text())
    report = json.loads((Path(plan['execution_output'])/'report.json').read_text())
    entries = []
    if report['status'] == 'finished' and report['training_performed']:
        backup = DURABLE_ROOT/'checkpoints'/plan_path.parent.name
        target = repo/'checkpoints/local-v8-update92'/plan_path.parent.name
        require(not target.exists(), 'Private checkpoint already exists')
        proof = verify_pair(backup/'after-resume.pt', backup/'after-policy.pt', plan, report)
        shutil.copytree(backup, target)
        require(verify_pair(target/'after-resume.pt', target/'after-policy.pt', plan, report) == proof,
                'Private copy independent restore differs')
        entries.append(str(target.relative_to(repo)))
    archive = repo/'results/local-v8-update92'/(plan_path.parent.name+'.tar.gz')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'x:gz', dereference=False) as output:
        output.add(plan_path.parent, arcname=plan_path.parent.name)
    receipt = archive.with_suffix('.json')
    migration.write_new(receipt, {'plan': binding(plan_path), 'report': binding(Path(plan['execution_output'])/'report.json'),
        'archive': binding(archive), 'status': report['status'], 'game_data_symlinks_followed': False})
    entries.extend((str(archive.relative_to(repo)), str(receipt.relative_to(repo))))
    git('add', '--', *entries)
    git('commit', '-m', 'Preserve v8 Teacher '+plan_path.parent.name)
    git('push', 'origin', 'HEAD:refs/heads/codex/local365-teacher-results')
    return git('rev-parse', 'HEAD')


def run_batch(parent, sequence, evaluate, session):
    from experiments.learner.builtin_v8_training import unused_schedule
    from experiments.learner.builtin_training_resume import _completion
    raw = json.loads(parent.read_text())
    if raw['schema'] == migration.SCHEMA:
        update = 92
    else:
        report = json.loads((Path(raw['execution_output'])/'report.json').read_text())
        update = _completion(raw, report)[0].updates
    name = f'v8-update92-heldout-{update:03d}' if evaluate else f'v8-update92-continuation-{sequence:03d}'
    output = DURABLE_ROOT/'runs'/name
    require(not output.exists(), 'Never retry an already prepared v8 batch')
    if not evaluate: unused_schedule(sequence)
    logs = scope.SESSION.parent/'logs'; logs.mkdir(exist_ok=True)
    command = [sys.executable, '-m', 'experiments.learner.builtin_v8_training', 'prepare',
        '--parent', str(parent), '--session', str(scope.SESSION), '--output', str(output)]
    command += ['--evaluate'] if evaluate else ['--sequence', str(sequence)]
    invoke(command, logs/(name+'-prepare.log'), session['compute_deadline_unix'])
    plan = output/'plan.json'
    invoke([sys.executable, '-m', 'experiments.learner.builtin_v8_training', 'execute',
        '--plan', str(plan), '--sha256', binding(plan)['sha256']], logs/(name+'-execute.log'), session['compute_deadline_unix']+60)
    return plan, json.loads((output/'execution/report.json').read_text())


def journal(value):
    path = scope.SESSION.parent/'campaign-report.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temporary, path)


def verify_local_update(plan_path, report):
    """Keep locally admitted/restored state distinct from private publication."""
    from experiments.learner.builtin_training_resume import _completion
    plan = json.loads(plan_path.read_text())
    require(report == json.loads((Path(plan['execution_output'])/'report.json').read_text())
            and report['plan_sha256'] == binding(plan_path)['sha256'], 'Completed report binding differs')
    progress, _, _ = _completion(plan, report)
    saved = report['after_checkpoint']
    for kind in ('policy', 'resume'):
        require(binding(saved[kind]['path']) == saved[kind], 'Verified checkpoint bytes changed')
    canonical = verify_pair(saved['resume']['path'], saved['policy']['path'], plan, report)
    backup = DURABLE_ROOT/'checkpoints'/plan_path.parent.name
    receipt_path = backup/'preservation-receipt.json'
    receipt = json.loads(receipt_path.read_text())
    require(receipt['sources']['plan.json'] == binding(plan_path)
            and receipt['sources']['report.json'] == binding(Path(plan['execution_output'])/'report.json'),
            'Independent preservation receipt belongs to another completed update')
    for kind in ('policy','resume'):
        require(receipt['sources']['after-'+kind+'.pt'] == saved[kind]
                and binding(backup/('after-'+kind+'.pt'))['sha256'] == saved[kind]['sha256'],
                'Independent checkpoint copy differs')
    restored = verify_pair(backup/'after-resume.pt', backup/'after-policy.pt', plan, report)
    require(canonical == restored == receipt['restore'], 'Independent local restore differs')
    return {'plan': binding(plan_path), 'policy': saved['policy'], 'resume': saved['resume'],
            'progress': vars(progress), 'restore': canonical, 'preservation_receipt': binding(receipt_path)}


def publish_failure(plan_path, repo, deadline, error, schedule, *, verified_checkpoint=None):
    """Retain failed attempts or publication failures with their actual admission state."""
    folder = plan_path.parent
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder/'execution/report.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    failure = {'schema': scope.SCOPE+'-failed-attempt', 'error': error,
        'intended_plan': str(plan_path), 'plan': binding(plan_path) if plan else None,
        'report': binding(report_path) if report else None,
        'attempts': report.get('attempts', []) if report else
            [dict(row, status='not_started', reason='controller_failure_before_report') for row in (plan or {}).get('schedule', schedule)],
        'checkpoint_admitted': verified_checkpoint is not None, 'verified_checkpoint': verified_checkpoint,
        'preservation_status': 'private_publication_failed' if verified_checkpoint is not None else 'failed_or_incomplete'}
    failure_path = folder/'controller-failure.json'
    migration.write_new(failure_path, failure)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo, text=True,
            timeout=max(.1, min(30, deadline-time.time()))).strip()
    preflight_private(repo)
    archive = repo/'results/local-v8-update92'/(folder.name+'-failed.tar.gz')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'x:gz', dereference=False) as output:
        output.add(folder, arcname=folder.name)
        for log in (scope.SESSION.parent/'logs').glob(folder.name+'-*.log'):
            output.add(log, arcname=folder.name+'/controller-logs/'+log.name)
    receipt = archive.with_suffix('.json')
    migration.write_new(receipt, {'failure': binding(failure_path), 'archive': binding(archive),
        'checkpoint_admitted': verified_checkpoint is not None, 'verified_checkpoint': verified_checkpoint})
    git('add','--',str(archive.relative_to(repo)),str(receipt.relative_to(repo)))
    git('commit','-m','Preserve failed v8 Teacher attempt '+folder.name)
    git('push','origin','HEAD:refs/heads/codex/local365-teacher-results')
    return git('rev-parse','HEAD')


def campaign(args):
    session = scope.session_for(scope.SESSION)
    require(os.getppid() == session['controller_pid'], 'Campaign worker requires its original live supervisor')
    with (scope.SESSION.parent/'campaign-worker.lock').open('x') as stream:
        stream.write(str(os.getpid())+'\n')
    release = scope.validate_release(session['release']['path'])
    receipt = scope.checked(release['migration_receipt'])
    parent = Path(receipt['plan']['path'])
    result = {'schema': scope.SCOPE+'-campaign', 'session': binding(scope.SESSION), 'status': 'running', 'batches': []}
    durations, sequence, update, evaluated = [300.0], 94, 92, set()
    pending = None
    result.update(latest_verified_parent=binding(parent), latest_private_parent=binding(parent),
                  latest_verified_progress={'updates':92,'rollouts':368,'decisions':21062})
    journal(result)
    try:
        while True:
            scope.session_for(scope.SESSION)
            if session['compute_deadline_unix']-time.time() < max(300, 2*max(durations[-5:])+120):
                result['status'] = 'stopped_for_deadline_reserve'; break
            evaluate = (update == 92 or update % 5 == 0) and update not in evaluated
            if not evaluate:
                from experiments.learner.builtin_v8_training import unused_schedule
                while True:
                    try: unused_schedule(sequence); break
                    except ValueError:
                        sequence += 1
                        require(sequence < 100000, 'No unused training sequence')
            name = f'v8-update92-heldout-{update:03d}' if evaluate else f'v8-update92-continuation-{sequence:03d}'
            pending = DURABLE_ROOT/'runs'/name/'plan.json'
            row = {'mode': 'evaluation' if evaluate else 'training', 'at_update': update,
                   'intended_plan': str(pending), 'status': 'preparing', 'private_commit': None}
            from experiments.learner.builtin_training import schedule as scheduled_rows
            row['schedule'] = scheduled_rows(update if evaluate else sequence, evaluate)
            result['batches'].append(row)
            result['pending_plan'] = str(pending)
            result['latest_parent'] = binding(parent)
            journal(result)  # Durable intent before prepare/execute/preservation can fail.
            actual_plan, report = run_batch(parent, sequence, evaluate, session)
            require(actual_plan == pending, 'Batch returned another intended plan')
            row.update(status='preserving', plan=binding(pending),
                report=binding(pending.parent/'execution/report.json'),
                training_samples=report['training_samples'], optimizer_steps=report['optimizer_steps'],
                outcomes=[attempt.get('outcomes') for attempt in report['attempts']])
            require(report['status'] == 'finished' and report['worker_complete'] is True
                and report['supervisor_cleanup_verified'] is True and report['within_shared_budget'] is True,
                'Batch failed; no replacement or further gameplay')
            row.update(checkpoint_admitted=False, publication_status='not_started')
            if not evaluate:
                verified = verify_local_update(pending, report)
                parent = pending
                update = verified['progress']['updates']
                row.update(checkpoint_admitted=True, verified_checkpoint=verified)
                result.update(latest_parent=binding(parent), latest_verified_parent=binding(parent),
                              latest_verified_progress=verified['progress'])
            row['publication_status'] = 'pending'
            journal(result)  # A subsequent push failure must not erase verified local progress.
            row['private_commit'] = publish_run(pending, args.private_repo, session['absolute_deadline_unix']-30)
            row.update(publication_status='published', status='passed')
            if evaluate: evaluated.add(update)
            else:
                result['latest_private_parent'] = binding(parent)
                sequence += 1
                durations.append(report['execution_wall_seconds'])
            pending = None
            result['latest_parent'] = binding(parent)
            result['pending_plan'] = None
            journal(result)
    except Exception as error:
        result.update(status='stopped_failure', error=f'{type(error).__name__}: {error}')
        if pending is not None:
            row = result['batches'][-1]
            row.update(status='publication_failed' if row.get('checkpoint_admitted') else 'failed',
                       preservation_error=result['error'])
            if row.get('publication_status') == 'pending': row['publication_status'] = 'failed'
            journal(result)
            if row.get('private_commit') is None:
                try:
                    row['failure_private_commit'] = publish_failure(pending, args.private_repo,
                        session['absolute_deadline_unix']-30, result['error'], row['schedule'],
                        verified_checkpoint=row.get('verified_checkpoint'))
                except Exception as preservation_error:
                    row['failure_preservation_error'] = f'{type(preservation_error).__name__}: {preservation_error}'
        raise
    finally:
        result.update(stopped_unix=time.time(), latest_parent=binding(parent),
                      pending_plan=str(pending) if pending else None)
        result['within_two_hours'] = result['stopped_unix'] <= session['absolute_deadline_unix']
        journal(result)


def supervise(args):
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    require(args.private_repo is not None, 'Private preservation repository required')
    preflight_private(args.private_repo)
    # Validate before burning the single-use session. Neither prepare nor help calls this.
    release = scope.validate_release(args.release)
    receipt = scope.checked(release['migration_receipt'])
    migration.load_parent(receipt['plan']['path'])
    migration.verify_private(release['private_migration_receipt']['path'], check_remote=True)
    session = scope.create_session(args.release)
    deadline = session['absolute_deadline_unix']-20
    monotonic_deadline = session['started_monotonic']+7180
    process = None
    result = {'schema': scope.SCOPE+'-controller', 'session': binding(scope.SESSION), 'controller_cleanup_verified': False}
    try:
        with (scope.SESSION.parent/'controller.log').open('x') as log:
            process = guarded_popen([sys.executable, str(Path(__file__).resolve()), 'campaign-worker',
                '--private-repo', str(args.private_repo)], deadline_monotonic=monotonic_deadline,
                deadline_wall=deadline, evidence_path=scope.SESSION.parent/'controller-deadline.jsonl',
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            result['returncode'] = process.wait(timeout=max(1, min(deadline-time.time(), monotonic_deadline-time.monotonic())))
    finally:
        try:
            limits = {'deadline_monotonic': session['started_monotonic']+7200,
                      'deadline_wall': session['absolute_deadline_unix']}
            if process is not None: close_owned_process(process, timeout=2, **limits)
            report_path = scope.SESSION.parent/'campaign-report.json'
            owned = json.loads(report_path.read_text()) if report_path.exists() else {'batches': []}
            if report_path.exists():
                require(owned.get('session') == binding(scope.SESSION), 'Cleanup journal belongs to another session')
            result['cleanup_incomplete'] = []
            for row in owned['batches']:
                plan_path = Path(row['intended_plan'])
                require(plan_path.parent.parent == DURABLE_ROOT/'runs'
                        and plan_path.parent.name.startswith('v8-update92-'), 'Unsafe owned cleanup path')
                for evidence in (plan_path.parent/'execution').glob('*deadline*.jsonl'):
                    if time.time() >= limits['deadline_wall'] or time.monotonic() >= limits['deadline_monotonic']:
                        result['cleanup_incomplete'].append(str(evidence)); continue
                    try:
                        cleanup_from_evidence(evidence, timeout=1, **limits)
                    except Exception as error:
                        result['cleanup_incomplete'].append({'path':str(evidence),'error':str(error)})
            result['controller_cleanup_verified'] = not result['cleanup_incomplete']
        except Exception as error:
            result['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            result['stopped_unix'] = time.time()
            result['within_two_hours'] = result['stopped_unix'] <= session['absolute_deadline_unix']
            migration.write_new(scope.SESSION.parent/'controller-result.json', result)
    require(result.get('returncode') == 0 and result['controller_cleanup_verified'] and result['within_two_hours'],
            'V8 controller stopped or cleanup incomplete; inspect durable result')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prepare = sub.add_parser('prepare-migration')
    for name in ('parent','runtime-bin','data','game-map','output'):
        prepare.add_argument('--'+name, type=Path, required=True)
    verify = sub.add_parser('verify-migration'); verify.add_argument('--plan', type=Path, required=True)
    private = sub.add_parser('preserve-migration')
    private.add_argument('--plan', type=Path, required=True); private.add_argument('--private-repo', type=Path, required=True)
    launch = sub.add_parser('launch'); launch.add_argument('--release', type=Path, required=True)
    launch.add_argument('--private-repo', type=Path, required=True)
    worker = sub.add_parser('campaign-worker'); worker.add_argument('--private-repo', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('HEROES_ARTIFACT_ROOT') == str(DURABLE_ROOT), 'Explicit durable artifact root required')
    if args.mode == 'prepare-migration': print(json.dumps(migration.prepare(args), indent=2))
    elif args.mode == 'verify-migration':
        state = migration.load_parent(args.plan)
        print(json.dumps({'checkpoint': state.checkpoint, 'progress': vars(state.progress), 'launch_ready': False}))
    elif args.mode == 'preserve-migration': print(json.dumps(migration.preserve_private(args.plan,args.private_repo),indent=2))
    elif args.mode == 'launch': supervise(args)
    else: campaign(args)


if __name__ == '__main__': main()
