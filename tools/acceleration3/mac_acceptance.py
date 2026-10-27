#!/usr/bin/env python3
"""Bounded Mac acceptance benchmark of the opt-in fast path. Run only with explicit user authorization.

    "$HOME/Projects/Heroes/.venv/bin/python" tools/acceleration3/mac_acceptance.py --dry-run
    "$HOME/Projects/Heroes/.venv/bin/python" tools/acceleration3/mac_acceptance.py [--order AFFA]
        [--updates 1] [--hours 4] [--parent-run RUN] [--bench-root DIR]

Campaign letters (each plays the eight fixed baseline games, then ``--updates`` training batches,
from disposable full copies of the selected verified checkpoint (699 by default), exactly through ``run_local_training.py start``):

    A  the default command (main's accepted implementation, released from ACCEPTED_IMPLEMENTATION)
    F  ``--fast-path`` with the macOS defaults (serial verification, no prefix helpers)
    O  ``--fast-path --overlap-verification`` (forked verification; fork safety on macOS is what
       this campaign tests)
    H  ``--fast-path --overlap-verification --prefix-helpers 2``
    N  ``--fast-path --no-text-replay`` (the macOS defaults with every text leaf occurrence computed
       separately: the in-lineage fallback if the text-leaf replay ever refuses)
    T  ``--fast-path --replay-threads 2`` (the macOS defaults with text-leaf replays on two threads
       during the backward: threads only, no fork)
    X  ``--fast-path --experiment shared_graph_encodings`` (numerically different; reported apart)

Every campaign of one invocation starts at the same benchmark sequence, so all play the same seeds.
A durable study ledger reserves a fresh diagnostic namespace for each invocation, including dry runs.
Each exact campaign (F, O, H, N, T, and any repeated A) is compared with the A campaigns batch by batch
while both still share the same parent; wherever their games are identical their checkpoints must
be bitwise identical, and every exact campaign needs at least one such comparable batch.
``summary.json`` gives each campaign's status (identical, diverged, not comparable). The default
order AF is the acceptance check; AFFA (or AOOA) is a counterbalanced timing comparison.

Safety and scope:
- Refuses unless this checkout is clean and contains both the accepted implementation and the fast
  path, no ``run_local_training.py`` process is running, and the official owner lock is free. It
  then holds that lock (read-only descriptor, exclusive) until it exits, so the manual 5,000-game
  run cannot start alongside it (it would refuse immediately with "Another local training script
  owns this artifact root") and this tool cannot start alongside that run; each benchmark runner
  inherits the locked descriptor, so the exclusion holds as long as any of them runs. If the lock file does
  not exist (no official run has ever started on this Mac), nothing is created and only the process
  check guards. The checkout must stay at the same clean commit; it is checked before every campaign.
- Writes only into a fresh benchmark root outside the official artifact root. Official files are
  read, never written: the parent run (selected by the runner's own final-parent rule), the pinned
  preparation inputs, the validation prerequisites, the machine description and the runtime. The
  runner's frozen releases are git worktrees of this checkout under the benchmark root.
- Seeds: benchmark namespaces start at 80M/80.05M and are reserved before work, never 40M/50M
  (original), 20M/30M (easier8x8), or the earlier 60M/70M benchmark. The benchmark-only shim also
  applies the same declared operational batch wall allowance to accepted and candidate code.
  PPO, rewards, rollout limits, replay and sample admission remain unchanged. Nothing produced
  is a continuation of the official lineage.
- Bounded: each runner has its own ``--hours`` hard stop; this tool additionally sends SIGTERM (the
  runner's manual stop) ten minutes after it and kills the campaign's process tree two minutes
  later. Ctrl-C works as with ``train-local.sh`` (stop after the current batch) and starts no
  further campaign; the summary is still written. After ``--updates`` checkpoints it requests the
  runner's own stop while the last batch runs, so the final evaluation pair is skipped.
- Exit status 0 only if every campaign ended cleanly (runner status stopped_requested, every
  requested training batch admitted, no bound reached, no leftover process, the expected release
  ran), the checkout did not change, nothing was interrupted, and every exact campaign was
  identical. ``summary.json`` lists every problem.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import campaign_summary

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OFFICIAL = Path.home() / 'Projects/Heroes-artifacts'
OFFICIAL_LOCK = OFFICIAL / 'local-script-training' / 'owner.lock'
OFFICIAL_CONFIG = OFFICIAL / 'local-script-training' / 'machine-original.json'
PARENT_RUN = OFFICIAL / 'local-script-training/original/run-20261026T181133Z-062cd75a'
PARENT_UPDATE = 699
SEED_LEDGER = Path.home() / 'Projects/Heroes-bench/mac699-seed-ledger.json'
NAMESPACE = {'engine_base': 80_000_000, 'policy_base': 80_050_000}
EXPECTED_PARENT = {'policy': '045d48215be54bd2bd3f438a12f2626ffe83b2a53445a6a8623254feeb0ac046',
                   'resume': 'c5f8e601040e738bbda9fee987f996829107dda3737e2954e1cc4ba371c0afcf'}
PARENT_FILES = {'plan': 'plan.json', 'report': 'report.json', 'policy': 'after-policy.pt', 'resume': 'after-resume.pt',
                'initial_checkpoint': 'initial-checkpoint-update698.pt'}
CAMPAIGNS = {'A': [], 'F': ['--fast-path', '--serial-verification'],
             'O': ['--fast-path', '--overlap-verification'],
             'H': ['--fast-path', '--overlap-verification', '--prefix-helpers', '2'],
             'N': ['--fast-path', '--serial-verification', '--no-text-replay'],
             'T': ['--fast-path', '--serial-verification', '--replay-threads', '2'],
             'X': ['--fast-path', '--serial-verification', '--experiment', 'shared_graph_encodings']}
EXACT = set('AFOHNT')
# The execution choices each campaign's session must record (the accepted runner records none).
EXECUTION = {letter: None if letter == 'A' else {
    'implementation': 'exact-fast-path-v1', 'overlapped_verification': overlapped, 'prefix_helpers': helpers,
    'text_leaf_replay': replay, 'replay_threads': threads, 'experiments': experiments}
    for letter, (overlapped, helpers, replay, threads, experiments) in {
        'A': (None, None, None, None, None), 'F': (False, 0, True, 0, []), 'O': (True, 0, True, 0, []),
        'H': (True, 2, True, 2, []), 'N': (False, 0, False, 0, []), 'T': (False, 0, True, 2, []),
        'X': (False, 0, True, 0, ['shared_graph_encodings'])}.items()}


def utc(value=None):
    return datetime.fromtimestamp(time.time() if value is None else value, timezone.utc).isoformat()


def binding(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def refuse(message):
    raise SystemExit('Refused: ' + message)


def preflight():
    if git('status', '--porcelain'):
        refuse('commit or remove local changes first; the runner releases only committed code')
    head = git('rev-parse', 'HEAD')
    sys.path.insert(0, str(ROOT / 'tools'))
    import run_local_training as runner
    if subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', runner.ACCEPTED_IMPLEMENTATION, head]).returncode:
        refuse('the accepted implementation is not in this checkout')
    rows = subprocess.check_output(['ps', '-A', '-o', 'pid=,command='], text=True).splitlines()
    others = [r.strip() for r in rows if 'run_local_training.py' in r and int(r.split()[0]) != os.getpid()]
    if others:
        refuse('a local training script is running: ' + '; '.join(others))
    return runner, head


def hold_official_lock():
    """Exclusive lock on the official owner lock for this process's lifetime (never created or written)."""
    if not OFFICIAL_LOCK.exists():
        return None, 'absent'
    descriptor = os.open(OFFICIAL_LOCK, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        refuse('the official local training lock is held; is the manual run active?')
    return descriptor, 'held'


def inside_official(path):
    """Whether path is in the official artifact root, by file identity (macOS paths are case-insensitive
    and may pass through firmlinks), with a case-insensitive comparison when the root does not exist."""
    path = Path(path).expanduser().resolve()
    official = OFFICIAL.expanduser()
    if official.exists():
        return any(p.exists() and os.path.samefile(p, official) for p in (path, *path.parents))
    folded = str(official.resolve()).casefold().rstrip('/') + '/'
    return (str(path).casefold().rstrip('/') + '/').startswith(folded)


def copy_pinned(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    a, b = binding(source), binding(destination)
    if (a['sha256'], a['bytes']) != (b['sha256'], b['bytes']):
        refuse(f'copy of {source} differs')
    return b


def stopped_parent(run):
    """Read-only selection of the pinned 699 state; failed batch 336 remains excluded."""
    run = Path(run).expanduser().resolve()
    session, report, shutdown = [json.loads((run / name).read_text()) for name in
                                ('session.json', 'campaign-report.json', 'shutdown.json')]
    if (Path(session['run_dir']).resolve() != run
            or report.get('status') not in ('finished', 'stopped_requested', 'stopped_failure')
            or report.get('owned_cleanup_verified') is not True
            or report.get('initial_state_unchanged') is not True
            or report.get('full_final_restore', {}).get('exact') is not True
            or report['full_final_restore'].get('progress', {}).get('updates') != PARENT_UPDATE
            or shutdown.get('owned_cleanup_verified') is not True
            or shutdown.get('within_hard_deadline') is not True
            or shutdown.get('worker_returncode') not in (0, 1)):
        refuse('parent campaign is not stopped with the verified full checkpoint 699')
    parent = report['latest_parent']
    for name, expected in EXPECTED_PARENT.items():
        if parent.get(name, {}).get('sha256') != expected:
            refuse('parent is not the pinned final 699 checkpoint: ' + name)
    return parent


def reserve_namespace(bench):
    """Persist a fresh study namespace before any game, even when preparation later fails.

    All implementations in one paired study intentionally share these diagnostic seeds;
    another invocation gets a different namespace. Never recycle interrupted attempts.
    """
    SEED_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with SEED_LEDGER.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = json.loads(SEED_LEDGER.read_text()) if SEED_LEDGER.exists() else {
            'schema': 'heroes-mac699-benchmark-seeds-v1', 'next_study': 0, 'reserved': []}
        if value.get('schema') != 'heroes-mac699-benchmark-seeds-v1':
            refuse('unrecognized benchmark seed ledger')
        study = value['next_study']
        if type(study) is not int or not 0 <= study < 10000:
            refuse('benchmark namespace exhausted or corrupt')
        namespace = {key: base + 100_000 * study for key, base in NAMESPACE.items()}
        receipt = dict(study=study, reserved_utc=utc(), benchmark=str(bench), seed_namespace=namespace)
        value['next_study'] = study + 1
        value['reserved'].append(receipt)
        temporary = SEED_LEDGER.with_suffix('.tmp')
        with temporary.open('w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, SEED_LEDGER)
        directory = os.open(SEED_LEDGER.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    return namespace, receipt


def prepare_inputs(runner, bench, parent_run):
    """Disposable copies of every official input a benchmark campaign reads."""
    sys.path.insert(0, str(ROOT / 'tools'))
    import prepare_local_training as prep
    inputs = bench / 'art' / 'inputs'
    official_parent = stopped_parent(parent_run)  # accepts the pinned successful state of the failed run, never its partial batch
    for name, item in official_parent.items():
        if binding(item['path']) != item:
            refuse('parent binding changed: ' + name)
    for name, sha in EXPECTED_PARENT.items():
        if official_parent[name]['sha256'] != sha:
            refuse(f'parent run {parent_run} is not the pinned final 699 checkpoint ({name})')
    # The official policy and its initial dependency are both named after-policy.pt; copies are named by role.
    if set(official_parent) != set(PARENT_FILES):
        refuse('unexpected parent descriptor fields: ' + ', '.join(sorted(official_parent)))
    parent = {name: copy_pinned(item['path'], inputs / 'parent699' / PARENT_FILES[name])
              for name, item in official_parent.items()}
    write_new(inputs / 'parent699.json', dict(parent=parent, official_source=official_parent,
              parent_run=str(parent_run), note='Disposable benchmark copy; never official lineage.'))
    for name, (relative, sha) in prep.INPUT_PINS.items():
        if copy_pinned(prep.DEFAULT_INPUT / relative, inputs / 'input412' / relative)['sha256'] != sha:
            refuse('preparation input differs from its pin: ' + name)
    for name, sha in prep.PREREQUISITE_PINS.items():
        if copy_pinned(OFFICIAL / 'inputs/validation' / name, inputs / 'validation' / name)['sha256'] != sha:
            refuse('validation prerequisite differs from its pin: ' + name)
    return inputs, official_parent


def frozen_site(bench):
    source = HERE / 'bench_site' / 'sitecustomize.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    folder = bench / 'site' / digest[:16]
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, folder / 'sitecustomize.py')
    return folder, digest


def machine_config(python, inputs, data, art, run_root, env, namespace):
    """The unchanged preparation code (imported, so the namespace mode applies), benchmark paths."""
    code = ('import json, sys; from pathlib import Path; sys.path.insert(0, "tools")\n'
            'import prepare_local_training as p\n'
            f'v = p.build_configuration(Path({str(inputs / "input412")!r}), p.DEFAULT_RUNTIME, Path({str(data)!r}),'
            f' Path({str(art)!r}), Path({str(run_root)!r}), "original")\n'
            'print(json.dumps(v))')
    out = subprocess.run([python, '-c', code], cwd=ROOT, env=env, check=True, capture_output=True, text=True)  # retain the outer study guardian's process group
    value = json.loads(out.stdout.strip().splitlines()[-1])
    if value['seed_namespace'] != namespace:
        refuse('benchmark namespace did not apply; nothing was started')
    return value


def tree(root):
    rows = {}
    for line in subprocess.run(['ps', '-A', '-o', 'pid=,ppid=,rss=,time='], capture_output=True, text=True).stdout.splitlines():
        pid, ppid, rss, cpu = line.split()[:4]
        rows[int(pid)] = (int(ppid), int(rss) * 1024, cpu)
    found, frontier = {root} if root in rows else set(), [root]
    while frontier:
        current = frontier.pop()
        for pid, (ppid, _, _) in rows.items():
            if ppid == current and pid not in found:
                found.add(pid); frontier.append(pid)
    return {pid: rows[pid] for pid in found}


def leftovers(bench):
    """Processes still naming this benchmark's artifact folder (runner, worker, engine), other than this tool."""
    rows = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True).stdout.splitlines()
    return sorted(int(r.split()[0]) for r in rows if str(bench / 'art') in r and int(r.split()[0]) != os.getpid())


class Sampler(threading.Thread):
    """Summed resident memory of the runner's process tree (forked children share pages, so it overcounts)."""
    def __init__(self, root, path, period=5.0):
        super().__init__(daemon=True); self.root, self.path, self.period = root, path, period
        self.stop = threading.Event(); self.peak = dict(total_rss=0, processes=0)

    def run(self):
        with open(self.path, 'w') as out:
            while not self.stop.is_set():
                rows = tree(self.root)
                total = sum(r[1] for r in rows.values())
                out.write(json.dumps(dict(utc=time.time(), total_rss=total, processes=len(rows),
                                          rows={p: r for p, r in rows.items()})) + '\n'); out.flush()
                self.peak.update(total_rss=max(self.peak['total_rss'], total), processes=max(self.peak['processes'], len(rows)))
                self.stop.wait(self.period)


def request_stop_during(run_root, count, stop):
    """Touch the runner's own stop file while its count-th training batch runs (it then completes)."""
    while not stop.wait(1.0):
        try:
            run_dir = Path(json.loads((run_root / 'latest-run.json').read_text())['run_dir'])
            with (run_dir / 'research-log.jsonl').open() as log:
                started = sum(1 for line in log if '"batch_intent"' in line and '"mode": "training"' in line)
        except (OSError, ValueError, KeyError):
            continue
        if started >= count:
            (run_dir / 'stop-requested').touch(exist_ok=True)
            return


INTERRUPT = dict(requested=None, runner=None)


def interrupted(signum, frame):
    """First Ctrl-C or SIGTERM: the runner stops after its current batch; no further campaign starts."""
    INTERRUPT['requested'] = INTERRUPT['requested'] or dict(signal=signum, utc=utc())
    runner = INTERRUPT['runner']
    if signum == signal.SIGTERM and runner is not None and runner.poll() is None:
        runner.send_signal(signal.SIGTERM)  # a terminal Ctrl-C already reached the runner itself


def run_campaign(args, bench, inputs, site, letter, index, env_base, data):
    name = f'{index:02d}-{letter}'
    campaign = bench / 'art' / 'campaigns' / name
    campaign.mkdir(parents=True)
    run_root = bench / 'art' / 'runs' / name
    spec = campaign / 'bench-spec.json'
    write_new(spec, dict(schema='heroes-local-mac-bench-spec-v1', mode='namespace', seed_namespace=args.namespace, batch_wall_seconds=args.batch_wall_seconds,
                         created_utc=utc(), purpose='Mac benchmark campaign; disposable copies; not official training.'))
    env = {**env_base, 'HEROES_LOCAL_BENCH_SPEC': str(spec)}
    config = machine_config(args.python, inputs, data, bench / 'art', run_root, env, args.namespace)
    write_new(campaign / 'machine.json', config)
    if INTERRUPT['requested']:
        return name, run_root, dict(name=name, letter=letter, not_started='interrupted during setup')
    command = [args.python, 'tools/run_local_training.py', 'start', '--config', str(campaign / 'machine.json'),
               '--profile', 'original', '--parent', str(inputs / 'parent699.json'), '--hours', str(args.hours),
               '--eval-every', '100', '--max-updates', str(args.updates), *CAMPAIGNS[letter]]
    record = dict(name=name, letter=letter, command=command, started_utc=utc())
    started = time.monotonic()
    with (campaign / 'runner.log').open('x') as log:
        # Same process group as this tool: Ctrl-C reaches the runner exactly as with train-local.sh
        # (a manual stop after the current batch).
        # The runner inherits the official lock, so the manual run stays excluded even if this tool dies.
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, pass_fds=() if args.lock_fd is None else (args.lock_fd,))
    INTERRUPT['runner'] = process
    sampler = Sampler(process.pid, campaign / 'resources.jsonl'); sampler.start()
    stop = threading.Event()
    threading.Thread(target=request_stop_during, args=(run_root, args.updates, stop), daemon=True).start()
    bound = min(args.hours * 3600 + 600, max(0, args.study_deadline_monotonic - time.monotonic() - 120))
    try:
        returncode = process.wait(timeout=bound)
        record['bound'] = 'not_reached'
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGTERM)  # the runner's manual stop
        try:
            returncode = process.wait(timeout=120)
            record['bound'] = 'stopped_by_sigterm'
        except subprocess.TimeoutExpired:
            for pid in tree(process.pid):
                try: os.kill(pid, signal.SIGKILL)
                except ProcessLookupError: pass
            returncode = process.wait()
            record['bound'] = 'killed'
    INTERRUPT['runner'] = None
    stop.set(); sampler.stop.set(); sampler.join()
    time.sleep(2.0)  # let the runner's own cleanup finish before looking for leftovers
    record.update(returncode=returncode, launch_seconds=time.monotonic() - started, finished_utc=utc(),
                  peaks=sampler.peak, leftover_processes=leftovers(bench))
    write_new(campaign / 'bench-run.json', record)
    return name, run_root, record


def problems(record, updates):
    """Why a campaign does not count as a clean, complete campaign (empty when it does)."""
    found = []
    run = record.get('run', {})
    if run.get('not_started'):
        return ['not started: ' + run['not_started']]
    if run.get('returncode') != 0:
        found.append(f"runner exit status {run.get('returncode')}")
    if run.get('bound') != 'not_reached':
        found.append(f"tool bound reached: {run.get('bound')}")
    if run.get('leftover_processes'):
        found.append(f"leftover processes {run['leftover_processes']}")
    if record.get('status') != 'stopped_requested':
        found.append(f"runner status {record.get('status')}")
    if record.get('release') != record.get('expected_release'):
        found.append(f"ran release {record.get('release')}, expected {record.get('expected_release')}")
    if record.get('execution') != record.get('expected_execution'):
        found.append(f"ran execution {record.get('execution')}, expected {record.get('expected_execution')}")
    trained = [b for b in record.get('batches', []) if b['mode'] == 'training']
    if sum(b['status'] == 'passed' for b in trained) != updates or len(trained) != updates:
        found.append(f'{sum(b["status"] == "passed" for b in trained)} of {updates} training batches admitted')
    if sum(b['mode'] == 'baseline' and b['status'] == 'passed' for b in record.get('batches', [])) != 2:
        found.append('baseline evaluation pair incomplete')
    return found


def release_of(run_dir):
    """The commit the campaign's runner actually released and ran (its session's source)."""
    return json.loads((Path(run_dir) / 'session.json').read_text())['source']['git_commit']


def execution_of(run_dir):
    """The execution choices the campaign's session bound; every batch plan must equal them and the worker applies them."""
    return json.loads((Path(run_dir) / 'session.json').read_text()).get('execution')


def chained_pairs(campaigns, reference, other):
    """The two campaigns' batches paired; a training pair is comparable only while both still share the
    same parent: identical games and identical admitted state in every earlier training pair, and
    both states present in this one."""
    rows, same_parent = [], True
    for pair in campaign_summary.compare({n: campaigns[n] for n in (reference, other)}):
        row = dict(pair, campaigns=[reference, other])
        if pair['mode'] == 'training':  # compare() orders training pairs by sequence
            row['comparable'] = same_parent and pair['games_identical'] is True and pair['states_identical'] is not None
            same_parent = row['comparable'] and pair['states_identical'] is True
        rows.append(row)
    return rows


def exact_verdict(campaigns):
    """Every exact campaign is checked against the accepted campaigns (the first exact one without A):
    identical (at least one comparable training pair, all with identical state), diverged (a comparable
    pair with different state) or not comparable (no comparable pair, e.g. different games)."""
    exact = [n for n, c in campaigns.items() if c['letter'] in EXACT]
    references = [n for n in exact if campaigns[n]['letter'] == 'A'] or exact[:1]
    pairs, status = [], {}
    for other in exact:
        if references and other == references[0]:
            continue
        rows = [row for reference in references if reference != other for row in chained_pairs(campaigns, reference, other)]
        pairs += rows
        comparable = [r for r in rows if r.get('comparable')]
        status[other] = ('diverged' if any(r['states_identical'] is not True for r in comparable)
                         else 'identical' if comparable else 'not_comparable')
    if not status:
        verdict = 'not_comparable: fewer than two exact campaigns'
    elif all(s == 'identical' for s in status.values()):
        verdict = True
    elif any(s == 'diverged' for s in status.values()):
        verdict = False
    else:
        verdict = 'not_comparable: ' + ', '.join(n for n, s in status.items() if s == 'not_comparable')
    return verdict, status, pairs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--order', default='AFT', help='campaign letters, e.g. AF (acceptance) or AFFA (timing)')
    p.add_argument('--updates', type=int, default=2, help='training batches per campaign (1 to 4)')
    p.add_argument('--hours', type=float, default=8.0,
                   help="each runner's own hard stop (0.25 to 12); the runner begins a training batch only while "
                        "more than six times its slowest recent batch remains, so --updates 2 or more needs a larger value")
    p.add_argument('--total-hours', type=float, default=8.0,
                   help='absolute wall budget for the entire study, including all paths and cleanup (0.5 to 24)')
    p.add_argument('--parent-run', type=Path, default=PARENT_RUN)
    p.add_argument('--batch-wall-seconds', type=int, default=10800,
                   help='equal operational allowance for accepted/candidate batches; 3600 to 21600')
    p.add_argument('--bench-root', type=Path)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--study-deadline-unix', type=float, help=argparse.SUPPRESS)
    p.add_argument('--study-deadline-monotonic', type=float, help=argparse.SUPPRESS)
    p.add_argument('--dry-run', action='store_true', help='checks and input copies only; no game')
    args = p.parse_args()
    if not args.order or set(args.order) - set(CAMPAIGNS) or len(args.order) > 6:
        p.error('--order takes one to six letters from ' + ''.join(CAMPAIGNS))
    if not 1 <= args.updates <= 4 or not 0.25 <= args.hours <= 12:
        p.error('--updates 1 to 4 and --hours 0.25 to 12')
    if not 0.5 <= args.total_hours <= 24:
        p.error('--total-hours must be 0.5 to 24')
    if not 3600 <= args.batch_wall_seconds <= 21600:
        p.error('--batch-wall-seconds must be 3600 to 21600')
    if args.hours * 3600 - 300 <= args.batch_wall_seconds:
        p.error('--hours must leave the declared batch allowance and five-minute finalization reserve')
    if sys.platform != 'darwin':
        refuse('this is the Mac acceptance tool; Linux benchmarks use run_local_bench.py')
    # From here a first Ctrl-C or SIGTERM only stops further work; the summary is still written.
    signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted)
    runner, head = preflight()
    lock, lock_state = hold_official_lock()
    if lock is not None:
        os.set_inheritable(lock, True)
    args.lock_fd = lock
    study_started = time.time()
    external_monotonic = args.study_deadline_monotonic
    args.study_deadline_monotonic = time.monotonic() + args.total_hours * 3600
    study_deadline_unix = study_started + args.total_hours * 3600
    if args.study_deadline_unix is not None or external_monotonic is not None:
        if args.study_deadline_unix is None or external_monotonic is None:
            refuse('both independent study deadlines are required')
        study_deadline_unix = min(study_deadline_unix, args.study_deadline_unix)
        args.study_deadline_monotonic = min(args.study_deadline_monotonic, external_monotonic)
    requested_hours = args.hours
    bench = (args.bench_root or Path.home() / 'Projects/Heroes-bench' /
             ('original5000-acceleration-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).expanduser().resolve()
    if inside_official(bench):
        refuse('the benchmark root must be outside the official artifact root')
    bench.mkdir(parents=True, exist_ok=False)
    args.namespace, namespace_receipt = reserve_namespace(bench)
    inputs, official_parent = prepare_inputs(runner, bench, args.parent_run.expanduser().resolve())
    site, site_digest = frozen_site(bench)
    data = Path(json.loads(OFFICIAL_CONFIG.read_text())['data']) if OFFICIAL_CONFIG.exists() else \
        Path.home() / 'Library/Application Support/vcmi'
    env = {**os.environ, 'PYTHONPATH': str(site), 'HEROES_ARTIFACT_ROOT': str(bench / 'art')}
    record = dict(schema='heroes-mac-fast-path-acceptance-v1', started_utc=utc(), checkout=str(ROOT), git_head=head,
                  accepted_implementation=runner.ACCEPTED_IMPLEMENTATION, order=args.order, updates=args.updates,
                  hours=args.hours, total_hours=args.total_hours,
                  study_deadline_unix=study_deadline_unix, study_deadline_utc=utc(study_deadline_unix),
                  study_deadline_monotonic=args.study_deadline_monotonic, official_lock=lock_state, parent_run=str(args.parent_run),
                  official_parent={k: v['sha256'] for k, v in official_parent.items()}, shim_sha256=site_digest,
                  tool_sources={name: binding(HERE / name) for name in
                      ('mac_acceptance.py', 'guarded_mac_acceptance.py', 'campaign_summary.py', 'bench_site/sitecustomize.py')},
                  seed_namespace=args.namespace, namespace_reservation=namespace_receipt,
                  batch_wall_seconds=args.batch_wall_seconds, python=args.python,
                  platform=dict(zip(('sysname', 'nodename', 'release', 'version', 'machine'), os.uname())), campaigns={})
    record['hypotheses'] = [
        'Exact acceleration reduces full verified original-map batch time on this Mac from full checkpoint699.',
        'Identical captured games from the same parent yield bitwise-identical full model/Adam/RNG/progress/pool state.',
        'A three-hour operational batch allowance avoids the earlier one-hour PPO timeout without changing learning settings.']
    record['interpretation'] = ('A/F use one Torch intra-op thread and no helpers; T uses two replay threads and is reported separately. '
        'Timing claims require completed comparable batches; a timeout is censored, not a completed baseline. '
        'Native asynchronous game differences are reported as not comparable, never evidence of exactness. '
        'No benchmark checkpoint is eligible for the official continuation.')
    write_new(bench / 'preregistered.json', record)
    if args.dry_run:
        record['dry_run'] = True
        write_new(bench / 'summary.json', record)
        print(json.dumps(dict(bench=str(bench), dry_run=True, official_lock=lock_state)))
        return 0
    try:
        for index, letter in enumerate(args.order):
            if INTERRUPT['requested']:
                break
            if git('rev-parse', 'HEAD') != head or git('status', '--porcelain'):
                record['checkout_changed'] = utc(); break
            remaining = min(study_deadline_unix - time.time(), args.study_deadline_monotonic - time.monotonic())
            args.hours = min(requested_hours, (remaining - 120) / 3600)
            if args.hours * 3600 <= args.batch_wall_seconds + 300:
                record['remaining_paths_unstarted'] = dict(reason='study_budget_reserve', letters=args.order[index:],
                                                          remaining_seconds=remaining)
                break
            name, run_root, run = run_campaign(args, bench, inputs, site, letter, index, env, data)
            expected = runner.ACCEPTED_IMPLEMENTATION if letter == 'A' else head
            try:
                record['campaigns'][name] = dict(letter=letter, run=run, expected_release=expected,
                                                 expected_execution=EXECUTION[letter], **campaign_summary.batches(run_root))
                record['campaigns'][name]['release'] = release_of(record['campaigns'][name]['run_dir'])
                record['campaigns'][name]['execution'] = execution_of(record['campaigns'][name]['run_dir'])
            except (OSError, ValueError, KeyError) as error:
                record['campaigns'][name] = dict(letter=letter, run=run, expected_release=expected,
                                                 expected_execution=EXECUTION[letter], status='no_session',
                                                 error=repr(error), batches=[])
            print(json.dumps({name: record['campaigns'][name]['status']}), flush=True)
        if git('rev-parse', 'HEAD') != head or git('status', '--porcelain'):
            record['checkout_changed'] = utc()
    except Exception as error:  # still write what was measured
        record['error'] = repr(error)
    finally:
        record['interrupted'] = INTERRUPT['requested']
        for campaign in record['campaigns'].values():
            campaign['problems'] = problems(campaign, args.updates)
        # Identical games from the same parent must give bitwise-identical learner state on every exact path.
        record['exact_campaigns_identical'], record['exact_campaign_status'], record['pairs'] = \
            exact_verdict(record['campaigns'])
        trained = [p for p in record['pairs'] if p['mode'] == 'training']
        record['paired_training_batches'] = dict(total=len(trained), comparable=sum(p['comparable'] for p in trained),
                                                 identical_games=sum(p['games_identical'] is True for p in trained))
        if 'X' in args.order:
            record['experiment_pairs'] = campaign_summary.compare(record['campaigns'])
        record['problems'] = ([f'{n}: {x}' for n, c in record['campaigns'].items() for x in c['problems']]
                              + (['interrupted'] if record['interrupted'] else [])
                              + (['checkout changed'] if record.get('checkout_changed') else [])
                              + ([record['error']] if record.get('error') else [])
                              + ([] if len(record['campaigns']) == len(args.order) else ['not every campaign ran']))
        record['passed'] = record['exact_campaigns_identical'] is True and not record['problems']
        record['official_parent_unchanged'] = all(binding(v['path']) == v for v in official_parent.values())
        if not record['official_parent_unchanged']:
            record['problems'].append('official parent changed')
            record['passed'] = False
        record['finished_utc'] = utc()
        write_new(bench / 'summary.json', record)
        print(json.dumps(dict(bench=str(bench), passed=record['passed'], problems=record['problems'],
                              exact_campaigns_identical=record['exact_campaigns_identical'])))
        if lock is not None:
            os.close(lock)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
