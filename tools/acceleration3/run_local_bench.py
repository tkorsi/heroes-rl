"""Run the manual local runner on this Linux host against disposable parent copies.

    run_local_bench.py --name NAME --checkout DIR [--runtime accepted] [--max-updates K]
        [--eval-every E] [--hours H] [--first-sequence S] [--stop-after-updates K] [-- START ARGS]

Writes a Linux machine configuration and benchmark specification for one campaign, then runs
``tools/run_local_training.py start`` from the given clean checkout exactly as ``train-local.sh``
does (frozen release checkout, ``prepare_vcmi.py`` and ``--check``, owned supervisor, worker,
batches, preservation), with a frozen copy of ``bench_site/sitecustomize.py`` on PYTHONPATH.
That shim changes only the three machine bindings it documents; every source file of the frozen
release stays byte-identical. Each campaign has its own run root, hence its own sequence ledger,
in the benchmark seed namespace (never 40M/50M or 20M/30M); campaigns with the same first
sequence play the same seeds. ``--stop-after-updates`` requests the runner's own manual stop once
that many new checkpoints are verified, which skips the final evaluation pair. The driver samples
every descendant through /proc and writes ``bench-run.json`` into the campaign folder.

Nothing here touches the official lineage, a Mac path, or the private training-state repository.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import threading
import time

HERE = Path(__file__).resolve().parent
BENCH = Path(os.environ.get('HEROES_LOCAL_BENCH_ROOT', '/home/claude/bench3')).resolve()
ART = BENCH / 'art'
DATA = Path(os.environ.get('HEROES_BENCH_DATA', '/home/claude/heroes-private/game-data')).resolve()
PYTHON = os.environ.get('HEROES_BENCH_PYTHON', '/home/claude/heroes-rl/.venv/bin/python')
WATCHDOG = os.environ.get('HEROES_BENCH_WATCHDOG', '/home/claude/overnight/watchdog.py')
NAMESPACE = {'engine_base': 60_000_000, 'policy_base': 70_000_000}
PARENT = {'plan': 'plan.json', 'report': 'report.json', 'policy': 'after-policy.pt', 'resume': 'after-resume.pt',
          'initial_checkpoint': 'initial-checkpoint-update628.pt'}
PREREQUISITES = ('local-builtin-revalidation-v2-2026-10-25.json', 'build-terminal-diagnostic-2026-10-25.json')
TICK = os.sysconf('SC_CLK_TCK')
PAGE = os.sysconf('SC_PAGE_SIZE')


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
        stream.write('\n')


def children(pid):
    found, frontier = set(), [pid]
    while frontier:
        current = frontier.pop()
        try:
            for task in os.listdir(f'/proc/{current}/task'):
                with open(f'/proc/{current}/task/{task}/children') as f:
                    for child in f.read().split():
                        if int(child) not in found:
                            found.add(int(child)); frontier.append(int(child))
        except OSError:
            continue
    return found


def sample(pid):
    try:
        with open(f'/proc/{pid}/stat') as f: stat = f.read().rsplit(')', 1)[1].split()
        with open(f'/proc/{pid}/cmdline', 'rb') as f: command = f.read().split(b'\0')[:5]
        return dict(pid=pid, rss=int(stat[21]) * PAGE, threads=int(stat[17]), cpu=(int(stat[11]) + int(stat[12])) / TICK,
                    fds=len(os.listdir(f'/proc/{pid}/fd')), command=' '.join(c.decode(errors='replace') for c in command))
    except OSError:
        return None


class Sampler(threading.Thread):
    """Resident memory, descriptors, threads and CPU of every descendant, once per period."""

    def __init__(self, root, path, period=1.0):
        super().__init__(daemon=True); self.root, self.path, self.period = root, path, period
        self.stop = threading.Event(); self.peak = dict(total_rss=0, processes=0, fds=0, threads=0)

    def run(self):
        with open(self.path, 'w') as out:
            while not self.stop.is_set():
                rows = [r for r in map(sample, [self.root, *children(self.root)]) if r]
                row = dict(t=time.monotonic(), utc=time.time(), total_rss=sum(r['rss'] for r in rows),
                           fds=sum(r['fds'] for r in rows), threads=sum(r['threads'] for r in rows), processes=rows)
                out.write(json.dumps(row) + '\n'); out.flush()
                self.peak.update(total_rss=max(self.peak['total_rss'], row['total_rss']),
                                 processes=max(self.peak['processes'], len(rows)),
                                 fds=max(self.peak['fds'], row['fds']), threads=max(self.peak['threads'], row['threads']))
                self.stop.wait(self.period)


class Stopper(threading.Thread):
    """Request the runner's own manual stop after K newly verified checkpoints."""

    def __init__(self, run_root, first_updates, count):
        super().__init__(daemon=True); self.run_root, self.target = run_root, first_updates + count
        self.stop = threading.Event(); self.requested = None

    def run(self):
        while not self.stop.wait(2.0):
            try:
                latest = json.loads((self.run_root / 'latest-checkpoint.json').read_text())
                run_dir = Path(json.loads((self.run_root / 'latest-run.json').read_text())['run_dir'])
            except (OSError, ValueError, KeyError):
                continue
            if latest['progress']['updates'] >= self.target:
                (run_dir / 'stop-requested').touch(exist_ok=True)
                self.requested = dict(utc=utc(), updates=latest['progress']['updates'])
                return


def frozen_site():
    """One immutable copy of the shim per content, shared by every variant."""
    source = HERE / 'bench_site' / 'sitecustomize.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    folder = BENCH / 'site' / digest[:16]
    if not (folder / 'sitecustomize.py').exists():
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, folder / 'sitecustomize.py')
    if hashlib.sha256((folder / 'sitecustomize.py').read_bytes()).hexdigest() != digest:
        raise ValueError('Frozen benchmark shim differs from its content address')
    return folder, digest


def in_checkout(checkout, code, env):
    out = subprocess.run([PYTHON, '-c', code], cwd=checkout, env=env, check=True, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def machine_config(checkout, runtime_bin, run_root, first_sequence, env):
    inputs = ART / 'inputs'
    parent = {k: binding(inputs / 'parent629' / v) for k, v in PARENT.items()}
    game_map = binding(inputs / 'maps' / 'hero-management-v2.vmap')
    files = ['vcmiclient', 'vcmireplay', 'vcmiserver', 'libvcmi.so',
             *sorted('AI/' + p.name for p in (runtime_bin / 'AI').glob('*.so'))]
    computed = in_checkout(checkout, (
        'import json, sys, torch; from pathlib import Path\n'
        'from experiments.headless.contextual_evidence import runtime_identity\n'
        'from experiments.episodes.scoring import engine_recipe_digests\n'
        'from experiments.learner.builtin_v8_migration import map_identity\n'
        'from experiments.learner.local_script_training import restore_parent\n'
        f'map_identity(Path({game_map["path"]!r}))\n'
        f'state = restore_parent({json.dumps(parent)})\n'
        f'print(json.dumps(dict(runtime=runtime_identity(Path({str(runtime_bin)!r})), recipe=engine_recipe_digests(Path(".")),'
        ' torch=str(torch.__version__), progress=vars(state.progress), steps=state.optimizer_steps)))'), env)
    return dict(schema='heroes-local-training-machine-v1', created_utc=utc(), backend='cold', default_profile='original',
                parent=parent, maps=[dict(game_map), dict(game_map)], runtime_bin=str(runtime_bin),
                runtime=computed['runtime'], runtime_files={name: binding(runtime_bin / name) for name in files},
                engine_recipe=computed['recipe'], data=str(DATA), artifact_root=str(ART), run_root=str(run_root),
                prerequisites={name: binding(inputs / 'validation' / name) for name in PREREQUISITES},
                python={'executable': PYTHON, 'torch': computed['torch']},
                checkpoint_restore={'progress': computed['progress'], 'optimizer_steps': computed['steps']},
                next_sequence=first_sequence, seed_namespace=dict(NAMESPACE), gameplay_performed=False, ppo_updates=0,
                training_authorized_by_preparation=False,
                scope='Disposable Linux benchmark copy of the final629 parent; benchmark seed namespace; never official training.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True); p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--variant', required=True, help='label recorded with the campaign')
    p.add_argument('--runtime', default='accepted'); p.add_argument('--max-updates', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=100); p.add_argument('--hours', type=float, default=3.0)
    p.add_argument('--first-sequence', type=int, default=266); p.add_argument('--stop-after-updates', type=int)
    p.add_argument('--check-only', action='store_true', help="run the runner's offline check, no game")
    p.add_argument('start_args', nargs='*', help='extra arguments for run_local_training.py start (after --)')
    a = p.parse_args()
    checkout = a.checkout.resolve()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=checkout, text=True).strip():
        raise SystemExit('Checkout is dirty')
    campaign = ART / 'campaigns' / a.name
    campaign.mkdir(parents=True, exist_ok=False)
    # The deadline watchdog may stop this driver and everything below it, and nothing else.
    subprocess.run(['python3', WATCHDOG, 'own', str(os.getpid()), 'driver-' + a.name], check=False, stdout=subprocess.DEVNULL)
    run_root = ART / 'runs' / a.name
    runtime_bin = (BENCH / 'runtime' / a.runtime / 'bin').resolve()
    site, site_digest = frozen_site()
    spec_path, config_path = campaign / 'bench-spec.json', campaign / 'machine.json'
    write_new(spec_path, dict(schema='heroes-local-linux-bench-spec-v1', created_utc=utc(), machine_config=str(config_path),
                              seed_namespace=dict(NAMESPACE), shim_sha256=site_digest,
                              purpose='Benchmark processes only; disposable copies; no official namespace or lineage.'))
    # The copied runtime's ELF RUNPATH names its build folder; the loader must use the hashed copy.
    env = {**os.environ, 'PYTHONPATH': str(site), 'HEROES_LOCAL_BENCH_SPEC': str(spec_path), 'HEROES_ARTIFACT_ROOT': str(ART),
           'LD_LIBRARY_PATH': str(runtime_bin)}
    env.pop('PYTHONDONTWRITEBYTECODE', None)
    prepared = time.monotonic()
    config = machine_config(checkout, runtime_bin, run_root, a.first_sequence, env)
    write_new(config_path, config)
    prepare_seconds = time.monotonic() - prepared
    command = ([PYTHON, 'tools/run_local_training.py', 'check', '--config', str(config_path), '--profile', 'original']
               if a.check_only else
               [PYTHON, 'tools/run_local_training.py', 'start', '--config', str(config_path), '--profile', 'original',
                '--hours', str(a.hours), '--eval-every', str(a.eval_every), '--max-updates', str(a.max_updates), *a.start_args])
    record = dict(name=a.name, variant=a.variant, checkout=str(checkout), git_head=head, runtime=a.runtime,
                  runtime_files={k: v['sha256'] for k, v in config['runtime_files'].items()}, shim_sha256=site_digest,
                  command=command, first_sequence=a.first_sequence, max_updates=a.max_updates, eval_every=a.eval_every,
                  hours=a.hours, stop_after_updates=a.stop_after_updates, machine_prepare_seconds=prepare_seconds,
                  started_utc=utc())
    (campaign / 'bench-run.json').write_text(json.dumps(record, indent=2) + '\n')
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    with (campaign / 'runner.log').open('x') as log:
        process = subprocess.Popen(command, cwd=checkout, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    subprocess.run(['python3', WATCHDOG, 'own', str(process.pid), 'bench-' + a.name], check=False)
    sampler = Sampler(process.pid, campaign / 'resources.jsonl'); sampler.start()
    stopper = None
    if a.stop_after_updates:
        stopper = Stopper(run_root, config['checkpoint_restore']['progress']['updates'], a.stop_after_updates); stopper.start()
    returncode = process.wait()
    sampler.stop.set(); sampler.join()
    if stopper: stopper.stop.set(); stopper.join()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    record.update(launch_seconds=time.monotonic() - started, returncode=returncode, finished_utc=utc(),
                  child_cpu_seconds=(after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime),
                  child_user_seconds=after.ru_utime - before.ru_utime, child_system_seconds=after.ru_stime - before.ru_stime,
                  child_max_rss_kib=after.ru_maxrss, peaks=sampler.peak, stop_requested=stopper.requested if stopper else None)
    (campaign / 'bench-run.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('name', 'variant', 'launch_seconds', 'returncode', 'child_cpu_seconds')}))


if __name__ == '__main__':
    main()
