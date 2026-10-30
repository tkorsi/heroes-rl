"""Manually finish the 1176-game goal anchored at verified checkpoint1585."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_1000_games import hours, positive_cpu_count, NATIVE_MANIFEST_SHA256, NATIVE_MANIFEST_RELATIVE


def command(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog='The runner verifies checkpoint1585 in the current official ancestry and counts '
               'only fully admitted descendant games. Repeated manual invocations finish the '
               'same 1176-game goal. Evaluation and failed batches never count. The latest '
               'checkpoint is never reset. The explicitly disclosed revealed-movement interaction '
               'guard is selected. No cloud or model calls are made.')
    parser.add_argument('--check', action='store_true', help='offline preflight only; no training')
    parser.add_argument('--hours', type=hours, default=10,
                        help='wall-clock limit including cleanup reserve (default: 10)')
    parser.add_argument('--cpu-cap', type=positive_cpu_count,
                        help='optional resource cap; default uses all available CPUs')
    options = [arg.split('=', 1)[0] for arg in argv if arg.startswith('--')]
    if len(options) != len(set(options)):
        parser.error('repeated options are not allowed')
    args = parser.parse_args(argv)
    manifest = Path.home() / 'Projects/Heroes-artifacts' / NATIVE_MANIFEST_RELATIVE
    result = [str(ROOT / 'continue-original-native.sh'), '--session-goal', 'checkpoint1585-remaining1176-v1',
              '--native-learner', str(manifest), '--native-sha256', NATIVE_MANIFEST_SHA256,
              '--interaction-guard', 'visible-interaction-budget-revealed-v1',
              '--hours', str(args.hours)]
    if args.check:
        result += ['--check']
    if args.cpu_cap is not None:
        result += ['--cpu-cap', str(args.cpu_cap)]
    return result


def run(argv):
    child, requested_signal = None, None

    def request_stop(signum, _frame):
        nonlocal requested_signal
        requested_signal = signum
        if child is not None:
            child.send_signal(signum)

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        arguments = command(argv)
        if requested_signal is not None:
            return 128 + requested_signal
        print('Goal: finish 1176 admitted games after checkpoint1585 (294 complete PPO updates). '
              'The verified remaining count is recalculated before every launch. '
              'Policy support: visible-interaction-budget-revealed-v1.', flush=True)
        # Preserve normal foreground signal handling. The owned controller handles
        # bounded batch completion and descendant cleanup; this wrapper waits.
        child = subprocess.Popen(arguments, env=os.environ.copy())
        if requested_signal is not None:
            child.send_signal(requested_signal)
        result = child.wait()
        return result if result >= 0 else 128 - result
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    started, status = time.monotonic(), 1
    try:
        status = run(sys.argv[1:] if argv is None else argv)
    except SystemExit as error:
        status = error.code if isinstance(error.code, int) else 1
    except OSError as error:
        print(f'Cannot start manual continuation: {error}', file=sys.stderr)
        status = 2
    finally:
        elapsed = time.monotonic() - started
        total = int(elapsed)
        print(f'Total running time: {total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d} '
              f'({elapsed:.2f} seconds); exit status: {status}.', flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
