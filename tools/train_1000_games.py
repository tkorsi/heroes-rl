"""Time a manual 1,000-game continuation without changing its ownership or learning contract."""
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
NATIVE_MANIFEST_SHA256 = '53970605c4560bbe14b655442df373d3577efef3de0c5f824b95cfa22a74b573'
NATIVE_MANIFEST_RELATIVE = 'native-learner-builds/remove-telemetry-v1-attempt1/manifest.json'


def hours(value):
    result = float(value)
    if not math.isfinite(result) or not 0.25 <= result <= 336:
        raise argparse.ArgumentTypeError('hours must be finite and in [0.25, 336]')
    return result


def positive_cpu_count(value):
    result = int(value)
    if not 1 <= result <= 256:
        raise argparse.ArgumentTypeError('CPU cap must be in [1, 256]')
    return result


def command(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog='Stops after 250 fully admitted four-game PPO updates, or earlier on a deadline, '
               'manual stop or failure. Evaluation games do not count. The existing cloud '
               'ownership fence and latest-state verification remain mandatory; this script '
               'never releases a fence or selects a fixed checkpoint. Total runtime includes '
               'startup, evaluation, checkpoint preservation and cleanup. Check target_games_met in the '
               'runner summary to distinguish completion from an early stop.')
    parser.add_argument('--check', action='store_true', help='offline preflight only; no training')
    parser.add_argument('--hours', type=hours, default=10,
                        help='wall-clock safety limit, including cleanup reserve (default: 10)')
    parser.add_argument('--cpu-cap', type=positive_cpu_count,
                        help='optional resource cap; by default use all available CPUs')
    parser.add_argument('--recovery-receipt', type=Path,
                        help='separately verified recovery receipt, when required by the runner')
    # Repeated options can hide accidental overrides in a copied launch command.
    options = [arg.split('=', 1)[0] for arg in argv if arg.startswith('--')]
    if len(options) != len(set(options)):
        parser.error('repeated options are not allowed')
    args = parser.parse_args(argv)
    manifest = Path.home() / 'Projects/Heroes-artifacts' / NATIVE_MANIFEST_RELATIVE
    result = [str(ROOT / 'continue-original-native.sh'), '--session-games', '1000',
              '--native-learner', str(manifest), '--native-sha256', NATIVE_MANIFEST_SHA256,
              '--hours', str(args.hours)]
    if args.check:
        result += ['--check']
    if args.cpu_cap is not None:
        result += ['--cpu-cap', str(args.cpu_cap)]
    if args.recovery_receipt is not None:
        result += ['--recovery-receipt', str(args.recovery_receipt.resolve())]
    return result


def run(argv):
    child = None
    requested_signal = None

    def request_stop(signum, _frame):
        nonlocal requested_signal
        requested_signal = signum
        if child is not None:
            # Only the direct owned controller receives the signal. Its existing
            # handler owns bounded batch completion and descendant cleanup.
            child.send_signal(signum)

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        arguments = command(argv)
        if requested_signal is not None:
            return 128 + requested_signal
        print('Requested: 1,000 additional admitted training games (250 complete PPO updates).', flush=True)
        # Inherit terminal/output and the foreground process group, as the normal
        # launcher does. A terminal interrupt may reach both processes; the
        # controller stop handler is idempotent. Never kill a process group.
        child = subprocess.Popen(arguments, env=os.environ.copy())
        if requested_signal is not None:
            child.send_signal(requested_signal)
        result = child.wait()
        return result if result >= 0 else 128 - result
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    started = time.monotonic()
    status = 1
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
