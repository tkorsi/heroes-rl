"""Inclusive sample shares from a py-spy ``--subprocesses -f raw`` profile.

    profile_tree.py PROFILE.txt [--rate HZ] [--process SUBSTRING] [--top N] [--match REGEX]

Each line is ``process A;process B;...;frame;frame count``. Prints seconds per innermost Python
process command and the functions with the largest inclusive time (each function counted once per
stack), optionally restricted to one process kind or to stacks containing a regular expression.
"""
import argparse
from collections import Counter
import re


def main():
    p = argparse.ArgumentParser(); p.add_argument('profile'); p.add_argument('--rate', type=float, default=50)
    p.add_argument('--process'); p.add_argument('--top', type=int, default=40); p.add_argument('--match')
    a = p.parse_args()
    by_process, inclusive, total = Counter(), Counter(), 0
    for line in open(a.profile):
        stack, _, count = line.rstrip('\n').rpartition(' ')
        count = int(count); frames = stack.split(';')
        processes = [f for f in frames if f.startswith('process ')]
        command = processes[-1].split(':', 1)[1].strip('"') if processes else '?'
        words = command.split(' ')
        kind = ' '.join(w for w in words[1:4] if not w.startswith('/home/claude/bench2/campaigns'))[:90]
        if a.process and a.process not in command: continue
        if a.match and not re.search(a.match, stack): continue
        by_process[kind] += count; total += count
        for f in set(f for f in frames if not f.startswith('process ')):
            inclusive[f] += count
    print(f'total {total / a.rate:.1f} s of samples')
    for kind, n in by_process.most_common(12): print(f'{n / a.rate:8.2f} s  {kind}')
    print('--- inclusive')
    for f, n in inclusive.most_common(a.top): print(f'{n / a.rate:8.2f} s  {f[:150]}')


if __name__ == '__main__':
    main()
