"""Compute, check or write the reviewed fast-path pins of the local adapter.

    fast_path_pins.py [--write]

For every learner/collector file that the accepted local runner binds (the 629 parent's
``source_sha256``: every ``experiments/learner/*.py`` and ``dependencies/learner.lock``), compares
its bytes at the accepted commits (3268f2f produced 629; a8ac411 is
``run_local_training.ACCEPTED_IMPLEMENTATION``) with this tree. ``FAST_PATH_SOURCES`` must list
exactly the files that differ (other than the adapter itself and the reviewed scope-routing
file), each with its accepted digests and this tree's digest; ``LOCAL_ADAPTER_TRANSITION_AST`` must
be this adapter's AST digest. ``FAST_PATH_TREE`` must list exactly the other files of
``REVIEWED_TREE`` (collector, game interface, admission verifiers, runner, lock) whose bytes differ from the a8ac411 tree,
with the accepted git blob ('' when new) and this tree's SHA-256. Without ``--write`` it only
reports and exits non-zero on a mismatch.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ACCEPTED = ('3268f2f8cae6a01277c1629c5d0a9fb8958b5786', 'a8ac411d32658387598376c792136bc586edc5f9',
            'a92250fe96feedcb7c21c169bd69ad9c7281b5ef',
            '6921700f0570d68c0275b7fe837bb03fe58f01d4')
ADAPTER = 'experiments/learner/local_script_training.py'


def at(commit, name):
    try:
        return subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None


def expected_pins():
    names = sorted({line for commit in ACCEPTED for line in subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', commit, '--', 'experiments/learner', 'dependencies/learner.lock'],
        cwd=ROOT, text=True).split() if line.endswith('.py') and line.count('/') == 2 or line == 'dependencies/learner.lock'})
    pins = {}
    for name in names:
        if name == ADAPTER:
            continue
        accepted = sorted({hashlib.sha256(data).hexdigest() for commit in ACCEPTED if (data := at(commit, name)) is not None})
        path = ROOT / name
        if not path.exists():
            raise SystemExit('Accepted learner file is missing here: ' + name)
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        # Retain older629 parent transitions even when this file already equals
        # the later a922 continuation. These pins certify bytes, not equivalent
        # policy support; the adapter classifies that intervention separately.
        if any(current != digest for digest in accepted):
            pins[name] = (tuple(accepted), current)
    return pins


def expected_tree():
    from experiments.learner import local_script_training as local
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True)
    accepted = {}
    for line in git('ls-tree', '-r', '--full-tree', ACCEPTED[1], '--', *local.REVIEWED_TREE).splitlines():
        meta, name = line.split('\t', 1)
        accepted[name] = meta.split()[2]
    names = set(accepted) | set(git('ls-files', '--full-name', '--', *local.REVIEWED_TREE).split('\n'))
    names |= local.discovered_reviewed_sources(ROOT)
    learner = set(expected_pins())
    tree = {}
    for name in sorted(names - {''} - learner - {ADAPTER}):
        path = ROOT / name
        if not path.is_file():
            raise SystemExit('Reviewed file is missing here: ' + name)
        data = path.read_bytes()
        if local._git_blob(data) != accepted.get(name):
            tree[name] = (accepted.get(name, ''), hashlib.sha256(data).hexdigest())
    return tree


def adapter_ast():
    from experiments.learner.local_script_training import _adapter_ast_digest
    return _adapter_ast_digest((ROOT / ADAPTER).read_bytes())


def render(pins):
    lines = ['FAST_PATH_SOURCES = {']
    for name, (accepted, current) in sorted(pins.items()):
        lines.append(f"    '{name}': (")
        lines.append('        (' + ''.join(f"'{a}', " for a in accepted).rstrip(' ') + ('' if len(accepted) > 1 else '') + '),')
        lines.append(f"        '{current}'),")
    lines.append('}')
    return '\n'.join(lines)


def render_tree(tree):
    lines = ['FAST_PATH_TREE = {']
    for name, (accepted, current) in sorted(tree.items()):
        lines.append(f"    '{name}': ('{accepted}',")
        lines.append(f"        '{current}'),")
    lines.append('}')
    return '\n'.join(lines)


def replace_block(text, head, block):
    start = text.index(head); end = text.index('\n}\n', start) + 2
    return text[:start] + block + text[end:]


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--write', action='store_true'); a = p.parse_args()
    from experiments.learner import local_script_training as local
    pins = expected_pins()
    adapter = ROOT / ADAPTER
    if a.write:
        text = replace_block(adapter.read_text(), 'FAST_PATH_SOURCES = {', render(pins))
        adapter.write_text(text)
        tree = expected_tree()
        text = replace_block(text, 'FAST_PATH_TREE = {', render_tree(tree))
        adapter.write_text(text)
        digest = adapter_ast()
        start = text.index("LOCAL_ADAPTER_TRANSITION_AST = '"); end = text.index("'\n", start + 33) + 1
        adapter.write_text(text[:start] + f"LOCAL_ADAPTER_TRANSITION_AST = '{digest}'" + text[end:])
        print('wrote', len(pins), 'learner pins,', len(tree), 'tree pins and adapter AST', digest)
        return 0
    tree = expected_tree()
    ok = ({k: (tuple(v[0]), v[1]) for k, v in local.FAST_PATH_SOURCES.items()} == pins and local.FAST_PATH_TREE == tree
          and local.LOCAL_ADAPTER_TRANSITION_AST == adapter_ast())
    for name, (accepted, current) in sorted(pins.items()):
        print(name, [x[:12] for x in accepted], '->', current[:12])
    for name, (accepted, current) in sorted(tree.items()):
        print(name, accepted[:12] or 'new', '->', current[:12])
    print('pins match' if ok else 'PINS DIFFER: run with --write after review')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
