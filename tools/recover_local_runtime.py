#!/usr/bin/env python3
"""Isolate an audited macOS build and rebuild the changed terminal/replay translation units.

The caller first copies the complete dependency/build into this checkout and
runs prepare_vcmi.py --check after installing the canonical accounting header.
Never invokes ninja's build mode or writes to the donor checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--donor-source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / 'vcmi'; build = source / 'build'
    donor = args.donor_source.resolve(); destination = args.destination.resolve()
    if destination.exists() or '/.codex/worktrees/' in str(destination):
        raise ValueError('Use a new durable destination outside managed worktrees')
    run(['python3', str(ROOT / 'tools/prepare_vcmi.py'), '--check'])
    # Extract the recorded compile/link recipes without running donor build rules.
    commands = run(['/opt/homebrew/bin/ninja', '-C', str(build), '-t', 'commands', 'vcmireplay', 'ExplorerRL', 'Nullkiller']).splitlines()
    units = [
        ('vcmireplay', 'experimentreplay/ContextualReplay.cpp'),
        ('vcmireplay', 'AI/ExplorerRL/ContextualEvidence.cpp'),
        ('ExplorerRL', 'AI/ExplorerRL/ContextualEvidence.cpp'),
        ('Nullkiller', 'AI/Nullkiller/AIGateway.cpp'),
    ]
    compiled_commands = []
    for target, unit in units:
        command = next(c for c in commands if ' -c ' + str(donor / unit) in c
                       and '/CMakeFiles/' + target + '.dir/' in c)
        compile_args = shlex.split(command.replace(str(donor), str(source)))
        cleaned = []
        i = 0
        while i < len(compile_args):
            if compile_args[i:i+2] in (['-Xclang', '-include-pch'], ['-Xclang', '-include']):
                i += 4
            else:
                cleaned.append(compile_args[i]); i += 1
        header = source / ('experimentreplay/StdInc.h' if target == 'vcmireplay' else 'AI/' + target + '/StdInc.h')
        cleaned += ['-include', str(header)]
        subprocess.run(cleaned, cwd=build, check=True)
        compiled_commands.append(cleaned)
    shutil.copytree(build / 'bin', destination, symlinks=True)
    linked_commands = []
    for output in ('bin/vcmireplay', 'bin/AI/libExplorerRL.dylib', 'bin/AI/libNullkiller.dylib'):
        link_line = next(c for c in commands if ' -o ' + output + ' ' in c)
        link = shlex.split(link_line.removeprefix(': && ').removesuffix(' && :'))
        link = [v.replace(str(donor / 'build/bin'), str(destination)) for v in link]
        link[link.index('-o') + 1] = str(destination / Path(output).relative_to('bin'))
        subprocess.run(link, cwd=build, check=True)
        linked_commands.append(link)
    patched = []
    for path in sorted(destination.rglob('*')):
        if not path.is_file() or path.is_symlink(): continue
        if path.name not in ('vcmiclient', 'vcmiserver', 'vcmireplay') and path.suffix != '.dylib': continue
        info = run(['otool', '-l', str(path)])
        if str(donor / 'build/bin') in info:
            subprocess.run(['install_name_tool', '-rpath', str(donor / 'build/bin'), str(destination), str(path)], check=True)
        subprocess.run(['codesign', '--force', '--sign', '-', str(path)], check=True)
        info = run(['otool', '-l', str(path)]) + run(['otool', '-L', str(path)])
        if str(donor) in info or '/.codex/worktrees/' in info:
            raise ValueError('Non-durable load dependency: ' + str(path))
        patched.append({'path': str(path.relative_to(destination)),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    receipt = {'schema': 'heroes-runtime-recovery-v1', 'donor_source': str(donor),
               'source': str(source), 'source_recipe_check': True,
               'rebuilt': units,
               'compile_commands': compiled_commands, 'link_commands': linked_commands, 'artifacts': patched,
               'load_commands_independent_of_donor_and_managed_worktrees': True}
    (destination.parent / 'recovery-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
