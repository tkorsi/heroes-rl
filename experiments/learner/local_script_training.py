"""User-launched local batches using the existing collector, verifier and PPO.

Historical checkpoint bytes and metadata are immutable. Explicit local bindings
relocate their files; new plans bind this executable source and native runtime.
The controller owns scheduling, the campaign clock and durable publication.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import torch

from . import builtin_first_run as pilot, builtin_training as batches
from . import builtin_training_resume as resume, checkpoints as cp
from . import interaction_guard
from .builtin_preservation import binding, durable_path, verify_pair
from .inference import source_identity
from experiments.headless.contextual_evidence import runtime_identity
from experiments.headless.env import VcmiLauncher

ROOT = Path(__file__).resolve().parents[2]
INTERFACE = 'adventure-capabilities-v1'
SCOPE = 'heroes-local-script-training-v1'
SESSION_SCHEMA = 'heroes-local-script-session-v1'
PROFILES = ('easier8x8', 'original')
SEED_NAMESPACES = {'easier8x8': {'engine_base': 20_000_000, 'policy_base': 30_000_000},
                   'original': {'engine_base': 40_000_000, 'policy_base': 50_000_000}}
require = resume.require
DESCRIPTOR_KEYS = frozenset(('plan', 'report', 'policy', 'resume', 'initial_checkpoint'))
HISTORICAL_ADAPTERS = frozenset(('experiments/learner/curriculum_training.py',
                               'experiments/learner/overnight_training.py'))
ROUTING_SCOPES = frozenset(('heroes-update146-curriculum-v3',
                          'heroes-overnight-curriculum-20261026', SCOPE))
ROUTING_MODULES = {'heroes-update146-curriculum-v3': 'curriculum_training',
                   'heroes-overnight-curriculum-20261026': 'overnight_training',
                   SCOPE: 'local_script_training'}
# The initial manual runner predates explicit map profiles. Only these released
# bytes may cross that orchestration transition; learner/PPO source stays exact.
ACCEPTED_ADAPTER_COMMIT = 'a8ac411d32658387598376c792136bc586edc5f9'
LOCAL_ADAPTER_ORIGINS = {
    'cf2021cd3b2225ced440b1459ad9220581a028c0': 'd2bba1396dceafd47d61dcec839e1ec4376d3bc18b763a7e8b1ee44308876518',
    '3268f2f8cae6a01277c1629c5d0a9fb8958b5786': 'd2bba1396dceafd47d61dcec839e1ec4376d3bc18b763a7e8b1ee44308876518',
    # The accepted implementation (run_local_training.ACCEPTED_IMPLEMENTATION) that the default path releases.
    ACCEPTED_ADAPTER_COMMIT: '499246f8ff707ddda8f166a4ed3dbd9c1d62928786f893e2d98da835bbcee3f3',
    # Official original-map run629->699: byte-identical accepted local adapter.
    'b4fe70cc0285a76716430b5f41b17b43ba828b12': '499246f8ff707ddda8f166a4ed3dbd9c1d62928786f893e2d98da835bbcee3f3',
    # Official checkpoint699 continuation, closed before the support intervention.
    'a92250fe96feedcb7c21c169bd69ad9c7281b5ef': '3195c79c5d0a0e62e499fd659a2815334236348b10ed4f7870959347f2d18363',
    # Guarded original-map run717->770, stopped on an unsupported mine query.
    '6921700f0570d68c0275b7fe837bb03fe58f01d4': 'a2dddd49fc5026fd2cd6842845dd00283400f4378d6092af88ee6492709827ef',
    # Official run770->916, stopped by terminal request replay accounting.
    'cb3849c376331aafdb940fb59dd45e143e5b4218': '5d42d69d0157849c85aa7201e1d6cfcf5850a0d96ac3272d1b25ed4d772ff9d9',
}
# These changes implement a declared policy-support intervention. Their exact
# source pins are checked alongside the execution pins, but they must never be
# reported as an identical-policy acceleration transition.
POLICY_SUPPORT_SOURCES = frozenset({
    'experiments/learner/builtin_rollout.py',
    'experiments/learner/contextual.py',
    'experiments/learner/contextual_model.py',
    'experiments/learner/contextual_rollout.py',
    'experiments/learner/evaluate_continuation.py',
    'experiments/learner/gameplay.py',
    'experiments/learner/ppo.py',
})
# Reviewed source pins: accepted historical digests and exact current bytes.
# POLICY_SUPPORT_SOURCES are classified as a behavior-support intervention;
# separate execution optimizations retain their exact-fast-path classification.
# A pin certifies reviewed bytes, not equivalent action support or trajectories.
FAST_PATH_SOURCES = {
    'experiments/learner/builtin_first_run.py': (
        ('80002aac82dfb9c5c4676584acde63b1c1825303a1f5a967fb506428b0e99fb2', 'a3ad9b28b05617259b0ec4422d166a728aeab523f9f14c2ce5313cad78160208',),
        '80002aac82dfb9c5c4676584acde63b1c1825303a1f5a967fb506428b0e99fb2'),
    'experiments/learner/builtin_gameplay.py': (
        ('ef604ae79abc47b31fa63d6e4f39da8e58959c349b4e1c0593a8d571b9020e0d', 'fc0ad6a7afdffe698f337b54bd83aba14b6aa76fa60bdd15a7cda4ae71de1831',),
        '21587802605cab4fd781734ac60e7de301013306331f18231d0f216574c69540'),
    'experiments/learner/builtin_rollout.py': (
        ('08d04449ee567146165417c8851a1f555d7921e921e0075284a773befa9516db', 'a9c2df339a63fa384bf6b53f1c182b197015f39182a59ca01c67e5970d37a67b', 'c2461ed510c8b9c58fa3fd86a53fc86e4b7e7caf316fbdec40092d53aa5073a6',),
        '08d04449ee567146165417c8851a1f555d7921e921e0075284a773befa9516db'),
    'experiments/learner/builtin_training.py': (
        ('00d51447c0a3fcf4f16ce020f0e970e6382ce889a037e228394e6fdcbbe70768', '3c00700df647f0a72ef17a6c2fab90a4d7e3b0f89dae394ad835e1192a527600',),
        '3c00700df647f0a72ef17a6c2fab90a4d7e3b0f89dae394ad835e1192a527600'),
    'experiments/learner/builtin_training_resume.py': (
        ('3853da06e4e510586cf1b4af0d251b63659dd9f14db394d871b1071d54ef53fb', '55a7c226b0791221512e2bf007975b45e5ceeddd71327a4ca89e00ca7ccb5423',),
        '55a7c226b0791221512e2bf007975b45e5ceeddd71327a4ca89e00ca7ccb5423'),
    'experiments/learner/contextual.py': (
        ('070203b248ac12949e192ac6c419832f66441010ac0f592dbf167fe3aa732dda', 'fc9188d3833f1508909d0d04922510111303d10fe3659d30f734a4875676242d',),
        '01d63bbf2279475c4f79fa1947b8662cd23814e4cced7616f68dff819e6f9358'),
    'experiments/learner/contextual_model.py': (
        ('0e887a870aea4a0a69e448c708b9917c163bcc5d41aeafdb1070ec396c5eea6f', '389434e4a181346498f4997f9e2ba1edc465d2e0b2583e8bca4a07119401bf2d', 'e30871a589a04555e2636c5ff2524700e0477c1bcd9d7fc936257fb40a326f72',),
        'e30871a589a04555e2636c5ff2524700e0477c1bcd9d7fc936257fb40a326f72'),
    'experiments/learner/contextual_rollout.py': (
        ('2213eb7d8e2d1556a6ae2dec3329c0b26d33a5422be2c7bd10dbc166a74ae346', 'afaec2277d4af982ce50dfb74e48be89956669ab7bc9aede2daa42dfb2d10637', 'bce5dbad0f69f8e415b93509f362e65b89fcddf3641bdb1ae9e8280bc7d0d205',),
        '2213eb7d8e2d1556a6ae2dec3329c0b26d33a5422be2c7bd10dbc166a74ae346'),
    'experiments/learner/evaluate_builtin_opponent.py': (
        ('3771cb05b93c0389eb7752274c94558ca9d3027538a93b3580837fe183926dc0', '6f3cb9727e9192a4e84ece61511d9c64a7281427a5355171b52d7dfb8babab45',),
        '6f3cb9727e9192a4e84ece61511d9c64a7281427a5355171b52d7dfb8babab45'),
    'experiments/learner/evaluate_continuation.py': (
        ('0a52b1e1a73ce78a4b7d1d3a59951bf7e93c32dd44f0dee100fa58dbdd655ed3', '6c64ef89d47f3986ff47a589a292d538ad75ae662589ae8cd643e46a1ba7470d',),
        '0a52b1e1a73ce78a4b7d1d3a59951bf7e93c32dd44f0dee100fa58dbdd655ed3'),
    'experiments/learner/first_run.py': (
        ('645e1ca237b2eee9f0338d0a8b8e388aac7ee5d952e205ea666e55481fecd81a', '8253af03d91cc6d5244a1d0be834c3aaadee1cca914923321ceb59035cb4da1c',),
        '645e1ca237b2eee9f0338d0a8b8e388aac7ee5d952e205ea666e55481fecd81a'),
    'experiments/learner/gameplay.py': (
        ('9688b93ba526704df51c7d376d27f46c013ec7bdc1ad09432e3de2a0f1fa99c4', 'f1b21b0b740a7087859d8602e0d7c59feeb88f8ccea6f2a17df57dfad402358f',),
        '9688b93ba526704df51c7d376d27f46c013ec7bdc1ad09432e3de2a0f1fa99c4'),
    'experiments/learner/inference.py': (
        ('b9537f61b0fa375741922c1308ee29acd5d4b218321d3775ad273d44a8b09c54', 'e90236841dbbfbc9c4e81d18708d7dad9496c65f6f91066f193762fc6635fc14',),
        'e90236841dbbfbc9c4e81d18708d7dad9496c65f6f91066f193762fc6635fc14'),
    'experiments/learner/ppo.py': (
        ('0a574d72b0fa94e02a1eab32150b547bfeb1a117b2a547094289a82bc0708e60', '4d9ff6194e1ab8fd46d4fa5adf643cfafa152ac593a60e7848cfc86d93bd7682', '6278c7675471aaa94f37745102e28dbf0ec3562e97e79c6ba42079ba0f794fcd',),
        '6278c7675471aaa94f37745102e28dbf0ec3562e97e79c6ba42079ba0f794fcd'),
}
# The rest of the code a fast-path run executes (collector, game interface, admission verifiers,
# runner, lock), compared with the accepted implementation's git blobs: only these files may differ,
# each with the accepted blob it replaces ('' = new) and the exact SHA-256 it must have here. The
# learner files above and this adapter (its AST digest below) have their own pins. Python files in
# the scanned folders count even when untracked. Generated by tools/acceleration3/fast_path_pins.py.
REVIEWED_TREE = ('experiments', 'validation/probes', 'dependencies/learner.lock', 'tools/run_local_training.py',
                 'tools/prepare_local_training.py', 'tools/prepare_vcmi.py', 'train-local.sh')
REVIEWED_SCAN = ('experiments', 'validation/probes')
FAST_PATH_TREE = {
    'experiments/episodes/scoring.py': ('2fee3829c13828963d0fec3a2aa215039656d4ff',
        '8bdd7158233685a3a90e08977c23702d60e1d2c596b07f9f303d29119964ec78'),
    'experiments/explorer/create_curriculum_scenario.py': ('',
        'd887ace1ba78c095f6485f2b0e894d40eb8a1eadd21482215b6625ed23db573a'),
    'experiments/explorer/create_garrison_scenarios.py': ('',
        '177f10c5a062476feff4f5503f7d62f878855a4fe66e41446299c77c1dd83aa0'),
    'experiments/headless/background_check.py': ('',
        'eb79612ff90cd684f3832b97b44c9a5fa84d16e905dca06b60c04d901801db5b'),
    'experiments/headless/builtin_env.py': ('e2290432ef893081d7e5beccb897c717ed4444b3',
        '1a4434db8b41df14fcfe8503f58fced9cc9db763c2baa55398f6edd3ed0b8c13'),
    'experiments/headless/builtin_evidence.py': ('0638d2ec9bc8c3a7a3588325316fbb59ae5f2515',
        'ec56c09ab106e29747e74a4acfe9754763a125886dbde7136797fab3214f6d8a'),
    'experiments/headless/builtin_replay.py': ('d7bde5cffd1124000838c1ba0f20b765b07efd39',
        '0cb7d936c5361c009b40ced053b56ca7f5f34e7d0cd5d7268439993c1e31a708'),
    'experiments/headless/contextual_env.py': ('e2a75d2327589a8ea27518823dc24318a95109ab',
        '4d555029eb5902f02a5883ca5386066fa150a4a91441b2163dfb8dc286de9ff5'),
    'experiments/headless/contextual_evidence.py': ('44b664e57028145c573d6397bd00f7c1227c5dc1',
        '494221e9a137fc7f8e1e237800b2cb79c572118125d4b0e4ef07810df65ff111'),
    'experiments/headless/contextual_protocol.py': ('e3e8b85713c15e54f7522e4f337a7ef8a184662d',
        '481120ae6411e353a1c02fb4ce00352cee0adf220f823263c4b6af5421742c96'),
    'experiments/headless/contextual_replay_history.py': ('e424d6d5bd0b23112f21d8ac641fbb4c4a9436da',
        '9e252f7f6b79f8a70ce020cc4114ae0919850307d4644082dbff8c9af69c179e'),
    'experiments/headless/digest_scope.py': ('',
        '482450e81c49a798595cd8a1f264590d051085e7aada90c4ad30163ad5e24a5b'),
    'experiments/headless/embedded_env.py': ('',
        '6e1dc2b22e460336c8f4282832559377db40a89b23a4eaa01052a8a1ba3de4a1'),
    'experiments/headless/embedded_evidence.py': ('',
        '71c6bc91c4fe74896e1fc6f43ef9d1f759f2d60e683c93a6623fdfad09c48b14'),
    'experiments/headless/env.py': ('3da283d9ebb185878d35ef989551ffd04cd18ab2',
        '1f5fb796cb67cd68bc9a755197f73128c16ee22bf19118428babf5ef281c519b'),
    'experiments/learner/interaction_guard.py': ('',
        '4ae5603f79f61339ce2df5a9a3005b27ceeee1d95d9bc7821bd9abfc8ccd699f'),
    'experiments/learner/prefix_pool.py': ('',
        '316e1fa5677563a640b34285e88d34b08a99f4c29d82b02dc1e5cb8b492b505a'),
    'tools/prepare_local_training.py': ('38812cd8ca73561212b45bbc30aae47c92fdc0a0',
        'cda0e3fbdcd4a227b9de5e1332391d000d9fdbb90ec16a28eb84272ba2d7b299'),
    'tools/run_local_training.py': ('0333046aa00dc09153998ed5f21825528546bfb7',
        '1a3849d1f8c1917f1c9a181f360c8fb057e3e4ebb9e710f1eb273d656cc883b4'),
    'validation/probes/audit_neural_rollout.py': ('9b0254de7e7d5d5007f722ce86385dbb007018ee',
        '5179f792366377667acd75d041fc16026c2907f8e3f8c82989e0aeebcd841773'),
    'validation/probes/validate_garrison_dialogs.py': ('',
        '1b4ebbebe5232bc13283ec052a7d18acdf80041552fe429f5d727fa0cc871f1a'),
    'validation/probes/validate_hero_policy.py': ('55badab454804f7fc2e1d544840ad830c1aab5ec',
        '574850634c7f617b0e9e5a9dada02cdb08e211ca7e05379af98133f2089d85f2'),
}
# The only execution choices a session may bind (``execution``); never an environment variable.
NUMERICAL_EXPERIMENTS = ('shared_graph_encodings',)
LOCAL_ADAPTER_TRANSITION_AST = '67b203b3620dd5e475102db143aa81354ffc5f613f5e2bc5a2a9a7064b45e9ea'


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def binding_descriptor(plan, report, policy, resume_file, initial_checkpoint):
    return {name: binding(path) for name, path in zip(
        ('plan', 'report', 'policy', 'resume', 'initial_checkpoint'),
        (plan, report, policy, resume_file, initial_checkpoint))}


def _bound(value):
    require(type(value) is dict and set(value) == {'path', 'sha256', 'bytes'},
            'Expected exact local file binding')
    require(binding(value['path']) == value, 'Bound local artifact changed')
    return Path(value['path'])


def _same_bytes(local, historical):
    require(all(local[k] == historical[k] for k in ('sha256', 'bytes')),
            'Relocation cannot change historical bytes')


def read_descriptor(descriptor):
    require(type(descriptor) is dict and set(descriptor) == DESCRIPTOR_KEYS,
            'Checkpoint descriptor requires full state and initial dependency')
    for value in descriptor.values():
        _bound(value)
    plan, report = (json.loads(Path(descriptor[k]['path']).read_text()) for k in ('plan', 'report'))
    require(report['plan_sha256'] == descriptor['plan']['sha256'], 'Parent plan/report mismatch')
    resume._completion(plan, report)
    for kind in ('policy', 'resume'):
        _same_bytes(descriptor[kind], report['after_checkpoint'][kind])
    _same_bytes(descriptor['initial_checkpoint'], resume._metadata(plan, report['plan_sha256'])['initial_checkpoint'])
    require(report['after_checkpoint'].get('exact_model_adam_rng_progress_pool') is True,
            'Parent lacks verified full state')
    require(plan['source']['interface'] == INTERFACE, 'Unsupported learned interface')
    return plan, report


def _without_scope_routes(contents):
    """Permit only the small adapter dispatches, not changes to the collector."""
    tree = ast.parse(contents)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'check_identity':
            while node.body and isinstance(node.body[0], ast.If):
                branch = node.body[0]
                test = branch.test
                if not (isinstance(test, ast.Compare) and len(test.ops) == 1
                        and isinstance(test.ops[0], ast.Eq) and len(test.comparators) == 1
                        and isinstance(test.comparators[0], ast.Constant)
                        and test.comparators[0].value in ROUTING_SCOPES
                        and ast.dump(test.left) == ast.dump(ast.parse('plan.get("scope")', mode='eval').body)
                        and not branch.orelse and len(branch.body) == 2
                        and isinstance(branch.body[0], ast.ImportFrom)
                        and isinstance(branch.body[1], ast.Return)):
                    break
                imported, returned = branch.body[0], branch.body[1].value
                if not (imported.level == 1 and imported.module == ROUTING_MODULES[test.comparators[0].value]
                        and len(imported.names) == 1 and imported.names[0].name == 'check_identity'
                        and isinstance(returned, ast.Call) and isinstance(returned.func, ast.Name)
                        and returned.func.id == (imported.names[0].asname or 'check_identity')
                        and len(returned.args) == 1 and isinstance(returned.args[0], ast.Name)
                        and returned.args[0].id == 'plan' and not returned.keywords):
                    break
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def _adapter_ast_digest(contents):
    tree = ast.parse(contents)
    tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == 'LOCAL_ADAPTER_TRANSITION_AST')]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def local_adapter_transition(commit, expected, historical, current):
    require(LOCAL_ADAPTER_ORIGINS.get(commit) == expected
            and hashlib.sha256(historical).hexdigest() == expected,
            'Unreviewed historical local adapter')
    require(_adapter_ast_digest(current) == LOCAL_ADAPTER_TRANSITION_AST,
            'Local adapter differs from the reviewed profile transition')
    return {'transition': 'reviewed_policy_support_adapter',
            'policy_support_contract': interaction_guard.VERSION,
            'native_interface_extension': 'map-garrison-v1',
            'policy_support_activation': 'explicit_session_and_plan_only',
            'historical_sha256': expected, 'current_sha256': hashlib.sha256(current).hexdigest(),
            'current_ast_sha256': LOCAL_ADAPTER_TRANSITION_AST,
            'historical_code_executed': False, 'historical_metadata_rewritten': False}


def fast_path_transition(name, expected, current):
    accepted, fast = FAST_PATH_SOURCES.get(name, ((), None))
    require(expected in accepted and hashlib.sha256(current).hexdigest() == fast,
            'Policy, PPO or collector implementation changed: ' + name)
    support_change = name in POLICY_SUPPORT_SOURCES
    return {'transition': 'reviewed_policy_support_transition' if support_change else 'reviewed_exact_fast_path',
            'historical_sha256': expected, 'current_sha256': fast,
            'policy_support_contract': interaction_guard.VERSION if support_change else None,
            'native_interface_extension': 'map-garrison-v1' if support_change else None,
            'policy_ppo_mathematics_unchanged': not support_change, 'historical_code_executed': False}


def _git_blob(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def discovered_reviewed_sources(root):
    """Discover authored Python, excluding only inactive generated virtualenvs.

    Accepted, tracked and explicitly pinned files are checked separately; this
    exclusion must never remove them from the reviewed source inventory.
    """
    root = Path(root).resolve()
    names = set()
    for folder in REVIEWED_SCAN:
        for directory, children, files in os.walk(root / folder, followlinks=False):
            children[:] = [name for name in children if name != '__pycache__']
            if '.venv' in children:
                environment = Path(directory) / '.venv'
                marker = environment / 'pyvenv.cfg'
                if (not environment.is_symlink() and marker.is_file() and not marker.is_symlink()
                        and subprocess.run(['git', '-C', str(root), 'check-ignore', '--no-index', '-q', '--',
                                            environment.relative_to(root).as_posix() + '/'],
                                           check=False, timeout=30).returncode == 0):
                    executable_paths = [sys.prefix, sys.exec_prefix, sys.executable, *sys.path]
                    executable_paths += [getattr(module, '__file__', None)
                                         for module in tuple(sys.modules.values()) if module is not None]
                    active = any(isinstance(path, str) and Path(path).resolve().is_relative_to(environment)
                                 for path in executable_paths)
                    require(not active, 'Generated virtualenv contributes executable source: '
                            + environment.relative_to(root).as_posix())
                    children.remove('.venv')
            names.update((Path(directory) / name).relative_to(root).as_posix()
                         for name in files if name.endswith('.py'))
    return names


def reviewed_tree(root=ROOT):
    """Every executable file matches accepted bytes or its reviewed source pin."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True, timeout=30)
    accepted = {}
    for line in git('ls-tree', '-r', '--full-tree', ACCEPTED_ADAPTER_COMMIT, '--', *REVIEWED_TREE).splitlines():
        meta, name = line.split('\t', 1)
        accepted[name] = meta.split()[2]
    names = set(accepted) | set(git('ls-files', '--full-name', '--', *REVIEWED_TREE).split('\n')) | set(FAST_PATH_TREE)
    names |= discovered_reviewed_sources(root)
    replaced = []
    for name in sorted(names - {''}):
        path = root / name
        require(path.is_file() and not path.is_symlink(), 'Missing or replaced reviewed source: ' + name)
        data = path.read_bytes()
        if name == 'experiments/learner/local_script_training.py':
            require(_adapter_ast_digest(data) == LOCAL_ADAPTER_TRANSITION_AST, 'Unreviewed local adapter')
        elif name in FAST_PATH_SOURCES:
            require(hashlib.sha256(data).hexdigest() == FAST_PATH_SOURCES[name][1], 'Unreviewed fast-path source: ' + name)
        elif name in FAST_PATH_TREE:
            require((accepted.get(name, ''), hashlib.sha256(data).hexdigest()) == FAST_PATH_TREE[name],
                    'Unreviewed fast-path source: ' + name)
            replaced.append(name)
        else:
            require(_git_blob(data) == accepted.get(name), 'Unreviewed fast-path source: ' + name)
    return replaced


def source_compatibility(parent):
    """Recognize only pinned execution changes and the disclosed support transition."""
    require(parent['interface'] == INTERFACE, 'Interface transition forbidden')
    require(parent['torch'].split('+')[0] == str(torch.__version__).split('+')[0],
            'Unreviewed PyTorch version transition')
    changes = {}
    for name, expected in parent['source_sha256'].items():
        historical = subprocess.check_output(['git', 'show', parent['git_commit'] + ':' + name],
                                             cwd=ROOT, timeout=10)
        require(hashlib.sha256(historical).hexdigest() == expected, 'Historical source is not committed')
        current_path = ROOT / name
        if name in HISTORICAL_ADAPTERS and not current_path.exists():
            changes[name] = 'Historical orchestration is not executed by this local adapter'
            continue
        require(current_path.is_file(), 'Missing learner source: ' + name)
        current = current_path.read_bytes()
        if hashlib.sha256(current).hexdigest() == expected:
            continue
        if name == 'experiments/learner/local_script_training.py':
            changes[name] = local_adapter_transition(parent['git_commit'], expected, historical, current)
            continue
        if name in FAST_PATH_SOURCES:
            changes[name] = fast_path_transition(name, expected, current)
            continue
        require(name == 'experiments/learner/builtin_first_run.py'
                and _without_scope_routes(historical) == _without_scope_routes(current),
                'Policy, PPO or collector implementation changed: ' + name)
        changes[name] = 'Recognized check_identity scope routing only; remaining AST exact'
    fast = sorted(k for k, v in changes.items() if type(v) is dict and v['transition'] == 'reviewed_exact_fast_path')
    support = sorted(k for k, v in changes.items() if type(v) is dict
                     and v['transition'] in ('reviewed_policy_support_transition', 'reviewed_policy_support_adapter'))
    return {'historical_commit': parent['git_commit'], 'changes': changes,
            'policy_ppo_unchanged': not (fast or support), 'reviewed_exact_fast_path': fast,
            'reviewed_policy_support_transition': support,
            'policy_support_contract': interaction_guard.VERSION if support else None,
            'native_interface_extension': 'map-garrison-v1' if support else None,
            'terminal_reward_and_ppo_formula_unchanged': True,
            'reviewed_fast_path_tree': reviewed_tree(), 'historical_metadata_rewritten': False}


def configuration(descriptor):
    plan, _ = read_descriptor(descriptor)
    cfg = deepcopy(plan['configuration'])
    require(cfg['interface'] == INTERFACE and cfg['max_days'] == 365
            and cfg['ordinary_choice_limit'] == 4960 and cfg['reserved_dialog_choices'] == 160
            and cfg['max_total_choices'] == 5120 and cfg['max_updates'] == 1
            and cfg['allow_verified_training_prefix'] is False,
            'Only the accepted complete-batch configuration is supported')
    return cfg


def operational_configuration(descriptor, batch_wall_seconds=None):
    """Change only the bounded update allowance; keep historical metadata intact.

    The four native games retain their original attempt/collection limits. A long
    exact PPO update gets a larger declared ceiling, still clamped by the fresh
    campaign's immutable compute deadline and the existing rollback watchdog.
    """
    cfg = configuration(descriptor)
    if batch_wall_seconds is not None:
        require(type(batch_wall_seconds) is int and 3600 <= batch_wall_seconds <= 21600,
                'Batch wall allowance must be an integer between 3600 and 21600 seconds')
        cfg['shared_wall_seconds'] = batch_wall_seconds
        cfg['phase_deadlines_seconds']['update'] = batch_wall_seconds - 60
    return cfg


def restore_parent(descriptor):
    """Restore full historical state without rewriting paths inside metadata."""
    original_torch, original_python = torch.get_rng_state().clone(), random.getstate()
    try:
        plan, report = read_descriptor(descriptor)
        proof = source_compatibility(plan['source'])
        options = {'historical_torch_version': plan['source']['torch']}
        verify_pair(descriptor['resume']['path'], descriptor['policy']['path'], plan, report,
                    migration=SimpleNamespace(checkpoint_options=options))
        metadata = resume._metadata(plan, report['plan_sha256'])
        progress, steps, sampler = resume._completion(plan, report)
        policy = cp.load_policy_checkpoint(descriptor['policy']['path'], expected_source_metadata=metadata,
            allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,), interface=INTERFACE, **options)
        model = policy.model.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), **plan['configuration']['adam'])
        generator = torch.Generator(device='cpu')
        state = cp.load_checkpoint(descriptor['resume']['path'], model, optimizer,
            expected_ppo_config=plan['configuration']['ppo'], expected_source_metadata=metadata,
            allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,), interface=INTERFACE,
            generators={'minibatch': generator}, at_rollout_boundary=True, **options)
        raw = torch.load(descriptor['resume']['path'], map_location='cpu', weights_only=True)
        require(state.progress == progress and state.sampler_state == sampler,
                'Restored progress or sampler differs')
        require(state.checkpoint == policy.checkpoint
                and cp.model_weight_digest(model) == policy.weights_sha256 == report['after_checkpoint']['weights_sha256'],
                'Restored policy/resume pair differs')
        for left, right in ((dict(model.state_dict()), raw['model_state']),
                (optimizer.state_dict(), raw['resume']['optimizer_state']),
                (state.opponent_pool.state_dict(), policy.opponent_pool.state_dict()),
                (torch.get_rng_state(), raw['rng']['torch_cpu']), (random.getstate(), raw['rng']['python']),
                (generator.get_state(), raw['rng']['torch_generators']['minibatch'])):
            resume._exact(left, right)
        require(optimizer.state and any(v['step'].item() > 0 for v in optimizer.state.values()),
                'Restored Adam state is absent')
        for value in descriptor.values():
            _bound(value)
        model.eval().requires_grad_(False)
        links = {k: descriptor[k] for k in ('plan', 'report', 'policy', 'resume')}
        links.update(progress=vars(progress), optimizer_steps=steps, source_proof=proof,
                     runtime_proof={'explicit_local_migration': True, 'historical_metadata_unchanged': True})
        return resume.ParentState(model, optimizer, generator, state.opponent_pool, progress, sampler,
                                  metadata, policy.checkpoint, policy.weights_sha256, links, steps)
    except BaseException:
        torch.set_rng_state(original_torch)
        random.setstate(original_python)
        raise


def _map_payloads(seat):
    from experiments.explorer.create_capability_scenarios import ordinary_payloads
    require(seat in (0, 1), 'Invalid seat')
    result = ordinary_payloads()
    color, other = ('red', 'blue')[seat], ('red', 'blue')[1-seat]
    hero = next(x for x in result['objects.json'] if x['instanceName'] == f'hero_{seat}')
    town = next(x for x in result['objects.json'] if x['instanceName'] == f'town_{1-seat}')
    town.update(x=6, y=5)
    hero.update(x=5, y=6)
    hero['options']['army'] = [{'type': 'core:pikeman', 'amount': 100}]
    result['objects.json'] = [hero, town]
    result['surface_terrain.json'] = [row[:8] for row in result['surface_terrain.json'][:8]]
    header = result['header.json']
    header['mapLevels']['surface'].update(width=8, height=8)
    header['players'][color].pop('mainTown', None)
    header['players'][other].pop('mainHero', None)
    header['players'][other]['heroes'] = {}
    header['players'][other]['mainTown'].update(x=4, y=5)
    header['name'] = f'Curriculum v1 stage3 learner {color}'
    header['description'] = ('Declared initial-state distribution: townless learner near opponent town; '
        'opponent starts without a hero. Native recruitment, fog, actions, battles and elimination rules unchanged.')
    return result


def map_identity(path, seat, profile='easier8x8'):
    require(type(seat) is int and seat in (0, 1) and profile in PROFILES,
            'Unknown map profile or seat')
    if profile == 'original':
        from .builtin_v8_migration import map_identity as original_identity
        return {**original_identity(path), **binding(path), 'profile': profile, 'seat': seat}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) == 3, 'Unexpected map archive members')
        actual = {name: json.loads(archive.read(name)) for name in names}
    require(actual == _map_payloads(seat), 'Map differs from the accepted open8x8 distribution')
    return {**binding(path), 'scenario': 'curriculum-v1-open8x8', 'stage': 3, 'seat': seat}


def check_clock(session, *, active=True):
    names = ('started_unix', 'started_monotonic', 'compute_deadline_unix',
             'compute_deadline_monotonic', 'absolute_deadline_unix', 'hard_deadline_monotonic')
    require(all(type(session[k]) in (int, float) and math.isfinite(session[k]) for k in names),
            'Invalid campaign clock')
    wall = session['absolute_deadline_unix'] - session['started_unix']
    mono = session['hard_deadline_monotonic'] - session['started_monotonic']
    require(wall > 300 and abs(wall-mono) < .001
            and abs((session['compute_deadline_unix']-session['started_unix'])
                    -(session['compute_deadline_monotonic']-session['started_monotonic'])) < .001
            and session['started_unix'] < session['compute_deadline_unix'] <= session['absolute_deadline_unix']-300,
            'Campaign clock or finalization reserve changed')
    if active:
        require(session['started_unix'] <= time.time() < session['compute_deadline_unix']
                and session['started_monotonic'] <= time.monotonic() < session['compute_deadline_monotonic'],
                'Campaign compute window expired or not started')


def check_execution(value):
    """The session's execution choices: exact by default; a numerical experiment only when named."""
    require(type(value) is dict
            and set(value) == {'implementation', 'overlapped_verification', 'prefix_helpers', 'text_leaf_replay',
                               'replay_threads', 'experiments'}
            and value['implementation'] == 'exact-fast-path-v1' and type(value['overlapped_verification']) is bool
            and type(value['prefix_helpers']) is int and 0 <= value['prefix_helpers'] <= 8
            and type(value['text_leaf_replay']) is bool
            and type(value['replay_threads']) is int and 0 <= value['replay_threads'] <= 8
            and (value['text_leaf_replay'] or value['replay_threads'] == 0)
            and type(value['experiments']) is list and value['experiments'] == sorted(set(value['experiments']))
            and set(value['experiments']) <= set(NUMERICAL_EXPERIMENTS), 'Unknown execution choice')
    return value


def execution_choice(*, overlapped_verification, prefix_helpers=0, text_leaf_replay=True, replay_threads=None,
                     experiments=()):
    """Replay threads default to the prefix helpers (the extra CPUs declared), and are 0 without the replay."""
    if replay_threads is None:
        replay_threads = prefix_helpers if text_leaf_replay else 0
    return check_execution({'implementation': 'exact-fast-path-v1', 'overlapped_verification': overlapped_verification,
                            'prefix_helpers': prefix_helpers, 'text_leaf_replay': text_leaf_replay,
                            'replay_threads': replay_threads, 'experiments': sorted(set(experiments))})


def apply_execution(plan):
    """Set this process's execution choices from a plan whose identity check passed."""
    from experiments.headless import background_check
    from . import ppo
    value = check_execution(plan['execution'])
    policy_support = check_policy_support(plan.get('policy_support'))
    background_check.FORK = value['overlapped_verification']
    ppo.PREFIX_HELPERS = value['prefix_helpers']
    ppo.REPLAY_TEXT_LEAVES = value['text_leaf_replay']
    ppo.TEXT_REPLAY_WORKERS = value['replay_threads']
    ppo.EXPERIMENTS = frozenset(value['experiments'])
    interaction_guard.ACTIVE_VERSION = policy_support
    return value


def check_policy_support(value):
    """An absent historical binding is disabled; only the registered version is accepted."""
    require(value is None or type(value) is str and value == interaction_guard.VERSION,
            'Unknown policy support intervention')
    return value


def check_policy_support_binding(plan, session):
    selected = check_policy_support(plan.get('policy_support'))
    require(selected == check_policy_support(session.get('policy_support')),
            'Batch/session policy support differs')
    require(selected == check_policy_support((plan.get('checkpoint_metadata') or {}).get('policy_support')),
            'Policy support differs from checkpoint metadata')
    return selected


def check_profile(session, assets):
    require(session['profile'] in PROFILES and session['profile'] == assets['default_profile'],
            'Explicit campaign profile differs from its machine configuration')
    namespace = assets.get('seed_namespace', SEED_NAMESPACES['easier8x8'])
    require(namespace == SEED_NAMESPACES[session['profile']]
            and session['training_engine_seed_base'] == namespace['engine_base']
            and session['training_policy_seed_base'] == namespace['policy_base'], 'Profile seed namespace changed')
    require(session['maps'] == assets['maps'], 'Campaign maps differ from its explicit machine configuration')
    require(len(session['maps']) == 2, 'Both seat maps required')
    if session['profile'] == 'original':
        require(session['maps'][0] == session['maps'][1], 'Original profile requires the same map for both seats')


def read_session(path, *, active=True):
    path = durable_path(path)
    session = json.loads(path.read_text())
    require(session['schema'] == SESSION_SCHEMA and session['scope'] == SCOPE, 'Unknown local session')
    check_policy_support(session.get('policy_support'))
    require(path == Path(session['run_dir'])/'session.json', 'Session path changed')
    check_clock(session, active=active)
    require(source_identity(INTERFACE) == session['source'] and not session['source']['tracked_worktree_dirty'],
            'Frozen learner source drift')
    require(pilot.implementation() == session['implementation'] and pilot.prerequisites() == session['prerequisites'],
            'Frozen implementation or acceptance evidence drift')
    require(operational_configuration(session['initial_parent'], session.get('batch_wall_seconds'))
            == session['configuration'], 'PPO configuration or declared operational budget changed')
    for value in (session.get('original_continuation') or {}).get('evidence', {}).values():
        _bound(value)
    _bound(session['asset_config'])
    assets = json.loads(Path(session['asset_config']['path']).read_text())
    check_profile(session, assets)
    for name, value in session.get('frozen_files', {}).items():
        require((ROOT/name).resolve().is_relative_to(ROOT) and binding(ROOT/name) == value,
                'Frozen controller file changed')
    require(runtime_identity(Path(session['runtime_bin'])) == session['runtime'], 'Runtime changed')
    require(len(session['maps']) == 2, 'Both seat maps required')
    for seat, value in enumerate(session['maps']):
        _bound(value)
        map_identity(value['path'], seat, session['profile'])
    return session


def schedule(session, sequence, seat, evaluate=False):
    require(type(sequence) is int and sequence >= 0 and type(seat) is int and seat in (0, 1),
            'Invalid sequence or seat')
    if not evaluate:
        require(sequence >= session['next_sequence'], 'Training sequence was already consumed')
    engine = 710001 if evaluate else session['training_engine_seed_base'] + 4*sequence
    policy = 720001 if evaluate else session['training_policy_seed_base'] + 4*sequence
    require(type(engine) is int and type(policy) is int and 0 < engine <= 2**31-5 and 0 < policy <= 2**31-5,
            'Seed range exceeds signed native limits')
    require(evaluate or not (set(range(engine, engine+4)) & set(range(710001, 710005))
                            or set(range(policy, policy+4)) & set(range(720001, 720005))),
            'Training seeds overlap held-out seeds')
    return [dict(episode=i+1, phase='evaluation' if evaluate else 'training', learner_player=seat,
                 engine_seed=engine+i, policy_seed=policy+i) for i in range(4)]


def check_lineage(parent_plan, execution, policy_support=None):
    """An exact session never continues a parent that a numerical experiment produced.

    A session running an experiment may start from an exact parent or continue its own experiment;
    the experiment that produced a checkpoint is named in its plan's checkpoint metadata.
    """
    produced = set((parent_plan.get('checkpoint_metadata') or {}).get('numerical_experiments', []))
    require(produced <= set(check_execution(execution)['experiments']),
            'Parent was produced by a numerical experiment this session does not run: ' + ', '.join(sorted(produced)))
    selected = check_policy_support(policy_support)
    inherited = check_policy_support((parent_plan.get('checkpoint_metadata') or {}).get('policy_support'))
    require(check_policy_support(parent_plan.get('policy_support')) == inherited,
            'Parent plan/metadata policy support differs')
    require(inherited is None or inherited == selected,
            'Guarded parent cannot silently change or drop its policy support')


def check_identity(plan):
    # One identity check hashes each unchanged file once (experiments.headless.digest_scope).
    from experiments.headless.digest_scope import scope
    with scope():
        return _check_identity(plan)


def _check_identity(plan):
    _bound(plan['local_session'])
    session = read_session(plan['local_session']['path'])
    evaluate = plan['mode'] == 'evaluation'
    require(plan['mode'] in ('training', 'evaluation') and plan['scope'] == SCOPE
            and plan['schema'] == (batches.EVALUATION if evaluate else batches.SCHEMA)+'-plan', 'Unknown batch scope')
    for key in ('profile', 'source', 'runtime', 'implementation', 'prerequisites', 'configuration', 'execution',
                'absolute_deadline_unix', 'compute_deadline_unix'):
        require(plan[key] == session[key], 'Batch/session identity differs: ' + key)
    check_policy_support_binding(plan, session)
    experiments = check_execution(plan['execution'])['experiments']
    require(plan['checkpoint_metadata'].get('numerical_experiments', []) == experiments,
            'Numerical experiment selection differs from its checkpoint metadata')
    require(plan['schedule'] == schedule(session, plan['batch_sequence'], plan['seat'], evaluate), 'Schedule changed')
    require(plan['game_map'] == session['maps'][plan['seat']]['path']
            and plan['map'] == map_identity(plan['game_map'], plan['seat'], session['profile']), 'Seat map changed')
    require(plan['binary'] == str(Path(session['runtime_bin'])/'vcmiclient') and plan['data'] == session['data'],
            'Runtime location changed')
    launcher = VcmiLauncher(Path(plan['binary']), Path(plan['data']), Path(plan['game_map']))
    require(launcher.identity() == plan['native'], 'Native binary or map identity changed')
    require(plan['native_opponents'] == [pilot.native_opponent(session['runtime'], seat) for seat in (0, 1)],
            'Native opponent changed')
    require(plan['checkpoint'] == plan['parent_descriptor']['policy'], 'Wrong policy checkpoint')
    for value in (plan['checkpoint'], plan['replay_binary'], *plan['parent_descriptor'].values()):
        _bound(value)
    parent_plan, parent_report = read_descriptor(plan['parent_descriptor'])
    check_lineage(parent_plan, plan['execution'], plan.get('policy_support'))
    progress, steps, _ = resume._completion(parent_plan, parent_report)
    require(all(plan['parent'][key] == plan['parent_descriptor'][key] for key in ('plan', 'report', 'policy', 'resume'))
            and plan['parent']['progress'] == vars(progress) and plan['parent']['optimizer_steps'] == steps,
            'Parent linkage or cumulative counters changed')
    resume._metadata(plan, '0'*64)  # Validate metadata linkage without creating a circular plan hash.
    require(Path(plan['execution_output']).parent.parent == Path(session['run_dir'])/'batches', 'Foreign batch output')


def prepare_batch(session_path, parent_descriptor, sequence, seat, evaluate=False, label=None):
    session_path = Path(session_path).resolve()
    session = read_session(session_path)
    state = restore_parent(parent_descriptor)
    check_lineage(read_descriptor(parent_descriptor)[0], session['execution'], session.get('policy_support'))
    require(operational_configuration(parent_descriptor, session.get('batch_wall_seconds'))
            == session['configuration'], 'Parent PPO configuration differs')
    rows = schedule(session, sequence, seat, evaluate)
    label = label or ('eval' if evaluate else 'train')
    require(label in ('train', 'eval', 'baseline', 'final', 'periodic'), 'Unknown batch label')
    require((label == 'train') == (not evaluate), 'Label/mode differs')
    output = Path(session['run_dir'])/'batches'/f'{label}-{sequence:06d}-seat{seat}'
    require(not output.exists(), 'Never overwrite or retry a consumed batch')
    if not evaluate:
        keys = {(r['engine_seed'], r['policy_seed'], r['learner_player']) for r in rows}
        for previous in (Path(session['run_dir'])/'batches').glob('*/plan.json'):
            old = json.loads(previous.read_text())
            require(not keys & {(r['engine_seed'], r['policy_seed'], r['learner_player'])
                                for r in old['schedule'] if r['phase'] == 'training'}, 'Training seeds reused')
    game_map = Path(session['maps'][seat]['path'])
    binary = Path(session['runtime_bin'])/'vcmiclient'
    launcher = VcmiLauncher(binary, Path(session['data']), game_map)
    plan = dict(schema=(batches.EVALUATION if evaluate else batches.SCHEMA)+'-plan', scope=SCOPE, profile=session['profile'],
        mode='evaluation' if evaluate else 'training', seat=seat, batch_sequence=sequence,
        source=session['source'], implementation=session['implementation'], prerequisites=session['prerequisites'],
        configuration=session['configuration'], execution=session['execution'], policy_support=session.get('policy_support'),
        runtime=session['runtime'], binary=str(binary), data=session['data'],
        game_map=str(game_map), map=map_identity(game_map, seat, session['profile']), native=launcher.identity(),
        replay_binary=binding(binary.parent/'vcmireplay'), native_opponents=[pilot.native_opponent(session['runtime'], s) for s in (0, 1)],
        schedule=rows, execution_output=str(output/'execution'), parent=state.bindings,
        parent_descriptor=parent_descriptor, checkpoint=parent_descriptor['policy'], local_session=binding(session_path),
        absolute_deadline_unix=session['absolute_deadline_unix'], compute_deadline_unix=session['compute_deadline_unix'],
        authorization={'scope': SCOPE, 'instruction': 'Explicit manual local script launch; no LLM supervision or automatic restart'},
        wall_scope='The immutable manual session includes evaluation, collection, checks, PPO, preservation and cleanup.',
        comparison_scope=('Original hero-management-v2 map; fixed paired held-out seeds are excluded from PPO and do not establish broader map generalization.'
            if session['profile'] == 'original' else
            'Same easier8x8 state distribution; fixed paired held-out seeds are not original-map generalization.'))
    plan['checkpoint_metadata'] = {**session['source'], 'parent_plan_sha256': state.bindings['plan']['sha256'],
        'parent_report_sha256': state.bindings['report']['sha256'], 'initial_checkpoint': state.bindings['policy'],
        'native_opponent': {k: v for k, v in plan['native_opponents'][0].items() if k != 'player'},
        'archived_neural_pool_executed': False, 'policy_support': session.get('policy_support')}
    if session['execution']['experiments']:
        # The plan, and through its metadata every checkpoint it produces, names the experiment.
        plan['checkpoint_metadata']['numerical_experiments'] = list(session['execution']['experiments'])
    check_identity(plan)
    write_new(output/'plan.json', plan)
    return output/'plan.json'


def read_plan(path, sha, *, check=True):
    """The plan at ``path``; ``check=False`` only for a caller whose next action is the full identity check."""
    path = durable_path(path)
    require(binding(path)['sha256'] == sha, 'Immutable batch plan changed')
    plan = json.loads(path.read_text())
    require(plan['execution_output'] == str(path.parent/'execution'), 'Batch execution path changed')
    require(plan.get('scope') == SCOPE, 'Unknown batch scope')
    if check:
        check_identity(plan)
    return plan


def worker_state(plan):
    """Runs right after builtin_training.worker's identity check of ``plan``, before any game."""
    apply_execution(plan)
    return restore_parent(plan['parent_descriptor'])


def valid_evaluation(plan, report):
    return (report.get('schema') == batches.EVALUATION+'-result' and plan.get('mode') == 'evaluation'
        and report.get('status') == 'finished' and report.get('training_performed') is False
        and all(report.get(k) is True for k in ('worker_complete', 'supervisor_cleanup_verified', 'within_shared_budget'))
        and all(report.get(k) == 0 for k in ('optimizer_updates', 'optimizer_steps', 'training_samples'))
        and len(report.get('attempts', [])) == len(plan.get('schedule', [])) == 4
        and all(all(row.get(k) == value for k, value in expected.items())
                and row.get('status') == 'passed' and row.get('training_samples') == 0
                and all(row.get(k) is True for k in ('completion_receipt', 'cleanup_verified', 'identity_verified', 'replay_verified'))
                for expected, row in zip(plan['schedule'], report['attempts'])))


def require_baselines(plan):
    session = read_session(plan['local_session']['path'])
    journal = json.loads((Path(session['run_dir'])/'campaign-report.json').read_text())
    accepted = set()
    for row in journal['batches']:
        if row.get('mode') != 'baseline' or row.get('status') != 'passed':
            continue
        bp = json.loads(_bound(row['plan']).read_text())
        br = json.loads(_bound(row['report']).read_text())
        check_policy_support_binding(bp, session)
        require(br['plan_sha256'] == row['plan']['sha256'], 'Baseline report/plan mismatch')
        require(bp['local_session'] == plan['local_session'] and bp['checkpoint'] == session['initial_parent']['policy']
                and bp['profile'] == session['profile']
                and bp['configuration'] == session['configuration'] and bp['runtime'] == session['runtime']
                and bp['source'] == session['source'] and bp['schedule'] == schedule(session, bp['batch_sequence'], bp['seat'], True)
                and valid_evaluation(bp, br), 'Unverified or foreign baseline')
        accepted.add(bp['seat'])
    require(accepted == {0, 1}, 'Both current-runtime frozen baselines must pass before PPO')


def descriptor_from_batch(plan_path):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    report_path = Path(plan['execution_output'])/'report.json'
    report = json.loads(report_path.read_text())
    resume._completion(plan, report)
    return binding_descriptor(plan_path, report_path, report['after_checkpoint']['policy']['path'],
                              report['after_checkpoint']['resume']['path'], plan['checkpoint']['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('execute', 'worker'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--started', type=float)
    parser.add_argument('--wall-started', type=float)
    args = parser.parse_args()
    if args.command == 'worker':
        require(args.started is not None and args.wall_started is not None, 'Worker clock required')
        # builtin_training.worker checks the plan's identity as its first action after reading it.
        result = batches.worker(args, plan_reader=lambda path, sha: read_plan(path, sha, check=False),
            state_reader=worker_state,
            heldout_checker=lambda *_: require_baselines(read_plan(args.plan, args.sha256)))
    else:
        result = pilot.execute(args, plan_reader=read_plan, worker_module=__spec__.name)
    print(json.dumps({k: result.get(k) for k in ('status', 'stop_reason', 'optimizer_updates', 'optimizer_steps', 'training_samples')}))
    return 0 if result.get('status') == 'finished' else 1


if __name__ == '__main__':
    sys.exit(main())
