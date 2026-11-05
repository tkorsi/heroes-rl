#!/usr/bin/env python3
"""Manually evaluate one frozen self-play game with doubled per-player choices.

Default check mode performs no gameplay and consumes no seeds. Evaluation
captures are diagnostic-only and never create training admission capabilities.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import benchmark_copied_training as copies
from tools import run_selfplay_training as campaign

BASE_COMMIT = 'b47c1b9aed1bbe8d2c52d87a02bcdac3e4931617'
CONTROLLER_COMMIT = '04cfe4d38a88d87c71a133a18dcbd895260624f6'
CONTROLLER_PATH = 'tools/run_selfplay_training.py'
STORAGE_PROGRESS_COMMIT = '3bd810060f0d1a211c985a723df4f67a10665a0e'
STORAGE_PROGRESS_PATHS = frozenset({
    CONTROLLER_PATH, 'experiments/learner/contextual_rollout.py',
    'experiments/learner/contextual_storage.py', 'experiments/learner/game_progress.py',
    'experiments/learner/selfplay_training.py', 'experiments/learner/local_script_training.py',
})
STORAGE_PROGRESS_NEW = frozenset({'experiments/learner/contextual_storage.py',
                                  'experiments/learner/game_progress.py'})
# Exact reviewed additions to policy support; legacy support retains its rules.
# These digests bind the reviewed opt-in observation and support release.
RECRUIT_ONLY_SOURCE_SHA256 = {
    'experiments/learner/local_script_training.py': 'ea50cf114414f73f119494aaa06af19ba26f4bf8ff54613bc8b94924f4031acd',
    'engine/ExperimentReplay/ContextualReplay.cpp': 'bebe4b98fc48abe048e32240767baacd8f443d71631dc3ca666178c2aada43e4',
    'engine/ExplorerRL/Contextual.cpp': '09ddb160a8c70e060f3e2b2093a2879e24aabab078a0b5f4770bbc0ec7a2db63',
    'engine/ExplorerRL/ContextualEvidence.cpp': '4e31338745b9cfbe867d99949112fb63340a3a389fe82458dff839ac51574154',
    'engine/ExplorerRL/ContextualEvidence.h': 'f43e34373fd36a52f5b3d377ed876af513b47db0ef5334b78a0adb5f0017631b',
    'engine/ExplorerRL/ExplorerRL.cpp': '97ca85b5363236143fa78fe3cce7b734acdff0a2b7fa4aae6b978fcb7c226d94',
    'engine/ExplorerRL/ExplorerRL.h': 'cac4746edd8d9f4443024202ca968db620d127d321fc52364691c5a494fc0b9d',
    'experiments/headless/contextual_env.py': 'a1ca9947b7c21262fa2b71520868da537c690fbaf3da8c55bacdea8419e640cd',
    'experiments/headless/contextual_protocol.py': '62dc02cb177181b7b51e5972d06c6cf73426ca09188324e89b255344eb78eebc',
    'experiments/headless/contextual_replay.py': '4a5173bd3d87ee1729f763daa067b5a2fad60e2cf319fbde852b3e43378355ab',
    'experiments/headless/contextual_replay_history.py': '5cf37e415f90c435c5fa910de10f9c35ede6feaad4012873c46d021cc5fca529',
    'experiments/learner/contextual.py': '69893b70595307a50cc936927eae7fd624ebe0355a040464864f798cfe676883',
    'experiments/learner/contextual_model.py': '14f505e78974cb1fd8c2e26afa956996f7cd23caef49a671a376fc00e44f4cfe',
    'experiments/learner/interaction_guard.py': 'd5b295bcdca6c37da6aa050007526b7b11e4da574ccae0693d61038a093418c4',
}
# Exact display-only delta after the native army-information acceptance. These
# files do not alter policy inputs, sampling, learning or replay verification.
DAY_PROGRESS_SOURCES = {
    'experiments/learner/contextual_rollout.py': {'before': '6c9439d52c304c504bc7a17227112f23b2cb7c47796fd404dcd8ccf1a9ca1b71',
        'after': '66cb89c252208f8de6468fa82b6fdc38af1eb25415ab04da73dacdb8d8c426ab'},
    'experiments/learner/game_progress.py': {'before': 'e5c1290de42be830392684a861e7508f4b34dc7469e4ac078d95566632ea057b',
        'after': '968ba409785f7a11f7d08506bb9286a8a6842307384100683667c754581ea015'},
    'experiments/learner/local_script_training.py': {'before': 'ea50cf114414f73f119494aaa06af19ba26f4bf8ff54613bc8b94924f4031acd',
        'after': 'e3f0da4db3583b1a80ea3a36c857ff5e1ab4a0066a9d37fa00905e8d58975c64'},
}
# This reviewed opt-in changes native artifact rules and their action offers.
# It requires its own native acceptance; it is not a display-only delta.
STARTING_EQUIPMENT_SOURCES = {
    'engine/ExplorerRL/Capabilities.cpp': '7af2b13258455d60107c866a2d72c52d16fc80f2879616f57a3516e26c4ef342',
    'engine/ExplorerRL/Notice.cpp': '1dea22d4c1c4bf667e90c6c5ead9d8c55a5969c7b06ff151b2c1f43f9d74fa95',
    'engine/patches/0031-no-starting-equipment.patch': '03959e119f28370abaecbc54ef5a264cbada063bf922f6ebc71f07352212775c',
    'experiments/headless/env.py': '4b7e4086b8bc6fcb334d71c704867a58e4cc12d70b438076cf54dfc1a3dcf563',
    'experiments/learner/contextual.py': 'dec7fe26a31823c6cd3b9aab2629a26dec696a4fdd0579dc000427730a830bd0',
    'experiments/learner/local_script_training.py': '8d3855d879581f7aedb24ff5d0ddfc9be05e004cdfb0246d0e87ed7c87e86aad',
}
STARTING_EQUIPMENT_NEW = frozenset({'engine/patches/0031-no-starting-equipment.patch'})
# Only the policy support and its source pins changed after native acceptance.
TOWN_COOLDOWN_SOURCES = {
    'experiments/learner/contextual_model.py': {'before': RECRUIT_ONLY_SOURCE_SHA256['experiments/learner/contextual_model.py'],
        'after': 'c7b4972f29aca29b5422b3027e5ef3fe68586e35f18365e6af540922541d097b'},
    'experiments/learner/interaction_guard.py': {'before': RECRUIT_ONLY_SOURCE_SHA256['experiments/learner/interaction_guard.py'],
        'after': '295accf489ed94e0a7e7e5d491706b1a7e9c8c09f1fd697cb0553ca7a944795f'},
    'experiments/learner/local_script_training.py': {'before': STARTING_EQUIPMENT_SOURCES['experiments/learner/local_script_training.py'],
        'after': 'efa4822f4ae560fbc2f0a78f25640f29f268c79d9ade7c4f8b50f73e0e6f82b0'},
}
HERO_RELATIVE_SOURCES = {
    'experiments/learner/contextual_model.py': {'before': TOWN_COOLDOWN_SOURCES['experiments/learner/contextual_model.py']['after'],
        'after': '12a3b8e9339a2b9fcf5809668a9f3dd3984a48337cb8db963e1c3da83d3816fe'},
    'experiments/learner/interaction_guard.py': {'before': TOWN_COOLDOWN_SOURCES['experiments/learner/interaction_guard.py']['after'],
        'after': 'eeec528257a70169353d7c8c8e15b22874405a20589e8bcdf7d63a241aea6639'},
    'experiments/learner/native_recipes.py': {'before': '6a35dc0694ef1b28e18855cf1476678d505ef7cb2af434a489bfee8250db5f26',
        'after': '61c4fa7dd90b12fe3c22e9ad4f9bf8342f25f9ddba4fea781bfcbaab434d6d93'},
    'experiments/learner/local_script_training.py': {'before': TOWN_COOLDOWN_SOURCES['experiments/learner/local_script_training.py']['after'],
        'after': '17ac5d27f4c57b5fc28ce2b4e5533316c142d13efa96ee36c726f5db14511b7b'},
}
# Exact semantics-preserving history/storage changes for the ten-game runner.
PARALLEL_HISTORY_SOURCES = {
    'experiments/learner/contextual.py': {'before': 'dec7fe26a31823c6cd3b9aab2629a26dec696a4fdd0579dc000427730a830bd0',
        'after': 'f4af40ce598dd6d0a35906168b22f62485dd27a20d7cef6270b2b01793552cff'},
    'experiments/learner/contextual_model.py': {'before': '12a3b8e9339a2b9fcf5809668a9f3dd3984a48337cb8db963e1c3da83d3816fe',
        'after': '811e59b3b39d48a29e37a5cfba37af20946d8a85c2fde8c420ca3e974d5fc0bf'},
    'experiments/learner/contextual_rollout.py': {'before': '66cb89c252208f8de6468fa82b6fdc38af1eb25415ab04da73dacdb8d8c426ab',
        'after': '5be32aca8d10122e353b6743b076fbab8258cf2acde4c8498ef0717681c085a4'},
    'experiments/learner/contextual_storage.py': {'before': '8d7c4b7bd1035291a9e95f136806ce33869a026d382a09b0064cd3569239496e',
        'after': 'c7b001a244592493831355fab4d425c855afa43b3f705c2eaa06303704e30736'},
    'experiments/learner/local_script_training.py': {'before': '17ac5d27f4c57b5fc28ce2b4e5533316c142d13efa96ee36c726f5db14511b7b',
        'after': '18ceabbd5e8e1486bf17d433e33ee220fba8a0f8505e9a3957bcf58d30cf933b'},
}
# Exact in-process history transport and immutable encoder-prefix optimization.
DIRECT_HISTORY_SOURCES = {
    'experiments/headless/contextual_env.py': {'before': 'a1ca9947b7c21262fa2b71520868da537c690fbaf3da8c55bacdea8419e640cd',
        'after': 'c532e2d45082021a4226b4493e792154c34ae5e56e82536bf8041b10fad909b6'},
    'experiments/learner/contextual_model.py': {'before': '811e59b3b39d48a29e37a5cfba37af20946d8a85c2fde8c420ca3e974d5fc0bf',
        'after': '4059e74ac24985b9f80897e6fc7c74d0034bcc627a8bce5b1bc0723d0188f9db'},
    'experiments/learner/contextual_rollout.py': {'before': '5be32aca8d10122e353b6743b076fbab8258cf2acde4c8498ef0717681c085a4',
        'after': '7996f0f7d0a1dce052caa78c777f7a9b515d9328aa94da6feee27472dd24e2f8'},
    'experiments/learner/local_script_training.py': {'before': '18ceabbd5e8e1486bf17d433e33ee220fba8a0f8505e9a3957bcf58d30cf933b',
        'after': 'd550cac1d1695736916d93cf40f311435c1cfed84f7a2334eb3885e021e3596c'},
}
# Reviewed Python admission/checkpoint/controller changes for a separate ten-game
# training contract. The native runtime, policy encoding and PPO kernel are unchanged.
TEN_BATCH_SOURCES = {
    'experiments/learner/checkpoints.py': {'before': 'a33dabaea5d34cdf2330cf07c8464b3f103463cc8309c5bc394387a692a0d68e',
        'after': '808a708c494e9d3099e82172e77ae393bf2971b2f6699ab43bfc140e313131ce'},
    'experiments/learner/selfplay_training.py': {'before': '01eda52e6ed91753262bbb2ac72a33f47698184bdba6c1f4e3c7cf5e27dd3e85',
        'after': 'c02baa2e2aec16b768afda8cfedcf10a7ebacf3d11bd41e39637fafb73d35427'},
    CONTROLLER_PATH: {'before': '3acb278be3534db303b3fcf3d0edfa146edbbd468f24e11e40df2bd152b1818e',
        'after': 'be95ae65753af7befaf0c47413152c3317d22d7a6f2e1e5ce95d4e1df01b8d37'},
}
# Explicit capture-only admission and prepared-native PPO path. Engine binaries
# and model encoding are unchanged; runtime receipts retain these exact deltas.
CYCLE_BATCH_SOURCES = {
    'experiments/learner/builtin_gameplay.py': {'before': '8a0b927474b7ce599bd260423249b075d3d89fcc7c08bb1b9756d617c4cdbc82',
        'after': '143d18cf0038708c24d61b81fb22f7593f21dd2f1c5018d33885e4c6d6f8881b'},
    'experiments/learner/checkpoints.py': {'before': '808a708c494e9d3099e82172e77ae393bf2971b2f6699ab43bfc140e313131ce',
        'after': 'a848db2c02665dd22a0499b5e185190279a8d4df653c7bb646fe02443a76a123'},
    'experiments/learner/game_progress.py': {'before': '968ba409785f7a11f7d08506bb9286a8a6842307384100683667c754581ea015',
        'after': '1f2eb870df6088c04926d675fac993e7788da2ed0a217f8bffd037528fcacee4'},
    'experiments/learner/ppo.py': {'before': 'b835502024f347863cf6110259e641903a1c294d9b29fa1eca6f461636849444',
        'after': 'd2bc6833098b596d52b26314ef0b2fbc87c827b1c5092d6c304f68552d5a2a8b'},
    'experiments/learner/selfplay_training.py': {'before': 'c02baa2e2aec16b768afda8cfedcf10a7ebacf3d11bd41e39637fafb73d35427',
        'after': '3f822e034aa5569cc7ccb9e78500de21aff4caeb7a654e630aa59142a774adfb'},
    'tools/run_selfplay_training.py': {'before': 'be95ae65753af7befaf0c47413152c3317d22d7a6f2e1e5ce95d4e1df01b8d37',
        'after': '461f726811f286011598c5d4e3f18b5c99cd283fe23fe76a1dc785e5476e050d'},
}
# Semantics-preserving actor/tail optimization and explicit capture-only binary
# handoff. Diagnostic validation and native binaries remain unchanged.
LOOP_SPEED_SOURCES = {
    'experiments/headless/contextual_env.py': {'before': 'c532e2d45082021a4226b4493e792154c34ae5e56e82536bf8041b10fad909b6',
        'after': '6fe0b8908964658aa2405c4aaa1347500ef6b5e97b99484fa01589bc4d0bd3cf'},
    'experiments/headless/env.py': {'before': '4b7e4086b8bc6fcb334d71c704867a58e4cc12d70b438076cf54dfc1a3dcf563',
        'after': 'e82f91dfea1c01444fc71a6658a510b5e0f3e7f770c32a17f54ac31169cc0786'},
    'experiments/learner/builtin_rollout.py': {'before': '08d04449ee567146165417c8851a1f555d7921e921e0075284a773befa9516db',
        'after': 'b9db3f2f20299ac1b2a608c0c621562f98bad85f5f678e9f0cf8633b6bd64696'},
    'experiments/learner/contextual_model.py': {'before': '4059e74ac24985b9f80897e6fc7c74d0034bcc627a8bce5b1bc0723d0188f9db',
        'after': '389d84b4693df9c18cfa4bdf416f2d9b00c07467a5ed794bbf6022ff13c13681'},
    'experiments/learner/contextual_rollout.py': {'before': '7996f0f7d0a1dce052caa78c777f7a9b515d9328aa94da6feee27472dd24e2f8',
        'after': '84636dcb0726a945a524556c3c196a89b0eb7f4699fc4379b5d7366e862e54a8'},
    'experiments/learner/contextual_storage.py': {'before': 'c7b001a244592493831355fab4d425c855afa43b3f705c2eaa06303704e30736',
        'after': 'e4c469394dab682ad3ad2d0518fbbaea102a10df844700228aee1fce447622d8'},
    'experiments/learner/selfplay_training.py': {'before': '3f822e034aa5569cc7ccb9e78500de21aff4caeb7a654e630aa59142a774adfb',
        'after': '2cb19a2ecd8c75368ae9a4b5a6998e18021d477b947b97844edab8ed32e2ff25'},
    'tools/run_selfplay_training.py': {'before': '461f726811f286011598c5d4e3f18b5c99cd283fe23fe76a1dc785e5476e050d',
        'after': '5be8888b2e7440ae0e4a0a3ec2541cd2ffab47bcbba9743f38348725ceb6c732'},
}
BASE = campaign.ARTIFACTS / 'selfplay-one-game-2x'
SCHEMA = 'heroes-selfplay-one-game-2x-v1'
BUDGET = dict(ordinary_choices_per_player=9920, reserved_dialog_choices_per_player=320,
              hard_choices_per_player=10240, max_days=365, capture_seconds=1200)
FIRST_WIN_BUDGET = {key: value * 16 if key != 'capture_seconds' else None
                    for key, value in BUDGET.items()}
ENGINE_SEED, POLICY_SEED = 1_960_000_000, 1_962_000_000
require, read, write, binding, bound = campaign.require, campaign.read, campaign.write, campaign.binding, campaign.bound


def verify_runtime_sources(implementation):
    """Bind the base plus exact reviewed storage/UI and support additions."""
    require(STORAGE_PROGRESS_PATHS <= implementation.keys(),
            'Evaluation source identity omitted a reviewed storage/progress dependency')
    require(RECRUIT_ONLY_SOURCE_SHA256.keys() <= implementation.keys(),
            'Evaluation source identity omitted a reviewed recruit-only dependency')
    require(STARTING_EQUIPMENT_SOURCES.keys() <= implementation.keys(),
            'Evaluation source identity omitted a reviewed starting-equipment dependency')
    def committed(commit, names):
        archive = subprocess.check_output(['git', 'archive', commit, *sorted(names)], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as source:
            files = {member.name: hashlib.sha256(source.extractfile(member).read()).hexdigest()
                     for member in source.getmembers() if member.isfile()}
        require(files.keys() == names, 'Committed evaluation source inventory differs')
        return files
    original = committed(BASE_COMMIT, implementation.keys() - STORAGE_PROGRESS_NEW - STARTING_EQUIPMENT_NEW)
    original.update(committed(STORAGE_PROGRESS_COMMIT, STORAGE_PROGRESS_PATHS))
    original.update(RECRUIT_ONLY_SOURCE_SHA256)
    original.update({name: pins['after'] for name, pins in DAY_PROGRESS_SOURCES.items()})
    original.update(STARTING_EQUIPMENT_SOURCES)
    original.update({name: pins['after'] for name, pins in TOWN_COOLDOWN_SOURCES.items()})
    original.update({name: pins['after'] for name, pins in HERO_RELATIVE_SOURCES.items()})
    original.update({name: pins['after'] for name, pins in PARALLEL_HISTORY_SOURCES.items()})
    original.update({name: pins['after'] for name, pins in DIRECT_HISTORY_SOURCES.items()})
    require(all(original.get(name) == pins['before'] for name, pins in TEN_BATCH_SOURCES.items()),
            'Ten-game source transition differs from its reviewed parent')
    original.update({name: pins['after'] for name, pins in TEN_BATCH_SOURCES.items()})
    require(all(original.get(name) == pins['before'] for name, pins in CYCLE_BATCH_SOURCES.items()),
            'Capture-only cycle source transition differs from its reviewed parent')
    original.update({name: pins['after'] for name, pins in CYCLE_BATCH_SOURCES.items()})
    require(all(original.get(name) == pins['before'] for name, pins in LOOP_SPEED_SOURCES.items()),
            'Main-loop source transition differs from its reviewed parent')
    original.update({name: pins['after'] for name, pins in LOOP_SPEED_SOURCES.items()})
    require(original == implementation, 'Evaluation changed an existing learner, adapter, engine or replay source')


def verify_native_validation_sources(accepted, current):
    """Retain the native receipt with only exact reviewed source transitions."""
    require(type(accepted) is dict and type(current) is dict,
            'Passive-army runtime lacks complete two-seat native validation sources')
    if accepted == current:
        return dict(kind='exact_native_validation_sources', changed=[])
    expected = dict(accepted)
    transitions = (('reviewed_display_only_day_progress', DAY_PROGRESS_SOURCES),
                   ('reviewed_policy_support_town_cooldown', TOWN_COOLDOWN_SOURCES),
                   ('reviewed_hero_relative_and_repeat_cooldown', HERO_RELATIVE_SOURCES),
                   ('reviewed_parallel_history_storage', PARALLEL_HISTORY_SOURCES),
                   ('reviewed_direct_history_transport', DIRECT_HISTORY_SOURCES),
                   ('reviewed_ten_game_training_contract', TEN_BATCH_SOURCES),
                   ('reviewed_capture_only_training_cycle', CYCLE_BATCH_SOURCES),
                   ('reviewed_capture_only_main_loop', LOOP_SPEED_SOURCES))
    for kind, transition in transitions:
        if all(expected.get(name) == pins['before'] for name, pins in transition.items()):
            expected.update({name: pins['after'] for name, pins in transition.items()})
        if expected == current:
            changed = {name: {'before': accepted[name], 'after': digest} for name, digest in current.items()
                       if accepted[name] != digest}
            return dict(kind=kind, changed=changed, native_capture_repeated=False)
    require(False, 'Passive-army runtime lacks complete two-seat native validation source agreement')


def source_identity():
    """Every reused source matches an explicit reviewed content identity."""
    from validation.probes.validate_capabilities import capability_implementation_identity
    from experiments.learner.inference import source_identity as learner_identity
    implementation = capability_implementation_identity()
    learner = learner_identity(campaign.INTERFACE)
    implementation.update(learner['source_sha256'])
    for name in (CONTROLLER_PATH, 'tools/benchmark_copied_training.py'):
        implementation[name] = binding(ROOT / name)['sha256']
    verify_runtime_sources(implementation)
    return dict(checkout=copies.checkout_identity(ROOT), source=learner,
        unchanged_runtime_base=BASE_COMMIT, implementation=implementation,
        reporting_controller_base=CONTROLLER_COMMIT,
        storage_progress_base=STORAGE_PROGRESS_COMMIT,
        recruit_only_sources=RECRUIT_ONLY_SOURCE_SHA256,
        day_progress_sources=DAY_PROGRESS_SOURCES,
        starting_equipment_sources=STARTING_EQUIPMENT_SOURCES,
        town_cooldown_sources=TOWN_COOLDOWN_SOURCES,
        hero_relative_sources=HERO_RELATIVE_SOURCES,
        parallel_history_sources=PARALLEL_HISTORY_SOURCES,
        direct_history_sources=DIRECT_HISTORY_SOURCES,
        ten_batch_sources=TEN_BATCH_SOURCES,
        cycle_batch_sources=CYCLE_BATCH_SOURCES,
        loop_speed_sources=LOOP_SPEED_SOURCES,
        driver=binding(__file__), launcher=binding(ROOT / 'play-one-selfplay-2x.sh'))


def policy_support(plan):
    from experiments.learner.interaction_guard import RECRUIT_ONLY_VERSIONS
    version = plan.get('policy_support')
    require(version in (campaign.GUARD, *RECRUIT_ONLY_VERSIONS),
            'Unknown or missing one-game policy support')
    return version


def seed_row(offset):
    require(type(offset) is int and 0 <= offset < 100000, 'Evaluation seed namespace exhausted')
    return dict(episode=1, engine_seed=ENGINE_SEED + offset,
                policy_seeds=[POLICY_SEED + 2 * offset, POLICY_SEED + 2 * offset + 1])


def seed_ledger():
    path = BASE / 'seed-reservations.json'
    if not path.exists():
        require(not BASE.exists() or not any(p.name != 'namespace.lock' for p in BASE.iterdir()),
                'Evaluation seed ledger is missing while prior artifacts remain; refusing seed reuse')
        return dict(schema=SCHEMA + '-seeds', reservations=[])
    ledger = read(path)
    require(ledger.get('schema') == SCHEMA + '-seeds' and type(ledger.get('reservations')) is list,
            'Invalid evaluation seed ledger')
    outputs = set()
    for index, row in enumerate(ledger['reservations']):
        require(row.get('offset') == index and row.get('schedule') == seed_row(index)
                and row.get('consumed_including_unstarted') is True and row['output'] not in outputs,
                'Evaluation seed reservation changed')
        require(read(BASE / f'reserved-{index:06d}.json') == row, 'Evaluation reservation receipt changed or is missing')
        outputs.add(row['output'])
    require({p.name for p in BASE.glob('reserved-*.json')}
            == {f'reserved-{i:06d}.json' for i in range(len(outputs))}, 'Unmatched evaluation reservation history')
    return ledger


def reserve(output):
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / 'namespace.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = seed_ledger()
        require(all(row['output'] != str(output) for row in ledger['reservations']), 'Evaluation output already consumed seeds')
        index = len(ledger['reservations'])
        reservation = dict(offset=index, output=str(output), schedule=seed_row(index), consumed_including_unstarted=True)
        ledger['reservations'].append(reservation)
        # Consumption precedes the receipt. An interrupted write fails closed.
        write(BASE / 'seed-reservations.json', ledger)
        receipt = BASE / f'reserved-{index:06d}.json'
        write(receipt, reservation, new=True)
        return binding(receipt)


def inspect_parent(lineage_path):
    from experiments.headless.contextual_evidence import runtime_identity
    from experiments.learner import selfplay_training as training
    from experiments.learner.native_execution import native_execution_choice
    identity = source_identity()
    lineage = campaign.inspect_lineage(lineage_path)
    selected = lineage['selected']; pair = selected['checkpoint']
    launch = read(bound(selected['launch']))
    require(pair['progress']['updates'] == 1753 and lineage['admitted_games'] == 100
            and launch['source']['checkout']['commit'] == BASE_COMMIT,
            'This evaluation requires the completed published checkpoint 1753 lineage')
    require(launch['policy_support'] == campaign.GUARD
            and runtime_identity(Path(launch['binary']).parent) == launch['runtime'], 'Accepted engine or policy support changed')
    bound(launch['replay_binary']); bound(launch['native_manifest'])
    execution = native_execution_choice(campaign.NATIVE, campaign.NATIVE_SHA, cpu_cap=16, require_acceptance=False)
    require(execution == launch['execution'], 'Accepted native learner execution profile changed')
    game_map = binding(launch['game_map'])
    require(game_map['sha256'] == launch['native']['map']['sha256'], 'Original map changed')
    state = training.restore_checkpoint_pair(pair, expected_metadata=pair['metadata'], expected_ppo_config=campaign.PPO,
        expected_optimizer_steps=lineage['optimizer_steps'])
    require(state.progress.updates == 1753 and state.weights_sha256 == pair['weights_sha256'], 'Restored evaluation policy changed')
    protected = dict(lineage['protected'])
    for value in (lineage['lineage'], lineage['seed_ledger'], game_map, launch['native_manifest']):
        protected[value['path']] = value
    campaign.verify_protected(protected)
    return dict(source=identity, parent=selected, protected=protected, checkpoint=1753,
        model_checkpoint=state.checkpoint, weights_sha256=state.weights_sha256,
        optimizer_steps=lineage['optimizer_steps'], runtime=launch['runtime'], native=launch['native'],
        binary=launch['binary'], data=launch['data'], game_map=launch['game_map'], replay_binary=launch['replay_binary'],
        execution=execution, native_manifest=launch['native_manifest'])


def prepare(args):
    require(math.isfinite(args.seconds) and 180 <= args.seconds <= 7200, 'Use a finite total budget from 180 to 7200 seconds')
    checked = inspect_parent(args.lineage)
    output = copies.private_path(args.output or BASE / (
        'run-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(4)), fresh=True)
    reservation = reserve(output)
    output.mkdir(parents=True, mode=0o700)
    now, wall = time.monotonic(), time.time()
    plan = dict(schema=SCHEMA + '-plan', **checked, output=str(output), budget=BUDGET, policy_support=campaign.GUARD,
        physical_games=1, learner_player_for_reporting=0, purpose='diagnostic', training_input=False,
        optimizer_updates=0, reservation=reservation, schedule=read(bound(reservation))['schedule'],
        supervisor_pid=os.getpid(), maximum_seconds=args.seconds,
        hard_deadline_monotonic=now + args.seconds, hard_deadline_unix=wall + args.seconds)
    write(output / 'plan.json', plan, new=True)
    return plan


def check_identity(plan):
    from experiments.headless.contextual_evidence import runtime_identity
    require(plan['schema'] == SCHEMA + '-plan' and plan['source'] == source_identity(), 'Frozen evaluation source changed')
    require(plan['budget'] == BUDGET and plan['physical_games'] == 1 and plan['purpose'] == 'diagnostic'
            and plan['training_input'] is False and plan['optimizer_updates'] == 0
            and plan['policy_support'] == campaign.GUARD, 'Evaluation contract changed')
    require(runtime_identity(Path(plan['binary']).parent) == plan['runtime'], 'Evaluation engine runtime changed')
    bound(plan['replay_binary'])
    bound(plan['native_manifest'])
    reservation = read(bound(plan['reservation']))
    require(reservation in seed_ledger()['reservations'] and plan['schedule'] == reservation['schedule'],
            'Evaluation seed binding differs')
    campaign.verify_protected(plan['protected'])


def frozen_models(plan):
    from experiments.learner import selfplay_training as training
    pair = plan['parent']['checkpoint']
    loaded = tuple(training.load_policy(bound(pair['policy']), pair['metadata']) for _ in range(2))
    require(all(item.checkpoint == plan['model_checkpoint'] and item.weights_sha256 == plan['weights_sha256']
                for item in loaded), 'Frozen model identity differs from the published parent')
    models = tuple(item.model.eval().requires_grad_(False) for item in loaded)
    require(not ({p.data_ptr() for p in models[0].parameters()} & {p.data_ptr() for p in models[1].parameters()}),
            'Both frozen players must own separate model tensors')
    return models


def verify_capture(folder, plan, models, remaining, *, identity_check=None, analyze=None, parallel_replay=False):
    from experiments.headless.contextual_protocol import declared_army_information
    from experiments.learner.contextual_rollout import behavior_guard_version
    from experiments.learner.contextual_storage import load_collected
    from experiments.learner.contextual_model import weights_sha256
    from experiments.learner import selfplay_training as training
    from experiments.headless import contextual_replay
    guard = policy_support(plan)
    budget = plan.get('budget', BUDGET)
    from experiments.headless.process_deadline import guarded_popen, close_owned_process
    replay_process = None
    try:
        manifest = read(folder / 'manifest.json')
        originals = training._files(folder, manifest)
        if parallel_replay:
            # Native replay independently validates this sealed capture. It
            # shares no model/RNG and cannot grant admission or update weights.
            seconds = remaining()
            with (Path(plan['output']) / 'replay-worker.log').open('xb') as log:
                replay_process = guarded_popen([sys.executable, '-m', 'experiments.headless.contextual_replay',
                    str(folder), '--binary', str(bound(plan['replay_binary'])), '--data', plan['data'],
                    '--output', str(folder / 'evaluation-verification'), '--timeout', str(seconds)],
                    deadline_monotonic=time.monotonic() + seconds, deadline_wall=time.time() + seconds,
                    evidence_path=Path(plan['output']) / 'replay-native-process-deadline.jsonl', cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        collected = load_collected(folder / 'neural-rollout.json')
        require(declared_army_information(manifest, campaign.INTERFACE) == plan.get('army_information'),
                'Evaluation army information extension differs from its plan')
        require(manifest.get('launch', {}).get('requested_settings', {}).get('server', {}).get(
                    'experimentEquipmentPolicy') == plan.get('equipment_policy'),
                'Evaluation starting-equipment rule differs from its plan')
        require(json.loads(collected.manifest_json) == manifest,
                'Persisted paired evaluation capture changed')
        require(manifest['episode'] == '1' and manifest['seed'] == plan['schedule']['engine_seed']
                and manifest['environment'] == campaign.INTERFACE and manifest['status'] in ('completed', 'truncated')
                and manifest['identity']['native_runtime_sha256'] == plan['runtime']
                and all(manifest['identity'].get(key) == value for key, value in plan['native'].items()),
                'Evaluation manifest runtime, seed or terminal boundary differs')
        for player, (rollout, model, config) in enumerate(zip(collected.rollouts, models, collected.model_configs_json)):
            require(rollout.player == player and rollout.purpose == 'diagnostic' and rollout.origin == 'native'
                    and rollout.role == ('learner' if player == 0 else 'frozen_opponent')
                    and rollout.optimization_eligible is False and rollout.opponent_pool == campaign.POOL
                    and rollout.behavior_checkpoint == rollout.opponent_checkpoint == plan['model_checkpoint']
                    and rollout.weights_sha256 == rollout.opponent_weights_sha256 == plan['weights_sha256'] == weights_sha256(model)
                    and json.loads(config) == model.config.to_dict(), 'Frozen evaluation policy identity or exclusion differs')
            require((behavior_guard_version(rollout.decisions) == guard or not rollout.decisions and not rollout.samples)
                    and len(rollout.decisions) <= budget['hard_choices_per_player']
                    and all(rollout.history[decision.decision_seq].day <= budget['max_days'] for decision in rollout.decisions),
                    'Evaluation guard, choices or sampled-day boundary differs')
        policies = training.reconstruct_policies(collected, models, plan['schedule']['policy_seeds'], remaining)
        if replay_process is None:
            replay = contextual_replay.verify_episode(folder, binary=bound(plan['replay_binary']), data=Path(plan['data']),
                output=folder / 'evaluation-verification', timeout=remaining())
        else:
            returncode = replay_process.wait(timeout=remaining())
            replay = read(folder / 'evaluation-verification/contextual-replay-verification.json')
            require(returncode == 0, f"Native replay worker failed: {replay.get('error', returncode)}")
        require(replay.get('schema') == contextual_replay.REPORT_SCHEMA and replay.get('verified') is True
                and replay.get('histories', {}).get('verified') is True
                and set(replay['histories']['players']) == {'red', 'blue'},
                f"Exact two-player native replay failed: {replay.get('error', 'invalid proof')}")
        require(read(folder / 'evaluation-verification/contextual-replay-verification.json') == replay
                and training._files(folder, manifest) == originals, 'Evidence changed during evaluation verification')
        remaining(); (check_identity if identity_check is None else identity_check)(plan)
        proof = dict(policies=policies, replay=replay, training_input=False, optimizer_updates=0, policy_support=guard,
                     choices_by_player=[len(rollout.decisions) for rollout in collected.rollouts],
                     parallel_native_replay=parallel_replay)
        if plan.get('until_first_win') and manifest['status'] == 'completed':
            proof['terminal_day'] = max(frame.day for rollout in collected.rollouts
                                        for frame in rollout.history if frame.kind == 'terminal')
        if analyze is not None:
            proof['analytics'] = analyze(collected)
            remaining()
        write(folder / 'evaluation-policy-reconstruction.json', proof, new=True)
        return proof, manifest
    finally:
        if replay_process is not None:
            close_owned_process(replay_process, timeout=10,
                deadline_monotonic=plan['hard_deadline_monotonic'], deadline_wall=plan['hard_deadline_unix'])


def collect(plan, models):
    from experiments.learner.contextual_rollout import ContextualCollector
    from experiments.learner.contextual_storage import save_collected
    from experiments.learner.game_progress import GameProgress, progress_path
    from experiments.headless.env import HeadlessSelfPlayEnv, Limits
    from experiments.headless.process_deadline import DeadlineVcmiLauncher
    from validation.probes.validate_hero_policy import wall_limit, retain_failed_evidence
    guard = policy_support(plan)
    output = Path(plan['output'])
    budget = plan.get('budget', BUDGET)
    progress = GameProgress(progress_path(output, 1), episode=1,
        ordinary=budget['ordinary_choices_per_player'], reserved=budget['reserved_dialog_choices_per_player'])
    deadline = plan['hard_deadline_monotonic'] - 30
    wall = plan['hard_deadline_unix'] - 30
    if budget['capture_seconds'] is not None:
        deadline = min(deadline, time.monotonic() + budget['capture_seconds'])
        wall = min(wall, time.time() + budget['capture_seconds'])
    launcher = DeadlineVcmiLauncher(Path(plan['binary']), Path(plan['data']), Path(plan['game_map']),
        deadline_monotonic=deadline, deadline_wall=wall, evidence_path=output / 'native-process-deadline.jsonl',
        **({'equipment_policy': plan['equipment_policy']} if 'equipment_policy' in plan else {}))
    class DayBoundedCollector(ContextualCollector):
        def act(self, request):
            require(request.record['day'] <= budget['max_days'], f"A sampled evaluation action exceeds the preserved day{budget['max_days']} bound")
            return super().act(request)
    collector = DayBoundedCollector(models, (plan['model_checkpoint'],) * 2, learner_player=0,
        opponent_pool=campaign.POOL, interface=campaign.INTERFACE, purpose='diagnostic', origin='native',
        policy_seeds=tuple(plan['schedule']['policy_seeds']), guard_version=guard)
    observation_options = ({'army_information': plan['army_information']} if 'army_information' in plan else {})
    env = HeadlessSelfPlayEnv(launcher, output, Limits(max_days=budget['max_days'], max_actions=budget['hard_choices_per_player'],
        watchdog_seconds=30, startup_seconds=30, shutdown_seconds=20, bridge_timeout=30),
        ledger=True, replay=True, interface=campaign.INTERFACE,
        agents={color: dict(checkpoint=plan['model_checkpoint'], kind='frozen-neural', purpose='diagnostic')
                for color in ('red', 'blue')}, **observation_options)
    try:
        with wall_limit(campaign.remaining(dict(hard_deadline_monotonic=deadline, hard_deadline_unix=wall))):
            collected = collector.run_episode(env, episode=1, seed=plan['schedule']['engine_seed'],
                max_requests_per_player=budget['ordinary_choices_per_player'],
                max_dialog_requests_per_player=budget['reserved_dialog_choices_per_player'], progress=progress)
            progress.phase('storing')
            save_collected(env.episode_dir / 'neural-rollout.json', collected)
            return env.episode_dir
    except Exception as error:
        progress.phase('failed')
        attempt = dict(status='failed', error=f'{type(error).__name__}: {error}', training_input=False)
        left = min(plan['hard_deadline_monotonic'] - time.monotonic(), plan['hard_deadline_unix'] - time.time())
        if left > 1:
            retain_failed_evidence(env, collector, attempt, error, cleanup_seconds=min(20, left))
        write(output / 'capture-failure.json', attempt, new=True)
        raise
    finally:
        try:
            env.close()
        finally:
            launcher.close_owned_groups(timeout=10); launcher.assert_closed()


def worker(path, sha, *, identity_check=None, model_loader=None, result_schema=SCHEMA, checkpoint=1753, before_verify=None):
    from validation.probes.validate_hero_policy import wall_limit
    from experiments.learner.game_progress import progress_path, set_phase
    identity_check = check_identity if identity_check is None else identity_check
    model_loader = frozen_models if model_loader is None else model_loader
    require(binding(path)['sha256'] == sha, 'Evaluation plan changed')
    plan = read(path); output = Path(plan['output'])
    require(os.getppid() == plan['supervisor_pid'], 'Only the lease-owning supervisor may start evaluation')
    write(output / 'worker-claim.json', dict(pid=os.getpid(), plan_sha256=sha), new=True)
    report = dict(schema=result_schema + '-result', status='running', plan=binding(path), checkpoint=checkpoint,
        physical_games=1, purpose='diagnostic', training_input=False, training_samples_admitted=0, optimizer_updates=0,
        budget=plan.get('budget', BUDGET), policy_support=policy_support(plan), owned_cleanup_verified=False)
    if 'army_information' in plan:
        report['army_information'] = plan['army_information']
    if 'equipment_policy' in plan:
        report['equipment_policy'] = plan['equipment_policy']
    write(output / 'report.json', report, new=True)
    try:
        with wall_limit(campaign.remaining(plan, 30)):
            identity_check(plan); campaign.configure_native(plan['execution'])
            models = model_loader(plan)
        # The capture and verification timers are sequential: wall_limit rejects
        # nesting, while the process guardian bounds the entire worker.
        folder = collect(plan, models)
        del models
        if before_verify is not None:
            before_verify(plan, folder)
        set_phase(progress_path(output, 1), 'verifying')
        with wall_limit(campaign.remaining(plan, 30)):
            proof, manifest = verify_capture(folder, plan, model_loader(plan), lambda: campaign.remaining(plan, 30),
                                             identity_check=identity_check,
                                             **({'parallel_replay': True} if before_verify is not None else {}))
            require(binding(path)['sha256'] == sha, 'Evaluation plan changed during execution')
            report.update(status='passed', episode_directory=str(folder), episode_status=manifest['status'],
                termination_reason=manifest['termination_reason'], outcomes=manifest['outcomes_by_player'],
                natural_terminal_after_horizon=manifest.get('natural_terminal_after_horizon'),
                rewards=manifest['rewards_by_player'], choices_by_player=proof['choices_by_player'],
                native_replay_verified=True, both_policies_reconstructed=True,
                proof=binding(folder / 'evaluation-policy-reconstruction.json'), owned_cleanup_verified=True)
            set_phase(progress_path(output, 1), 'verified')
    except Exception as error:
        set_phase(progress_path(output, 1), 'failed')
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        try:
            report['protected_inputs_unchanged'] = campaign.verify_protected(plan['protected'])
        except Exception as error:
            report.update(status='failed', protection_error=str(error))
        write(output / 'report.json', report)
    return report


def supervise(plan, *, worker_script=None, result_schema=SCHEMA, checkpoint=1753):
    from experiments.headless.process_deadline import guarded_popen, close_owned_process, cleanup_from_evidence
    from experiments.learner.game_progress import ProgressDisplay
    output, process, errors = Path(plan['output']), None, []
    display = ProgressDisplay(output / 'progress')
    plan_binding = binding(output / 'plan.json')
    try:
        with (output / 'worker.log').open('xb') as log:
            script = Path(__file__).resolve() if worker_script is None else Path(worker_script).resolve()
            process = guarded_popen([sys.executable, str(script), '_worker', '--plan', str(output / 'plan.json'),
                '--sha256', plan_binding['sha256']],
                deadline_monotonic=plan['hard_deadline_monotonic'], deadline_wall=plan['hard_deadline_unix'],
                evidence_path=output / 'worker-process-deadline.jsonl', cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            display.wait(process, timeout=campaign.remaining(plan) + 1)
    finally:
        if process is not None:
            try:
                close_owned_process(process, timeout=10)
            except Exception as error:
                errors.append(str(error))
        for evidence in output.glob('*native-process-deadline.jsonl'):
            try:
                cleanup_from_evidence(evidence, timeout=10)
            except Exception as error:
                errors.append(str(error))
        display.finish(failed=bool(errors) or process is None or process.returncode != 0)
        protected, protection_error = False, None
        try:
            protected = campaign.verify_protected(plan['protected'])
        except Exception as error:
            protection_error = f'{type(error).__name__}: {error}'
        plan_unchanged, plan_error = False, None
        try:
            plan_unchanged = binding(output / 'plan.json') == plan_binding
        except Exception as error:
            plan_error = f'{type(error).__name__}: {error}'
        write(output / 'shutdown.json', dict(owned_cleanup_verified=not errors, errors=errors,
            protected_inputs_unchanged=protected, protection_error=protection_error,
            plan_unchanged=plan_unchanged, plan_error=plan_error,
            returncode=process.returncode if process else None), new=True)
    report = read(output / 'report.json') if (output / 'report.json').exists() else {}
    require(process is not None and process.returncode == 0 and not errors and protected and plan_unchanged
            and report.get('plan') == plan_binding and report.get('status') == 'passed'
            and report.get('schema') == result_schema + '-result' and report.get('checkpoint') == checkpoint
            and report.get('policy_support') == policy_support(plan)
            and report.get('army_information') == plan.get('army_information')
            and report.get('equipment_policy') == plan.get('equipment_policy'),
            'Evaluation stopped; preserve this evidence and use fresh seeds for another manual attempt')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true'); modes.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--lineage', type=Path, default=campaign.LINEAGE)
    parser.add_argument('command', nargs='?', choices=('_worker',))
    parser.add_argument('--plan', type=Path); parser.add_argument('--sha256')
    args = parser.parse_args(argv)
    if args.command == '_worker':
        require(args.plan is not None and args.sha256 is not None and not args.run and not args.check, 'Worker requires a bound plan')
        return int(worker(args.plan, args.sha256)['status'] != 'passed')
    require(math.isfinite(args.seconds) and 180 <= args.seconds <= 7200, 'Use a finite total budget from 180 to 7200 seconds')
    started = time.monotonic()
    try:
        if not args.run:
            checked = inspect_parent(args.lineage)
            result = dict(status='passed', checkpoint=checked['checkpoint'], parent=checked['parent'],
                source=checked['source'], budget=BUDGET, total_seconds=args.seconds,
                next_seed=seed_row(len(seed_ledger()['reservations'])), gameplay_performed=False,
                seeds_reserved=False, optimizer_updates=0, exact_model_adam_rng_progress_sampler_pool=True)
        else:
            try:
                with campaign.activity_lease():
                    result = supervise(prepare(args))
            except ValueError as error:
                if str(error) == 'Another local campaign owns the machine':
                    raise ValueError('Another campaign or its post-run report still owns the machine; '
                                     'wait for it to finish; no game or seeds started') from error
                raise
        print(json.dumps(result, indent=2)); return 0
    finally:
        print(f'Total running time: {time.monotonic() - started:.2f} seconds.', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
