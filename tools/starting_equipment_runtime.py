"""Pinned acceptance for the explicitly artifact-free game rules."""
from pathlib import Path

from tools import run_selfplay_training as campaign
from tools import run_tiny_selfplay_one_game as verified_runtime

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'no-artifacts-v1'
RELEASE = dict(
    directory=campaign.ARTIFACTS / 'runtime/local-v8-no-artifacts-v1-repair1',
    freeze_sha='PENDING_NATIVE_BUILD',
    preflight_sha='PENDING_NATIVE_BUILD',
    validation=campaign.ARTIFACTS / 'runtime/local-v8-no-artifacts-v1-repair1/native-acceptance.json',
    validation_sha='PENDING_NATIVE_VALIDATION',
    schema='heroes-starting-artifacts-native-v1',
)


def inspect_runtime(baseline):
    from experiments.headless.env import VcmiLauncher
    checked = verified_runtime.inspect_runtime(baseline, release=RELEASE)
    acceptance = campaign.read(campaign.bound(checked['runtime_origin']['native_validation']))
    campaign.require(acceptance.get('equipment_policy') == VERSION,
                     'Native validation did not accept the all-artifact prohibition')
    pairs = acceptance.get('pairs', [])
    campaign.require(len(pairs) == 9 and all(row.get('status') == 'passed'
                     and row.get('all_equipment_absent') is True
                     and row.get('map_resources_positions_troops_unchanged') is True for row in pairs)
                     and acceptance.get('resolved_bonus_branches') == [0, 1, 2]
                     and acceptance.get('wrong_setting_mutant', {}).get('rejected') is True,
                     'Starting-equipment acceptance lacks paired initialization and replay rejection controls')
    global_controls = acceptance.get('global_equipment_controls', {})
    campaign.require(global_controls.get('players') == [0, 1]
                     and global_controls.get('hired_heroes_checked', 0) >= 2
                     and all(global_controls.get(key) is True for key in (
                         'all_artifacts_absent', 'purchases_unavailable',
                         'spellbook_unavailable', 'artifact_acquisition_disabled')),
                     'Equipment acceptance lacks all-hero restrictions and artifact acquisition controls')
    launcher = VcmiLauncher(Path(checked['binary']), Path(baseline['data']),
                            Path(checked['game_map']), equipment_policy=VERSION)
    launcher.preflight()
    checked['native'] = launcher.identity()
    checked['equipment_policy'] = VERSION
    return checked
