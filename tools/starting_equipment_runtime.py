"""Pinned acceptance for the explicitly artifact-free game rules."""
from pathlib import Path

from tools import run_selfplay_training as campaign
from tools import run_tiny_selfplay_one_game as verified_runtime

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'no-artifacts-v1'
RELEASE = dict(
    directory=campaign.ARTIFACTS / 'runtime/local-v8-taverns-static-neutrals-v1',
    freeze_sha='4c0a91f831e866dc3e78b1f0a55bd2d6d8ba6bb7c61eaa55085e46d744238e50',
    preflight_sha='dcd7042f42218ed4eb9255c4fbfa012fc31e66304f748480b4bbf5a227e3679b',
    validation=campaign.ARTIFACTS / 'runtime/local-v8-taverns-static-neutrals-v1/native-acceptance.json',
    validation_sha='98c26da6406e238fd3eeff9a7818fcc07c98c15fab02ec5be850bb581a41b40f',
    schema='heroes-starting-artifacts-native-v1',
)
CYCLE_MAP = ROOT / 'experiments/explorer/maps/hero-management-v3.vmap'
CYCLE_MAP_SHA = '2ddd04177e17f588922cc7733ec4fa049be10bb0f020f7387fd7285ea1604056'
CYCLE_ACCEPTANCE = RELEASE['directory'] / 'scenario-acceptance.json'
CYCLE_ACCEPTANCE_SHA = '0c328103d291c3576d9f76e0ae87fdf950f733528bb5ebe082c5fc191ba092b2'


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


def inspect_cycle_scenario(checked):
    """Select the independently accepted temporary map for training cycles."""
    from experiments.headless.env import VcmiLauncher
    from tools import run_selfplay_one_game as evaluation
    game_map, receipt = campaign.binding(CYCLE_MAP), campaign.binding(CYCLE_ACCEPTANCE)
    campaign.require(game_map['sha256'] == CYCLE_MAP_SHA and receipt['sha256'] == CYCLE_ACCEPTANCE_SHA,
                     'Temporary cycle map or its native acceptance changed')
    accepted = campaign.read(campaign.bound(receipt))
    campaign.require(accepted.get('schema') == 'heroes-taverns-static-neutrals-native-v1'
                     and accepted.get('status') == 'passed' and accepted.get('runtime') == checked['runtime']
                     and accepted.get('map') == game_map and accepted.get('players') == [0, 1]
                     and accepted.get('optimizer_updates') == 0 and accepted.get('training_samples_admitted') == 0
                     and all(accepted.get(key) is True for key in (
                         'initial_state_preserved', 'starting_taverns_and_hiring_both_players',
                         'monthly_spawn_positive_and_negative_controls', 'weekly_growth_preserved',
                         'native_replay_verified', 'both_policies_reconstructed', 'owned_cleanup_verified')),
                     'Temporary cycle rules lack complete native acceptance')
    source_validation = evaluation.verify_native_validation_sources(
        accepted.get('source_implementation'), evaluation.source_identity()['implementation'])
    launcher = VcmiLauncher(Path(checked['binary']), Path(checked['data']), CYCLE_MAP,
                            equipment_policy=VERSION)
    launcher.preflight()
    checked.update(game_map=str(CYCLE_MAP), native=launcher.identity())
    checked['runtime_origin']['scenario_validation'] = receipt
    checked['runtime_origin']['scenario_source_validation'] = source_validation
    checked['protected'].update({item['path']: item for item in (game_map, receipt)})
    return checked
