#!/usr/bin/env python3
"""One manual frozen game with trained checkpoint 1753 and recruit-only armies.

The large model, original map and doubled budgets remain unchanged. The explicit
observation/support intervention does not rewrite the checkpoint's provenance.
Check mode starts no game, reserves no seed and performs no optimizer update.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.learner.contextual import OWN_ARMY_INFORMATION
from experiments.learner.interaction_guard import RECRUIT_ONLY_VERSION
from experiments.learner.game_progress import color_text, terminal_color_enabled
from tools import run_selfplay_one_game as evaluation
from tools import run_selfplay_training as campaign
from tools import run_tiny_selfplay_one_game as runtime_validation

require, read, write, binding, bound = campaign.require, campaign.read, campaign.write, campaign.binding, campaign.bound
SCHEMA = 'heroes-recruit-only-one-game-v1'
CHECKPOINT = 1753
PARAMETERS = 74344
MODEL_CONFIG = dict(schema='heroes-context-model-v6', hidden_size=64, feature_size=32, max_map_side=256)
BUDGET = evaluation.BUDGET
GUARD = RECRUIT_ONLY_VERSION


def _human_terminal(stream):
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def _duration(seconds):
    minutes, seconds = divmod(round(max(0, seconds), 1), 60)
    hours, minutes = divmod(int(minutes), 60)
    parts = ([f'{hours}h'] if hours else []) + ([f'{minutes}m'] if minutes else [])
    return ' '.join(parts if parts and seconds == 0 else [*parts, f'{seconds:.1f}s'])


def _panel(title, status, sections, *, color=False, columns=80, status_color='1;36'):
    """Render presentation only; plain and colored layouts have identical text."""
    width = max(20, min(columns, 88))
    lines = ['']
    lines.extend(color_text(line, '1;36', color) for line in textwrap.wrap(title, width=width))
    lines.append(color_text('-' * min(width, 60), '2', color))
    lines.extend(color_text(line, status_color, color) for line in textwrap.wrap(status, width=width))
    for heading, rows in sections:
        lines.extend(('', color_text(heading, '1', color)))
        for label, value, code in rows:
            if not label:
                lines.extend(color_text(line, code, color) for line in textwrap.wrap(str(value), width=width,
                    initial_indent='  ', subsequent_indent='  '))
            elif width < 50:
                lines.append('  ' + color_text(label, code, color))
                lines.extend(textwrap.wrap(str(value), width=width,
                    initial_indent='    ', subsequent_indent='    '))
            else:
                prefix = f'  {label:<15}'
                wrapped = textwrap.wrap(str(value), width=width - len(prefix)) or ['']
                lines.append('  ' + color_text(f'{label:<15}', code, color) + wrapped[0])
                lines.extend(' ' * len(prefix) + line for line in wrapped[1:])
    return '\n'.join(lines) + '\n'


def render_setup(*, ready, seconds=3600, color=False, columns=80):
    """Human summary of the already-validated selected experiment."""
    sections = [
        ('Setup', [
            ('Model', f'Trained large network / checkpoint {CHECKPOINT:,} / {PARAMETERS:,} parameters', '36'),
            ('Game', 'One self-play game: Red and Blue use the same trained policy.', '36'),
            ('Learning OFF', 'Frozen weights; no training update.', '1;33'),
        ]),
        ('Game rules', [
            ('Allowed', 'Buy creatures; move a whole hero into or out of a town garrison.', '1;32'),
            ('Disabled', 'Rearrange, split, merge or transfer individual troop stacks.', '1;33'),
            ('Disabled', 'Open stack windows or upgrade existing stacks. Temporary restriction.', '1;33'),
            ('Combat', 'Automatic battles.', '36'),
            ('Movement', 'Revealed tiles only.', '36'),
            ('Information', 'Your armies and combat stats are visible without clicks.', '36'),
        ]),
        ('Limits', [
            ('Game days', f"{BUDGET['max_days']:,}; the current day appears in live progress.", '36'),
            ('Each player', f"{BUDGET['ordinary_choices_per_player']:,} ordinary choices + "
                            f"{BUDGET['reserved_dialog_choices_per_player']:,} reserved for mandatory dialogs.", '36'),
            ('Time', f"{_duration(BUDGET['capture_seconds'])} for gameplay; {_duration(seconds)} for the entire run.", '36'),
        ]),
    ]
    if ready:
        sections.append(('Launch when ready', [('', './play-one-selfplay.sh --run', '1;32')]))
    return _panel('HEROES RL / SELF-PLAY',
        'READY - checks passed; no game started.' if ready else 'STARTING - one frozen self-play game.',
        sections, color=color, columns=columns, status_color='1;32' if ready else '1;36')


def render_result(result, *, color=False, columns=80):
    """Keep successful verification distinct from a naturally won game."""
    verified = result.get('status') == 'passed' and all(result.get(key) is True for key in (
        'native_replay_verified', 'both_policies_reconstructed', 'owned_cleanup_verified'))
    rows = []
    if result.get('episode_status') == 'truncated':
        reasons = {'game_day_limit': 'Day limit reached', 'collection_window': 'Choice limit reached',
                   'agent_action_limit': 'Choice limit reached'}
        reason = reasons.get(result.get('termination_reason'), 'Collection stopped at its limit')
        rows.append(('Outcome', f'{reason}; result is unscored.', '1;33'))
    elif result.get('episode_status') == 'completed':
        outcomes = result.get('outcomes', {})
        winner = next((color.title() for color in ('red', 'blue') if outcomes.get(color) == 'win'), None)
        rows.append(('Outcome', f'{winner} wins.' if winner else 'Game completed; see the report for its outcome.', '1;32'))
    else:
        rows.append(('Outcome', 'No verified game result is available.', '1;33'))
    choices = result.get('choices_by_player')
    if isinstance(choices, (list, tuple)) and len(choices) == 2:
        rows.append(('Choices', f'Red {choices[0]:,} / Blue {choices[1]:,}', '36'))
    rows.append(('Learning OFF', ('Weights unchanged; no training update.'
                                 if result.get('protected_inputs_unchanged') is True else
                                 'No training update requested; inspect the report for state checks.'), '1;33'))
    if verified:
        rows.append(('Checks', 'Replay and both policies verified; owned processes cleaned up.', '1;32'))
    plan_path = (result.get('plan') or {}).get('path')
    if plan_path:
        path = str(Path(plan_path).parent / 'report.json')
        home = str(Path.home())
        if path.startswith(home + '/'):
            path = '~' + path[len(home):]
        rows.append(('Report', path, '2'))
    return _panel('HEROES RL / RESULT', 'VERIFIED - recording checks passed.' if verified else 'RUN STOPPED',
                  [('Result', rows)], color=color, columns=columns, status_color='1;32' if verified else '1;31')


def render_failure(error, *, output=None, color=False, columns=80):
    rows = [('Reason', str(error), '33')]
    if output is not None:
        rows.append(('Run files', str(output), '2'))
    return _panel('HEROES RL / RESULT', 'RUN STOPPED - no automatic retry.',
                  [('Details', rows)], color=color, columns=columns, status_color='1;31')


def experiment():
    return dict(experiment='trained1753-recruit-only-v1', temporary=True,
                policy_support=GUARD, army_information=OWN_ARMY_INFORMATION,
                checkpoint=CHECKPOINT, model_config=MODEL_CONFIG, parameters=PARAMETERS,
                recruitment='unchanged_native_purchase', whole_hero_garrison='allowed',
                combat='automatic_unchanged', rewards=dict(win=1, loss=-1, ordinary_transition=0),
                training_launcher_accepted=False,
                restoration_reminder='Restore troop-slot management in a separately versioned experiment.')


def source_identity():
    return dict(runtime=evaluation.source_identity(), driver=binding(__file__),
                launcher=binding(ROOT / 'play-one-selfplay.sh'),
                runtime_validator=binding(runtime_validation.__file__))


def frozen_models(plan):
    models = evaluation.frozen_models(plan)
    require(plan['checkpoint'] == CHECKPOINT and plan['model_config'] == MODEL_CONFIG
            and plan['parameters'] == PARAMETERS
            and all(model.config.to_dict() == MODEL_CONFIG
                    and sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS
                    for model in models), 'The trained large model architecture changed')
    return models


def inspect_candidate():
    baseline = evaluation.inspect_parent(campaign.LINEAGE)
    require(baseline['checkpoint'] == CHECKPOINT, 'The completed trained checkpoint changed')
    checked = {**baseline, 'source': source_identity(), 'protected': dict(baseline['protected']),
               'model_config': dict(MODEL_CONFIG), 'parameters': PARAMETERS,
               'experiment': experiment(), 'training_launcher_accepted': False}
    # This validator only binds the accepted native runtime. It never loads the
    # retired tiny policy, its preparation receipt or its seed namespace.
    checked.update(runtime_validation.inspect_runtime(baseline))
    checked['protected'].update(checked.pop('runtime_protected'))
    frozen_models(checked)
    campaign.verify_protected(checked['protected'])
    return checked


def prepare(args):
    require(math.isfinite(args.seconds) and 180 <= args.seconds <= 7200,
            'Use a finite total budget from 180 to 7200 seconds')
    checked = inspect_candidate()
    output = evaluation.copies.private_path(args.output or evaluation.BASE / (
        'run-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(4)), fresh=True)
    # Share the established large-model ledger, including its consumed attempts.
    reservation = evaluation.reserve(output)
    output.mkdir(parents=True, mode=0o700)
    plan = dict(schema=SCHEMA + '-plan', **checked, output=str(output), budget=BUDGET, policy_support=GUARD,
        physical_games=1, learner_player_for_reporting=0, purpose='diagnostic', training_input=False,
        optimizer_updates=0, reservation=reservation, schedule=read(bound(reservation))['schedule'],
        supervisor_pid=os.getpid(), maximum_seconds=args.seconds,
        hard_deadline_monotonic=time.monotonic() + args.seconds, hard_deadline_unix=time.time() + args.seconds)
    write(output / 'plan.json', plan, new=True)
    return plan


def check_identity(plan):
    require(plan['schema'] == SCHEMA + '-plan' and plan['budget'] == BUDGET
            and plan['physical_games'] == 1 and plan['purpose'] == 'diagnostic'
            and plan['training_input'] is False and plan['optimizer_updates'] == 0
            and plan['policy_support'] == GUARD and plan['army_information'] == OWN_ARMY_INFORMATION,
            'Trained recruit-only evaluation contract changed')
    checked = inspect_candidate()
    require(all(plan.get(key) == value for key, value in checked.items()),
            'Frozen trained policy, runtime, source or original-state protection changed')
    reservation = read(bound(plan['reservation']))
    require(reservation in evaluation.seed_ledger()['reservations']
            and reservation['output'] == plan['output'] and plan['schedule'] == reservation['schedule'],
            'Trained evaluation seed binding differs')
    campaign.verify_protected(plan['protected'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true'); modes.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--json', action='store_true', help='Print the machine-readable result, including in a terminal')
    parser.add_argument('command', nargs='?', choices=('_worker',))
    parser.add_argument('--plan', type=Path); parser.add_argument('--sha256')
    args = parser.parse_args(argv)
    if args.command == '_worker':
        require(args.plan is not None and args.sha256 is not None and not args.run and not args.check,
                'Trained worker requires a bound plan')
        return int(evaluation.worker(args.plan, args.sha256, identity_check=check_identity,
            model_loader=frozen_models, result_schema=SCHEMA, checkpoint=CHECKPOINT)['status'] != 'passed')
    require(math.isfinite(args.seconds) and 180 <= args.seconds <= 7200,
            'Use a finite total budget from 180 to 7200 seconds')
    human = _human_terminal(sys.stdout) and not args.json
    color = human and terminal_color_enabled(sys.stdout)
    columns = shutil.get_terminal_size(fallback=(80, 24)).columns
    started = time.monotonic()
    plan = None
    try:
        if not args.run:
            checked = inspect_candidate()
            result = dict(status='passed', checkpoint=CHECKPOINT, model_checkpoint=checked['model_checkpoint'],
                model_config=MODEL_CONFIG, parameters=PARAMETERS, source=checked['source'], budget=BUDGET,
                policy_support=GUARD, army_information=OWN_ARMY_INFORMATION, experiment=experiment(),
                total_seconds=args.seconds, next_seed=evaluation.seed_row(len(evaluation.seed_ledger()['reservations'])),
                gameplay_performed=False, seeds_reserved=False, optimizer_updates=0,
                exact_model_adam_rng_progress_sampler_pool=True, training_launcher_accepted=False)
        else:
            with campaign.activity_lease():
                plan = prepare(args)
                if human:
                    print(render_setup(ready=False, seconds=args.seconds, color=color, columns=columns), flush=True)
                result = evaluation.supervise(plan, worker_script=Path(__file__).resolve(),
                                              result_schema=SCHEMA, checkpoint=CHECKPOINT)
        if human:
            print(render_result(result, color=color, columns=columns) if args.run else
                  render_setup(ready=True, seconds=args.seconds, color=color, columns=columns), flush=True)
        else:
            print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        if not human:
            raise
        print(render_failure(error, output=plan.get('output') if plan else None,
                             color=color, columns=columns), flush=True)
        return 2
    finally:
        elapsed = time.monotonic() - started
        if human:
            print(color_text('Total running time  ', '1', color) + _duration(elapsed), flush=True)
        else:
            print(f'Total running time: {elapsed:.2f} seconds.',
                  file=sys.stderr if args.json else sys.stdout, flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
