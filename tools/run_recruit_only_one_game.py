#!/usr/bin/env python3
"""One manual frozen game from checkpoint 1753 with a town cooldown.

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
import subprocess
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.learner.contextual import OWN_ARMY_INFORMATION
from experiments.learner.interaction_guard import HERO_RELATIVE_VERSION, TOWN_COOLDOWN_VERSION
from experiments.learner.game_progress import color_text, terminal_color_enabled
from tools import run_selfplay_one_game as evaluation
from tools import run_selfplay_training as campaign
from tools import starting_equipment_runtime as runtime_validation

require, read, write, binding, bound = campaign.require, campaign.read, campaign.write, campaign.binding, campaign.bound
SCHEMA = 'heroes-equipment-restricted-one-game-v1'
CHECKPOINT = 1753
PARAMETERS = 74344
MODEL_CONFIG = dict(schema='heroes-context-model-v6', hidden_size=64, feature_size=32, max_map_side=256)
BUDGET = evaluation.BUDGET
GUARD = TOWN_COOLDOWN_VERSION


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


def render_setup(*, ready, guard=GUARD, seconds=3600, color=False, columns=80):
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
            ('Artifacts OFF', 'No artifacts, spellbooks or war machines for any hero. Temporary rule.', '1;33'),
            ('All game', 'No equipment from starts, hires, purchases, loot or rewards.', '1;33'),
            ('Town cooldown', 'After closing a town, complete one other action before opening any town.', '1;33'),
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
    if guard == HERO_RELATIVE_VERSION:
        sections[1][1].extend([
            ('Coordinates', 'Movement targets are scored relative to the selected hero.', '36'),
            ('Repeat cooldown', 'The same command cannot be chosen twice in a row. A different move target is allowed.', '1;33'),
        ])
    if ready:
        command = './play-one-selfplay.sh --run'
        if guard == HERO_RELATIVE_VERSION:
            command += ' --hero-relative'
        sections.append(('Launch when ready', [('', command, '1;32')]))
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


def experiment(guard=GUARD):
    relative = guard == HERO_RELATIVE_VERSION
    return dict(experiment='trained1753-hero-relative-cooldown-v1' if relative else 'trained1753-town-cooldown-v1',
                temporary=True, policy_support=guard, army_information=OWN_ARMY_INFORMATION,
                equipment_policy=runtime_validation.VERSION,
                movement_coordinates='hero_relative' if relative else 'absolute',
                repeat_cooldown='same_command_with_required_exits' if relative else None,
                town_reopen_cooldown_actions=1,
                later_hired_heroes='empty_inventory',
                artifacts='disabled_for_all_heroes_and_acquisition_paths',
                checkpoint=CHECKPOINT, model_config=MODEL_CONFIG, parameters=PARAMETERS,
                recruitment='unchanged_native_purchase', whole_hero_garrison='allowed',
                combat='automatic_unchanged', rewards=dict(win=1, loss=-1, ordinary_transition=0),
                training_launcher_accepted=False,
                restoration_reminder='Review the town cooldown and restore artifact acquisition and troop-slot management in separately versioned experiments.')


def source_identity():
    return dict(runtime=evaluation.source_identity(), driver=binding(__file__),
                launcher=binding(ROOT / 'play-one-selfplay.sh'),
                runtime_validator=binding(runtime_validation.__file__),
                runtime_auditor=binding(runtime_validation.verified_runtime.__file__))


def frozen_models(plan):
    models = evaluation.frozen_models(plan)
    require(plan['checkpoint'] == CHECKPOINT and plan['model_config'] == MODEL_CONFIG
            and plan['parameters'] == PARAMETERS
            and all(model.config.to_dict() == MODEL_CONFIG
                    and sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS
                    for model in models), 'The trained large model architecture changed')
    return models


def inspect_candidate(guard=GUARD):
    baseline = evaluation.inspect_parent(campaign.LINEAGE)
    require(baseline['checkpoint'] == CHECKPOINT, 'The completed trained checkpoint changed')
    checked = {**baseline, 'source': source_identity(), 'protected': dict(baseline['protected']),
               'model_config': dict(MODEL_CONFIG), 'parameters': PARAMETERS,
               'experiment': experiment(guard), 'training_launcher_accepted': False}
    # This validator only binds the accepted native runtime. It never loads the
    # retired tiny policy, its preparation receipt or its seed namespace.
    checked.update(runtime_validation.inspect_runtime(baseline))
    require(checked.get('equipment_policy') == runtime_validation.VERSION,
            'The manual experiment requires the all-artifact prohibition')
    checked['protected'].update(checked.pop('runtime_protected'))
    frozen_models(checked)
    campaign.verify_protected(checked['protected'])
    return checked


def prepare(args, guard=GUARD, *, checked=None, deadline=None):
    require(math.isfinite(args.seconds) and 180 <= args.seconds <= 7200,
            'Use a finite total budget from 180 to 7200 seconds')
    checked = inspect_candidate(guard) if checked is None else checked
    first_win = deadline is not None
    output = evaluation.copies.private_path(args.output or evaluation.BASE / (
        'run-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(4)), fresh=True)
    # Share the established large-model ledger, including its consumed attempts.
    reservation = evaluation.reserve(output)
    output.mkdir(parents=True, mode=0o700)
    plan = dict(schema=SCHEMA + '-plan', **checked, output=str(output),
        budget=evaluation.FIRST_WIN_BUDGET if first_win else BUDGET, policy_support=guard,
        physical_games=1, learner_player_for_reporting=0, purpose='diagnostic', training_input=False,
        optimizer_updates=0, reservation=reservation, schedule=read(bound(reservation))['schedule'],
        supervisor_pid=os.getpid(), maximum_seconds=args.seconds,
        hard_deadline_monotonic=time.monotonic() + args.seconds, hard_deadline_unix=time.time() + args.seconds)
    if first_win:
        plan.update(until_first_win=True, hard_deadline_monotonic=deadline[0], hard_deadline_unix=deadline[1])
    write(output / 'plan.json', plan, new=True)
    return plan


def check_identity(plan):
    guard = plan.get('policy_support')
    first_win = plan.get('until_first_win') is True
    budget = evaluation.FIRST_WIN_BUDGET if first_win else BUDGET
    require(plan['schema'] == SCHEMA + '-plan' and plan['budget'] == budget
            and (not first_win or guard == HERO_RELATIVE_VERSION)
            and plan['physical_games'] == 1 and plan['purpose'] == 'diagnostic'
            and plan['training_input'] is False and plan['optimizer_updates'] == 0
            and guard in (GUARD, HERO_RELATIVE_VERSION) and plan['army_information'] == OWN_ARMY_INFORMATION
            and plan.get('equipment_policy') == runtime_validation.VERSION,
            'Trained recruit-only evaluation contract changed')
    checked = inspect_candidate(guard)
    require(all(plan.get(key) == value for key, value in checked.items()),
            'Frozen trained policy, runtime, source or original-state protection changed')
    reservation = read(bound(plan['reservation']))
    require(reservation in evaluation.seed_ledger()['reservations']
            and reservation['output'] == plan['output'] and plan['schedule'] == reservation['schedule'],
            'Trained evaluation seed binding differs')
    campaign.verify_protected(plan['protected'])


def victory(manifest):
    outcomes = manifest.get('outcomes_by_player', {})
    rewards = manifest.get('rewards_by_player', {})
    if manifest.get('status') == 'completed' and not manifest.get('natural_terminal_after_horizon'):
        for winner, loser in (('red', 'blue'), ('blue', 'red')):
            if outcomes == {winner: 'win', loser: 'loss'} and rewards == {winner: 1, loser: -1}:
                return winner
    return None


def wait_for_verification(plan, folder):
    output = Path(plan['output'])
    require(folder == output / 'episode-000001', 'Unexpected capture directory')
    (output / 'capture-ready').touch(exist_ok=False)
    while not (output / 'verify-now').exists():
        campaign.remaining(plan, 30)
        time.sleep(0.25)


def first_win(args, checked, *, human, color):
    from experiments.headless.process_deadline import (
        guarded_popen, close_owned_process, cleanup_from_evidence, _release_owned_process,
    )
    started = time.monotonic()
    deadline = (started + args.seconds, time.time() + args.seconds)
    root = evaluation.copies.private_path(args.output or evaluation.BASE / (
        'first-win-' + secrets.token_hex(6)), fresh=True)
    root.mkdir(parents=True, mode=0o700)
    print(f'Private evidence: {root}', file=sys.stderr, flush=True)
    active, results, cleanup_errors = {}, [], []
    winner = None
    launched = 0
    reason = 'budget'
    env = {**os.environ, **{key: '1' for key in
           ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}}

    def close(folder, process):
        limits = dict(timeout=10, deadline_monotonic=deadline[0], deadline_wall=deadline[1])
        for operation in [lambda: close_owned_process(process, **limits)] + [
                lambda receipt=receipt: cleanup_from_evidence(receipt, **limits)
                for receipt in folder.glob('*native-process-deadline.jsonl')]:
            try:
                operation()
            except Exception as error:
                cleanup_errors.append(f'{folder.name}: {error}')

    def display(pending, *, final=False):
        elapsed = time.monotonic() - started
        done = sum(row['status'] == 'verified' for row in results)
        cutoffs = sum(row.get('cutoff', False) for row in results)
        failures = sum(row['status'] == 'failed' for row in results)
        heading = f'FIRST WIN  •  CHECKPOINT {CHECKPOINT:,}'
        cells = [('Elapsed', _duration(elapsed)), ('Completed', str(done)),
                 ('Active', str(len(active))), ('Cutoffs', str(cutoffs)),
                 ('Failures', str(failures)), ('Winner', winner['colour'] if winner else
                  'awaiting verification' if pending else 'searching' if not final else 'none')]
        if human:
            width = shutil.get_terminal_size((80, 24)).columns
            bar = '━' * min(30, int(30 * elapsed / args.seconds))
            view = _panel(heading, 'Finished' if final else 'Frozen evaluation',
                          [('Run', [(key, value, '1;36') for key, value in cells]
                                    + [('Time budget', bar.ljust(30, '─'), '1;32')])],
                          color=color, columns=width)
            print(('\x1b[H\x1b[2J' if not final else '\n') + view, flush=True)
        elif not args.json:
            print(' | '.join(f'{key} {value}' for key, value in cells), flush=True)

    try:
        last_display = -60
        while time.monotonic() < deadline[0] - 32 and time.time() < deadline[1] - 32:
            pending, ready = [], []
            for folder, (process, pin, schedule) in list(active.items()):
                capture = folder / 'episode-000001'
                try:
                    manifest = read(capture / 'manifest.json')
                except (FileNotFoundError, json.JSONDecodeError):
                    manifest = {}
                candidate = victory(manifest)
                code = process.poll()
                if code is None:
                    if candidate:
                        pending.append(folder)
                    if (folder / 'capture-ready').exists() and not (folder / 'verify-now').exists():
                        ready.append(folder)
                    continue
                close(folder, process)
                del active[folder]
                try:
                    report = read(folder / 'report.json')
                    require(code == 0 and report['status'] == 'passed' and report['plan'] == pin
                            and binding(folder / 'plan.json') == pin
                            and report['native_replay_verified'] is True
                            and report['both_policies_reconstructed'] is True
                            and report['protected_inputs_unchanged'] is True
                            and report['training_input'] is False and report['optimizer_updates'] == 0
                            and report['training_samples_admitted'] == 0 and not cleanup_errors,
                            report.get('error', 'Evaluation worker did not pass'))
                    proof = read(bound(report['proof']))
                    candidate = victory(dict(status=report['episode_status'],
                        outcomes_by_player=report['outcomes'], rewards_by_player=report['rewards'],
                        natural_terminal_after_horizon=report.get('natural_terminal_after_horizon')))
                    if candidate:
                        require(type(proof['terminal_day']) is int and proof['terminal_day'] > 0,
                                'Verified terminal day missing')
                        if winner is None:
                            winner = dict(colour=candidate, day=proof['terminal_day'], seeds=schedule,
                                          evidence=str(capture), proof=report['proof'])
                    results.append(dict(game=folder.name, status='verified',
                                        cutoff=report['episode_status'] == 'truncated' or
                                        bool(report.get('natural_terminal_after_horizon'))))
                except (OSError, ValueError, KeyError) as error:
                    results.append(dict(game=folder.name, status='failed', error=str(error)))
            if winner or cleanup_errors:
                break
            if shutil.disk_usage(root).free < args.disk_reserve_gib * 1024**3:
                reason = 'disk reserve reached'
                break
            verifying = {folder for folder in active if (folder / 'verify-now').exists()}
            # One ordinary verifier and a reserved priority slot for a victory.
            # Waiting captures occupy worker slots, so the queue cannot grow unbounded.
            if pending:
                if not verifying.intersection(pending):
                    priority = next((folder for folder in ready if folder in pending), None)
                    if priority is not None and len(verifying) < 2:
                        (priority / 'verify-now').touch(exist_ok=False)
            elif ready and not verifying:
                (ready[0] / 'verify-now').touch(exist_ok=False)
            if not pending and launched < args.max_games and len(active) < args.workers:
                launched += 1
                folder = root / f'game-{launched:06d}'
                game_args = argparse.Namespace(output=folder, seconds=args.seconds)
                plan = prepare(game_args, HERO_RELATIVE_VERSION, checked=checked, deadline=deadline)
                pin = binding(folder / 'plan.json')
                with (folder / 'worker.log').open('xb') as log:
                    process = guarded_popen([sys.executable, str(Path(__file__).resolve()), '_first_win_worker',
                        '--plan', pin['path'], '--sha256', pin['sha256']],
                        deadline_monotonic=deadline[0], deadline_wall=deadline[1],
                        evidence_path=folder / 'worker-process-deadline.jsonl', cwd=ROOT, env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                active[folder] = (process, pin, plan['schedule'])
                # Fill empty slots promptly, then poll at a low frequency.
                continue
            if not active:
                break
            if time.monotonic() - last_display >= (1 if human else 15):
                display(pending)
                last_display = time.monotonic()
            time.sleep(0.25)
    except KeyboardInterrupt:
        reason = 'interrupted by operator'
    finally:
        for process, _, _ in active.values():
            _release_owned_process(process)
        for folder, (process, _, _) in active.items():
            close(folder, process)
            results.append(dict(game=folder.name, status='interrupted'))
        active.clear()
        protected = False
        try:
            protected = campaign.verify_protected(checked['protected'])
        except Exception as error:
            cleanup_errors.append(f'Protected inputs: {error}')
        summary = dict(winner=winner, reason='verified victory' if winner else reason,
                       games=results, launched=launched, elapsed_seconds=time.monotonic() - started,
                       owned_cleanup_verified=not cleanup_errors, cleanup_errors=cleanup_errors,
                       protected_inputs_unchanged=protected, optimizer_updates=0, training_samples_admitted=0)
        write(root / 'report.json', summary, new=True)
    display(False, final=True)
    message = (f"{winner['colour'].title()} won on day {winner['day']}. Seeds {winner['seeds']}. "
               f"Evidence {winner['evidence']}" if winner else 'No verified winner within the budget.')
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(message, flush=True)
        print(f'Report: {root / "report.json"}', flush=True)
    return 2 if cleanup_errors or not protected else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true'); modes.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--until-first-win', action='store_true', help='Search with frozen checkpoint 1753 and 16× gameplay limits')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--max-games', type=int, default=100)
    parser.add_argument('--disk-reserve-gib', type=int, default=30)
    parser.add_argument('--hero-relative', action='store_true', help='Use hero-relative coordinates and the general repeat cooldown')
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--json', action='store_true', help='Print the machine-readable result, including in a terminal')
    parser.add_argument('command', nargs='?', choices=('_worker', '_first_win_worker'))
    parser.add_argument('--plan', type=Path); parser.add_argument('--sha256')
    args = parser.parse_args(argv)
    if args.command in ('_worker', '_first_win_worker'):
        require(args.plan is not None and args.sha256 is not None and not args.run and not args.check and not args.hero_relative,
                'Trained worker requires a bound plan')
        options = {'before_verify': wait_for_verification} if args.command == '_first_win_worker' else {}
        return int(evaluation.worker(args.plan, args.sha256, identity_check=check_identity,
            model_loader=frozen_models, result_schema=SCHEMA, checkpoint=CHECKPOINT,
            **options)['status'] != 'passed')
    require(math.isfinite(args.seconds) and 180 <= args.seconds <= 7200,
            'Use a finite total budget from 180 to 7200 seconds')
    human = _human_terminal(sys.stdout) and not args.json
    color = human and terminal_color_enabled(sys.stdout)
    columns = shutil.get_terminal_size(fallback=(80, 24)).columns
    started = time.monotonic()
    if args.until_first_win:
        require(1 <= args.workers <= 10 and 1 <= args.max_games <= 10000 and args.disk_reserve_gib >= 1,
                'Use 1–10 workers, 1–10,000 games and a positive disk reserve')
        if args.run:
            with campaign.activity_lease():
                return first_win(args, inspect_candidate(HERO_RELATIVE_VERSION), human=human, color=color)
        checked = inspect_candidate(HERO_RELATIVE_VERSION)
        print(json.dumps(dict(status='ready', checkpoint=CHECKPOINT, map=checked['game_map'],
            experiment=checked['experiment'], budget=evaluation.FIRST_WIN_BUDGET,
            workers=args.workers, max_games=args.max_games, seconds=args.seconds,
            disk_reserve_gib=args.disk_reserve_gib), indent=2))
        return 0
    plan = None
    guard = HERO_RELATIVE_VERSION if args.hero_relative else GUARD
    try:
        if not args.run:
            checked = inspect_candidate(guard)
            result = dict(status='passed', checkpoint=CHECKPOINT, model_checkpoint=checked['model_checkpoint'],
                model_config=MODEL_CONFIG, parameters=PARAMETERS, source=checked['source'], budget=BUDGET,
                policy_support=guard, army_information=OWN_ARMY_INFORMATION, experiment=experiment(guard),
                equipment_policy=checked['equipment_policy'],
                total_seconds=args.seconds, next_seed=evaluation.seed_row(len(evaluation.seed_ledger()['reservations'])),
                gameplay_performed=False, seeds_reserved=False, optimizer_updates=0,
                exact_model_adam_rng_progress_sampler_pool=True, training_launcher_accepted=False)
        else:
            with campaign.activity_lease():
                plan = prepare(args, guard)
                if human:
                    print(render_setup(ready=False, guard=guard, seconds=args.seconds, color=color, columns=columns), flush=True)
                result = evaluation.supervise(plan, worker_script=Path(__file__).resolve(),
                                              result_schema=SCHEMA, checkpoint=CHECKPOINT)
        if human:
            print(render_result(result, color=color, columns=columns) if args.run else
                  render_setup(ready=True, guard=guard, seconds=args.seconds, color=color, columns=columns), flush=True)
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
