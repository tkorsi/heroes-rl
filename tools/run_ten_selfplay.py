#!/usr/bin/env python3
"""Ten concurrent frozen games, strict verification and visible-history analytics.

The twenty-minute limit includes preflight, gameplay, storage, verification and
reporting. A timeout fails the batch; it never converts partial play to a result.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

STARTED, STARTED_WALL = time.monotonic(), time.time()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_recruit_only_one_game as one
from experiments.learner import game_progress as progress
from experiments.headless.process_deadline import (
    guarded_popen, close_owned_process, cleanup_from_evidence, _release_owned_process,
)
from validation.probes.validate_hero_policy import wall_limit

evaluation, campaign = one.evaluation, one.campaign
require, read, write, binding, bound = one.require, one.read, one.write, one.binding, one.bound
GAMES, SECONDS = 10, 1200
GUARD = one.HERO_RELATIVE_VERSION
SCHEMA = 'heroes-ten-frozen-selfplay-v1'


def emit(*args, **kwargs):
    """A closed terminal must not prevent private reports or owned cleanup."""
    try:
        print(*args, **kwargs)
    except OSError:
        pass


def source():
    return dict(driver=binding(__file__), launcher=binding(ROOT / 'play-ten-selfplay.sh'))


def check_plan(plan):
    one.check_identity(plan)
    require(plan['batch_source'] == source() and 1 <= plan['batch_slot'] <= GAMES,
            'Ten-game runner source or slot changed')


def analyze(collected):
    """Read already-verified typed histories once; never feed metrics to policy."""
    players = {}
    for rollout, color in zip(collected.rollouts, ('red', 'blue')):
        actions, results, events, moves = Counter(), Counter(), Counter(), Counter()
        submitted, visits, previous_heroes = {}, {}, {}
        unresolved_moves = set()
        known, positions, action_days, progress_days = set(), set(), set(), set()
        first = previous = None
        previous_seq = -1
        captures, position_changes, revisits, gained = [], 0, 0, 0
        for f in rollout.history:
            if f.kind == 'submitted':
                submitted[f.decision_seq] = f.action
                actions[f.action.kind] += 1
                action_days.add(f.day)
                if f.action.kind == 'move_to':
                    action = f.action
                    obs = rollout.history[f.decision_seq].observation
                    hero = next(h for h in obs.heroes if h.handle == action.subject)
                    target = (action.x, action.y, hero.position[2])
                    unresolved_moves.add(f.decision_seq)
                    moves['attempts'] += 1
                    moves['revealed_targets'] += any(t.position == target for t in obs.tiles)
                    moves['targets_left_edge'] += action.x == 0
                    moves['targets_visible_opponent_town'] += any(
                        obj.kind == 98 and obj.owner == 'opponent' and obj.position == target for obj in obs.objects)
            if f.result is not None:
                results[f.result.status] += 1
                action = submitted[f.decision_seq]
                if action.kind == 'move_to':
                    obs = rollout.history[f.decision_seq].observation
                    hero = next(h for h in obs.heroes if h.handle == action.subject)
                    unresolved_moves.remove(f.decision_seq)
                    moves[f.result.status] += 1
                    moves['endpoint_changes'] += f.result.position is not None and f.result.position != hero.position
                    moves['missing_endpoints'] += f.result.position is None
            if f.event is not None:
                events[f.event.kind] += 1
                if f.event.kind == 'visit' and f.event.object_kind == 98:
                    visits[f.event.subject] = (f.seq, f.event.owner)
            obs = f.observation
            if obs is None:
                continue
            tiles = {t.position for t in obs.tiles}
            heroes = {h.handle: h.position for h in obs.heroes}
            if first is None:
                first = obs
            else:
                if tiles - known:
                    progress_days.add(f.day)
                old_towns = {t.handle for t in previous.towns}
                added = {t.handle for t in obs.towns} - old_towns
                gained += len(added)
                old_owners = {o.handle: o.owner for o in previous.objects if o.kind == 98}
                new_owners = {o.handle: o.owner for o in obs.objects if o.kind == 98}
                for town in sorted(added):
                    visit = visits.get(town)
                    recent_visit = visit and previous_seq < visit[0] < f.seq and visit[1] != 'self'
                    changed_owner = town in old_owners and old_owners[town] != 'self' and new_owners.get(town) == 'self'
                    if recent_visit or changed_owner:
                        captures.append(dict(day=f.day, seq=f.seq, town=town))
                        progress_days.add(f.day)
                for hero, position in heroes.items():
                    if hero in previous_heroes and previous_heroes[hero] != position:
                        position_changes += 1
                        revisits += (hero, position) in positions
                        progress_days.add(f.day)
            known.update(tiles)
            positions.update(heroes.items())
            previous, previous_seq, previous_heroes = obs, f.seq, heroes
        if unresolved_moves:
            moves['no_result'] += len(unresolved_moves)
            moves['missing_endpoints'] += len(unresolved_moves)
        require(first is not None and previous is not None, 'Analytics requires visible observations')
        players[color] = dict(choices=sum(actions.values()), last_day=max(f.day for f in rollout.history),
            actions=dict(actions), results=dict(results), events=dict(events), movement=dict(moves),
            new_tiles=len(known - {t.position for t in first.tiles}), position_changes=position_changes,
            revisited_positions=revisits, town_captures=len(captures), capture_evidence=captures,
            town_roster_gains=gained, towns_first_last=[len(first.towns), len(previous.towns)],
            heroes_first_last=[len(first.heroes), len(previous.heroes)],
            hires=results['hired'], recruitments=results['recruited'], buildings=results['built'],
            days_without_observed_travel_reveal_or_capture=len(action_days - progress_days))
    return dict(players=players, training_input=False, policy_input=False, reward_input=False,
        limitations=['Revisits and days without travel do not establish aimlessness.',
                     'Final rosters are last delivered player-visible observations.',
                     'Battle events are player perspectives; do not sum them as unique battles.'])


def worker(path, sha):
    require(binding(path)['sha256'] == sha, 'Worker plan changed')
    plan = read(path)
    output = Path(plan['output'])
    require(os.getppid() == plan['supervisor_pid'], 'Only the lease owner can launch a game')
    write(output / 'worker-claim.json', dict(pid=os.getpid(), plan_sha256=sha), new=True)
    report = dict(schema=SCHEMA + '-game', status='running', slot=plan['batch_slot'],
                  plan=binding(path), checkpoint=one.CHECKPOINT, training_input=False,
                  training_samples_admitted=0, optimizer_updates=0, timings={})
    write(output / 'report.json', report, new=True)
    started = time.monotonic()
    try:
        with wall_limit(campaign.remaining(plan, 15)):
            check_plan(plan)
            campaign.configure_native(plan['execution'])
            models = one.frozen_models(plan)
        report['timings']['preflight_seconds'] = time.monotonic() - started
        tick = time.monotonic()
        folder = evaluation.collect(plan, models)
        report['timings']['collection_and_storage_seconds'] = time.monotonic() - tick
        progress.set_phase(progress.progress_path(output, 1), 'verifying')
        tick = time.monotonic()
        with wall_limit(campaign.remaining(plan, 15)):
            proof, manifest = evaluation.verify_capture(folder, plan, one.frozen_models(plan),
                lambda: campaign.remaining(plan, 15), identity_check=check_plan, analyze=analyze, parallel_replay=True)
            require(binding(path)['sha256'] == sha, 'Worker plan changed during execution')
            report.update(status='passed', episode_directory=str(folder), episode_status=manifest['status'],
                termination_reason=manifest['termination_reason'], outcomes=manifest['outcomes_by_player'],
                choices_by_player=proof['choices_by_player'], analytics=proof['analytics'],
                native_replay_verified=True, both_policies_reconstructed=True,
                proof=binding(folder / 'evaluation-policy-reconstruction.json'))
        report['timings']['verification_and_analytics_seconds'] = time.monotonic() - tick
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        try:
            report['protected_inputs_unchanged'] = campaign.verify_protected(plan['protected'])
        except Exception as error:
            report.update(status='failed', protection_error=str(error))
        report['timings']['worker_total_seconds'] = time.monotonic() - started
        write(output / 'report.json', report)
        progress.set_phase(progress.progress_path(output, 1), 'verified' if report['status'] == 'passed' else 'failed')
    return 0 if report['status'] == 'passed' else 1


class Display:
    """Ten fixed terminal rows, using the existing tiny progress sidecars."""
    def __init__(self, folders, started):
        self.folders, self.started = folders, started
        self.tty = sys.stdout.isatty() and os.environ.get('TERM', '').lower() != 'dumb'
        self.color = progress.terminal_color_enabled(sys.stdout)
        self.lines, self.last, self.phases = 0, -100, None

    def values(self):
        values = []
        for folder in self.folders:
            try:
                value = progress._read(progress.progress_path(folder, 1))
            except (OSError, ValueError):
                value = dict(choices=[0, 0], days=[], ordinary=9920, phase='starting')
            values.append(value)
        return values

    def poll(self, *, final=False):
        values = self.values()
        elapsed = time.monotonic() - self.started
        phases = tuple(v['phase'] for v in values)
        if not final and elapsed - self.last < (1 if self.tty else 15) and phases == self.phases:
            return
        width = max(20, shutil.get_terminal_size((100, 24)).columns - 1)
        if self.tty and self.lines:
            emit(f'\x1b[{self.lines}F\x1b[J', end='')
        rows = [f'TEN FROZEN GAMES | {elapsed:6.1f}s / {SECONDS}s | checkpoint 1753',
                'Choices used (not time remaining); 9,920 ordinary + 320 dialog / player']
        rows = [row[:width] for row in rows]
        for slot, value in enumerate(values, 1):
            day = max((d for d in value.get('days', ()) if d is not None), default=0)
            counts = value['choices']
            used = min(8, 8 * max(counts) // 9920)
            row = f'{slot:02} Day {day:3} [{"#" * used}{"-" * (8-used)}] R {counts[0]:5} B {counts[1]:5}  {value["phase"]}'
            code = '1;32' if value['phase'] == 'verified' else '1;31' if value['phase'] == 'failed' else '36'
            rows.append(progress.color_text(row[:width], code, self.color))
        emit('\n'.join(rows), flush=True)
        self.lines, self.last, self.phases = len(rows), elapsed, phases


def overlap_witness(folders, values):
    """Observe ten real native leaders simultaneously, not merely ten Pythons."""
    if not all(v['phase'] == 'collecting' for v in values):
        return None
    leaders = []
    try:
        for folder in folders:
            rows = [json.loads(line) for line in (folder / 'native-process-deadline.jsonl').read_text().splitlines()]
            row = next(r for r in rows if r['event'] == 'exec_attempt')
            leaders.append(row['leader_pid'])
        snapshot = subprocess.check_output(['ps', '-p', ','.join(map(str, leaders)),
            '-o', 'pid=,pgid=,stat=,comm='], text=True, timeout=2)
        live = [line.strip().split(None, 3) for line in snapshot.splitlines()]
        if (len(live) != GAMES or {int(r[0]) for r in live} != set(leaders)
                or any(r[0] != r[1] or r[2].startswith('Z') or 'vcmi' not in r[3].lower() for r in live)):
            return None
        return dict(observed_unix=time.time(), native_processes=live,
                    native_leader_pids=leaders, collecting_slots=list(range(1, GAMES + 1)))
    except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
        return None


def prepare(output, checked, started, wall):
    output = evaluation.copies.private_path(output or campaign.ARTIFACTS / 'selfplay-ten' / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(4)), fresh=True)
    output.mkdir(parents=True, mode=0o700)
    plans = []
    for slot in range(1, GAMES + 1):
        folder = output / f'game-{slot:02}'
        reservation = evaluation.reserve(folder)
        folder.mkdir(mode=0o700)
        plan = dict(schema=one.SCHEMA + '-plan', **checked, output=str(folder),
            budget=one.BUDGET, policy_support=GUARD, physical_games=1, learner_player_for_reporting=0,
            purpose='diagnostic', training_input=False, optimizer_updates=0,
            reservation=reservation, schedule=read(bound(reservation))['schedule'],
            supervisor_pid=os.getpid(), maximum_seconds=SECONDS, batch_slot=slot, batch_source=source(),
            hard_deadline_monotonic=started + SECONDS - 30, hard_deadline_unix=wall + SECONDS - 30)
        write(folder / 'plan.json', plan, new=True)
        progress.GameProgress(progress.progress_path(folder, 1), episode=1, ordinary=9920, reserved=320).phase('starting')
        plans.append(plan)
    write(output / 'plan.json', dict(schema=SCHEMA, games=[binding(Path(p['output']) / 'plan.json') for p in plans],
        source=source(), checkpoint=one.CHECKPOINT, policy_support=GUARD, budget=one.BUDGET,
        seconds=SECONDS, started_unix=wall, training_input=False, optimizer_updates=0), new=True)
    return output, plans


def run(output, started, wall):
    checked = one.inspect_candidate(GUARD)
    output, plans = prepare(output, checked, started, wall)
    folders = [Path(p['output']) for p in plans]
    bindings = [binding(folder / 'plan.json') for folder in folders]
    display = Display(folders, started)
    processes, errors, cleanup_errors, witness = [], [], [], None
    cleanup = dict(deadline_monotonic=started + SECONDS - 2, deadline_wall=wall + SECONDS - 2)
    emit(f'Private results: {output}\nFrozen weights; no PPO. All ten slots launch together.', flush=True)
    try:
        for folder, plan, pin in zip(folders, plans, bindings):
            with (folder / 'worker.log').open('xb') as log:
                process = guarded_popen([sys.executable, str(Path(__file__).resolve()), '_worker',
                    '--plan', pin['path'], '--sha256', pin['sha256']],
                    deadline_monotonic=plan['hard_deadline_monotonic'], deadline_wall=plan['hard_deadline_unix'],
                    evidence_path=folder / 'worker-process-deadline.jsonl', cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                processes.append(process)
        while any(p.poll() is None for p in processes):
            display.poll()
            if witness is None:
                witness = overlap_witness(folders, display.values())
                if witness is not None:
                    write(output / 'parallel-native-witness.json', witness, new=True)
            if time.monotonic() >= started + SECONDS - 30 or time.time() >= wall + SECONDS - 30:
                raise TimeoutError('Twenty-minute batch exhausted its work budget')
            time.sleep(0.5)
    except (Exception, KeyboardInterrupt) as error:
        errors.append(f'{type(error).__name__}: {error}')
    finally:
        # Release every owner before waiting; cancellation is concurrent too.
        for process in processes:
            _release_owned_process(process)
        for folder, process in zip(folders, processes):
            try:
                close_owned_process(process, timeout=10, **cleanup)
                for receipt in folder.glob('*native-process-deadline.jsonl'):
                    cleanup_from_evidence(receipt, timeout=10, **cleanup)
            except Exception as error:
                cleanup_errors.append(f'{folder.name}: {error}')
        for folder in folders:
            progress.finish_unfinished(folder / 'progress')
        display.poll(final=True)
    errors.extend(cleanup_errors)
    protected = False
    try:
        protected = campaign.verify_protected(checked['protected'])
    except Exception as error:
        errors.append(f'Protected input: {error}')
    games = []
    for slot, (folder, pin) in enumerate(zip(folders, bindings), 1):
        try:
            report = read(folder / 'report.json')
            require(binding(folder / 'plan.json') == pin and report['plan'] == pin, 'Plan changed')
            require(slot <= len(processes) and processes[slot-1].returncode == 0, report.get('error', 'Worker did not exit successfully'))
            require(report.get('schema') == SCHEMA + '-game' and report.get('status') == 'passed'
                and report.get('checkpoint') == one.CHECKPOINT and report.get('slot') == slot
                and all(report.get(key) is True for key in ('native_replay_verified', 'both_policies_reconstructed',
                    'protected_inputs_unchanged')) and report.get('optimizer_updates') == 0, 'Game did not verify')
            proof = read(bound(report['proof']))
            require(proof['analytics'] == report['analytics'] and proof['choices_by_player'] == report['choices_by_player'],
                    'Analytics or choices differ from verified proof')
            games.append(report)
        except Exception as error:
            errors.append(f'Game {slot}: {error}')
    elapsed = time.monotonic() - started
    passed = len(games) == GAMES and witness is not None and protected and not errors and elapsed < SECONDS
    report = dict(schema=SCHEMA + '-result', status='passed' if passed else 'failed',
        checkpoint=one.CHECKPOINT, games_requested=GAMES, games_verified=len(games),
        total_seconds=elapsed, deadline_seconds=SECONDS, all_ten_native_overlap=witness is not None,
        overlap=binding(output / 'parallel-native-witness.json') if witness else None,
        protected_inputs_unchanged=protected, owned_cleanup_verified=not any(p.poll() is None for p in processes) and not cleanup_errors,
        training_input=False, optimizer_updates=0, errors=errors, games=games,
        red_wins=sum(g['outcomes'].get('red') == 'win' for g in games),
        blue_wins=sum(g['outcomes'].get('blue') == 'win' for g in games),
        cutoffs=sum(g['episode_status'] == 'truncated' for g in games))
    lines = ['# Ten-game self-play results', '',
        f"Status: {report['status']}. {len(games)}/10 verified in {elapsed:.1f} seconds. "
        f"Ten native games observed concurrently: {witness is not None}. Zero PPO updates.", '',
        '| Game | Result | Day | Choices R/B | Moves R/B | New tiles R/B | Captures R/B | Hires R/B | Seconds |',
        '|---|---|---|---|---|---|---|---|---|']
    for game in games:
        red, blue = (game['analytics']['players'][c] for c in ('red', 'blue'))
        pair = lambda field: f'{red[field]}/{blue[field]}'
        outcome = next((f'{c.title()} win' for c in ('red', 'blue') if game['outcomes'].get(c) == 'win'), game['termination_reason'])
        lines.append(f"| {game['slot']} | {outcome} | {max(red['last_day'], blue['last_day'])} | {pair('choices')} | "
            f"{red['movement'].get('endpoint_changes', 0)}/{blue['movement'].get('endpoint_changes', 0)} | "
            f"{pair('new_tiles')} | {pair('town_captures')} | {pair('hires')} | {game['timings']['worker_total_seconds']:.1f} |")
    lines.extend(['', 'Moves count observed endpoint changes, not clicks. Day/choice cutoffs are unscored.',
                  'Per-player action counts, movement failures, battles and timing phases are in report.json.'])
    (output / 'analytics.md').write_text('\n'.join(lines) + '\n')
    emit('\n'.join(lines[2:]), flush=True)
    emit(f'Analytics: {output / "analytics.md"}\nFull report: {output / "report.json"}', flush=True)
    if errors:
        emit('\n'.join(errors), file=sys.stderr, flush=True)
    elapsed = time.monotonic() - started
    # Include analytics creation and terminal rendering, reserving one second
    # for the small final report and return to the shell.
    passed = passed and elapsed < SECONDS - 1
    report.update(total_seconds=elapsed, status='passed' if passed else 'failed')
    write(output / 'report.json', report, new=True)
    emit(f"Batch {report['status']}: {elapsed:.2f} seconds including analytics.", flush=True)
    return 0 if passed else 1


def main():
    started, wall = STARTED, STARTED_WALL
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('command', nargs='?', choices=('_worker',))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.command == '_worker':
        require(args.plan is not None and args.sha256 is not None and not args.run and not args.check, 'Bound plan required')
        return worker(args.plan, args.sha256)
    try:
        if not args.run:
            one.inspect_candidate(GUARD)
            emit('READY: 10 parallel frozen games from checkpoint 1753; hero-relative exact-command cooldown.')
            emit('Unchanged 9,920 + 320 choices/player, 365 days, all replay checks. No seeds consumed.')
            emit('Total limit: 20 minutes including verification and analytics. Timeouts fail; no automatic retry.')
            emit('Launch: ./play-ten-selfplay.sh --run')
            return 0
        with campaign.activity_lease():
            with wall_limit(SECONDS - 5 - (time.monotonic() - started)):
                return run(args.output, started, wall)
    finally:
        emit(f'Total running time: {time.monotonic() - started:.2f} seconds.', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
