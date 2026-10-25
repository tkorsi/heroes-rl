#!/usr/bin/env python3
"""Audit persisted real-game histories; read only, never launch games or train.

Exit 0 means the recorded evidence is internally consistent with the declared
projection status. It does not mean every episode is learner-eligible, nor that
visibility completeness has been proved against native VCMI state.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sightings(frames: list[dict]) -> dict:
    """Deduplicate (handle, position) within each adjacent own-decision interval."""
    decisions = [f for f in frames if f['kind'] == 'decision']
    transient, intermediate, intervals = [], [], set()
    def enemies(frame):
        return {o['handle']: o for o in frame['view']['objects']
                if o['owner'] == 'opponent' and o['kind'] == 34}
    for before, after in zip(decisions, decisions[1:]):
        left, right, seen = enemies(before), enemies(after), set()
        for frame in frames[before['seq'] + 1:after['seq']]:
            if frame['kind'] != 'view':
                continue
            for handle, obj in enemies(frame).items():
                key = handle, tuple(obj['position'])
                if key in seen:
                    continue
                seen.add(key)
                sample = {'decision_before_seq': before['seq'], 'decision_after_seq': after['seq'],
                          'sighting_seq': frame['seq'], 'day': frame['day'],
                          'handle': handle, 'position': obj['position']}
                if handle not in left and handle not in right:
                    transient.append(sample)
                    intervals.add((before['seq'], after['seq']))
                if ((handle not in left or left[handle]['position'] != obj['position'])
                        and (handle not in right or right[handle]['position'] != obj['position'])):
                    intermediate.append(sample)
    return {'fully_transient_position_samples': len(transient),
            'fully_transient_decision_intervals': len(intervals),
            'intermediate_position_samples': len(intermediate),
            'fully_transient_examples': transient[:12], 'intermediate_examples': intermediate[:6]}


def audit_player(manifest_path, manifest, color, root, PlayerHistory, BridgeValidator, contract):
    episode, player = str(manifest['episode']), ('red', 'blue').index(color)
    ledger = manifest_path.parent / 'agents' / color / 'bridge.jsonl'
    result = {'episode': episode, 'player': color, 'ledger': str(ledger),
              'ledger_sha256': digest(ledger), 'manifest_status': manifest['status'], 'failures': []}
    failures = result['failures']
    history, validator = PlayerHistory(player, color, episode), BridgeValidator(player, episode)
    observed, responded, resolved = {}, {}, {}
    counts, decision_to_action, terminal = Counter(), {}, None
    records = ledger.read_text().splitlines()
    for line_number, line in enumerate(records, 1):
        try:
            row = json.loads(line)
            message = row['message']
            if row['direction'] == 'engine':
                kind = validator.check(message)
                counts['engine_' + kind] += 1
                if kind == 'ready':
                    history.start(message['map_size'])
                    continue
                if kind == 'error':
                    raise ValueError('Engine error message: ' + str(message.get('message')))
                before = len(history.v1.frames)
                history.ingest(message)
                if kind == 'observation':
                    observed[message['action_id']] = message
                    if len(history.v1.frames) > before and history.v1.frames[-1]['kind'] == 'decision':
                        decision_to_action[history.v1.frames[-1]['seq']] = message['action_id']
                elif kind == 'action_result':
                    resolved[message['action_id']] = message
                    counts['actions_accepted' if message['accepted'] else 'actions_rejected'] += 1
                elif kind == 'terminal':
                    terminal = message
            elif row['direction'] == 'policy':
                delivered = message.get('delivered', True)
                if type(delivered) is not bool:
                    raise ValueError('Malformed delivered flag')
                response = {k: v for k, v in message.items() if k != 'delivered'}
                value = response['action'] if 'action' in response else response['choice']
                if validator.respond(value) != response:
                    raise ValueError('Policy response does not match validated request')
                if not delivered:
                    validator.refuse(response)
                    counts['responses_undelivered'] += 1
                elif 'action' in response:
                    counts['actions_sent'] += 1
                    responded[response['action_id']] = response['action']
                    history.act(response['action'], response['action_id'])
                else:
                    counts['decisions_sent'] += 1
            else:
                raise ValueError('Unknown ledger direction')
        except Exception as error:
            failures.append({'phase': 'ledger_reconstruction', 'line': line_number, 'error': repr(error)})
            break

    frames, summary = history.v1.frames, history.v1.summary()
    recorded = manifest['players'][color]
    if summary != recorded['history_v1']:
        failures.append({'phase': 'manifest', 'error': 'Reconstructed projection summary differs from manifest'})
    derived = {k: counts[k] for k in ('actions_sent', 'decisions_sent', 'responses_undelivered',
                                     'actions_accepted', 'actions_rejected')}
    derived.update(actions_unconsumed=history.actions_unconsumed,
                   actions_without_result=history.actions_without_result,
                   visible_updates=validator.next_visible_sequence, records=len(history.records))
    counter_differences = {k: {'derived': v, 'manifest': recorded.get(k)}
                           for k, v in derived.items() if recorded.get(k) != v}
    if counter_differences:
        failures.append({'phase': 'counters', 'differences': counter_differences})
    cursor, decoded = contract.HistoryCursor(episode, player), 0
    decoder_error = None
    for frame in frames:
        try:
            cursor.consume(contract.decode_frame(frame))
            decoded += 1
        except Exception as error:
            decoder_error = {'seq': frame.get('seq'), 'error': repr(error)}
            break
    if decoder_error:
        failures.append({'phase': 'learner_decode', **decoder_error})

    emitted_views = sum(f['kind'] == 'view' for f in frames)
    missing_views = validator.next_visible_sequence - emitted_views
    exclusions = summary.get('unrepresented_events', {})
    supported_exclusions = {'battle_start', 'battle_complete', 'visit', 'dialog', 'dialog_answer', 'level_up'}
    unexpected_exclusions = {k: v for k, v in exclusions.items() if v and k not in supported_exclusions}
    declared_invalid = summary.get('valid') is False
    incomplete = (declared_invalid or summary.get('complete') is False or summary.get('error') is not None
                  or missing_views != 0 or bool(unexpected_exclusions))
    if (missing_views != 0 or unexpected_exclusions) and not declared_invalid:
        failures.append({'phase': 'completeness', 'error': 'Omitted represented views are not declared invalid',
                         'visible_updates_without_view_frames': missing_views,
                         'unexpected_unrepresented_events': unexpected_exclusions})

    consumed_ids = set(resolved)
    watermark = terminal.get('last_action_id') if terminal else None
    if watermark is not None and watermark >= 0:
        consumed_ids.add(watermark)
    pending_deliveries = set(responded) - set(resolved)
    expected_unconsumed = len(pending_deliveries - consumed_ids) if terminal else 0
    expected_without_result = len(pending_deliveries & consumed_ids) if terminal else 0
    for name, expected in (('actions_unconsumed', expected_unconsumed),
                           ('actions_without_result', expected_without_result)):
        if derived[name] != expected:
            failures.append({'phase': 'consumption', 'counter': name, 'expected': expected, 'actual': derived[name]})
    frame_actions = [f for f in frames if f['kind'] == 'action']
    projected_action_ids = []
    for frame in frame_actions:
        action_id = decision_to_action.get(frame['decision_seq'])
        if action_id is None or responded.get(action_id) != frame['action']:
            failures.append({'phase': 'action_mapping', 'seq': frame['seq'], 'decision_seq': frame['decision_seq']})
        projected_action_ids.append(action_id)
    unconsumed_projected = [i for i in projected_action_ids if terminal and i not in consumed_ids]
    # Invalid diagnostic prefixes are retained for inspection but never eligible for terminal credit.
    if terminal and not incomplete and unconsumed_projected:
        failures.append({'phase': 'consumption', 'error': 'Valid projection contains action(s) not consumed',
                         'action_ids': unconsumed_projected})
    if terminal and not incomplete and set(projected_action_ids) != consumed_ids:
        failures.append({'phase': 'consumption', 'error': 'Valid complete projection does not match consumed actions',
                         'projected': projected_action_ids, 'consumed': sorted(consumed_ids)})
    eligible = not incomplete and decoder_error is None and not failures
    credit = None
    if manifest['status'] == 'completed' and eligible:
        if not terminal or not projected_action_ids or projected_action_ids[-1] != watermark:
            failures.append({'phase': 'terminal_credit', 'error': 'No matching last consumed action'})
            eligible = False
        else:
            expected_reward = 1 if terminal['outcome'] == 'win' else -1
            if manifest['rewards_by_player'][color] != expected_reward:
                failures.append({'phase': 'terminal_credit', 'error': 'Final reward disagrees with own terminal'})
                eligible = False
            else:
                credit = {'action_id': watermark, 'frame_seq': frame_actions[-1]['seq'],
                          'decision_seq': frame_actions[-1]['decision_seq'], 'reward': expected_reward}
    if manifest['status'] != 'completed' and manifest['rewards_by_player'][color] is not None:
        failures.append({'phase': 'terminal_credit', 'error': 'Non-natural episode carries terminal reward'})
    if failures:
        eligible, credit = False, None
    cutoff = recorded.get('cutoff')
    cutoff_audit = None
    if cutoff and cutoff.get('history_v1_seq') is not None:
        seq = cutoff['history_v1_seq']
        boundary = frames[seq] if 0 <= seq < len(frames) else None
        cutoff_audit = {'seq': seq, 'is_decision': boundary is not None and boundary['kind'] == 'decision',
                        'later_action_frames': sum(f['kind'] == 'action' and f['decision_seq'] >= seq for f in frames)}
        if not cutoff_audit['is_decision']:
            failures.append({'phase': 'cutoff', 'error': 'Cutoff does not name a decision frame'})
            eligible, credit = False, None
    result.update(decoded_frames=decoded, reconstructed_frames=len(frames), history_v1=summary,
        manifest_projection_matches=summary == recorded['history_v1'], counters=derived,
        counter_differences=counter_differences, decoder_error=decoder_error,
        emitted_view_frames=emitted_views, visible_updates_without_view_frames=missing_views,
        declared_invalid=declared_invalid, projection_incomplete=incomplete,
        unexpected_unrepresented_events=unexpected_exclusions,
        eligible_by_adapter_checks=eligible,
        terminal_action_watermark=watermark, delivered_action_ids=sorted(responded),
        consumed_action_ids=sorted(consumed_ids), projected_action_ids=projected_action_ids,
        unconsumed_projected_action_ids=unconsumed_projected,
        terminal_credit_candidate=credit, cutoff=cutoff_audit,
        sightings=sightings(frames))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Exact final PR source checkout')
    parser.add_argument('--learner-parent', type=Path, required=True, help='Folder containing learner/contract.py')
    parser.add_argument('--run', type=Path, required=True, help='run_series or run_headless output directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, learner, run, output = (p.resolve() for p in (args.source, args.learner_parent, args.run, args.output))
    if output.exists():
        parser.error('Refusing to overwrite existing audit output')
    sys.path.insert(0, str(root / 'experiments/headless'))
    sys.path.insert(0, str(learner))
    from player_history import PlayerHistory
    from bridge_protocol import BridgeValidator
    from learner import contract
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    manifests = sorted(run.glob('worker-*/episodes/episode-*/manifest.json')) or sorted(run.glob('episodes/episode-*/manifest.json'))
    if not manifests:
        parser.error('No episode manifests found')
    episodes, results, failures = [], [], []
    for path in manifests:
        manifest = json.loads(path.read_text())
        identity = manifest.get('identity', {}).get('project', {})
        if identity.get('commit') != head:
            failures.append({'episode': manifest['episode'], 'phase': 'provenance',
                             'error': 'Run source commit differs from current adapter checkout', 'recorded': identity})
        if identity.get('dirty'):
            failures.append({'episode': manifest['episode'], 'phase': 'provenance', 'error': 'Run project was dirty'})
        episodes.append({'episode': manifest['episode'], 'status': manifest['status'],
                         'termination_reason': manifest['termination_reason'],
                         'manifest': str(path), 'manifest_sha256': digest(path), 'project_identity': identity})
        for color in ('red', 'blue'):
            try:
                results.append(audit_player(path, manifest, color, root, PlayerHistory, BridgeValidator, contract))
            except Exception as error:
                failures.append({'episode': manifest['episode'], 'player': color,
                                 'phase': 'audit_exception', 'error': repr(error)})
    failed_players = sum(bool(r['failures']) for r in results)
    report = {'reviewed_commit': head, 'run': str(run), 'episodes': episodes, 'players': results,
        'failures': failures,
        'totals': {'episodes': len(episodes), 'player_streams': len(results),
            'decoded_frames': sum(r['decoded_frames'] for r in results),
            'incomplete_projections': sum(r['projection_incomplete'] for r in results),
            'declared_invalid_projections': sum(r['declared_invalid'] for r in results),
            'eligible_by_adapter_checks': sum(r['eligible_by_adapter_checks'] for r in results),
            'terminal_credit_candidates': sum(r['terminal_credit_candidate'] is not None for r in results),
            'failed_player_audits': failed_players, 'other_failures': len(failures),
            'fully_transient_position_samples': sum(r['sightings']['fully_transient_position_samples'] for r in results),
            'fully_transient_decision_intervals': sum(r['sightings']['fully_transient_decision_intervals'] for r in results),
            'intermediate_position_samples': sum(r['sightings']['intermediate_position_samples'] for r in results)},
        'limitations': ['Decoder success checks emitted records, not completeness. Missing represented views or declared invalid projections cannot receive learner terminal credit.',
            'Eligibility here is limited to adapter consistency checks, not permission to train or proof of exhaustive native information filtering.',
            'Terminal credits rely on the engine consumption watermark; this script cannot establish its native concurrency correctness.',
            'Sightings count unique (handle, position) samples per adjacent own-decision interval, not distinct opponents or total appearance events.',
            'No games, VCMI build, optimizer updates or training are run.'],
        'source_sha256': {name: digest(root / 'experiments/headless' / name)
                          for name in ('player_history.py', 'history_v1.py', 'bridge_protocol.py')},
        'learner_contract_sha256': digest(learner / 'learner/contract.py'),
        'probe': {'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__)), 'argv': sys.argv}}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'report': str(output), **report['totals']}, indent=2))
    return 1 if failed_players or failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
