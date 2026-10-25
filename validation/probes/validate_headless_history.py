#!/usr/bin/env python3
"""Check desired history behavior with synthetic messages/fake engine only.

A represented terminal sighting must survive projection, or that projection
must be declared invalid and excluded. Decoding alone does not establish
completeness. Exit 1 reports an unmet desired behavior; JSON is still saved.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--learner-parent', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--tests', action='store_true', help='Also run test_headless.py; off by default')
a = parser.parse_args()
root, learner, output = a.source.resolve(), a.learner_parent.resolve(), a.output.resolve()
os.environ['HEROES_LEARNER_PATH'] = str(learner)
sys.path.insert(0, str(learner))
from learner.contract import HistoryCursor, decode_frame
spec = importlib.util.spec_from_file_location('history_review_fixture', root / 'tests/test_headless.py')
t = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t
spec.loader.exec_module(t)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_frames(frames, episode='1', player=0):
    cursor = HistoryCursor(episode, player)
    for frame in frames:
        cursor.consume(decode_frame(frame))
    return {'frames': len(frames), 'next_seq': cursor.next_seq,
            'pending_action': cursor.pending_action,
            'unacted_decision': cursor.decision.seq if cursor.decision else None}

class Stream:
    def __init__(self):
        self.history = t.player_history.PlayerHistory(0, 'red', '1')
        self.validator = t.bridge_protocol.BridgeValidator(0, '1')
        self.hero, self.enemy = t.obj(1, 0, 0, owner=0), t.obj(2, 2, 1)
        self.put({'type': 'ready', 'protocol_version': 3, 'player': 0, 'episode': '1',
                  'adventure_actions': 9, 'observation_events': 'visible_update-v2',
                  'map_size': [20, 20, 1]})
        self.observe(0, 1, [self.hero, self.enemy], [t.update(0, appeared=[self.hero, self.enemy],
                           revealed=t.columns(0, 2), own=t.OWN)])
    def put(self, m):
        kind = self.validator.check(m)
        if kind == 'ready': self.history.start(m['map_size'])
        else: self.history.ingest(m)
    def observe(self, action_id, day, objects, events):
        m = t.observation(action_id, t.columns(0, 2), objects, events, day=day)
        m.update(width=20, height=20)
        self.put(m)
    def act(self, action_id, action=8):
        self.validator.respond(action)
        self.history.act(action, action_id)
    def result(self, action_id, day, events=()):
        self.put({'type': 'action_result', 'protocol_version': 3, 'player': 0, 'episode': '1',
                  'action_id': action_id, 'action': 8, 'accepted': True,
                  'before': {'day': day, 'hero_id': 1}, 'after': {'day': day, 'hero_id': 1},
                  'events': list(events)})
        self.put({'type': 'turn_end', 'protocol_version': 3, 'player': 0, 'episode': '1'})
    def terminal(self, consumed, events):
        self.put({'type': 'terminal', 'protocol_version': 3, 'player': 0, 'episode': '1',
                  'outcome': 'loss', 'last_action_id': consumed,
                  'state': {'day': 1, 'hero_id': -1}, 'events': events})

s = Stream()
s.act(0); s.result(0, 1)
s.observe(1, 3, [s.hero, s.enemy], [])
visible = s.history.memory()[1]
s.act(1); s.result(1, 4, [t.update(1, disappeared=[2], day=4)])
hidden = s.history.memory()[1]
s.observe(2, 6, [s.hero], [])
later = s.history.memory()[1]
assert visible['day'] == hidden['day'] == later['day'] == 3
assert visible['visible'] and not hidden['visible'] and not later['visible']
assert visible['fact'] == hidden['fact'] == later['fact']
assert visible['sequence'] == hidden['sequence'] == later['sequence'] == 0
memory_result = {'desired_behavior_passed': True, 'visible_day3': visible,
                 'hidden_day4': hidden, 'still_hidden_day6': later,
                 'strict_cursor': check_frames(s.history.v1.frames)}

terminal_cases = []
for submitted, consumed in ((False, -1), (True, -1), (True, 0)):
    s = Stream()
    before_delivery = copy.deepcopy(s.history.v1.frames)
    if submitted: s.act(0, 2)
    before_terminal = copy.deepcopy(s.history.v1.frames)
    transient = t.obj(3, 1, 2)
    s.terminal(consumed, [t.update(1, appeared=[transient]), t.update(2, disappeared=[3])])
    frames = s.history.v1.frames
    raw_has_sighting = any(e.get('appeared') for e in s.history.records[-1]['events'])
    projected_has_sighting = any(o['handle'] == '3' for f in frames if 'view' in f for o in f['view']['objects'])
    summary = s.history.v1.summary()
    assert raw_has_sighting, 'The probe failed to deliver its terminal sighting'
    projection_loss = not projected_has_sighting
    desired = not projection_loss or summary.get('valid') is False
    if submitted and consumed == 0:
        assert projected_has_sighting, 'Consumed-action terminal views must remain represented'
    assert before_delivery == frames[:len(before_delivery)], 'Previously delivered decision prefix mutated'
    terminal_cases.append({'submitted': submitted, 'last_action_id': consumed,
        'raw_terminal_contains_transient_sighting': raw_has_sighting,
        'projected_history_contains_transient_sighting': projected_has_sighting,
        'represented_terminal_views_lost': projection_loss,
        'projection_declared_invalid': summary.get('valid') is False,
        'desired_behavior_passed': desired,
        'summary': summary, 'strict_cursor': check_frames(frames),
        'already_delivered_prefix_unchanged': True,
        'provisional_public_history_was_retracted': len(frames) < len(before_terminal)})

# Freeze delivered request contents and process them immediately through the real learner cursor.
# This catches both mutation and retrospective-only validation of the exposed Request stream.
delivery_cases = []
for mode in ('reply_unconsumed', 'terminal_before_reply', 'terminal_race'):
    with tempfile.TemporaryDirectory(prefix='heroes-history-review-') as directory:
        env = t.env_module.HeadlessSelfPlayEnv(t.FakeLauncher(mode=mode, end_day=2), Path(directory) / 'run',
                  t.env_module.Limits(watchdog_seconds=10, startup_seconds=20, shutdown_seconds=10))
        players = {'red': t.ScriptedAgent([8, 2]), 'blue': t.ScriptedAgent()}
        cursors = {c: HistoryCursor('1', i) for i, c in enumerate(('red', 'blue'))}
        delivered = {c: [] for c in cursors}
        retained = []
        try:
            step = env.reset(1, 7)
            while not step.done:
                frozen = copy.deepcopy(step.frames)
                retained.append((step, frozen))
                for frame in frozen: cursors[step.color].consume(decode_frame(frame))
                delivered[step.color].extend(frozen)
                step = env.step(players[step.color].act(step))
            for color in cursors:
                for frame in step.final_frames[color]: cursors[color].consume(decode_frame(frame))
                delivered[color].extend(copy.deepcopy(step.final_frames[color]))
                assert delivered[color] == env.history_v1(color)
            assert all(request.frames == frozen for request, frozen in retained)
            delivery_cases.append({'mode': mode, 'status': step.status,
                  'desired_behavior_passed': True,
                  'incremental_decoding_passed': True, 'delivered_frames_immutable': True,
                  'delivered_frames_equal_final_history': True,
                  'player_summaries': {c: step.manifest['players'][c]['history_v1'] for c in cursors}})
        finally: env.close()

test_result = {'requested': a.tests, 'run': False}
if a.tests:
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_headless.py', '-v']
    tests = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=90)
    log = output.with_name('history-tests.log')
    log.write_text(tests.stdout + tests.stderr)
    match = re.search(r'Ran (\d+) tests', tests.stderr)
    test_result = {'requested': True, 'run': True, 'command': command,
                   'returncode': tests.returncode, 'desired_behavior_passed': tests.returncode == 0,
                   'tests_run': int(match.group(1)) if match else None,
                   'log': str(log), 'log_sha256': sha(log)}
all_desired = (memory_result['desired_behavior_passed']
               and all(c['desired_behavior_passed'] for c in terminal_cases + delivery_cases)
               and (not a.tests or test_result['desired_behavior_passed']))
report = {
    'reviewed_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'scope': 'Synthetic messages, fake-engine adapter checks, and strict root learner contract. No VCMI build/game, training, source or test edits.',
    'last_seen_time': memory_result,
    'terminal_history_cases': terminal_cases,
    'incremental_delivery_cases': delivery_cases,
    'desired_behavior_passed': all_desired,
    'terminal_history_requirement': {
        'requirement': 'Retain all represented terminal views or explicitly mark the projection invalid/ineligible.',
        'integration_gate': 'Do not attest histories_complete merely because the emitted prefix passes HistoryCursor.',
        'implementation_note': 'No schema extension is assumed. A fail-closed projection is acceptable while full raw events remain retained.',
        'desired_behavior_passed': all(c['desired_behavior_passed'] for c in terminal_cases)},
    'retraction_assessment': 'Existing fake-engine cases preserve already-delivered Request prefixes, including incremental root HistoryCursor consumption. The adapter does retract a provisional action from its internal/public full-history snapshot. No additional ordinary Request-stream mutation was reproduced; any collector must not learn from provisional snapshots before reconciliation.',
    'targeted_tests': test_result,
    'source_sha256': {p: sha(root / p) for p in ['experiments/headless/history_v1.py', 'experiments/headless/player_history.py', 'experiments/headless/bridge_protocol.py']},
    'learner_contract_sha256': sha(learner / 'learner/contract.py'),
    'probe': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__), 'argv': sys.argv}
}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps({'report': str(output), 'desired_behavior_passed': all_desired,
                  'last_seen_fix_passed': True,
                  'invalidly_suppressed_terminal_cases': sum(not c['desired_behavior_passed'] for c in terminal_cases),
                  'incremental_delivery_cases_passed': len(delivery_cases),
                  'targeted_tests_run': test_result['run']}))
raise SystemExit(0 if all_desired else 1)
