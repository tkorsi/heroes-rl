#!/usr/bin/env python3
"""Check exported timeline coverage and temporal boundaries against capture files."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('bundle', type=Path)
p.add_argument('episode', type=Path)
p.add_argument('--output', type=Path)
a = p.parse_args()
b = json.loads(a.bundle.read_text())
def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
server = rows(a.episode / 'engine/replay.jsonl')
tape = rows(a.episode / 'engine/contextual-replay.jsonl')
endpoints = {r['seq']: r['client_out'] for r in server if r['kind'] == 'applied'}
original = rows(a.episode / 'agents/blue/history-v7.jsonl')
actions = [r['action'] for r in original if r['kind'] == 'submitted']
events = b['events']
assert [e['id'] for e in events] == list(range(len(events)))
assert [f['seq'] for f in b['frames']] == list(range(-1, len(endpoints)))
assert [json.loads(e['raw']) for e in events if e['source'].startswith('contextual')] == tape
assert [json.loads(e['raw'])['response']['action'] for e in events if e['category'] == 'decision'] == actions
assert [b['frames'][e['frame']]['seq'] for e in events if e['category'] == 'state'] == list(range(-1,len(endpoints)))
assert [e['packet'] for e in events] == sorted(e['packet'] for e in events)
known_observation = None
known_hero_details = None
known_hero_packet = 0
known_slots = [{'slot':0,'creature':0,'count':25}]
for e in events:
    if e['source'].startswith('contextual'):
        for record in json.loads(e['raw']).get('message',{}).get('records',[]):
            if record.get('observation') is not None:
                known_observation = record['observation']
                window = known_observation.get('window') or {}
                if window.get('context', {}).get('kind') == 'hero' and window['context'].get('subject') == 'object-0':
                    known_hero_details, known_hero_packet = window, e['packet']
                for army in (known_observation.get('window') or {}).get('armies',[]):
                    if army['subject'] == 'object-0' and army['role'] in ('hero','visiting_hero','garrison_hero'):
                        known_slots = [{k:s[k] for k in ('slot','creature','count')} for s in army['slots'] if s['count']]
    assert json.loads(e['hero_details']) == known_hero_details and e['hero_details_packet'] == known_hero_packet, 'Future hero window leaked'
    assert json.loads(e['observation']) == known_observation, 'Future observation leaked'
    if e['army']['status'] != 'absent':
        assert e['army']['slots'] == known_slots, 'Future army leaked'
    seq = b['frames'][e['frame']]['seq']
    assert seq == -1 or endpoints[seq] <= e['packet'], 'Future map state leaked'
    assert e['army']['source_packet'] <= e['packet']
    if e['category'] == 'decision':
        policy = json.loads(e['policy'])
        assert policy['action'] == json.loads(e['raw'])['response']['action']
        assert abs(sum(c['log_prob'] for c in policy['conditionals']) - policy['log_prob']) < 1e-5
assert len(b['object_details']) == len(b['frames'][0]['objects'])
assert all(b['object_details'][str(o[0])]['quantity'] == 500 for o in b['frames'][0]['objects'] if o[4] == 79 and o[5] == 6)
assert b['frames'][-1]['boundary'] == 'game_end'
assert events[-1]['army']['status'] == 'absent'
result = dict(schema='heroes-replay-inspector-validation-v1', passed=True,
              bundle_sha256=hashlib.sha256(a.bundle.read_bytes()).hexdigest(),
              states=len(b['frames']), events=len(events), decisions=len(actions),
              contextual_records=len(tape), history_frames=len(original),
              checks=['exact contextual coverage', 'exact learner action coverage', 'all native endpoints',
                      'no future map, observation or army', 'conditional log-probability sum', 'natural terminal', 'no future hero window', 'object metadata coverage and gold amounts'])
if a.output:
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
