#!/usr/bin/env python3
"""Export the verified schematic match into an offline native inspector bundle."""
import argparse
import copy
import math
import zipfile
import json
from pathlib import Path
from .render_builtin_match import inspect_inputs, digest

NAMES = {180: 'End turn', 182: 'Move hero', 183: 'Arrange army',
         185: 'Build structure', 187: 'Recruit creatures', 188: 'Upgrade creatures',
         198: 'Automatic battle action', 260: 'Interface ready'}

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2)

def export(episode, trace, report):
    data = inspect_inputs(episode, trace, report)
    server = rows(episode / 'engine/replay.jsonl')
    tape = rows(episode / 'engine/contextual-replay.jsonl')
    rollout_path = episode / 'builtin-rollout.json'
    rollout = json.loads(rollout_path.read_text())
    applied = {r['seq']: r for r in server if r['kind'] == 'applied'}
    inputs = {r['seq']: r for r in server if r['kind'] == 'input'}
    endpoints = [(r['client_out'], r['seq'] + 1) for r in server if r['kind'] == 'applied']
    events, observation, packet, decision = [], None, 0, 0
    known_army = copy.deepcopy(data['hero_army'][-1])
    hero_details, hero_details_packet = None, 0
    with zipfile.ZipFile(episode / 'profile/Maps/headless.vmap') as archive:
        map_objects = json.loads(archive.read('objects.json'))
    object_details = {}
    kinds = {'hero':34, 'town':98, 'mine':53, 'monster':54, 'resource':79}
    for obj in map_objects:
        mask = obj['template']['mask']
        anchors = [(x,y) for y,row in enumerate(mask) for x,c in enumerate(row) if c == 'A']
        assert len(anchors) == 1
        ax, ay = anchors[0]
        position = [obj['x'] - (len(mask[0])-1-ax), obj['y'] - (len(mask)-1-ay), obj.get('l',0)]
        matches = [o for o in data['frames'][0]['objects'] if o[4] == kinds[obj['type']] and o[1:4] == position]
        assert len(matches) == 1, 'Ambiguous map object identity'
        options = obj['options']
        lines = []
        if 'army' in options:
            lines.append('Стартовая армия: ' + (', '.join(f"{a['amount']} × {a['type'].replace('core:pikeman','копейщики')}" for a in options['army']) or 'пусто'))
        if 'primarySkills' in options:
            skills = options['primarySkills']
            lines.append(f"Стартовые характеристики: атака {skills['attack']}, защита {skills['defence']}, сила магии {skills['spellpower']}, знания {skills['knowledge']}")
        for key,label in [('artifacts','Стартовые артефакты'),('secondarySkills','Стартовые навыки'),('spellBook','Стартовые заклинания')]:
            if key in options: lines.append(label + ': ' + (pretty(options[key]) if options[key] else 'нет'))
        if 'buildings' in options: lines.append('Стартовые здания: ' + ', '.join(options['buildings']['allOf']))
        if obj['type'] == 'monster': lines.append(f"Стартовое количество: {options['amount']} копейщиков")
        object_details[str(matches[0][0])] = dict(initial='\n'.join(lines), quantity=options.get('amount'), subtype=obj['subtype'])
    def add(title, actor, category, frame, raw, policy=None, source=''):
        army = copy.deepcopy(known_army)
        state = data['frames'][frame]
        hero = next((o for o in state['objects'] if o[0] == army['hero_id']), None)
        army['position'] = hero[1:4] if hero else None
        if hero is None:
            army.update(status='absent', slots=[], freshness='absent')
        elif state.get('battles'):
            army.update(status='battle', freshness='pre_battle')
        events.append(dict(hero_details=pretty(hero_details), hero_details_packet=hero_details_packet, army=army, id=len(events), title=title, actor=actor, category=category,
                           frame=frame, packet=packet, source=source, raw=pretty(raw),
                           observation=pretty(observation), policy=pretty(policy) if policy else '',
                           summary=(f"P(action) = {math.exp(policy['log_prob']):.6g}  •  V = {policy['value']:.5f}" if policy else '')))
    add('Initial state', 'Engine', 'state', 0, server[:2], source='native initial state')
    emitted = set()
    for transaction in tape:
        if transaction['kind'] == 'packet_complete':
            packet = transaction['completed']
        frame = max([index for end, index in endpoints if end <= packet] or [0])
        for record in transaction.get('message', {}).get('records', []):
            if record.get('observation') is not None:
                observation = record['observation']
                window = observation.get('window') or {}
                if window.get('context', {}).get('kind') == 'hero' and window['context'].get('subject') == 'object-0':
                    hero_details, hero_details_packet = copy.deepcopy(window), packet
                for army in (observation.get('window') or {}).get('armies', []):
                    if army['subject'] == 'object-0' and army['role'] in ('hero', 'visiting_hero', 'garrison_hero'):
                        known_army.update(slots=[{k: slot[k] for k in ('slot', 'creature', 'count')} for slot in army['slots'] if slot['count']], source_packet=packet, freshness='observed')
        kind = transaction['kind']
        actor, title, policy = 'Engine', kind, None
        category = 'event'
        if kind == 'response':
            actor, category = 'Our network · Blue', 'decision'
            action = transaction['response']['action']
            title = action['kind']
            if action.get('x') is not None: title += f" ({action['x']}, {action['y']})"
            policy = rollout['decisions'][decision]
            assert policy['action'] == action, 'Policy decision/action mismatch'
            decision += 1
        elif kind == 'request':
            actor = 'Nullkiller · Red' if transaction.get('source') == 'native_opponent' else 'Adapter / automatic'
            if transaction['pack_type'] == 198: actor = 'Autobattle · ' + ('Red' if transaction['player'] == 0 else 'Blue')
            title = NAMES.get(transaction['pack_type'], f"Command type {transaction['pack_type']}")
        elif kind == 'message':
            title = 'Learner receives: ' + transaction['message'].get('type', 'message')
            actor = 'Adapter → network'
        elif kind == 'packet_complete':
            title = f"Packet #{packet} · type {transaction['type']}"
        else:
            title += ' ' + transaction.get('operation', '')
        add(title, actor, category, frame, transaction, policy, f"contextual #{transaction.get('seq', 'init')}")
        for end, index in endpoints:
            seq = index - 1
            if end <= packet and seq not in emitted:
                emitted.add(seq)
                inp = inputs[seq]
                actor = 'Nullkiller · Red' if inp['player'] == 0 else 'Blue / automatic' if inp['player'] == 1 else 'Engine / automatic'
                if inp['type'] == 198: actor = 'Autobattle · ' + ('Red' if inp['player'] == 0 else 'Blue')
                title = NAMES.get(inp['type'], f"Command type {inp['type']}")
                add(title, actor, 'state', index, {'input': inp, 'applied': applied[seq]}, source=f'native input #{seq}')
    assert len(emitted) == len(inputs), 'Missing native endpoints'
    assert decision == len(rollout['decisions']) == 13
    documents = {}
    for name in ['manifest.json', 'builtin-rollout.json', 'behavior.json', 'policy-reconstruction.json',
                 'builtin-finalization.json', 'builtin-verification/builtin-replay-verification.json',
                 'builtin-verification/native-report.json', 'agents/blue/history-v7.jsonl',
                 'agents/blue/bridge.jsonl', 'engine/replay.jsonl', 'engine/contextual-replay.jsonl']:
        path = episode / name
        if path.exists(): documents[name] = path.read_text()
    documents['native-trace.jsonl'] = trace.read_text()
    documents['fresh-native-report.json'] = report.read_text()
    documents['trace-generation.json'] = (trace.parent / 'generation.json').read_text()
    documents['provenance.json'] = pretty(data['provenance'])
    return dict(schema='heroes-replay-inspector-v1', title='Edric vs Nullkiller',
                subtitle='Castle vs Castle · episode 2 · frozen random policy · no PPO updates',
                header=data['header'], frames=data['frames'], object_details=object_details,
                armies=[data['hero_army'][f['seq']] for f in data['frames']], events=events,
                documents=documents, rollout_sha256=digest(rollout_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle = export(args.episode.resolve(), args.trace.resolve(), args.report.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, ensure_ascii=False))
    print(json.dumps({'output': str(args.output), 'events': len(bundle['events']), 'states': len(bundle['frames'])}))
