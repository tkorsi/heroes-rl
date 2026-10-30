#!/usr/bin/env python3
"""Export a passed one-game evaluation as a native-state Swift viewer timeline.

This presentation contains every recorded native input and state. It does not
export policy-only UI choices, policy probabilities or hero inventory panels.
Raw captures remain private, and no policy inference or training is performed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.headless import replay
from experiments.headless.process_deadline import guarded_popen, close_owned_process
from tools.run_selfplay_training import binding, bound, read, require, write

NAMES = {180: 'End turn', 182: 'Move hero', 183: 'Arrange army', 185: 'Build structure',
         187: 'Recruit creatures', 188: 'Upgrade creatures', 198: 'Automatic battle', 260: 'Interface ready'}


def replay_facts(value):
    """Wall-clock measurements vary; retain every replay assertion and counter."""
    if isinstance(value, dict):
        return {k: replay_facts(v) for k, v in value.items() if k != 'timing_seconds'}
    if isinstance(value, list):
        return [replay_facts(v) for v in value]
    return value


def export(run, output):
    run, output = run.resolve(), output.resolve()
    require(not output.exists() and not output.is_relative_to(run), 'Use a fresh output outside the source run')
    report = read(run / 'report.json'); shutdown = read(run / 'shutdown.json')
    require(report.get('checkpoint') == 1753 and report.get('episode_status') == 'truncated'
            and report.get('termination_reason') == 'collection_window' and report.get('purpose') == 'diagnostic',
            'This exporter labels the checkpoint 1753 choice-cutoff evaluation')
    require(report['status'] == 'passed' and report['native_replay_verified'] and report['both_policies_reconstructed']
            and shutdown['returncode'] == 0 and shutdown['owned_cleanup_verified'] and not shutdown['errors'],
            'The evaluation must have passed and shut down before export')
    plan = read(bound(report['plan'])); proof = read(bound(report['proof']))
    episode = Path(report['episode_directory']).resolve()
    require(episode.is_relative_to(run), 'Episode escaped the run')
    manifest_path = episode / 'manifest.json'; manifest = read(manifest_path)
    require(binding(manifest_path)['sha256'] == proof['replay']['inputs']['manifest']['sha256'], 'Manifest changed')
    inputs = {str(p): binding(p) for p in [run / 'report.json', run / 'shutdown.json', manifest_path,
              bound(report['plan']), bound(report['proof']), bound(plan['replay_binary'])]}
    for name, expected in manifest['replay']['files'].items():
        p = (episode / name).resolve()
        require(p.is_relative_to(episode), 'Capture path escaped episode')
        actual = binding(p)
        require(all(actual[k] == expected[k] for k in ('sha256', 'bytes')), 'Capture changed')
        inputs[str(p)] = actual
    checked = episode / 'evaluation-verification'
    log, tape = checked / 'replay.jsonl', checked / 'contextual-replay.jsonl'
    for p in (log, tape): inputs[str(p)] = binding(p)
    output.mkdir(parents=True, mode=0o700)
    env, _ = replay.replay_profile(output, episode, Path(plan['data']), manifest['launch']['requested_settings'])
    binary = Path(plan['replay_binary']['path'])
    native, trace = output / 'native-report.json', output / 'trace.jsonl'
    command = [str(binary), '--log', str(log), '--contextual', str(tape), '--report', str(native), '--trace', str(trace), '--prefix']
    inherited = {k:v for k,v in os.environ.items() if not k.startswith(('EXPLORER_RL_', 'VCMI_RECORD_', 'VCMI_EXPERIMENT_', 'VCMI_USER_DATA_DIR', 'XDG_'))}
    started = time.monotonic(); process = None
    with (output / 'native.log').open('xb') as console:
        try:
            process = guarded_popen(command, deadline_monotonic=started+120, deadline_wall=time.time()+120,
                evidence_path=output/'deadline.jsonl', cwd=binary.parent, env={**inherited, **env},
                stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT)
            process.wait(timeout=121)
            require(process.returncode == 0, 'Native trace reconstruction failed')
        finally:
            if process is not None: close_owned_process(process, timeout=5)
    verification = read(native)
    require(verification['verified'] is True
            and replay_facts(verification['summary']) == replay_facts(proof['replay']['native']['summary']),
            'Trace replay differs from the accepted verification')
    with trace.open() as stream:
        header = json.loads(next(stream)); frames = [json.loads(line) for line in stream]
    require(header['levels'] == 1, 'Only single-level maps are supported')
    applied, commands = {}, {}
    with log.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] == 'applied': applied[row['seq']] = row
            if row['kind'] == 'input': commands[row['seq']] = row
    require([f['seq'] for f in frames] == list(range(-1, len(commands))) and len(applied) == len(commands),
            'Missing native states or commands')
    events, armies = [], []
    for i, frame in enumerate(frames):
        require(len(frame['players']['1']['visible']) == header['width']*header['height'], 'Invalid fog state')
        hero = next((o for o in frame['objects'] if o[4] == 34 and o[6] == 1 and o[5] == 2), None)
        army = dict(name='Blue hero · inventory not exported', status='unavailable', position=hero[1:4] if hero else None,
                    slots=[], source_packet=0, freshness='unavailable')
        armies.append(army)
        if i == 0:
            title, actor, packet, raw = 'Initial native state', 'Engine', 0, {'scope':'native-state timeline'}
        else:
            seq=frame['seq']; cmd=commands[seq]
            title=NAMES.get(cmd['type'], f"Native command {cmd['type']}")
            actor={0:'Self-play · Red',1:'Self-play · Blue'}.get(cmd['player'],'Engine / automatic')
            packet=applied[seq]['client_out']; raw={'input':cmd,'applied':applied[seq]}
        events.append(dict(id=i,title=title,actor=actor,category='state',frame=i,packet=packet,
            source=f"native input #{frame['seq']}",raw=json.dumps(raw,indent=2),
            observation='',policy='',summary='Native state after command; detailed policy/observations not exported',army=army))
    limitations=['Complete native input/state timeline; policy-only UI choices are not included.',
                 'Hero armies, inventories and policy probabilities are not exported.',
                 'Full map is spectator data; right map shows recorded Blue visibility.',
                 'Game stopped at its collection choice limit with no winner.']
    bundle=dict(schema='heroes-replay-inspector-v1',title='Self-play · checkpoint 1753',
        subtitle='Red vs Blue · doubled choices · no winner (choice limit) · native state timeline',
        header=header,frames=frames,armies=armies,events=events,object_details={},
        documents={'manifest.json':json.dumps(manifest,indent=2),'report.json':json.dumps(report,indent=2),
                   'export-scope.json':json.dumps(limitations,indent=2)})
    for value in inputs.values(): bound(value)
    target=output/'match.json'
    target.write_text(json.dumps(bundle,separators=(',',':')))
    result=dict(status='passed',schema='heroes-selfplay-inspector-export-v1',run=str(run),
        bundle=binding(target),trace=binding(trace),native_report=binding(native),inputs=inputs,
        states=len(frames),events=len(events),all_native_inputs_included=len(commands),
        elapsed_seconds=time.monotonic()-started,limitations=limitations,gameplay_started=False,optimizer_updates=0)
    write(output/'export-validation.json',result,new=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export(args.run,args.output),indent=2))
