"""Freeze canonical Linux runtime copies and the prospective diagnostic manifest."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.prepare_transfer_eval import BASE,ROOT,binding,write

COLD=Path('/workspace/releases/teacher-overnight-30629a84bc30')
ACCEL=Path('/workspace/releases/teacher-transfer-accelerated-941f60b')


def inventory(root):
    return {str(p.relative_to(root)):binding(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix!='.a'}


def python_value(root,code):
    return json.loads(subprocess.check_output([sys.executable,'-c',code],cwd=root,timeout=60))


def main():
    value=json.loads((BASE/'inputs.json').read_text());destination=BASE/'runtime/accelerated'
    assert not destination.exists();destination.mkdir(parents=True)
    source=ACCEL/'vcmi/build/bin';binary=destination/'bin';binary.mkdir()
    for path in source.iterdir():
        if path.suffix=='.a':continue
        target=binary/path.name
        if path.is_dir():shutil.copytree(path,target,symlinks=False)
        else:shutil.copyfile(path,target);shutil.copymode(path,target)
    embedded=destination/'embedded';embedded.mkdir()
    for path in (ACCEL/'vcmi/build/embedded').glob('*.so'):shutil.copy2(path,embedded/path.name)
    assert len(list(embedded.glob('*.so')))==1
    bins={'cold':Path('/workspace/Heroes-artifacts/runtime/overnight-20261026/bin'),'accelerated':binary}
    roots={'cold':COLD,'accelerated':ACCEL};runtimes={};sources={}
    for key,root in roots.items():
        subprocess.run([sys.executable,str(root/'tools/prepare_vcmi.py'),'--check'],cwd=root,check=True,timeout=90)
        directory=bins[key];env={**os.environ,'LD_LIBRARY_PATH':str(directory)+':'+os.environ.get('LD_LIBRARY_PATH','')}
        for path in (directory/'vcmiclient',directory/'vcmireplay',*[p for p in embedded.glob('*.so') if key=='accelerated']):
            checked=subprocess.check_output(['ldd',str(path)],env=env,stderr=subprocess.STDOUT,timeout=30).decode()
            assert 'not found' not in checked,checked
            write(BASE/f'ldd-{key}-{path.name}.json',dict(binary=str(path),linked_libraries=checked))
        sources[key]=python_value(root,"import json;from experiments.learner.inference import source_identity;print(json.dumps(source_identity('adventure-capabilities-v1')))")
        assert not sources[key]['tracked_worktree_dirty']
        files=python_value(root,"import json;from pathlib import Path;from experiments.headless.contextual_evidence import runtime_identity;print(json.dumps(runtime_identity(Path("+repr(str(directory))+"))))")
        runtimes[key]=dict(source=str(root),binary=str(directory/'vcmiclient'),replay_binary=str(directory/'vcmireplay'),files=files,
            all_runtime_files=inventory(directory),module_files=inventory(embedded) if key=='accelerated' else {},
            dependency_lock=binding(root/'dependencies/vcmi.json'))
    data=Path('/workspace/Heroes-artifacts/game-data');prior=json.loads(Path('/workspace/Heroes-artifacts/overnight-20261026/campaign-v1/assets-verification.json').read_text())
    # Freshly rehash all eight assets; metadata/stat checks bind them for every worker.
    actual={};stats={}
    for path in sorted((data/'Data').iterdir()):
        if not path.is_file():continue
        actual[str(path)]=binding(path);s=path.stat();stats[str(path)]={k:getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}
    assert len(actual)==8
    old_by_sha={x['sha256'] for x in prior['files'].values()} if isinstance(prior.get('files'),dict) else set()
    if not old_by_sha:
        def digests(v):
            if isinstance(v,dict):
                for k,x in v.items():
                    if k=='sha256':yield x
                    else:yield from digests(x)
            elif isinstance(v,list):
                for x in v:yield from digests(x)
        old_by_sha=set(digests(prior))
    assert {x['sha256'] for x in actual.values()}<=old_by_sha,'Private game Data hashes differ from verified transfer'
    for checkpoint in value['checkpoints'].values():checkpoint.pop('original_source',None)
    value.update(schema='heroes-frozen-transfer-prospective-v1',data=str(data),data_bindings=actual,data_stat=stats,runtimes=runtimes,sources=sources,
        harness_files={p:binding(ROOT/p)['sha256'] for p in ('tools/prepare_transfer_eval.py','tools/transfer_eval_worker.py','tools/transfer_eval_watchdog.py','tools/run_transfer_eval.py','tools/freeze_transfer_manifest.py')},
        archived_strict={key:dict(passed=json.loads((BASE/f'archived-{key if key=="cold" else "accelerated"}-strict/report.json').read_text())['status']=='passed',
            report=binding(BASE/f'archived-{key if key=="cold" else "accelerated"}-strict/report.json')) for key in ('cold','accelerated')},
        hypotheses=['412 transfer survives changed nearby-town coordinates/distances beyond the repeatedly evaluated original8x8 starts.',
            '412 performance on original hero-management-v2 may differ from146 and254; matched outcomes, recruitment and movement provide descriptive evidence.',
            'Selected acceleration preserves exact same-Linux history, decisions, estimates and terminal outcomes in four both-seat matched gate games.'],
        clock=dict(start='first new game launch intent, conservatively including imports',compute_seconds=5100,hard_seconds=5400,reset_permitted=False),
        limits=dict(gate_games=8,transfer_games=192,max_days=365,ordinary_choices=4960,dialog_reserve=160,total_choices=5120,max_embedded_games_per_process=4,
            actual_gate_games_per_process=2,actual_transfer_games_per_process=1,torch_threads=1,game_wall_seconds=300),
        configuration=dict(interface='adventure-capabilities-v1',opponent='Nullkiller Easy',battle_ai='BattleAI',reward=dict(natural_win=1,natural_loss=-1,ordinary=0),
            model_sampling_unchanged=True,training_samples=0,ppo_updates=0,optimizer_steps=0),
        strict_gate=dict(required=['archived_same_Linux_numeric_reconstruction','fresh_capture_replay_policy_reconstruction_both_seats',
            'exact_paired_history_decisions_estimates_outcomes','checkpoint_source_runtime_map_identity','owned_cleanup'],
            numeric_tolerance=0,failure_action='preserve failure and use independently accepted original cold runtime; never waive gate'),
        dependency_locks={str(p.relative_to(ROOT)):binding(p) for p in (ROOT/'dependencies').glob('*') if p.is_file()})
    write(ROOT/'configs/teacher-transfer-eval-20261026.json',value)
    print(json.dumps(dict(cold_source=sources['cold']['commit'],accelerated_source=sources['accelerated']['commit'],data_files=len(actual),
        transfer_games=len(value['transfer_schedule']),gate_games=len(value['gate_schedule']),new_games=0)))


if __name__=='__main__':main()
