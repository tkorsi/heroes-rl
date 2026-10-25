#!/usr/bin/env python3
"""Restore-test and publish one complete Linux batch to the private state repo.

No historical paths or checkpoint metadata are rewritten. Raw capture archives
retain symlinks as metadata and never follow the game's Data directory links.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.learner.builtin_preservation import DURABLE_ROOT, durable_path, verify_pair
from experiments.learner.builtin_training_resume import binding, require


def git(repo, *args):
    return subprocess.check_output(['git', *args], cwd=repo, text=True).strip()


def publish(plan_path, repo):
    import torch
    torch.set_num_threads(1)
    plan_path = durable_path(plan_path)
    plan = json.loads(plan_path.read_text())
    report_path = Path(plan['execution_output'])/'report.json'
    report = json.loads(report_path.read_text())
    require(report['plan_sha256'] == binding(plan_path)['sha256'], 'Snapshot plan/report binding differs')
    run_id = plan_path.parent.name
    backup = DURABLE_ROOT/'checkpoints'/run_id
    receipt = json.loads((backup/'preservation-receipt.json').read_text())
    require(receipt['restore']['exact_model_adam_rng_progress_pool'] is True, 'Missing independent restore receipt')
    for name in ('after-resume.pt', 'after-policy.pt'):
        require(binding(backup/name)['sha256'] == receipt['sources'][name]['sha256'], 'Independent backup changed')
    verify_pair(backup/'after-resume.pt',backup/'after-policy.pt',plan,report)
    repo = Path(repo).resolve()
    require(repo != ROOT and not repo.is_relative_to(ROOT), 'Private snapshots must be outside the public source')
    require(git(repo,'branch','--show-current') == 'codex/cloud-teacher-results', 'Use the private results branch')
    require(git(repo,'remote','get-url','origin') == 'https://github.com/tkorsi/heroes-rl-training-state.git',
            'Refuse snapshot publication to another repository')
    destination = repo/'checkpoints/cloud'/run_id
    destination.parent.mkdir(parents=True,exist_ok=True)
    sources = {'plan.json':plan_path,'report.json':report_path,
               'after-resume.pt':backup/'after-resume.pt','after-policy.pt':backup/'after-policy.pt',
               'preservation-receipt.json':backup/'preservation-receipt.json'}
    if not destination.exists():
        staged = Path(tempfile.mkdtemp(prefix='.cloud-preserve-',dir=destination.parent))
        try:
            for name,path in sources.items():
                shutil.copyfile(path,staged/name)
                require(binding(staged/name)['sha256'] == binding(path)['sha256'],'Snapshot bytes differ')
            verify_pair(staged/'after-resume.pt',staged/'after-policy.pt',plan,report)
            os.rename(staged,destination)
        finally:
            if staged.exists(): shutil.rmtree(staged)
    for name,path in sources.items():
        require(binding(destination/name)['sha256'] == binding(path)['sha256'],'Existing snapshot differs')
    proof = verify_pair(destination/'after-resume.pt',destination/'after-policy.pt',plan,report)
    archive = repo/'results/cloud'/(run_id+'.tar.gz')
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        staged = archive.with_suffix('.tar.gz.staging')
        with tarfile.open(staged,'w:gz',dereference=False) as tf:
            tf.add(plan_path.parent,arcname=run_id)
        os.rename(staged,archive)
    manifest = {'schema':'heroes-private-cloud-snapshot-v1','run':run_id,'restore':proof,
        'plan':binding(plan_path),'report':binding(report_path),'raw_archive':binding(archive),
        'checkpoint_files':{name:binding(destination/name) for name in sources},
        'history_rewritten':False,'game_data_symlinks_followed':False}
    manifest_path = archive.with_suffix('.json')
    if manifest_path.exists():
        require(json.loads(manifest_path.read_text()) == manifest,'Existing snapshot manifest differs')
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    paths = [str(p.relative_to(repo)) for p in (*destination.iterdir(),archive,manifest_path)]
    git(repo,'add','--',*paths)
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo,text=True).strip():
        git(repo,'commit','-m',f'Preserve restore-tested cloud Teacher update {proof["progress"]["updates"]}')
    git(repo,'push','origin','HEAD:refs/heads/codex/cloud-teacher-results')
    return {'private_commit':git(repo,'rev-parse','HEAD'),'restore':proof,
            'manifest':str(manifest_path.relative_to(repo)),'raw_archive_sha256':manifest['raw_archive']['sha256']}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--repo',type=Path,default=Path('/workspace/heroes-rl-training-state'))
    args=parser.parse_args()
    print(json.dumps(publish(args.plan,args.repo),indent=2))
