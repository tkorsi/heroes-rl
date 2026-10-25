#!/usr/bin/env python3
"""Read-only inventory of the exact update92 parent for a future v8 migration.

Does not instantiate a v8 policy, rewrite metadata, create a session, collect
experience or run an optimizer. Run against this preserved v7 checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
PINS={'after-policy.pt':'1d6f9a68b61fd60d6c6e49155b2c8b479c0324b2d3c5bd89ddda3e97fbaab344',
      'after-resume.pt':'1c21b70deaf35de410501cf1b6e1b37768dfc054cbcd23bfe86df6c05f483ce4',
      'plan.json':'aa65d37019dc62252fee9d6269719c31d48b9df011fc9886abf59c314835517e',
      'report.json':'a76c8e1ec41e7ce4196a1f54c4bb343ee1a1a92a25e2913bf6b94402e05c353a'}


def fingerprint(value):
    import torch
    def encode(v):
        if isinstance(v,torch.Tensor):
            raw=bytes(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist())
            return {'type':'tensor','dtype':str(v.dtype),'shape':list(v.shape),'sha256':hashlib.sha256(raw).hexdigest()}
        if type(v) is dict:return {'type':'dict','items':[[encode(k),encode(v[k])] for k in sorted(v,key=repr)]}
        if type(v) in (list,tuple):return {'type':type(v).__name__,'items':[encode(x) for x in v]}
        if v is None or type(v) in (int,float,str,bool):return {'type':type(v).__name__,'value':v}
        raise TypeError(type(v).__name__)
    return hashlib.sha256(json.dumps(encode(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def audit(root):
    import torch
    from experiments.learner.builtin_training_resume import binding,require
    from experiments.learner.builtin_preservation import verify_pair
    torch.set_num_threads(1)
    name='local365-continuation-092';run=root/'runs'/name
    plan_path=run/'plan.json';report_path=run/'execution/report.json'
    paths={'plan.json':plan_path,'report.json':report_path,
           'after-policy.pt':run/'execution/after-policy.pt','after-resume.pt':run/'execution/after-resume.pt'}
    for name,path in paths.items():require(binding(path)['sha256']==PINS[name],'Wrong update92 parent: '+name)
    plan=json.loads(plan_path.read_text());report=json.loads(report_path.read_text())
    locations={'canonical':run/'execution','backup':root/'checkpoints'/run.name,
               'private_snapshot':root/'private-local365-state/checkpoints/local365'/run.name}
    proofs={}
    for label,folder in locations.items():
        for name in ('after-policy.pt','after-resume.pt'):
            require(binding(folder/name)['sha256']==PINS[name],'Changed '+label+' checkpoint')
        proofs[label]=verify_pair(folder/'after-resume.pt',folder/'after-policy.pt',plan,report)
    raw=torch.load(paths['after-resume.pt'],weights_only=True,map_location='cpu')
    resume=raw['resume'];state=resume['optimizer_state'];parameters=[]
    require(resume['progress']=={'updates':92,'rollouts':368,'decisions':21062},'Wrong progress')
    require(resume['sampler_state']['optimizer_steps']==783,'Wrong total Adam progress')
    for names,group in zip(resume['parameter_names'],state['param_groups'],strict=True):
        for name,identifier in zip(names,group['params'],strict=True):
            tensor=raw['model_state'][name];slots=state['state'].get(identifier,{})
            parameters.append({'name':name,'shape':list(tensor.shape),'dtype':str(tensor.dtype),
                'tensor_fingerprint':fingerprint(tensor),'optimizer_id':identifier,
                'adam_step':None if 'step' not in slots else slots['step'].item(),
                'adam_slots':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'fingerprint':fingerprint(v)} for k,v in slots.items()}})
    require({x['name'] for x in parameters}==set(raw['model_state']),'Model/optimizer names differ')
    consumed={}
    for name in ['campaign-session.json','campaign-session.lock','controller-result.json']:
        consumed[name]=binding(root/'local365'/name)
    return {'schema':'heroes-update92-v8-resume-audit-v1','status':'parent_verified_target_pending',
        'gameplay_started':False,'optimizer_run':False,'migrated_checkpoint_written':False,
        'parent':{k:binding(v) for k,v in paths.items()},'restoration':proofs,
        'contracts':raw['contracts'],'model_config':raw['model_config'],'torch_version':raw['torch_version'],
        'progress':resume['progress'],'sampler_state':resume['sampler_state'],
        'optimizer_class':resume['optimizer_class'],'parameter_names':resume['parameter_names'],
        'optimizer_groups_without_ids':[{k:v for k,v in g.items() if k!='params'} for g in state['param_groups']],
        'parameters':parameters,'fingerprints':{k:fingerprint(v) for k,v in {
            'model':raw['model_state'],'adam':state,'rng':raw['rng'],'pool':raw['opponent_pool']}.items()},
        'archived_pool':[{'checkpoint':e['checkpoint'],'weights_sha256':e['weights_sha256'],'model_config':e['model_config']} for e in raw['opponent_pool']],
        'consumed_old_session':consumed,'target_v8_contract':None,'target_v8_source_commit':None,
        'migration_ready':False,'launch_ready':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    value=audit(a.artifact_root.resolve())
    with a.output.open('x') as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':value['status'],'parameters':len(value['parameters']),'progress':value['progress'],'adam_steps':value['sampler_state']['optimizer_steps'],'launch_ready':False}))
