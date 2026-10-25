#!/usr/bin/env python3
"""Compare the actual update-31 policy on original and consolidated CPU sources.

Synthetic dialog histories exercise exact inference, action probabilities,
serialization and gradients. This is not native gameplay or PPO admission.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def worker(source,policy,fixtures,output):
    sys.path.insert(0,str(source))
    import torch
    from experiments.learner import contextual as c, contextual_model as m, contextual_rollout as rollout
    torch.set_num_threads(1)
    saved=torch.load(policy,weights_only=True,map_location='cpu')
    model=m.ContextualActorCritic(m.ModelConfig(**saved['model_config']))
    model.load_state_dict(saved['model_state']);model.eval()
    histories=[[c.decode_frame(f) for f in h] for h in json.loads(fixtures.read_text())]
    results=[];serialization=[]
    for frames in histories:
        serialization.append([rollout._frame_bytes(f) for f in frames])
        def run(grad):
            encoder=m.ContextualEncoder(frames[0].episode,frames[0].player,frames[0].schema)
            state=model.initial_state();generator=torch.Generator().manual_seed(731);trace=[];loss=None
            with torch.set_grad_enabled(grad):
                for frame in frames:
                    out=model.forward_step(encoder.encode(frame),state);state=out.state
                    trace.append(state.detach().clone())
                    if out.decision is not None:
                        dist=out.decision.distribution;action=dist.sample(generator=generator);ev=dist.evaluate(action)
                        trace.append((c.encode_action(action),out.decision.value.detach().clone(),ev.log_prob.detach().clone(),
                            [(x.name,x.selected_index,x.support_size,x.log_prob.detach().clone()) for x in ev.conditionals]))
                        term=out.decision.value+ev.log_prob;loss=term if loss is None else loss+term
                if grad and loss is not None:loss.backward()
            return trace
        results.append(run(False));results.append(run(False));model.zero_grad(set_to_none=True)
        run(True);results.append({name:p.grad.detach().clone() if p.grad is not None else None for name,p in model.named_parameters()})
    torch.save({'trace':results,'serialized':serialization,'weights':m.weights_sha256(model)},output)


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--source',type=Path)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--fixtures',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--artifacts',type=Path)
    a=p.parse_args()
    if a.worker:return worker(a.source,a.policy,a.fixtures,a.output)
    sys.path.insert(0,str(ROOT))
    import torch
    from experiments.learner.builtin_local365 import PARENT_COMMIT,performance_changes,PARENT_POLICY
    from experiments.learner.builtin_training_resume import _exact,binding
    from experiments.learner import contextual as c
    from tests.test_contextual_dialogs import FAMILIES,history_at,panel
    from tests.test_perf_canonical_cache import finalized_pairs
    a.artifacts.mkdir(parents=True,exist_ok=False)
    old=a.artifacts/'old-source';old.mkdir()
    archive=subprocess.check_output(['git','archive',PARENT_COMMIT,'experiments'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:tf.extractall(old,filter='data')
    fixture=a.artifacts/'fixtures.json'
    histories=[history_at(panel(f)).frames for f in FAMILIES]+[h.frames for pair in finalized_pairs() for h in pair]
    fixture.write_text(json.dumps([[c.encode_frame(f) for f in h] for h in histories]))
    for name,source in [('old',old),('new',ROOT)]:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--source',str(source),'--policy',str(a.policy),
            '--fixtures',str(fixture),'--output',str(a.artifacts/(name+'.pt'))],cwd=a.artifacts,check=True)
    before=torch.load(a.artifacts/'old.pt',weights_only=True);after=torch.load(a.artifacts/'new.pt',weights_only=True)
    _exact(before,after)
    assert binding(a.policy)['sha256']==PARENT_POLICY
    evidence={'schema':'heroes-local365-parity-v1','status':'passed','source_changes':performance_changes(),
        'checker_sha256':binding(Path(__file__).resolve())['sha256'],
        'policy_sha256':PARENT_POLICY,'exact_trace_and_gradients':True,'histories':len(histories),
        'checks':['canonical frame bytes','trained weights digest','cold and warm recurrent state/value/action/conditional log-probability','parameter gradients'],
        'scope':'Synthetic dialog families and both-seat terminal/cutoff/failure histories on the same Mac CPU runtime; fresh native both-seat gate still required.',
        'fixtures':binding(fixture),'old_result':binding(a.artifacts/'old.pt'),'new_result':binding(a.artifacts/'new.pt')}
    a.output.write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'passed','histories':len(histories)}))

if __name__=='__main__':main()
