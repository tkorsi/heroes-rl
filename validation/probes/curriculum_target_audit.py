"""Offline, player-visible target support/probability audit; no sampled actions."""
import json
from pathlib import Path
import torch
from validation.probes.validate_curriculum_policy import load_fixed
from experiments.learner.builtin_rollout import BuiltinCollector
from experiments.learner.contextual import Action
from experiments.learner.evaluate_builtin_opponent import native_opponent
from experiments.learner.builtin_training_resume import binding


def audit(folder):
    folder=Path(folder);plan=json.loads((folder/'plan.json').read_text());seat=plan['map']['seat']
    path=folder/'episode-000001/agents'/('red','blue')[seat]/'history-v8.jsonl'
    frames=[]
    for line in path.read_text().splitlines():
        frame=json.loads(line);frames.append(frame)
        if frame['kind']=='decision':break
    loaded,_=load_fixed(plan['checkpoint']['path'])
    collector=BuiltinCollector(loaded.model,loaded.checkpoint,learner_player=seat,
        opponent_identity=native_opponent(plan['runtime'],seat),policy_seed=plan['schedule'][0]['policy_seed'],
        interface='adventure-capabilities-v1')
    collector.start('1');collector.session.consume(frames)
    dist=collector.session._policies[frame['seq']];width,height,_=dist.offers.size
    hero=frame['observation']['heroes'][0]
    town=next(o for o in frame['observation']['objects'] if o['owner']=='opponent' and o['kind']==98)
    x,y,_=town['position'];action=Action('move_to',subject=hero['handle'],x=x,y=y,schema='heroes-context-actions-v8')
    with torch.no_grad():
        evaluated=dist.evaluate(action)
        move_mass=sum(float(dist.log_prob(Action('move_to',subject=hero['handle'],x=px,y=py,
            schema='heroes-context-actions-v8')).exp()) for py in range(height) for px in range(width))
        target=float(evaluated.log_prob.exp())
    return {'history':binding(path),'checkpoint':plan['checkpoint'],'seat':seat,'target_visible':True,
        'target':town,'hero':hero,'offers_permits':dist.offers.permits(action),
        'coordinate_index':y*width+x,'enumerated_coordinates':width*height,
        'inverse_mapping':[ (y*width+x)%width,(y*width+x)//width ],
        'joint_target_probability':target,'move_kind_probability':move_mass,
        'target_probability_given_move':target/move_mass,
        'conditionals':[{'name':v.name,'support_size':v.support_size,'selected_index':v.selected_index,
                        'log_prob':float(v.log_prob)} for v in evaluated.conditionals],
        'native_reachability':'requires separate mechanical control; not inferred from offers',
        'policy_weights_unchanged':True,'actions_sampled':0,'training_samples':0}
