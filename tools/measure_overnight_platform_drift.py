"""Offline Mac/Linux behavior comparison; never creates training receipts.

Historical observations use their original behavior checkpoint, which may be
the final checkpoint's initial_checkpoint dependency. Full unrecorded logits
cannot be compared. Selected joint/conditional log probabilities, values,
supports and sampled actions are compared on identical recorded frames.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from experiments.learner import checkpoints as cp, contextual as c
from experiments.learner.builtin_training_resume import binding,require,_metadata
from experiments.learner.builtin_rollout import load_builtin_episode
from experiments.learner.contextual_model import ContextualEncoder,weights_sha256

ABSOLUTE=1e-5
RELATIVE=1e-4


def measure(episode,model):
    require(weights_sha256(model)==episode.weights_sha256,'Wrong historical behavior weights')
    encoder=ContextualEncoder(episode.history[0].episode,episode.player,episode.history[0].schema)
    state=model.initial_state();generator=torch.Generator().manual_seed(episode.policy_seed)
    values={v.decision_seq:v for v in episode.estimates}
    choices={v.decision_seq:v for v in episode.decisions}
    metrics={k:dict(count=0,maximum_absolute=0.,maximum_relative=0.,outside_tolerance=0) for k in ('value','joint_log_probability','conditional_log_probability','selected_probability')}
    action_mismatches=0
    def compare(kind,left,right):
        require(math.isfinite(left) and math.isfinite(right),'Nonfinite platform comparison')
        difference=abs(left-right);relative=difference/max(abs(right),1e-30)
        row=metrics[kind];row['count']+=1
        row['maximum_absolute']=max(row['maximum_absolute'],difference)
        row['maximum_relative']=max(row['maximum_relative'],relative)
        row['outside_tolerance']+=not math.isclose(left,right,abs_tol=ABSOLUTE,rel_tol=RELATIVE)
    with torch.no_grad():
        for frame in episode.history:
            step=model.forward_step(encoder.encode(frame),state);state=step.state
            if frame.seq in values:
                require(step.decision is not None,'Missing value decision')
                compare('value',step.decision.value.item(),values[frame.seq].value)
            if frame.seq in choices:
                old=choices[frame.seq]
                require(step.decision is not None and step.decision.distribution.support==old.support,'Platform action support drift')
                distribution=step.decision.distribution
                action_mismatches+=distribution.sample(generator=generator)!=old.action
                actual=distribution.evaluate(old.action)
                compare('joint_log_probability',actual.log_prob.item(),old.log_prob)
                compare('selected_probability',math.exp(actual.log_prob.item()),math.exp(old.log_prob))
                require(len(actual.conditionals)==len(old.conditionals),'Conditional support count drift')
                for left,right in zip(actual.conditionals,old.conditionals):
                    require((left.name,left.selected_index,left.support_size)==(right.name,right.selected_index,right.support_size),'Conditional action order/support drift')
                    compare('conditional_log_probability',left.log_prob.item(),right.log_prob)
    require(weights_sha256(model)==episode.weights_sha256,'Comparison changed weights')
    return dict(metrics=metrics,action_mismatches=action_mismatches,
        passed=action_mismatches==0 and all(r['outside_tolerance']==0 for r in metrics.values()),
        absolute_tolerance=ABSOLUTE,relative_tolerance=RELATIVE,
        limitations='Full unrecorded logits are unavailable; comparison covers recorded values and selected probabilities.',
        training_samples=0,optimizer_steps=0)


def main():
    p=argparse.ArgumentParser()
    for key in ('episode','checkpoint','plan','report','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    plan=json.loads(a.plan.read_text());report=json.loads(a.report.read_text())
    require(binding(a.plan)['sha256']==report['plan_sha256'],'Historical plan/report mismatch')
    expected=report['after_checkpoint']['policy']
    require(all(binding(a.checkpoint)[k]==expected[k] for k in ('sha256','bytes')),'Historical checkpoint bytes differ')
    loaded=cp.load_policy_checkpoint(a.checkpoint,expected_source_metadata=_metadata(plan,report['plan_sha256']),
        interface=plan['source']['interface'],allowed_provenance=(cp.BUILTIN_GAMEPLAY_PROVENANCE,),
        historical_torch_version=plan['source']['torch'])
    episode=load_builtin_episode(a.episode/'builtin-rollout.json')
    result=dict(episode=binding(a.episode/'builtin-rollout.json'),checkpoint=binding(a.checkpoint),
        historical_plan=binding(a.plan),**measure(episode,loaded.model))
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=result['passed'],output=str(a.output))))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
