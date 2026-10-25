"""Read-only post-run reconciliation; never starts native games or optimization."""
from pathlib import Path
import hashlib, json, math, random, sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.learner import contextual as c
from experiments.learner.checkpoints import load_policy_checkpoint, load_checkpoint
from experiments.learner.contextual_model import ContextualActorCritic, encode_history, weights_sha256
from experiments.learner.contextual_rollout import CollectedEpisode
from experiments.learner.gameplay import (CollectionReceipt, validate_collection_receipt,
    read_plan, assert_same_state as executor_assert_same_state)

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/first-training-prepared-2026-10-25'
OUT=BASE/'approved-execution'
PLAN_HASH='ff22a542eb881b913e34772dd124b34802a36fe55f625e248984d331c23512e6'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def artifact(path):return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size}
def check(value,message):
    if not value:raise AssertionError(message)
def read(path):return json.loads(path.read_text())

def assert_same_state(left,right):
    from collections.abc import Mapping
    if isinstance(left,torch.Tensor):
        check(isinstance(right,torch.Tensor) and torch.equal(left,right),'Tensor state differs')
    elif isinstance(left,Mapping):
        check(isinstance(right,Mapping) and left.keys()==right.keys(),'Mapping state differs')
        for key in left:assert_same_state(left[key],right[key])
    elif type(left) in (tuple,list):
        check(type(left) is type(right) and len(left)==len(right),'Sequence state differs')
        for a,b in zip(left,right):assert_same_state(a,b)
    else:check(type(left) is type(right) and left==right,'Scalar state differs')

def behavior_audit(model,trajectory):
    byseq={d.decision_seq:d for d in trajectory.decisions}
    values={v.decision_seq:v.value for v in trajectory.estimates}
    counts={'decisions':0,'values':0,'conditionals':0,'max_log_prob_difference':0.0,
            'max_value_difference':0.0,'max_conditional_difference':0.0}
    state=model.initial_state()
    with torch.no_grad():
        for seq,frame in enumerate(encode_history(trajectory.history)):
            result=model.forward_step(frame,state);state=result.state
            if seq in values:
                check(result.decision is not None,'Missing own value at stored estimate')
                delta=abs(result.decision.value.item()-values[seq])
                counts['values']+=1;counts['max_value_difference']=max(counts['max_value_difference'],delta)
                check(math.isclose(result.decision.value.item(),values[seq],rel_tol=2e-5,abs_tol=2e-5),'Value mismatch')
            if seq in byseq:
                saved=byseq[seq]
                check(result.decision is not None and result.decision.distribution.support==saved.support,'Support mismatch')
                actual=result.decision.distribution.evaluate(saved.action)
                delta=abs(actual.log_prob.item()-saved.log_prob)
                counts['decisions']+=1;counts['max_log_prob_difference']=max(counts['max_log_prob_difference'],delta)
                check(math.isclose(actual.log_prob.item(),saved.log_prob,rel_tol=2e-5,abs_tol=2e-5),'Joint probability mismatch')
                check(len(actual.conditionals)==len(saved.conditionals),'Conditional length mismatch')
                for a,b in zip(actual.conditionals,saved.conditionals):
                    check((a.name,a.selected_index,a.support_size)==(b.name,b.selected_index,b.support_size),'Conditional support mismatch')
                    delta=abs(a.log_prob.item()-b.log_prob)
                    counts['conditionals']+=1;counts['max_conditional_difference']=max(counts['max_conditional_difference'],delta)
                    check(math.isclose(a.log_prob.item(),b.log_prob,rel_tol=2e-5,abs_tol=2e-5),'Conditional probability mismatch')
    return counts

def group_audit(before,after):
    def group(name):
        if name.startswith('value.'):return 'value_head'
        if name.startswith(('policy_query.','coordinate.','quantity.')):return 'policy_heads'
        return 'shared_encoder_and_recurrent_core'
    def digest(state,keys):
        h=hashlib.sha256()
        for k in sorted(keys):
            value=state[k].detach().cpu().contiguous().clone()
            h.update(k.encode()+b'\0');h.update(bytes(value.reshape(-1).view(torch.uint8).tolist()))
        return h.hexdigest()
    result={}
    for category in ('shared_encoder_and_recurrent_core','policy_heads','value_head'):
        keys=[k for k in before if group(k)==category]
        changed=[k for k in keys if not torch.equal(before[k],after[k])]
        result[category]={'initial_sha256':digest(before,keys),'final_sha256':digest(after,keys),
            'changed_tensors':changed,'tensor_count':len(keys),
            'maximum_absolute_change':max((after[k]-before[k]).abs().max().item() for k in keys)}
    return result

def main():
    torch.set_num_threads(1)
    report=read(OUT/'report.json')
    check(report['status'] in ('completed','stopped'),'Executor still running')
    plan=read_plan(BASE/'plan.json',PLAN_HASH)
    cfg=plan['configuration']
    preparation=read(ROOT/'validation/first-training-preparation-2026-10-25.json')
    schedule=preparation['attempt_schedule']
    check(len(report['attempts'])<=10,'Extra attempts')
    check(report['wall_seconds']<=cfg['total_wall_seconds']+5,'Budget exceeded beyond cleanup tolerance')
    initial=load_policy_checkpoint(BASE/'initial-policy.pt',interface=c.ARMIES_ENVIRONMENT)
    trained=None
    checkpoints={'initial':artifact(BASE/'initial-policy.pt'),'initial_weights_sha256':initial.weights_sha256,
                 'frozen_opponent_identities':initial.opponent_pool.identities}
    if (OUT/'resume.pt').exists():
        raw_resume=torch.load(OUT/'resume.pt',map_location='cpu',weights_only=True)
        if (OUT/'trained-policy.pt').exists():
            raw_policy=torch.load(OUT/'trained-policy.pt',map_location='cpu',weights_only=True)
            trained=load_policy_checkpoint(OUT/'trained-policy.pt',interface=c.ARMIES_ENVIRONMENT,allowed_provenance=('approved_gameplay',))
            check(trained.source_metadata['plan_sha256']==PLAN_HASH,'Trained plan mismatch')
            assert_same_state(raw_resume['model_state'],raw_policy['model_state'])
            assert_same_state(raw_resume['opponent_pool'],raw_policy['opponent_pool'])
            check(trained.opponent_pool.identities==initial.opponent_pool.identities,'Frozen pool identity differs')
        init_raw=torch.load(BASE/'initial-policy.pt',map_location='cpu',weights_only=True)
        assert_same_state(init_raw['opponent_pool'],raw_resume['opponent_pool'])
        model=ContextualActorCritic(initial.model.config)
        optimizer=torch.optim.Adam(model.parameters(),**cfg['adam'])
        generators={'minibatch':torch.Generator(device='cpu')};python_generators={'schedule':random.Random()}
        metadata={'plan_sha256':PLAN_HASH,'source':plan['source'],'configuration':cfg}
        resumed=load_checkpoint(OUT/'resume.pt',model,optimizer,expected_ppo_config=cfg['ppo'],
            expected_source_metadata=metadata,generators=generators,python_generators=python_generators,
            at_rollout_boundary=True,interface=c.ARMIES_ENVIRONMENT,allowed_provenance=('approved_gameplay',))
        assert_same_state(model.state_dict(),raw_resume['model_state'])
        assert_same_state(optimizer.state_dict(),raw_resume['resume']['optimizer_state'])
        assert_same_state(torch.get_rng_state(),raw_resume['rng']['torch_cpu'])
        assert_same_state(random.getstate(),raw_resume['rng']['python'])
        assert_same_state(generators['minibatch'].get_state(),raw_resume['rng']['torch_generators']['minibatch'])
        assert_same_state(python_generators['schedule'].getstate(),raw_resume['rng']['python_generators']['schedule'])
        check(resumed.opponent_pool.identities==initial.opponent_pool.identities,'Resumed frozen pool mismatch')
        check(resumed.progress.updates==report['updates']==1,'Progress update mismatch')
        steps=[int(item['step'].item()) for item in raw_resume['resume']['optimizer_state']['state'].values()]
        check(max(steps)==report['optimizer_steps']<=16,'Optimizer step cap/evidence mismatch')
        check(weights_sha256(model)==raw_resume['weights_sha256'],'Resumed model digest differs')
        original_comparator_error=None
        try:executor_assert_same_state(model.state_dict(),model.state_dict())
        except Exception as error:original_comparator_error=f'{type(error).__name__}: {error}'
        changed=[k for k,v in raw_resume['model_state'].items() if not torch.equal(v,init_raw['model_state'][k])]
        checkpoints.update(resume=artifact(OUT/'resume.pt'),trained=artifact(OUT/'trained-policy.pt') if (OUT/'trained-policy.pt').exists() else None,
            trained_weights_sha256=raw_resume['weights_sha256'],parameter_groups=group_audit(init_raw['model_state'],raw_resume['model_state']),changed_model_tensor_count=len(changed),
            model_tensor_count=len(raw_resume['model_state']),adam_step_values=sorted(set(steps)),
            progress=vars(resumed.progress),sampler_state=resumed.sampler_state,
            exact_model_optimizer_rng_progress_pool_reload=True,optimizer_step_executed_by_audit=False,
            executor_resume_verified=report.get('resume_verified',False),
            original_executor_comparator_error=original_comparator_error,
            independent_comparator_supports_ordered_mappings=True)
    rows=[];totals={'choices':0,'learner_training_samples':0,'natural_attempts':0,'valid_cutoffs':0,
                    'native_replays':0,'all_sample_reward_values':set(),'values':0,'conditionals':0,
                    'max_log_prob_difference':0.0,'max_value_difference':0.0,'max_conditional_difference':0.0}
    for row,wanted in zip(report['attempts'],schedule):
        for key in ('phase','engine_seed','learner_player','policy_seeds'):check(row[key]==wanted[key],'Schedule mismatch '+key)
        folder=OUT/f"episode-{row['episode']:06d}"
        if row['status']=='collection_error':
            rows.append({'episode':row['episode'],'phase':row['phase'],'status':row['status'],'error':row['error']});continue
        manifest=read(folder/'manifest.json');raw=read(folder/'neural-rollout.json');collected=CollectedEpisode.from_dict(raw)
        check(json.loads(collected.manifest_json)==manifest,'Rollout/manifest mismatch')
        check(manifest['seed']==row['engine_seed'] and int(manifest['episode'])==row['episode'],'Manifest schedule mismatch')
        check(manifest['identity']['map_sha256']==plan['map_identity']['sha256'],'Map mismatch')
        check(manifest['environment']==c.ARMIES_ENVIRONMENT,'Wrong profile')
        for a,b in (('status','status'),('termination_reason','termination_reason'),('outcomes_by_player','outcomes'),('rewards_by_player','rewards')):
            check(manifest[a]==row[b],'Report/manifest disagreement '+a)
        choicecount=sum(len(item.decisions) for item in collected.rollouts)
        maxday=max(item.history[d.decision_seq].day for item in collected.rollouts for d in item.decisions)
        check(choicecount<=256 and maxday<=30,'Per-attempt cap exceeded')
        totals['choices']+=choicecount
        if manifest['status']=='truncated':
            totals['valid_cutoffs']+=1
            check(all(v is None for v in manifest['outcomes_by_player'].values()) and all(v is None for v in manifest['rewards_by_player'].values()),'Cutoff scored')
        else:totals['natural_attempts']+=1
        replay_data=[]
        for name in ('collection-replay', *(() if row['phase']!='training' else ('admission-replay',))):
            path=folder/name/'contextual-replay-verification.json'
            if not path.exists():continue
            replay=read(path)
            check(replay['verified'] is True and replay['native']['verified'] is True,'Replay failed')
            check(all(v==0 for v in replay['native']['summary']['unchecked'].values()),'Unchecked replay records')
            check(replay['inputs']['manifest']['sha256']==sha(folder/'manifest.json'),'Replay manifest binding mismatch')
            totals['native_replays']+=1;replay_data.append({'kind':name,**artifact(path)})
        check(len(replay_data)==(2 if row['phase']=='training' and report['updates'] else 1),'Replay count mismatch')
        policy=trained if row['phase']=='after' else initial
        check(policy is not None and row['weights_sha256']==policy.weights_sha256,'Wrong learner weights')
        learner=collected.rollouts[row['learner_player']]
        opponent=collected.rollouts[1-row['learner_player']]
        check(learner.role=='learner' and opponent.role=='frozen_opponent','Role mismatch')
        check(learner.weights_sha256==policy.weights_sha256 and opponent.weights_sha256==initial.opponent_pool.identities[0][1],'Policy weight identity mismatch')
        check(learner.purpose==('training' if row['phase']=='training' else 'diagnostic'),'Wrong purpose')
        if row['phase']=='training':
            receipt_raw=read(folder/'collection-receipt.json');receipt_raw['policy_seeds']=tuple(receipt_raw['policy_seeds'])
            receipt=CollectionReceipt(**receipt_raw)
            validate_collection_receipt(receipt,folder,collected,episode=row['episode'],seed=row['engine_seed'],seat=row['learner_player'],
                policy_seeds=tuple(row['policy_seeds']),learner_checkpoint=initial.checkpoint,learner_digest=initial.weights_sha256,
                opponent_checkpoint=initial.opponent_pool.identities[0][0],opponent_digest=initial.opponent_pool.identities[0][1],
                opponent_pool='first-training-frozen-v1',max_requests=cfg['max_requests'],max_days=cfg['max_days'])
            check(learner.optimization_eligible and not opponent.optimization_eligible,'Wrong optimizer eligibility')
            totals['learner_training_samples']+=len(learner.samples)
        else:check(not learner.optimization_eligible and not opponent.optimization_eligible,'Evaluation admitted')
        recompute=[]
        for trajectory in collected.rollouts:
            model=policy.model if trajectory.role=='learner' else initial.opponent_pool.get(initial.opponent_pool.identities[0][0]).model
            results=behavior_audit(model,trajectory);recompute.append(results)
            totals['values']+=results['values'];totals['conditionals']+=results['conditionals']
            for key in ('max_log_prob_difference','max_value_difference','max_conditional_difference'):totals[key]=max(totals[key],results[key])
            totals['all_sample_reward_values'].update(s.reward for s in trajectory.samples)
            if trajectory.boundary=='cutoff':
                check(all(s.reward==0 for s in trajectory.samples),'Nonzero cutoff reward')
                tail=trajectory.samples[-1];check(tail.successor_seq is not None and tail.bootstrap_value is not None,'Missing cutoff bootstrap')
        rows.append({'episode':row['episode'],'phase':row['phase'],'engine_seed':row['engine_seed'],'learner_player':row['learner_player'],
            'status':row['status'],'termination_reason':row['termination_reason'],'total_choices':choicecount,'maximum_sampled_day':maxday,
            'learner_samples':len(learner.samples),'learner_optimizer_eligible':learner.optimization_eligible,
            'frozen_opponent_optimizer_eligible':opponent.optimization_eligible,'replays':replay_data,'recomputed':recompute})
    if report['status']=='completed':check(len(rows)==10 and report['updates']==1 and report['resume_verified'],'Completed report missing required phases')
    if report['updates']:
        check(totals['learner_training_samples']==report['ppo']['samples']==checkpoints['progress']['decisions'],'PPO sample count mismatch')
        check(checkpoints['progress']['rollouts']==2,'Unexpected optimizer trajectory count')
    totals['all_sample_reward_values']=sorted(totals['all_sample_reward_values'],key=lambda x:-999 if x is None else x)
    result={'schema':'heroes-first-gameplay-independent-review-v1','status':'passed','execution_status':report['status'],
        'report':artifact(OUT/'report.json'),'plan':artifact(BASE/'plan.json'),'audit_source':artifact(Path(__file__).resolve()),'wall_seconds':report['wall_seconds'],
        'schedule_and_caps_verified':True,'source_binary_map_evidence_identities_rechecked':True,
        'attempts':rows,'totals':totals,'checkpoints':checkpoints,
        'scope':'Read-only artifact, contract, checkpoint and inference reconstruction audit. No native process, optimizer step, training retry or new campaign was run by this audit.',
        'limitations':['Execution stopped after the update; four planned after-evaluation attempts were not launched and no paired performance conclusion is available.',
            'The original executor OrderedDict comparator failed; this independent audit verifies saved state with a Mapping-aware comparator without resuming gameplay.',
            'Bootstrap-only experience contains no observed terminal win/loss feedback when natural_attempts is zero.',
            'Inference reconstruction uses the same frozen implementation, in a fresh process; it is not an independently implemented policy.',
            'Checkpoint reload verifies exact state continuity, not interrupted native process restart support.']}
    (OUT/'independent-review.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'totals':totals,'checkpoints':checkpoints,'wall_seconds':report['wall_seconds']},indent=2))
if __name__=='__main__':main()
