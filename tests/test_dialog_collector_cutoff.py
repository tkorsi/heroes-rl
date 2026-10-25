"""Synthetic collector orchestration, never a simulated native validation claim.

The fixture supplies a typed query chain. Actual frozen neural distributions
choose every action, including the answers after the requested collection bound.
"""
from copy import deepcopy
from dataclasses import replace
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch": raise
    raise unittest.SkipTest("Optional torch is not installed") from error

from experiments.learner import contextual as c
from experiments.learner import contextual_model as m
from experiments.learner.contextual_rollout import CollectedEpisode, ContextualCollector, ContextualSession
from tests.test_contextual_dialogs import observation, panel, decide
from tests.test_contextual_rollout import initial, request, end_record
from tests.test_contextual_management_model import last_output


class SyntheticDialogChain:
    """A hand-authored transport sequence, with no engine, optimizer or files."""
    interface = c.DIALOGS_ENVIRONMENT

    def __init__(self):
        self.pair = None
        self.unread = 0
        self.actions = []
        self.deferrals = []
        self.drain = None
        self.truncations = []

    def reset(self, *, episode, seed):
        self.base = replace(observation(), towns=(), heroes=(observation().heroes[0],))
        red = c.PlayerHistory(str(episode),0,schema=c.DIALOGS_SCHEMA)
        decide(red,self.base)
        self.pair = red,initial(1,str(episode),schema=c.DIALOGS_SCHEMA)
        return self.next_request()

    def next_request(self):
        result = request(self.pair[0],self.unread)
        self.unread = len(self.pair[0].frames)
        return result

    def delivered(self, number):
        return panel("chest",f"synthetic-query-{number}")

    def query_observation(self, delivered):
        return replace(self.base,window=observation(delivered).window)

    def open_query(self, delivered):
        red = self.pair[0]
        red.emit("context_opened",context=self.query_observation(delivered).context,
                 revision=red.revision+1,reason="adventure_dialog")
        decide(red,self.query_observation(delivered))

    def step(self, selected):
        red = self.pair[0]
        decision = red.decision
        if not decision.offers.permits(selected): raise AssertionError("Real sampler left the current support")
        red.emit("submitted",decision_seq=decision.seq,action=selected)
        red.emit("consumed",decision_seq=decision.seq,consumer="engine")
        self.actions.append(selected)
        if len(self.actions) == 1:
            # These fixed synthetic model/RNG seeds yield a normal move choice.
            # No action is substituted or forced by the fixture.
            if selected.kind != "move_to": raise AssertionError("Fixture seed no longer reaches its query trigger")
            red.emit("issued",decision_seq=decision.seq,command_index=0)
            delivered = self.delivered(1)
            red.emit("event",event=c.VisibleEvent("dialog_requested","hero",dialog=delivered))
            red.emit("acknowledged",decision_seq=decision.seq,command_index=0,accepted=True)
            red.emit("result",decision_seq=decision.seq,result=c.Result("interaction",True,(1,1,0),100,
                     (selected.x,selected.y) == (1,1)))
            self.open_query(delivered)
        else:
            if selected.kind != "choose_dialog" or self.drain is None:
                raise AssertionError("Post-budget query must use a recorded actual policy answer")
            self.drain["choices"].append({"player":0,"decision_seq":decision.seq})
            red.emit("context",decision_seq=decision.seq,context=c.Context(),revision=red.revision+1)
            red.emit("issued",decision_seq=decision.seq,command_index=0)
            if len(self.actions) == 2:
                delivered = self.delivered(2)
                red.emit("event",event=c.VisibleEvent("dialog_requested","hero",dialog=delivered))
            red.emit("acknowledged",decision_seq=decision.seq,command_index=0,accepted=True)
            red.emit("result",decision_seq=decision.seq,result=c.Result("dialog_answered",True))
            if len(self.actions) == 2: self.open_query(delivered)
            else: decide(red,self.base)
        return self.next_request()

    def defer_dialog_cutoff(self, reason):
        red = self.pair[0]
        if red.context.kind != "dialog": raise AssertionError("Only active queries may defer the budget")
        self.deferrals.append((reason,red.decision.seq))
        if self.drain is None:
            self.drain = {"reason":reason,"trigger":{"player":0,"decision_seq":red.decision.seq,
                          "actions_sent":[len(self.actions),0]},"limit":8,"choices":[],"learner_included":True}

    def truncate(self, reason):
        if self.pair[0].context.kind != "map": raise AssertionError("Collector tried to truncate a strategic dialog")
        self.truncations.append((reason,self.pair[0].decision.seq))
        for history in self.pair:
            history.emit("cutoff",decision_seq=history.decision.seq,reason=reason)
        c.finalize_cutoff(self.pair,evidence_ref="synthetic-not-engine-evidence",histories_complete=(True,True),
                          failure_free=True,shutdown_verified=True,continuation_limit=8,continuation_steps=(0,0))
        end = end_record(self.pair)
        end.manifest.update(termination_reason=reason,dialog_cutoff_drain=deepcopy(self.drain))
        return end

    def history(self, color):
        return self.pair[("red","blue").index(color)].frames


class DialogCollectorCutoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def collect(self):
        torch.manual_seed(172)
        model = m.ContextualActorCritic(m.ModelConfig(schema=m.DIALOGS_MODEL_SCHEMA,hidden_size=8,feature_size=4))
        identity = m.weights_sha256(model)
        collector = ContextualCollector((model,model),("fixed-learner","fixed-opponent"),learner_player=0,
            opponent_pool="synthetic-fixed-pool",interface=c.DIALOGS_ENVIRONMENT,purpose="diagnostic",origin="synthetic",policy_seeds=(1,2))
        env = SyntheticDialogChain()
        episode = collector.run_episode(env,episode="synthetic-dialog-collector",seed=17,max_requests=1)
        self.assertEqual(m.weights_sha256(model),identity)
        return episode,collector,env,model

    def test_run_episode_samples_records_and_remembers_every_extra_choice(self):
        episode,collector,env,model = self.collect()
        self.assertEqual([action.kind for action in env.actions],["move_to","choose_dialog","choose_dialog"])
        self.assertEqual(len(env.deferrals),2)
        self.assertEqual(len(env.truncations),1)
        red = episode.rollouts[0]
        self.assertEqual([decision.action for decision in red.decisions],env.actions)
        self.assertEqual(len(red.estimates),4)  # Also value the safe bootstrap decision, without sampling it.
        self.assertEqual(len(red.samples),3)
        self.assertTrue(all(sample.reward == 0 for sample in red.samples))
        self.assertTrue(all(not rollout.optimization_eligible for rollout in episode.rollouts))
        self.assertEqual(env.drain["choices"],[{"player":0,"decision_seq":decision.decision_seq} for decision in red.decisions[1:]])
        self.assertEqual([seq for _,seq in env.deferrals],[decision.decision_seq for decision in red.decisions[1:]])
        self.assertNotIn(env.truncations[0][1],collector.sessions[0]._choices)
        independent = ContextualSession(model,red.episode,0,"fixed-learner","fixed-opponent",schema=c.DIALOGS_SCHEMA,seed=1)
        for decision in red.decisions:
            independent.consume(red.history[:decision.decision_seq+1])
            # Same unmodified distribution, RNG path, chosen joint log P and recurrent prefix.
            self.assertEqual(independent.choose(decision.decision_seq),decision)
            with torch.no_grad(): scored = last_output(model,red.history[:decision.decision_seq+1])
            self.assertAlmostEqual(scored.decision.value.item(),decision.value,places=6)
            self.assertAlmostEqual(scored.decision.distribution.log_prob(decision.action).item(),decision.log_prob,places=6)
        with torch.no_grad(): folded = last_output(model,red.history)
        torch.testing.assert_close(folded.state,collector.sessions[0].state,rtol=0,atol=0)
        self.assertEqual(episode.to_dict()["audit"],{"source":"synthetic","native_evidence_verified":False})
        self.assertEqual(CollectedEpisode.from_dict(episode.to_dict()),episode)

    def test_serialized_drain_declaration_must_match_retained_behavior(self):
        episode,_,_,_ = self.collect()
        for mutation in (lambda raw: raw["manifest"].pop("dialog_cutoff_drain"),
                         lambda raw: raw["manifest"]["dialog_cutoff_drain"]["choices"].clear(),
                         lambda raw: raw["manifest"]["dialog_cutoff_drain"].update(learner_included=False),
                         lambda raw: raw["manifest"]["dialog_cutoff_drain"]["trigger"]["actions_sent"].__setitem__(0,0)):
            raw = episode.to_dict(); mutation(raw)
            with self.assertRaises(ValueError): CollectedEpisode.from_dict(raw)


if __name__ == "__main__": unittest.main()
