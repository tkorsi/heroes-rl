"""Synthetic one-seat boundaries and receipt grammar; never native game evidence."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.headless import builtin_evidence as e
from experiments.headless.builtin_env import BuiltinHeadlessEnv
from experiments.headless.contextual_protocol import ContextualProducer
from experiments.learner import contextual as c
from tests.test_contextual_dialogs import history_at, panel, action, begin_answer, finish_answer, decide, observation


def initial(player=0, profile=c.DIALOGS_PROFILE):
    history = c.PlayerHistory("synthetic-one-seat", player, schema=profile.history_schema)
    obs = c.Observation((2, 2, 1), (c.Tile((0, 0, 0), 1, 0, False, False),), (), (0,) * 7, (), ())
    history.emit("decision", observation=obs, revision=0, offers=c.Offers.from_observation(obs, schema=profile.action_schema))
    return history


def manifest_for(history, status="truncated"):
    color, player = e.COLORS[history.player], history.player
    outcomes = {color: history.terminal, e.COLORS[1-player]: "loss" if history.terminal == "win" else "win"} if status == "completed" else dict.fromkeys(e.COLORS)
    return {"schema_version": 1, "kind": e.MANIFEST_KIND, "episode": history.episode,
        "environment": c.DIALOGS_ENVIRONMENT, "history_schema": c.DIALOGS_SCHEMA, "action_schema": c.DIALOGS_ACTION_SCHEMA,
        "learner_player": player, "controllers": e.controllers(player), "opponent_identity": e.opponent_identity(player, "a"*64),
        "status": status, "termination_reason": "natural_terminal" if status == "completed" else "collection_window" if status == "truncated" else "protocol_error",
        "returncode": 0 if status == "completed" else -9, "dialog_cutoff_drain": None,
        "outcomes_by_player": outcomes, "rewards_by_player": {key: 1 if val == "win" else -1 if val == "loss" else None for key,val in outcomes.items()},
        "players": {color: {"history_v7": {"canonical_sha256": history.digest(), "prefix_sha256": history.digest(),
            "boundary": {"completed":"natural", "truncated":"cutoff", "failed":"failure"}[status], "complete": True}}}}


def as_producer(history):
    result = ContextualProducer(history.episode, history.player, interface=c.DIALOGS_ENVIRONMENT)
    result.live = result.history = history
    result.ready = True
    result.decision_ids = {f.seq: f.seq for f in history.frames if f.kind == "decision"}
    result.current_id = history.decision.seq if history.decision else None
    return result


class BuiltinBoundaryTests(unittest.TestCase):
    def configured(self, history):
        env = BuiltinHeadlessEnv(object(), Path("/unused-synthetic"), learner_player=history.player)
        sent = []
        side = SimpleNamespace(player=history.player, color=e.COLORS[history.player], validator=as_producer(history),
            history=history, actions_sent=sum(f.kind=="submitted" for f in history.frames), unread_frames_from=0,
            connection=SimpleNamespace(sendall=sent.append), ledger=io.StringIO(), responses_undelivered=0)
        env.sides = {history.player: side}
        env.started = time.monotonic()
        env.agent_wait = 0
        env.costs = {"respond":0,"ledger":0}
        env.reason = env.detail = None
        env._issue(history.player)
        return env, side, sent

    def test_cutoff_only_own_safe_decision_never_advances_native_opponent(self):
        for player in (0,1):
            h = initial(player)
            env, side, sent = self.configured(h)
            with patch.object(env, "_settle", return_value=None), patch.object(env, "_finish", return_value="cutoff") as finish:
                self.assertEqual(env.truncate(), "cutoff")
                finish.assert_called_once_with(None)
            self.assertEqual(side.validator.history.boundary,"cutoff_pending")
            self.assertEqual(side.validator.continuation_steps,0)
            self.assertEqual(sent,[])
            self.assertIsNone(env.request)
            with self.assertRaises(ValueError): env.history(e.COLORS[1-player])
            self.assertTrue(e.validate_manifest_history(side.validator.history,manifest_for(side.validator.history))["consistent"])

    def test_active_dialog_rejects_direct_cutoff_and_never_chooses_cleanup(self):
        h = history_at(); env, side, sent = self.configured(h)
        before = h.digest()
        with self.assertRaises(ValueError): env.truncate()
        self.assertEqual(h.digest(),before)
        env.defer_dialog_cutoff("collection_window")
        self.assertEqual(env.dialog_cutoff_drain["trigger"]["actions_sent"],1)
        self.assertEqual(sent,[])
        with self.assertRaises(ValueError): env._send_current(action(selection=panel().choices[0].ref),continuation=True)
        env._send_current(action(selection=panel().choices[1].ref))
        self.assertEqual(len(sent),1)
        self.assertFalse(json.loads(side.ledger.getvalue())["message"]["outside_learner_prefix"])
        self.assertEqual(side.validator.continuation_steps,0)

    def test_natural_result_requires_own_terminal_but_no_fictitious_opponent_stream(self):
        for player in (0,1):
            h = initial(player)
            h.emit("superseded",decision_seq=h.decision.seq,reason="terminal")
            h.emit("terminal",outcome="loss")
            manifest = manifest_for(h,"completed")
            result=e.validate_manifest_history(h,manifest)
            self.assertEqual(result["boundary"],"natural")
            self.assertIsNone(h.boundary)
            self.assertFalse(result["native_execution_verified"])
            bad=deepcopy(manifest); bad["players"][e.COLORS[1-player]]={}
            with self.assertRaises(ValueError): e.validate_manifest_history(h,bad)
            bad=deepcopy(manifest); bad["outcomes_by_player"][e.COLORS[player]]="win"
            with self.assertRaises(ValueError): e.validate_manifest_history(h,bad)

    def test_cutoff_failure_and_controller_tampering_never_become_scores(self):
        h=initial(1); h.emit("cutoff",decision_seq=h.decision.seq,reason="collection_window")
        good=manifest_for(h)
        for mutate in (lambda m:m["rewards_by_player"].update(blue=0),
                       lambda m:m["controllers"].update(explorer_players=[True]),
                       lambda m:m["opponent_identity"].update(player=False),
                       lambda m:m["players"]["blue"]["history_v7"].update(prefix_sha256="b"*64)):
            bad=deepcopy(good);mutate(bad)
            with self.assertRaises(ValueError):e.validate_manifest_history(h,bad)
        h.emit("failure",reason="protocol_error")
        self.assertEqual(e.validate_manifest_history(h,manifest_for(h,"failed"))["boundary"],"failure")

    def test_drain_metadata_binds_only_actual_extra_own_choices(self):
        h=history_at(); trigger=h.decision.seq
        seq=begin_answer(h);finish_answer(h,seq);decide(h,observation())
        h.emit("cutoff",decision_seq=h.decision.seq,reason="collection_window")
        manifest=manifest_for(h)
        manifest["dialog_cutoff_drain"]={"reason":"collection_window","trigger":{"player":0,"decision_seq":trigger,"actions_sent":1},
            "limit":8,"choices":[{"player":0,"decision_seq":trigger}],"learner_included":True}
        e.validate_manifest_history(h,manifest)
        for mutate in (lambda d:d["choices"].clear(),lambda d:d.update(limit=9),lambda d:d["trigger"].update(actions_sent=0)):
            bad=deepcopy(manifest);mutate(bad["dialog_cutoff_drain"])
            with self.assertRaises(ValueError):e.validate_manifest_history(h,bad)


class BuiltinReceiptTests(unittest.TestCase):
    def records(self,player):
        common={"schema":e.native.SCHEMA,"episode":"synthetic-receipt","interface":c.DIALOGS_ENVIRONMENT,"barrier_id":1,
            "origin":"request","request":{"player":1-player,"id":3,"type":101},"client_packets":7,
            "outcomes":["win","loss"],"controllers":e.controllers(player)}
        return [{**common,"kind":"server"},{**common,"kind":"client","callbacks_complete":True,"observed_count":7,
            "native_terminal_callbacks":[0,1],"explorer_terminal_callbacks":[player]}]

    def test_strict_mixed_receipt_grammar_both_native_only_one_external_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"engine").mkdir()
            for player in (0,1):
                original=self.records(player)
                for mutation in (None,lambda v:v[1].update(explorer_terminal_callbacks=[0,1]),
                                 lambda v:v[1].update(native_terminal_callbacks=[player]),
                                 lambda v:v[0]["controllers"].update(explorer_players=[0,1]),
                                 lambda v:v[1].update(observed_count=8)):
                    rows=deepcopy(original)
                    if mutation:mutation(rows)
                    for path,row in zip(e.native.FILES,rows):(root/path).write_text(json.dumps(row)+"\n")
                    if mutation:
                        with self.assertRaises(ValueError):e.terminal_barrier(root,episode="synthetic-receipt",interface=c.DIALOGS_ENVIRONMENT,outcomes=["win","loss"],learner_player=player)
                    else:
                        self.assertTrue(e.terminal_barrier(root,episode="synthetic-receipt",interface=c.DIALOGS_ENVIRONMENT,outcomes=["win","loss"],learner_player=player)["verified"])


if __name__ == "__main__":unittest.main()
