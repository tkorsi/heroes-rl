"""Synthetic offline evidence fixtures; no native game/replay or model executes."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from experiments.headless import builtin_evidence as e
from experiments.learner import contextual as c
from tests.test_builtin_env import initial, manifest_for
from validation.probes import summarize_builtin_preflight as s


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class PreflightFixture:
    def __init__(self, directory, *, passed=False, version="old"):
        self.directory = directory
        directory.mkdir(parents=True)
        for name in ("fixture.vmap", "fixture.json"):
            (directory/name).write_text("synthetic fixture")
        recorded = {"path":str(directory/"removed-runtime"), "bytes":10, "sha256":("a" if version=="old" else "b")*64}
        self.plan = {"schema":s.PROBE_SCHEMA,"output":str(directory),"policy_actions":0,
            "training_performed":False,"training_samples":0,"source":recorded,
            "executables":{"binary":recorded,"replay_binary":recorded},
            "fixture":{"map":s.binding(directory/"fixture.vmap"),"manifest":s.binding(directory/"fixture.json"),
                       "ordinary_cohort_member":False,"natural_outcome_validation":False},
            "schedule":[{"episode":seat+1,"learner_player":seat,"engine_seed":121901} for seat in (0,1)]}
        write(directory/"plan.json",self.plan)
        self.report={"schema":s.PROBE_SCHEMA+"-result","status":"failed","plan":s.binding(directory/"plan.json"),
            "ordinary_cohort_member":False,"training_performed":False,"training_samples":0,"optimizer_steps":0,
            "attempts":[{**row,"status":"not_started"} for row in self.plan["schedule"]]}
        self.report["attempts"][0].update(status="failed",error="synthetic startup failure",policy_actions=0)
        if passed:
            self.passed_case(0)
            self.passed_case(1)
            self.report["status"]="passed"
        self.save()

    def save(self): write(self.directory/"report.json",self.report)

    def passed_case(self, player):
        episode, color = player+1,s.COLORS[player]
        folder=self.directory/f"episode-{episode:06d}"
        h=c.PlayerHistory(str(episode),player,schema=c.DIALOGS_SCHEMA)
        h.append(replace(initial(player).frames[0],episode=str(episode)))
        h.emit("cutoff",decision_seq=h.decision.seq,reason="collection_window")
        history=folder/f"agents/{color}/history-v7.jsonl"
        history.parent.mkdir(parents=True)
        history.write_bytes(b"".join((json.dumps(c.encode_frame(f),sort_keys=True,separators=(",",":"))+"\n").encode() for f in h.frames))
        live=history.with_name("live-history-v7.jsonl");live.write_bytes(history.read_bytes().splitlines(keepends=True)[0])
        bridge=history.with_name("bridge.jsonl");bridge.write_text(json.dumps({"direction":"engine","message":{"synthetic":True}})+"\n")
        for name in ("logs/process.log","engine/replay.jsonl","engine/contextual-replay.jsonl","engine/experiment-metadata.json"):
            path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("synthetic evidence\n")
        manifest=manifest_for(h)
        captured={"files":{name:e.native.file_identity(folder/name) for name in ("engine/replay.jsonl","engine/contextual-replay.jsonl")}}
        identity={"binary_sha256":self.plan["executables"]["binary"]["sha256"],
                  "map_sha256":self.plan["fixture"]["map"]["sha256"],
                  "native_runtime_sha256":{"vcmireplay":self.plan["executables"]["replay_binary"]["sha256"]},
                  "contextual_python_sha256":{"historical_adapter.py":"c"*64}}
        manifest.update(seed=121901,identity=identity,replay=captured)
        declaration=manifest["players"][color]
        declaration.update(dict.fromkeys(("actions_sent","actions_accepted","actions_rejected","responses_undelivered"),0))
        declaration["history_v7"].update(path=str(history.relative_to(folder)),sha256=h.digest())
        declaration["live_evidence"]={"path":str(live.relative_to(folder)),"sha256":e.native.file_identity(live)["sha256"]}
        final={"schema":e.FINAL_SCHEMA,"episode":str(episode),"learner_player":player,"controllers":e.controllers(player),
            "returncode":-9,"identity":identity,"native_capture":captured,"dialog_cutoff_drain":None,"history_sha256":h.digest(),
            "stream_complete":True,"engine_evidence_error":None,"declared_stop":True,"shutdown_verified":True,
            "reason":"collection_window","engine_verified_outcomes":None,
            "raw_bridge_sha256":{color:e.native.file_identity(bridge)["sha256"]},
            "process_log_sha256":e.native.file_identity(folder/"logs/process.log")["sha256"]}
        write(folder/"builtin-finalization.json",final)
        manifest["evidence_ref"]="builtin-finalization.json#sha256="+e.native.file_identity(folder/"builtin-finalization.json")["sha256"]
        write(folder/"manifest.json",manifest)
        counts=dict(transactions=2,messages=1,responses=0,requests=2,client_packets=3,native_opponent_requests=1)
        wire={**counts,"controllers":e.controllers(player)}
        native={"schema":s.replay.REPORT_SCHEMA,"verified":True,"summary":{"controllers":e.controllers(player),
            "server_verified":True,"loaded":counts,"checked":counts,"unchecked":dict.fromkeys(counts,0),
            "boundary":{"kind":"cutoff","client_packets":3,"barrier_id":None},"captured_prefix_only":True,
            "terminal_barrier":None,"server":{"inputs":2}}}
        replay_path=folder/"builtin-verification/builtin-replay-verification.json"
        write(replay_path.parent/"native-report.json",native)
        checked={"schema":s.replay.REPORT_SCHEMA,"episode":str(episode),"verified":True,"opponent_policy_reexecuted":False,
            "inputs":{"manifest":e.native.file_identity(folder/"manifest.json"),"verifier_sources":{"historical.py":{"sha256":"d"*64}}},
            "native":{**native,"returncode":0,"seconds":.1},"wire":wire,"history":{"verified":True},
            "effective_engine":{"verified":True,"metadata_file":e.native.file_identity(folder/"engine/experiment-metadata.json")},
            "native_opponent_requests":1}
        write(replay_path,checked)
        self.report["attempts"][player]={**self.plan["schedule"][player],"status":"passed","policy_actions":0,
            "held_decision_seq":0,"held_decision_day":1,"cutoff_seq":1,"own_history_frames":2,"visible_events_retained":0,
            "outcomes":dict.fromkeys(s.COLORS),"rewards":dict.fromkeys(s.COLORS),"manifest":s.binding(folder/"manifest.json"),
            "replay":s.binding(replay_path),"replay_verified":True,"effective_engine":checked["effective_engine"],"native_opponent_requests":1}


class BuiltinPreflightSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()

    def test_all_cohorts_preserved_with_separate_historical_runtime_and_logs(self):
        failed=PreflightFixture(self.root/"failed")
        fixed=PreflightFixture(self.root/"fixed",passed=True,version="new")
        log=self.root/"build.log";log.write_text("synthetic retained build output")
        result=s.summarize([failed.directory,fixed.directory],validation_logs=[log])
        self.assertEqual(result["counts"],{"failed":1,"not_started":1,"passed":2})
        self.assertFalse(result["natural_outcome_validation"])
        self.assertFalse(result["ordinary_cohort_member"])
        self.assertIsNone(result["cohorts"][0]["attempts"][0]["observed_policy_actions"])
        self.assertEqual(result["cohorts"][1]["attempts"][0]["observed_policy_actions"],0)
        self.assertNotEqual(result["cohorts"][0]["historical_executables"],result["cohorts"][1]["historical_executables"])
        self.assertEqual(result["validation_logs"],[s.binding(log)])

    def test_report_bindings_schedule_and_scores_cannot_be_tampered(self):
        fixture=PreflightFixture(self.root/"cohort",passed=True)
        original=deepcopy(fixture.report)
        for mutate in (lambda r:r["plan"].update(sha256="f"*64),
                       lambda r:r["attempts"][0]["manifest"].update(path=str(self.root/"other.json")),
                       lambda r:r["attempts"][0]["replay"].update(sha256="f"*64),
                       lambda r:r["attempts"][0].update(held_decision_seq=7),
                       lambda r:r["attempts"][0].update(policy_actions=True),
                       lambda r:r["attempts"][0]["rewards"].update(red=0),
                       lambda r:r["attempts"].pop()):
            fixture.report=deepcopy(original);mutate(fixture.report);fixture.save()
            with self.assertRaises(ValueError):s.summarize([fixture.directory])

    def test_raw_evidence_change_is_rejected_even_when_report_unchanged(self):
        fixture=PreflightFixture(self.root/"cohort",passed=True)
        path=fixture.directory/"episode-000001/agents/red/bridge.jsonl"
        path.write_text(path.read_text()+"\n")
        with self.assertRaisesRegex(ValueError,"Bound file differs"):s.summarize([fixture.directory])

    def test_native_coverage_failure_cannot_be_promoted_by_rehashing_summary(self):
        fixture=PreflightFixture(self.root/"cohort",passed=True)
        path=Path(fixture.report["attempts"][0]["replay"]["path"])
        checked=s.read(path);checked["native"]["summary"]["unchecked"]["requests"]=1;write(path,checked)
        fixture.report["attempts"][0]["replay"]=s.binding(path);fixture.save()
        with self.assertRaisesRegex(ValueError,"coverage"):s.summarize([fixture.directory])

    def test_not_started_or_duplicate_cohorts_cannot_hide_executed_artifacts(self):
        fixture=PreflightFixture(self.root/"cohort")
        with self.assertRaisesRegex(ValueError,"distinct"):s.summarize([fixture.directory,fixture.directory])
        (fixture.directory/"episode-000002").mkdir()
        with self.assertRaisesRegex(ValueError,"Not-started"):s.summarize([fixture.directory])


if __name__ == "__main__":unittest.main()
