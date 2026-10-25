"""Synthetic offline evidence checks, never native execution or outcome evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from validation.probes import summarize_builtin_terminal as s
from tests import test_builtin_replay as replay_fixtures


class TerminalSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)

    def test_supplemental_success_cannot_admit_or_rescore_original(self):
        good={"schema":s.replay.DIAGNOSTIC_SCHEMA,"verified":False,"diagnostic_verified":True,
            "original_cohort_admission":False,"original_cohort_result_unchanged":True,"optimization_eligible":False,
            "outcomes_by_player":dict.fromkeys(s.COLORS),"rewards_by_player":dict.fromkeys(s.COLORS),
            "training_performed":False,"training_samples":0}
        s.check_supplemental_flags(good)
        for change in (lambda d:d.update(verified=True),lambda d:d.update(schema=s.replay.REPORT_SCHEMA),
                       lambda d:d.update(original_cohort_admission=True),lambda d:d.update(original_cohort_result_unchanged=False),
                       lambda d:d.update(optimization_eligible=True),lambda d:d.update(training_samples=1),
                       lambda d:d["outcomes_by_player"].update(red="loss")):
            changed=deepcopy(good);change(changed)
            with self.assertRaises(ValueError):s.check_supplemental_flags(changed)
        good.update(diagnostic_verified=False,error="retained failed supplemental replay")
        s.check_supplemental_flags(good)

    def test_compatibility_checks_historical_capture_not_current_sources(self):
        identity={"native_runtime_sha256":{"vcmireplay":"a"*64,"libvcmi.dylib":"b"*64},
                  "engine_recipe":{"engine/ExperimentReplay/ContextualReplay.cpp":"c"*64}}
        good={"mode":"diagnostic_verifier_only","runtime_changes":{"vcmireplay":{"captured":"a"*64,"current":"d"*64}},
              "recipe_changes":{"engine/ExperimentReplay/ContextualReplay.cpp":{"captured":"c"*64,"current":"e"*64}},
              **dict.fromkeys(("nonverifier_runtime_unchanged","client_unchanged","adapter_sources_unchanged","nonverifier_recipe_unchanged"),True)}
        s.check_compatibility(good,identity)
        for change in (lambda d:d["runtime_changes"].update({"libvcmi.dylib":{"captured":"b"*64,"current":"e"*64}}),
                       lambda d:d["recipe_changes"].update({"engine/ExplorerRL/main.cpp":{"captured":None,"current":"e"*64}}),
                       lambda d:d["runtime_changes"]["vcmireplay"].update(captured="b"*64),
                       lambda d:d.update(adapter_sources_unchanged=False)):
            changed=deepcopy(good);change(changed)
            with self.assertRaises(ValueError):s.check_compatibility(changed,identity)

    def schedule(self):
        schedule=[{"episode":i+1,"learner_player":i%2,"engine_seed":100+i//2,"policy_seed":200+i//2} for i in range(8)]
        plan={"schedule":schedule,"frozen":{"checkpoint_file_sha256":"a"*64},"checkpoint":{"sha256":"a"*64},
              "execution_output":str(self.root)}
        rows=[{**row,"status":"failed" if i==0 else "not_started","outcomes":dict.fromkeys(s.COLORS),
               "rewards":dict.fromkeys(s.COLORS),"training_samples":0,"optimizer_steps":0} for i,row in enumerate(schedule)]
        report={"status":"stopped","attempts":rows,"training_samples":0,"optimizer_steps":0,"optimizer_updates":0,
                "training_performed":False,"frozen_before":plan["frozen"],"frozen_after_file_sha256":"a"*64}
        return plan,report,{"attempts":deepcopy(rows)}

    def test_original_failed_and_unstarted_rows_are_all_retained(self):
        plan,report,compact=self.schedule()
        result=s.stopped_schedule(plan,report,compact)
        self.assertEqual(result["counts"],{"failed":1,"not_started":7})
        self.assertEqual(len(result["attempts"]),8)
        self.assertTrue(all(r["outcomes"] == dict.fromkeys(s.COLORS) for r in result["attempts"]))
        for change in (lambda r:r["attempts"].pop(),lambda r:r["attempts"][0].update(status="passed"),
                       lambda r:r["attempts"][0]["rewards"].update(red=-1),lambda r:r.update(frozen_after_file_sha256="b"*64),
                       lambda r:r.update(optimizer_steps=1),lambda r:r["attempts"][1].update(policy_seed=999)):
            bad=deepcopy(report);change(bad)
            with self.assertRaises(ValueError):s.stopped_schedule(plan,bad,compact)
        (self.root/"episode-000002").mkdir()
        with self.assertRaises(ValueError):s.stopped_schedule(plan,report,compact)

    def test_compact_accounting_hashes_packet_bytes_without_dropping_identity(self):
        original={"authoritative_inputs":2,"final_terminal_input":None,"terminal_unapplied":[{
            "pack":"deadbeef","player":1,"request_id":3,"pack_type":182,"receipt":{"index":7},"seq":10}]}
        result=s.compact_accounting(original)
        self.assertNotIn("pack",result["terminal_unapplied"][0])
        self.assertEqual(result["terminal_unapplied"][0]["request_id"],3)
        self.assertEqual(len(result["terminal_unapplied"][0]["pack_sha256"]),64)
        self.assertIn("pack",original["terminal_unapplied"][0])

    def replay_fixture(self):
        folder=self.root/"episode";output=self.root/"diagnostic";output.mkdir()
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value)+"\n")
        native_fixture=replay_fixtures.NativeReportTests();native_fixture.setUp()
        accounting={"authoritative_inputs":4,"terminal_unapplied":[],"final_terminal_input":None}
        native_fixture.wire["request_accounting"]=accounting
        native_fixture.native["summary"]["request_accounting"]=accounting
        manifest={"identity":{},"learner_player":1,"players":{"blue":{
            label:{"path":f"agents/blue/{label}.jsonl"} for label in ("history_v7","live_evidence")}},
            "outcomes_by_player":{"red":"win","blue":"loss"}}
        for item in manifest["players"]["blue"].values():
            write(folder/item["path"],{"synthetic":True});item["sha256"]=s.bind(folder/item["path"])["sha256"]
        for name in ("manifest.json","agents/blue/bridge.jsonl","logs/process.log","engine/experiment-metadata.json",
                     "engine/replay.jsonl","engine/contextual-replay.jsonl"):
            write(folder/name,manifest if name == "manifest.json" else {"synthetic":True})
        for name in ("replay.jsonl","contextual-replay.jsonl"):
            (output/name).write_bytes((folder/"engine"/name).read_bytes())
        final={"native_capture":{"files":{}},"raw_bridge_sha256":{"blue":s.bind(folder/"agents/blue/bridge.jsonl")["sha256"]},
               "process_log_sha256":s.bind(folder/"logs/process.log")["sha256"],"terminal_barrier":{"client":native_fixture.barrier}}
        report={"diagnostic_verified":True,"inputs":{"manifest":s.bind(folder/"manifest.json")},
            "opponent_policy_reexecuted":False,"history":{"verified":True,"opponent_history":None},
            "native":native_fixture.native,"wire":native_fixture.wire,
            "effective_engine":{"metadata_file":s.bind(folder/"engine/experiment-metadata.json")},
            "prefix":{"server_partial_bytes":0,"server_unpaired_inputs":0,"contextual_partial_bytes":0}}
        write(output/"native-report.json",{k:v for k,v in report["native"].items() if k not in ("returncode","seconds")})
        path=output/"builtin-replay-diagnostic.json";write(path,report)
        return path,folder,manifest,final,report

    def test_native_counts_and_full_saved_report_are_checked(self):
        path,folder,manifest,final,report=self.replay_fixture()
        with patch.object(s.evidence,"read_finalization",return_value=final):
            checked=s.checked_replay(path,folder,manifest,supplemental=True)
            self.assertTrue(checked["passed"])
            self.assertEqual(checked["counts"]["requests"],4)
            for mutate in (lambda d:d["native"]["summary"]["unchecked"].update(requests=1),
                           lambda d:d["native"]["summary"]["server"].update(inputs=3),
                           lambda d:d["wire"]["request_accounting"].update(authoritative_inputs=3),
                           lambda d:d["native"]["summary"].update(unbound_extra="tampered")):
                changed=deepcopy(report);mutate(changed);path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):s.checked_replay(path,folder,manifest,supplemental=True)

    def test_failed_supplemental_is_retained_without_success_claim(self):
        path,folder,manifest,final,report=self.replay_fixture()
        report.update(diagnostic_verified=False,error="retained synthetic native failure")
        report.pop("native");path.write_text(json.dumps(report))
        with patch.object(s.evidence,"read_finalization",return_value=final):
            result=s.checked_replay(path,folder,manifest,supplemental=True)
        self.assertFalse(result["passed"])
        self.assertEqual(result["error"],report["error"])
        self.assertNotIn("counts",result)

    def test_replay_copies_allow_json_formatting_but_no_changed_or_partial_rows(self):
        path,folder,manifest,final,_=self.replay_fixture()
        copy=path.parent/"contextual-replay.jsonl"
        copy.write_text('{"synthetic": true}\n')
        with patch.object(s.evidence,"read_finalization",return_value=final):
            self.assertTrue(s.checked_replay(path,folder,manifest,supplemental=True)["passed"])
            for text in ('{"synthetic": false}\n','{"synthetic": true}'):
                copy.write_text(text)
                with self.assertRaises(ValueError):s.checked_replay(path,folder,manifest,supplemental=True)

    def test_bound_artifact_tampering_is_rejected(self):
        path=self.root/"report.json";path.write_text('{}\n');before=s.bind(path)
        path.write_text('{"changed":true}\n')
        with self.assertRaises(ValueError):s.bind(path,before)

