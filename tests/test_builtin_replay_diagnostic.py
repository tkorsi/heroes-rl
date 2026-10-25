"""Verifier compatibility and report admission tests; no native execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.headless import builtin_replay as replay
from tests import test_builtin_replay as fixtures


class IdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binary = self.root / "bin/vcmireplay"
        for name in ("vcmireplay", "vcmiclient", "libvcmi.dylib", "AI/libExplorerRL.dylib",
                     "AI/libNullkiller.dylib", "AI/libBattleAI.dylib"):
            self.file("bin/" + name, "synthetic " + name)
        self.file("dependencies/vcmi.json", json.dumps({"commit":"a"*40}))
        for name in ("engine/patches/rules.patch", "engine/ExplorerRL/Contextual.cpp", "engine/ExperimentReplay/ContextualReplay.cpp"):
            self.file(name, "synthetic source " + name)
        sources = replay.evidence.ADAPTER_SOURCES | {"experiments/headless/builtin_env.py", "experiments/headless/builtin_evidence.py"}
        for name in sources: self.file(name, "synthetic python " + name)
        recipe = replay.legacy.scoring.engine_recipe_digests(self.root)
        runtime = replay.evidence.runtime_identity(self.binary.parent)
        bridge = {k:v for k,v in recipe.items() if k.startswith("engine/ExplorerRL/")}
        identity = {"native_runtime_sha256":runtime, "binary_sha256":self.sha("bin/vcmiclient"),
            "explorer_rl_sha256":runtime["AI/libExplorerRL.dylib"], "engine_recipe":recipe,
            "bridge_source_sha256":hashlib.sha256(json.dumps(bridge,sort_keys=True,separators=(",", ":")).encode()).hexdigest(),
            "contextual_python_sha256":{name:self.sha(name) for name in sources}, "engine_revision":"a"*40,
            "engine":{"commit":"a"*40}}
        self.manifest = {"identity":identity, "opponent_identity":{"binary_sha256":runtime["AI/libNullkiller.dylib"]}}

    def file(self, name, text):
        path = self.root / name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)

    def sha(self, name): return replay.evidence.file_identity(self.root / name)["sha256"]

    def check(self, diagnostic=False):
        with patch.object(replay,"ROOT",self.root):
            return replay.verify_identity(self.manifest,self.binary,diagnostic_verifier_compatibility=diagnostic)

    def test_default_stays_exact_and_opt_in_allows_only_verifier_changes(self):
        self.assertEqual(self.check()["mode"],"exact_capture")
        self.file("bin/vcmireplay","synthetic new verifier")
        self.file("engine/ExperimentReplay/ContextualReplay.cpp","synthetic verifier repair")
        with self.assertRaises(ValueError): self.check()
        result=self.check(True)
        self.assertEqual(set(result["runtime_changes"]),{"vcmireplay"})
        self.assertEqual(set(result["recipe_changes"]),{"engine/ExperimentReplay/ContextualReplay.cpp"})
        self.assertTrue(result["nonverifier_runtime_unchanged"])

    def test_diagnostic_rejects_client_rules_ai_adapter_and_recipe_drift(self):
        for name in ("bin/vcmiclient", "bin/libvcmi.dylib", "bin/AI/libExplorerRL.dylib",
                     "bin/AI/libNullkiller.dylib", "bin/AI/libBattleAI.dylib",
                     "engine/patches/rules.patch", "engine/ExplorerRL/Contextual.cpp",
                     "experiments/headless/builtin_env.py", "experiments/learner/contextual.py",
                     "dependencies/vcmi.json"):
            path=self.root/name; old=path.read_bytes(); path.write_bytes(old+b" changed")
            with self.subTest(name=name),self.assertRaises(ValueError):self.check(True)
            path.write_bytes(old)
        self.file("bin/AI/libUnexpected.dylib","extra runtime")
        with self.assertRaisesRegex(ValueError,"inventory"):self.check(True)

    def test_missing_source_inventory_and_inconsistent_bridge_binding_fail(self):
        original=deepcopy(self.manifest)
        for mutate in (lambda m:m["identity"]["contextual_python_sha256"].pop("experiments/headless/builtin_env.py"),
                       lambda m:m["identity"].update(bridge_source_sha256="b"*64),
                       lambda m:m["identity"]["native_runtime_sha256"].update(vcmireplay=None),
                       lambda m:m["identity"]["engine_recipe"].update({"engine/ExperimentReplay/../forged": "b"*64}),
                       lambda m:m["opponent_identity"].update(binary_sha256="b"*64)):
            self.manifest=deepcopy(original);mutate(self.manifest)
            with self.assertRaises(ValueError):self.check(True)
        with self.assertRaises(ValueError):self.check("true")


class DiagnosticAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.episode=self.root/"captured";self.episode.mkdir()
        self.output=self.root/"supplemental"
        self.manifest={"episode":"1", "learner_player":0, "replay":{"schema":replay.SCHEMA,"files":{}},
            "status":"completed", "seed":120501, "launch":{"requested_settings":{}},"identity":{}}
        for name in ("builtin-finalization.json","logs/process.log","engine/experiment-metadata.json",
                     "agents/red/bridge.jsonl","agents/red/history-v7.jsonl","agents/red/live-history-v7.jsonl"):
            self.file(name,'{"synthetic":true}\n')
        self.file("engine/replay.jsonl",'{"kind":"init","seed":120501,"no_cheats":"1"}\n{"kind":"start"}\n')
        self.file("engine/contextual-replay.jsonl",'{"synthetic":true}\n')
        game_map=self.root/"map.vmap";game_map.write_text("synthetic map")
        self.manifest["identity"]["map_sha256"]=replay.evidence.file_identity(game_map)["sha256"]
        self.file("manifest.json",json.dumps(self.manifest))
        self.file("builtin-verification/builtin-replay-verification.json",json.dumps({"schema":replay.REPORT_SCHEMA,
            "verified":False,"error":"preserved original failure","inputs":{"manifest":replay.evidence.file_identity(self.episode/"manifest.json")}}))
        report_fixture=fixtures.NativeReportTests();report_fixture.setUp()
        self.wire,self.native,self.barrier=report_fixture.wire,report_fixture.native,report_fixture.barrier
        self.final={"terminal_barrier":{"client":self.barrier}}

    def file(self,name,text):
        path=self.episode/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)

    def run_diagnostic(self, *, native_effect=None, bound_error=None, diagnostic=True, output=None):
        with patch.object(replay,"bound_evidence",side_effect=bound_error,return_value=self.final) as bound, \
             patch.object(replay,"effective_environment",return_value={"verified":True}), \
             patch.object(replay,"cross_check_wire",return_value=self.wire), \
             patch.object(replay.legacy,"replay_profile",return_value=({},self.root/"map.vmap")), \
             patch("experiments.headless.builtin_replay_history.verify_history",return_value={"verified":True}), \
             patch.object(replay,"run_native",side_effect=native_effect,return_value=self.native) as native:
            result=replay.verify_episode(self.episode,binary=self.root/"vcmireplay",data=self.root,
                output=output or self.output,diagnostic_verifier_compatibility=diagnostic)
        return result,bound,native

    def test_success_is_supplemental_only_and_preserves_original_files(self):
        before=replay.diagnostic_inputs(self.episode,self.manifest)
        result,bound,native=self.run_diagnostic()
        self.assertTrue(result["diagnostic_verified"])
        self.assertFalse(result["verified"])
        self.assertFalse(result["original_cohort_admission"])
        self.assertTrue(result["original_cohort_result_unchanged"])
        self.assertFalse(result["optimization_eligible"])
        self.assertEqual(result["schema"],replay.DIAGNOSTIC_SCHEMA)
        self.assertEqual(result["training_samples"],0)
        self.assertEqual(result["outcomes_by_player"],dict.fromkeys(replay.COLORS))
        self.assertEqual(before,replay.diagnostic_inputs(self.episode,self.manifest))
        self.assertTrue(bound.call_args.kwargs["diagnostic_verifier_compatibility"])
        self.assertTrue((self.output/"builtin-replay-diagnostic.json").is_file())
        self.assertFalse((self.output/"builtin-replay-verification.json").exists())
        native.assert_called_once()

    def test_default_failure_never_retries_with_compatibility(self):
        result,bound,native=self.run_diagnostic(bound_error=ValueError("runtime changed"),diagnostic=False)
        self.assertFalse(result["verified"])
        self.assertNotIn("diagnostic_verified",result)
        self.assertEqual(result["schema"],replay.REPORT_SCHEMA)
        bound.assert_called_once()
        self.assertFalse(bound.call_args.kwargs["diagnostic_verifier_compatibility"])
        native.assert_not_called()

    def test_original_changes_during_diagnosis_cannot_pass(self):
        def mutate(*args,**kwargs):
            self.file("agents/red/bridge.jsonl",'{"changed":true}\n')
            return self.native
        result,_,_=self.run_diagnostic(native_effect=mutate)
        self.assertFalse(result["diagnostic_verified"])
        self.assertFalse(result["verified"])
        self.assertFalse(result["original_cohort_result_unchanged"])
        self.assertIn("Original evidence changed",result["error"])

    def test_explicit_external_fresh_output_and_preserved_failure_are_required(self):
        for directory in (None,self.episode/"supplemental",self.episode,self.root):
            with self.subTest(directory=directory),self.assertRaises(ValueError):
                replay.verify_episode(self.episode,binary=self.root/"vcmireplay",data=self.root,
                    output=directory,diagnostic_verifier_compatibility=True)
        self.output.mkdir()
        with self.assertRaises(FileExistsError):self.run_diagnostic()
        original=self.episode/"builtin-verification/builtin-replay-verification.json"
        record=json.loads(original.read_text());record["verified"]=True;original.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,"preserved failed"):self.run_diagnostic(output=self.root/"fresh")

