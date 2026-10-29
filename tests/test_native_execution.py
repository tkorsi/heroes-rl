"""Native execution policy and frozen-launch plumbing; no native games or updates."""
from contextlib import nullcontext, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.learner import local_script_training as local, native_backend, native_execution, resources

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_execution_controller", ROOT / "tools/run_local_training.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
OBSERVED = {"schema": resources.SCHEMA, "available_cpus": 16, "platform": "Darwin"}
MANIFEST = Path("/private/synthetic-native/manifest.json")
SHA = "a" * 64


def choice(cap=None):
    with patch.object(native_backend, "verified_manifest", return_value={}), \
            patch.object(resources, "discover_resources", return_value=OBSERVED):
        return native_execution.native_execution_choice(MANIFEST, SHA, cpu_cap=cap, require_acceptance=False)


class NativeExecutionSchemaTests(unittest.TestCase):
    def test_plan_preserves_four_games_and_binds_phase_budgets(self):
        value = choice()
        self.assertIs(native_execution.check_native_execution(value), value)
        self.assertIs(local.check_execution(value), value)
        self.assertEqual(value["resource_plan"]["batch_size"], 4)
        self.assertEqual(value["native_learner"]["prefix_threads"], 16)
        self.assertEqual((value["collection_workers"], value["verification_workers"]), (4, 4))
        limited = choice(2)
        self.assertEqual(limited["native_learner"]["prefix_threads"], 2)
        self.assertEqual((limited["collection_workers"], limited["verification_workers"]), (1, 1))

    def test_pipeline_requires_room_for_all_independent_phases_and_coordinator(self):
        for cap in (1, 2, 4, 8, 9, 16):
            value = choice(cap)
            self.assertEqual(value['implementation'], 'native-cpu-v2')
            self.assertEqual(value['pipeline_verification'], cap >= 9)
            self.assertEqual(value['resource_plan']['phases_run_concurrently'], cap >= 9)
            self.assertLessEqual(value['resource_plan']['collection_verification_cpu_budget'], cap)
            for bad in (not value['pipeline_verification'], 1, None):
                with self.subTest(cap=cap, bad=bad), self.assertRaises(ValueError):
                    native_execution.check_native_execution({**value, 'pipeline_verification': bad})

    def test_historical_native_v1_resource_contract_still_validates_exactly(self):
        value = choice()
        value['implementation'] = 'native-cpu-v1'
        value.pop('pipeline_verification')
        value['resource_plan'] = resources.make_resource_plan(resources=OBSERVED)
        self.assertIs(local.check_execution(value), value)
        with self.assertRaises(ValueError):
            native_execution.check_native_execution({**value, 'pipeline_verification': False})

    def test_fork_python_replay_numerical_experiments_and_unknown_fields_are_rejected(self):
        changes = {"overlapped_verification": True, "prefix_helpers": 1, "text_leaf_replay": True,
                   "replay_threads": 2, "experiments": ["shared_graph_encodings"],
                   "implementation": "implicit-native", "unbound": True}
        for field, value in changes.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                native_execution.check_native_execution({**choice(), field: value})
        for missing in choice():
            value = choice()
            del value[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                native_execution.check_native_execution(value)

    def test_mutated_resource_snapshot_derivatives_or_worker_counts_are_rejected(self):
        changes = (("budget_cpus", 8), ("batch_size", 16), ("torch_intraop_threads", 2),
                   ("verification_workers", 1), ("native_prefix_threads", 4),
                   ("phases_run_concurrently", False), ("requested_cpu_cap", 32))
        for field, changed in changes:
            value = choice()
            value["resource_plan"][field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                native_execution.check_native_execution(value)
        for field in ("collection_workers", "verification_workers"):
            for changed in (True, 0, 2, 8):
                value = choice()
                value[field] = changed
                with self.subTest(field=field, value=changed), self.assertRaises(ValueError):
                    native_execution.check_native_execution(value)

    def test_manifest_binding_requires_absolute_path_digest_and_exact_worker_budget(self):
        for field, changed in (("manifest", "relative/manifest.json"), ("sha256", "not-a-hash"),
                               ("prefix_threads", True), ("prefix_threads", 8), ("prefix_threads", 0)):
            value = choice()
            value["native_learner"][field] = changed
            with self.subTest(field=field, value=changed), self.assertRaises(ValueError):
                native_execution.check_native_execution(value)


class NativeAcceptanceTests(unittest.TestCase):
    def test_transitive_acceptance_helpers_are_bound_before_official_selection(self):
        helpers = ('tools/benchmark_copied_training.py', 'tools/acceleration3/ppo_bench.py',
                   'tools/validate_garrison_regressions.py')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            subprocess.run(['git', 'init', '-q', str(root)], check=True, capture_output=True)
            for name in helpers:
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('original acceptance helper\n')
            subprocess.run(['git', '-C', str(root), 'add', '--', *helpers], check=True, capture_output=True)
            with patch.object(native_execution, '__file__', str(root/'experiments/learner/native_execution.py')), \
                    patch.object(native_backend, 'verified_manifest', return_value={}):
                identities = native_execution.acceptance_source_hashes()
                self.assertEqual(set(identities), set(helpers))
                report = dict(accepted_for_manual_training=True, native_manifest_sha256=SHA,
                              acceptance_source_sha256=identities)
                accepted = root/'validation/native-mac-learner-2026-10-29.json'
                accepted.parent.mkdir(); accepted.write_text(json.dumps(report))
                self.assertEqual(native_execution.require_native_acceptance(MANIFEST, SHA), {})
                for name in helpers:
                    path = root/name; original = path.read_bytes()
                    path.write_bytes(original+b'changed executable helper\n')
                    with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'acceptance gate'):
                        native_execution.require_native_acceptance(MANIFEST, SHA)
                    path.write_bytes(original)
                self.assertEqual(native_execution.require_native_acceptance(MANIFEST, SHA), {})

    def test_official_worker_cannot_bypass_acceptance_or_mutate_globals_before_it(self):
        from experiments.headless import background_check
        from experiments.learner import ppo, interaction_guard
        value = choice()
        names = ('PREFIX_HELPERS', 'REPLAY_TEXT_LEAVES', 'TEXT_REPLAY_WORKERS', 'EXPERIMENTS', 'NATIVE_EXECUTION')
        before = [getattr(ppo, name) for name in names] + [background_check.FORK, interaction_guard.ACTIVE_VERSION]
        with patch.object(native_backend, 'verified_manifest', return_value={}), \
                patch.object(Path, 'read_text', return_value=json.dumps({'accepted_for_manual_training': False})), \
                patch.object(local, 'restore_parent') as restore, \
                self.assertRaisesRegex(ValueError, 'acceptance gate'):
            local.worker_state({'execution': value, 'policy_support': None, 'parent_descriptor': {}})
        restore.assert_not_called()
        self.assertEqual([getattr(ppo, name) for name in names] +
                         [background_check.FORK, interaction_guard.ACTIVE_VERSION], before)

    def test_manual_choice_requires_matching_accepted_actual_build(self):
        accepted = {"accepted_for_manual_training": True, "native_manifest_sha256": SHA,
                    "acceptance_source_sha256": native_execution.acceptance_source_hashes()}
        with patch.object(native_backend, "verified_manifest", return_value={}) as verified, \
                patch.object(resources, "discover_resources", return_value=OBSERVED), \
                patch.object(Path, "read_text", return_value=json.dumps(accepted)) as report:
            value = native_execution.native_execution_choice(MANIFEST, SHA)
        verified.assert_called_once_with(MANIFEST, SHA)
        report.assert_called_once()
        self.assertEqual(value["native_learner"]["sha256"], SHA)

    def test_unaccepted_missing_or_other_build_report_fails_closed(self):
        reports = ({"accepted_for_manual_training": False, "native_manifest_sha256": SHA},
                   {"accepted_for_manual_training": True, "native_manifest_sha256": "b"*64},
                   {"accepted_for_manual_training": True, "native_manifest_sha256": SHA,
                    "acceptance_source_sha256": {"stale": "0"*64}}, {})
        for result in reports:
            with self.subTest(result=result), patch.object(native_backend, "verified_manifest", return_value={}), \
                    patch.object(resources, "discover_resources", return_value=OBSERVED), \
                    patch.object(Path, "read_text", return_value=json.dumps(result)), self.assertRaises(ValueError):
                native_execution.native_execution_choice(MANIFEST, SHA)
        with patch.object(native_backend, "verified_manifest", return_value={}), \
                patch.object(resources, "discover_resources", return_value=OBSERVED), \
                patch.object(Path, "read_text", side_effect=FileNotFoundError("acceptance absent")), \
                self.assertRaises(FileNotFoundError):
            native_execution.native_execution_choice(MANIFEST, SHA)

    def test_diagnostics_skip_acceptance_only_and_still_verify_binary_identity(self):
        with patch.object(native_backend, "verified_manifest", side_effect=ValueError("actual binary differs")), \
                patch.object(Path, "read_text") as report, self.assertRaisesRegex(ValueError, "actual binary differs"):
            native_execution.native_execution_choice(MANIFEST, SHA, require_acceptance=False)
        report.assert_not_called()


class NativeCommandTests(unittest.TestCase):
    def arguments(self, *extra):
        return ["run_local_training.py", "--config", "/private/synthetic-machine.json", "--profile", "original", *extra]

    def parse(self, *extra):
        with patch.object(controller.sys, "argv", self.arguments(*extra)), \
                patch.object(controller, "start", return_value=0) as start, \
                patch.object(controller.Path, "exists", return_value=True), \
                patch.object(controller, "read", return_value={"artifact_root": "/private/synthetic", "default_profile": "original"}), \
                patch.object(resources, "discover_resources", return_value=OBSERVED), patch.dict(controller.os.environ):
            controller.main()
        return start.call_args.args[0]

    def native_arguments(self):
        return ("--fast-path", "--native-learner", str(MANIFEST), "--native-sha256", SHA)

    def test_native_cli_selects_serial_aten_and_parallel_independent_phases(self):
        args = self.parse(*self.native_arguments(), "--cpu-cap", "8")
        self.assertEqual(args.cpu_cap, 8)
        self.assertEqual(args.collection_workers, 4)
        self.assertEqual((args.overlap_verification, args.text_leaf_replay, args.replay_threads), (False, False, 0))
        limited = self.parse(*self.native_arguments(), "--cpu-cap", "2")
        self.assertEqual(limited.collection_workers, 1)

    def test_native_flags_and_conflicting_modes_are_rejected_before_launch(self):
        cases = [("--native-learner", str(MANIFEST)), ("--fast-path", "--native-learner", str(MANIFEST)),
                 ("--native-sha256", SHA), ("--cpu-cap", "4")]
        cases += [(*self.native_arguments(), *extra) for extra in
                  (("--prefix-helpers", "1"), ("--overlap-verification",), ("--replay-threads", "2"),
                   ("--experiment", "shared_graph_encodings"), ("--cpu-cap", "0"), ("--cpu-cap", "17"))]
        for extra in cases:
            with self.subTest(extra=extra), patch.object(controller.sys, "argv", self.arguments(*extra)), \
                    patch.object(resources, "discover_resources", return_value=OBSERVED), \
                    patch.object(controller, "start") as start, redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                controller.main()
            self.assertEqual(error.exception.code, 2)
            start.assert_not_called()

    def test_selected_execution_keeps_acceptance_gate_and_exact_manifest(self):
        args = self.parse(*self.native_arguments(), "--cpu-cap", "8")
        with patch.object(native_execution, "native_execution_choice", return_value={"bound": True}) as selected:
            self.assertEqual(controller.selected_execution(args), {"bound": True})
        selected.assert_called_once_with(MANIFEST, SHA, cpu_cap=8)

    def test_frozen_source_reentry_preserves_native_build_and_cpu_cap(self):
        args = self.parse(*self.native_arguments(), "--cpu-cap", "8", "--hours", "1")
        config = {"artifact_root": "/private/synthetic", "run_root": "/private/synthetic/run"}
        release = Path("/private/synthetic/frozen-release")
        class Reentered(BaseException): pass
        with patch.object(controller, "read", return_value=config), \
                patch.dict(sys.modules, {"prepare_local_training": SimpleNamespace(validate_config=lambda value: value)}), \
                patch.object(controller, "campaign_locks", return_value=nullcontext()), \
                patch.object(controller, "exclusive_lock", return_value=nullcontext()), \
                patch.object(controller, "release_checkout", return_value=(release, "f"*40)), \
                patch.object(controller.os, "chdir"), patch.dict(controller.os.environ), \
                patch.object(controller.os, "execv", side_effect=Reentered) as launch, self.assertRaises(Reentered):
            controller.start(args)
        executable, command = launch.call_args.args
        self.assertEqual(executable, sys.executable)
        self.assertEqual(command[1:3], [str(release / "tools/run_local_training.py"), "_launch"])
        for flag, value in (("--native-learner", str(MANIFEST)), ("--native-sha256", SHA), ("--cpu-cap", "8")):
            self.assertEqual(command[command.index(flag)+1], value)
        self.assertIn("--no-text-replay", command)
        self.assertNotIn("--overlap-verification", command)


if __name__ == "__main__":
    unittest.main()
