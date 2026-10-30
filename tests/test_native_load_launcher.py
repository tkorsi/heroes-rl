"""User-facing native load diagnostics; every benchmark execution is mocked."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.learner import native_execution
from tools import benchmark_native_training as benchmark
from tools import test_native_load as launcher


SHA = "a" * 64


def evidence():
    request = {"resources": {"budget_cpus": 8, "resources": {"available_cpus": 16}}}
    names = ("before_full_state_sha256", "model_sha256", "gradient_sha256", "adam_sha256",
             "generator_sha256", "full_state_sha256", "sampler_order", "report",
             "state_parts_sha256", "audit", "parent", "checkout_head", "driver")
    summary = {"status": "passed", "exact": {name: True for name in names},
               "attempts": [{"path": name, "status": "completed", "returncode": 0,
                             "owned_cleanup_verified": True} for name in ("native", "python")],
               "native_seconds": 2.0, "python_seconds": 4.0, "kernel_speedup": 2.0}
    return request, summary


class NativeLoadResultTests(unittest.TestCase):
    def test_numerical_result_needs_no_runtime_telemetry(self):
        result = launcher.load_result(*evidence())
        self.assertEqual(result["available_cpus"], 16)
        self.assertEqual(result["cpu_budget"], 8)
        self.assertFalse(result["performance_telemetry"])
        self.assertNotIn("native_mean_cpu_cores", result)
        self.assertNotIn("native_peak_concurrent_frame_workers", result)
        self.assertEqual(result["native_ppo_seconds"], 2.0)
        self.assertEqual(result["exact_state_checks_passed"], 13)
        self.assertEqual(result["official_updates"], 0)
        self.assertEqual(result["seed_reservations"], 0)

    def test_failed_or_incomplete_exact_comparison_cannot_be_reported_as_passed(self):
        mutations = (lambda s: s.update(status="failed"),
                     lambda s: s["exact"].update(adam_sha256=False),
                     lambda s: s["exact"].pop("parent"),
                     lambda s: s["attempts"][1].update(owned_cleanup_verified=False),
                     lambda s: s.update(attempts=[]),
                     lambda s: s["attempts"][1].update(path="native"),
                     lambda s: s["attempts"][1].update(status="failed"),
                     lambda s: s["attempts"][1].update(returncode=7))
        for mutate in mutations:
            request, summary = evidence()
            mutate(summary)
            with self.subTest(summary=summary), self.assertRaises(ValueError):
                launcher.load_result(request, summary)


class NativeLoadLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.home = self.base / "user"
        self.root = self.base / "checkout"
        self.artifacts = self.home / "Projects/Heroes-artifacts"
        self.output_root = self.artifacts / "native-load-tests"
        (self.root / "validation").mkdir(parents=True)
        (self.root / "validation/native-mac-learner-2026-10-29.json").write_text(
            json.dumps({"native_manifest_sha256": SHA}))
        self.batch = self.artifacts / (
            "local-script-training/original/run-20261028T192106Z-44be2ecc/"
            "batches/train-000846-seat0/execution")
        self.batch.parent.mkdir(parents=True)
        (self.batch.parent / "plan.json").write_text("{}")
        self.selected = {"resource_plan": evidence()[0]["resources"]}
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.patches = (
            patch.object(launcher, "ROOT", self.root),
            patch.object(Path, "home", return_value=self.home),
            patch.object(native_execution, "native_execution_choice", return_value=self.selected),
            patch.object(benchmark, "main"),
        )
        self.root_patch, self.home_patch, self.accept, self.run = [p.start() for p in self.patches]
        for p in self.patches:
            self.addCleanup(p.stop)

    def invoke(self, *args):
        with redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            return launcher.main(list(args))

    def written_output(self):
        command = self.run.call_args.args[0]
        return Path(command[command.index("--output") + 1])

    def mock_completed_pair(self, command):
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        for name, value in zip(("request.json", "summary.json"), evidence()):
            (output / name).write_text(json.dumps(value))
        print("Preserved mocked pair evidence")
        return 0

    def test_acceptance_refusal_happens_before_work_or_output_creation(self):
        self.accept.side_effect = ValueError("native acceptance refused")
        with self.assertRaisesRegex(ValueError, "acceptance refused"):
            self.invoke()
        self.run.assert_not_called()
        self.assertFalse(self.output_root.exists())

    def test_unknown_arguments_cannot_execute_a_benchmark(self):
        with self.assertRaises(SystemExit) as failure:
            self.invoke("--hours", "10")
        self.assertEqual(failure.exception.code, 2)
        self.accept.assert_not_called()
        self.run.assert_not_called()
        self.assertFalse(self.output_root.exists())

    def test_missing_preserved_fixture_never_selects_another_run(self):
        (self.batch.parent / "plan.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "no replacement or current run"):
            self.invoke()
        self.run.assert_not_called()
        self.assertFalse(self.output_root.exists())

    def test_symlinked_output_cannot_write_console_inside_source_or_official_lineage(self):
        for forbidden in (self.root, self.artifacts / "local-script-training"):
            before = sorted(str(path.relative_to(forbidden)) for path in forbidden.rglob("*"))
            self.output_root.symlink_to(forbidden, target_is_directory=True)
            try:
                with self.subTest(forbidden=forbidden), self.assertRaisesRegex(ValueError, "outside source and official"):
                    self.invoke()
                self.run.assert_not_called()
                self.assertEqual(sorted(str(path.relative_to(forbidden)) for path in forbidden.rglob("*")), before)
            finally:
                self.output_root.unlink()

    def test_success_records_actual_capacity_and_preserves_exact_benchmark_parameters(self):
        self.run.side_effect = self.mock_completed_pair
        self.assertEqual(self.invoke("--cpu-cap", "8", "--order", "PN"), 0)
        command = self.run.call_args.args[0]
        for name, value in (("--batch", str(self.batch)), ("--native-manifest-sha256", SHA),
                            ("--wall-seconds", "120"), ("--cpu-cap", "8"), ("--order", "PN")):
            self.assertEqual(command[command.index(name) + 1], value)
        self.accept.assert_called_once_with(
            self.artifacts / "native-learner-builds/remove-telemetry-v1-attempt1/manifest.json", SHA, cpu_cap=8)
        output = self.written_output()
        result = json.loads((output / "load-report.json").read_text())
        self.assertFalse(result["performance_telemetry"])
        self.assertIn("Learner performance telemetry is disabled", self.stdout.getvalue())
        self.assertNotIn("Native mean CPU", self.stdout.getvalue())
        self.assertIn("Official updates and new seeds: 0", self.stdout.getvalue())
        self.assertIn("Preserved mocked pair evidence", output.with_suffix(".log").read_text())
        self.assertEqual(output.with_suffix(".log").stat().st_mode & 0o777, 0o600)

    def test_failure_preserves_original_exit_code_and_evidence_without_success_report(self):
        def failed_pair(command):
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            (output / "summary.json").write_text(json.dumps({"status": "failed", "error": "test failure"}))
            print("original failure diagnostic")
            return 7
        self.run.side_effect = failed_pair
        self.assertEqual(self.invoke(), 7)
        output = self.written_output()
        self.assertFalse((output / "load-report.json").exists())
        self.assertEqual(json.loads((output / "summary.json").read_text())["error"], "test failure")
        self.assertIn("original failure diagnostic", output.with_suffix(".log").read_text())
        self.assertIn("exit 7", self.stderr.getvalue())
        self.assertNotIn("checks passed", self.stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
