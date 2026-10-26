"""Benchmark plan binding: execution options come only from a plan that matches its campaign."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch  # noqa: F401  (the benchmark module imports the learner)
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Optional PyTorch is not installed") from error

from experiments.learner import bench_training as bench

EXPERIMENT = ["shared_graph_encodings"]


class BenchmarkPlanBindingTests(unittest.TestCase):
    def options(self, context, plan):
        with patch.object(bench, "context", return_value=context):
            return bench.execution_options(plan)

    def test_execution_options_come_only_from_a_plan_that_matches_its_campaign(self):
        context = {"experiments": EXPERIMENT, "prefix_helpers": 3}
        plan = {"experiments": EXPERIMENT, "prefix_helpers": 3, "checkpoint_metadata": {"numerical_experiments": EXPERIMENT}}
        self.assertEqual(self.options(context, plan), (frozenset(EXPERIMENT), 3))
        self.assertEqual(self.options({}, {"checkpoint_metadata": {}}), (frozenset(), 0))
        for bad in ({**plan, "experiments": []}, {**plan, "checkpoint_metadata": {}}, {**plan, "prefix_helpers": 0},
                    {**plan, "prefix_helpers": 2}, {k: v for k, v in plan.items() if k != "experiments"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.options(context, bad)
        # An exact campaign rejects a plan that selects the experiment, and unknown choices are rejected.
        with self.assertRaisesRegex(ValueError, "Numerical experiment selection changed"):
            self.options({}, plan)
        for context in ({"experiments": ["other"]}, {"prefix_helpers": 16}, {"prefix_helpers": 1.0}):
            with self.subTest(context=context), self.assertRaises(ValueError):
                self.options(context, {"checkpoint_metadata": {}})

    def test_the_worker_reader_requires_the_benchmark_scope_and_the_exact_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            folder = base / "batches/train-0010-seat0"
            folder.mkdir(parents=True)
            path = folder / "plan.json"
            def write(plan):
                path.write_text(json.dumps(plan))
                return hashlib.sha256(path.read_bytes()).hexdigest()
            plan = {"scope": "heroes-official", "execution_output": str(folder / "execution")}
            with patch.object(bench, "BASE", base):
                with self.assertRaisesRegex(ValueError, "Unreleased source"):
                    bench.read_plan(path, write(plan), check=False)
                sha = write({**plan, "scope": bench.SCOPE})
                self.assertEqual(bench.read_plan(path, sha, check=False)["scope"], bench.SCOPE)
                with self.assertRaisesRegex(ValueError, "Plan changed"):
                    bench.read_plan(path, "0" * 64, check=False)
                with patch.object(bench, "check_identity") as checked:
                    bench.read_plan(path, sha)
                    bench.read_plan(path, sha, check=False)
                self.assertEqual(checked.call_count, 1)


if __name__ == "__main__":
    unittest.main()
