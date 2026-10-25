import unittest

from experiments.perf import contextual_benchmark as bench


class PerformanceBenchmarkTests(unittest.TestCase):
    def test_rows_are_deterministic_and_swap_policy_seeds_with_models(self):
        cfg = bench.config()
        rows = bench.rows(cfg, 4)
        self.assertEqual([row["engine_seed"] for row in rows], [140001, 140001, 140002, 140002])
        self.assertEqual([row["learner_player"] for row in rows], [0, 1, 0, 1])
        self.assertEqual(rows[0]["policy_seeds"], [140101, 140102])
        self.assertEqual(rows[1]["policy_seeds"], [140102, 140101])
        self.assertEqual([row["episode"] for row in rows], [1, 2, 3, 4])
        self.assertEqual(rows, bench.rows(cfg, 4))
        with self.assertRaises(ValueError):
            bench.rows(cfg, cfg["row_generator"]["max_rows"] + 1)

    def test_seeds_are_disjoint_from_scientific_cohorts(self):
        import json
        cfg = bench.config()
        rows = bench.rows(cfg, cfg["row_generator"]["max_rows"])
        used = {seed for row in rows for seed in (row["engine_seed"], *row["policy_seeds"])}
        scientific = set()
        for name in ("long-dialog-evaluation-v1.json", "builtin-opponent-evaluation-v1.json"):
            value = json.loads((bench.ROOT / "configs" / name).read_text())
            for key in ("engine_seeds", "model_policy_seeds", "learner_policy_seeds"):
                for item in value.get(key, []):
                    scientific.update(item if isinstance(item, list) else [item])
        self.assertTrue(scientific)
        self.assertFalse(used & scientific)

    def test_assignment_covers_each_row_once(self):
        plan = bench.assignment(7, 3)
        self.assertEqual(plan, {0: [0, 3, 6], 1: [1, 4], 2: [2, 5]})
        self.assertEqual(sorted(i for rows in plan.values() for i in rows), list(range(7)))

    def test_timer_passes_results_and_exceptions_through(self):
        timer = bench.StageTimer()

        def inner(value):
            if value < 0:
                raise KeyError(value)
            return value * 2

        timed = timer.timed(inner, "inner")
        self.assertEqual(timed(3), 6)
        with self.assertRaises(KeyError):
            timed(-1)
        self.assertEqual(timer.calls["inner"], 2)
        self.assertIs(timed.__perf_original__, inner)

    def test_nested_reentry_counts_wall_once(self):
        timer = bench.StageTimer()

        def recurse(n):
            return 0 if n == 0 else 1 + wrapped(n - 1)
        wrapped = timer.timed(recurse, "recurse")
        before = timer.snapshot()
        self.assertEqual(wrapped(5), 5)
        delta = bench.StageTimer.delta(before, timer.snapshot())
        self.assertEqual(delta["stages"]["recurse"]["calls"], 6)
        self.assertLessEqual(delta["stages"]["recurse"]["seconds"], delta["wall_seconds"] + 1e-6)

    def test_wrap_refuses_double_instrumentation(self):
        timer = bench.StageTimer()

        class Owner:
            def method(self):
                return 1
        timer.wrap(Owner, "method", "owner")
        self.assertEqual(Owner().method(), 1)
        with self.assertRaises(ValueError):
            timer.wrap(Owner, "method", "owner")

    def test_summary_keeps_outcome_denominators_separate(self):
        phase = {"wall_seconds": 2.0, "cpu_self_seconds": 1.0, "cpu_children_seconds": 0.5,
                 "stages": {"policy.act": {"seconds": 0.75, "calls": 10}}}
        attempts = [
            {"row": 0, "status": "passed", "episode_status": "truncated", "sampled_choices": 10,
             "actions": {"red": {"actions_accepted": 4, "actions_rejected": 1}, "blue": {"actions_accepted": 5}},
             "phases": {"collection": phase}, "environment": {"engine_wait_seconds": 0.5}},
            {"row": 1, "status": "passed", "episode_status": "completed", "sampled_choices": 6,
             "phases": {"collection": phase}},
            {"row": 2, "status": "native_failure", "episode_status": "failed", "sampled_choices": 3,
             "phases": {"collection": phase}},
            {"row": 3, "status": "not_started"},
        ]
        summary = bench.summarize_attempts(attempts, 10.0)
        self.assertEqual(summary["cutoffs_verified"], 1)
        self.assertEqual(summary["natural_endings_verified"], 1)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["not_started"], 1)
        self.assertEqual(summary["sampled_choices"], 19)
        self.assertEqual(summary["actions_accepted"], 9)
        self.assertEqual(summary["actions_rejected"], 1)
        self.assertEqual(summary["verified_attempts_per_hour"], 720.0)
        self.assertEqual(summary["totals_seconds"]["collection.policy.act"], 2.25)
        self.assertEqual(summary["totals_seconds"]["environment.engine_wait_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()
