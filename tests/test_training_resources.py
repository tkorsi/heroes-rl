"""CPU constraints and execution budgets do not alter scientific batch semantics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.learner import resources


class TrainingResourcesTests(unittest.TestCase):
    def test_mac_topology_and_plan_are_serializable(self):
        values = {"hw.logicalcpu": 16, "hw.physicalcpu": 16, "hw.perflevel0.logicalcpu": 12,
                  "hw.perflevel1.logicalcpu": 4, "hw.memsize": 128 * 1024 ** 3}
        with patch.object(resources.platform, "system", return_value="Darwin"), \
             patch.object(resources, "_sysctl", side_effect=values.get):
            observed = resources.discover_resources()
        plan = resources.make_resource_plan(resources=observed)
        self.assertEqual(plan["budget_cpus"], 16)
        self.assertEqual(plan["collection_workers"], 4)
        self.assertEqual(plan["verification_workers"], 4)
        self.assertEqual(plan["native_prefix_threads"], 16)
        self.assertEqual(plan["torch_intraop_threads"], 1)
        self.assertEqual(plan["batch_size"], 4)
        self.assertEqual(plan["resources"]["performance_cpus"], 12)
        self.assertEqual(plan["resources"]["effective_memory_limit_bytes"], 128 * 1024 ** 3)
        self.assertEqual(json.loads(json.dumps(plan)), plan)

    def test_explicit_cap_limits_every_phase_and_rejects_invalid_caps(self):
        observed = {"schema": resources.SCHEMA, "available_cpus": 16}
        plan = resources.make_resource_plan(2, resources=observed)
        for name in ("collection_workers", "verification_workers"):
            self.assertEqual(plan[name], 1)
        self.assertEqual(plan["native_prefix_threads"], 2)
        for value in (0, -1, 17, True, 2.0, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resources.make_resource_plan(value, resources=observed)

    def test_cpu_list_parser_rejects_malformed_and_unbounded_ranges(self):
        self.assertEqual(resources._cpu_set("0-3,7,10-11"), {0, 1, 2, 3, 7, 10, 11})
        for value in (None, "", "1-0", "-1", "3,", "x", "0-100000000"):
            self.assertIsNone(resources._cpu_set(value))

    def fixture(self, directory, *, unified=True, mount_root="/"):
        proc, mount = Path(directory) / "proc", Path(directory) / "cgroup"
        (proc / "self").mkdir(parents=True)
        (mount / "child").mkdir(parents=True)
        group = "/tenant/child" if mount_root == "/tenant" else "/child"
        (proc / "self/cgroup").write_text(f"0::{group}\n" if unified else f"4:cpu,cpuset,memory:{group}\n")
        kind, opts = ("cgroup2", "rw") if unified else ("cgroup", "rw,cpu,cpuset,memory")
        (proc / "self/mountinfo").write_text(f"30 20 0:1 {mount_root} {mount} rw - {kind} cgroup {opts}\n")
        return proc, mount

    def test_cgroup_v2_parent_quota_and_remaining_memory_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, mount = self.fixture(directory)
            (mount / "cpu.max").write_text("250000 100000")
            (mount / "child/cpu.max").write_text("max 100000")
            (mount / "cpuset.cpus.effective").write_text("0-7")
            (mount / "child/cpuset.cpus.effective").write_text("2-5")
            (mount / "memory.max").write_text("1000")
            (mount / "memory.current").write_text("700")
            (mount / "child/memory.max").write_text("800")
            (mount / "child/memory.current").write_text("200")
            result = resources._cgroup_constraints(proc)
        self.assertEqual(result["cpu_quota_cores"], 2.5)
        self.assertEqual(result["cpu_set"], {2, 3, 4, 5})
        self.assertEqual(result["memory_limit_bytes"], 800)
        self.assertEqual(result["memory_remaining_bytes"], 300)

    def test_cgroup_v1_delegated_mount_is_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, mount = self.fixture(directory, unified=False, mount_root="/tenant")
            (mount / "cpu.cfs_quota_us").write_text("300000")
            (mount / "cpu.cfs_period_us").write_text("100000")
            (mount / "child/cpuset.cpus").write_text("1,3,5")
            (mount / "memory.limit_in_bytes").write_text("4096")
            result = resources._cgroup_constraints(proc)
        self.assertEqual(result["cpu_quota_cores"], 3)
        self.assertEqual(result["cpu_set"], {1, 3, 5})
        self.assertEqual(result["memory_limit_bytes"], 4096)

    def test_linux_affinity_intersects_cpuset_and_fractional_quota_has_one_worker(self):
        constraints = dict(cpu_quota_cores=3.8, cpu_set={2, 3, 4, 5}, memory_limit_bytes=4096,
                           memory_remaining_bytes=1024)
        with patch.object(resources.platform, "system", return_value="Linux"), \
             patch.object(resources.os, "cpu_count", return_value=16), \
             patch.object(resources.os, "sched_getaffinity", return_value={0, 1, 2, 3}, create=True), \
             patch.object(resources, "_cgroup_constraints", return_value=constraints), \
             patch.object(resources, "_read", return_value="MemTotal: 16 kB\nMemAvailable: 8 kB"):
            observed = resources.discover_resources()
            self.assertEqual(observed["available_cpus"], 2)
            self.assertEqual(observed["effective_memory_limit_bytes"], 4096)
            self.assertEqual(observed["available_memory_bytes"], 1024)
            constraints["cpu_quota_cores"] = 0.5
            self.assertEqual(resources.discover_resources()["available_cpus"], 1)


if __name__ == "__main__":
    unittest.main()
