"""Read host capacity and plan independent training work without changing learning.

The plan is an execution budget, not a promise to keep every core busy. The
four-game batch, recurrent dependencies and ordered gradient accumulation stay
unchanged. Discovery never imports Torch, forks, or changes process environment.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import platform
import re
import subprocess


SCHEMA = "heroes-training-resources-v1"
PLAN_SCHEMA = "heroes-training-resource-plan-v1"


def _read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def _positive_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _cpu_set(value):
    """Parse the kernel's CPU-list syntax; reject incomplete or excessive input."""
    if not value:
        return None
    result = set()
    for part in value.split(","):
        if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?", part):
            return None
        bounds = [int(item) for item in part.split("-")]
        low, high = bounds[0], bounds[-1]
        if low > high or high >= 65536:
            return None
        result.update(range(low, high + 1))
    return result or None


def _unescape_mount(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _cgroup_constraints(proc_root):
    """Read the current cgroup and its ancestors, including delegated mounts."""
    membership = {}
    for line in (_read(proc_root / "self/cgroup") or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        for controller in parts[1].split(","):
            membership[controller] = parts[2]
    quotas, cpu_sets, memory_limits, memory_remaining = [], [], [], []
    observed = set()
    for line in (_read(proc_root / "self/mountinfo") or "").splitlines():
        halves = line.split(" - ", 1)
        if len(halves) != 2:
            continue
        before, after = halves[0].split(), halves[1].split()
        if len(before) < 6 or len(after) < 3 or after[0] not in ("cgroup", "cgroup2"):
            continue
        unified = after[0] == "cgroup2"
        controllers = {""} if unified else set(after[2].split(",")) & {"cpu", "cpuset", "memory"}
        root = Path(_unescape_mount(before[3]))
        mount = Path(_unescape_mount(before[4]))
        if not root.is_absolute() or not mount.is_absolute() or ".." in root.parts or ".." in mount.parts:
            continue
        for controller in controllers:
            member = membership.get(controller)
            if member is None:
                continue
            member_path = Path(member)
            if not member_path.is_absolute() or ".." in member_path.parts:
                continue
            try:
                relative = member_path.relative_to(root)
            except ValueError:
                # A cgroup namespace may expose membership relative to its root.
                if root != Path("/") and member_path == Path("/"):
                    relative = Path(".")
                else:
                    continue
            current = mount / relative
            while True:
                key = (str(current), controller)
                if key not in observed:
                    observed.add(key)
                    if unified or controller == "cpu":
                        if unified:
                            fields = (_read(current / "cpu.max") or "").split()
                            quota, period = (map(_positive_int, fields) if len(fields) == 2 else (None, None))
                        else:
                            quota = _positive_int(_read(current / "cpu.cfs_quota_us"))
                            period = _positive_int(_read(current / "cpu.cfs_period_us"))
                        if quota is not None and period is not None:
                            quotas.append(quota / period)
                    if unified or controller == "cpuset":
                        names = ("cpuset.cpus.effective", "cpuset.cpus") if unified else ("cpuset.effective_cpus", "cpuset.cpus")
                        for name in names:
                            values = _cpu_set(_read(current / name))
                            if values is not None:
                                cpu_sets.append(values)
                                break
                    if unified or controller == "memory":
                        limit = _positive_int(_read(current / ("memory.max" if unified else "memory.limit_in_bytes")))
                        usage = _read(current / ("memory.current" if unified else "memory.usage_in_bytes"))
                        if limit is not None:
                            memory_limits.append(limit)
                            if usage is not None and usage.isdigit():
                                memory_remaining.append(max(0, limit - int(usage)))
                if current == mount:
                    break
                current = current.parent
    cpus = set.intersection(*cpu_sets) if cpu_sets else None
    return {
        "cpu_quota_cores": min(quotas) if quotas else None,
        "cpu_set": cpus,
        "memory_limit_bytes": min(memory_limits) if memory_limits else None,
        "memory_remaining_bytes": min(memory_remaining) if memory_remaining else None,
    }


def _sysctl(name):
    try:
        value = subprocess.run(["/usr/sbin/sysctl", "-n", name], check=True,
                               capture_output=True, text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return _positive_int(value)


def discover_resources():
    """Return JSON-safe observed capacity, respecting Linux affinity and cgroups."""
    system = platform.system()
    logical = max(1, os.cpu_count() or 1)
    physical = performance = efficiency = memory = available_memory = None
    affinity = None
    if system != "Darwin" and hasattr(os, "sched_getaffinity"):
        try:
            affinity = set(os.sched_getaffinity(0)) or None
        except OSError:
            pass
    constraints = dict(cpu_quota_cores=None, cpu_set=None, memory_limit_bytes=None, memory_remaining_bytes=None)
    if system == "Darwin":
        logical = _sysctl("hw.logicalcpu") or logical
        physical = _sysctl("hw.physicalcpu")
        performance = _sysctl("hw.perflevel0.logicalcpu")
        efficiency = _sysctl("hw.perflevel1.logicalcpu")
        memory = _sysctl("hw.memsize")
    elif system == "Linux":
        proc = Path("/proc")
        constraints = _cgroup_constraints(proc)
        info = {}
        for line in (_read(proc / "meminfo") or "").splitlines():
            match = re.fullmatch(r"(MemTotal|MemAvailable):\s+(\d+) kB", line)
            if match:
                info[match[1]] = int(match[2]) * 1024
        memory, available_memory = info.get("MemTotal"), info.get("MemAvailable")
    candidates = [logical]
    allowed_cpus = constraints["cpu_set"]
    if affinity is not None:
        candidates.append(len(affinity))
        if allowed_cpus is not None:
            allowed_cpus = allowed_cpus & affinity
    if allowed_cpus is not None:
        candidates.append(len(allowed_cpus))
    quota = constraints["cpu_quota_cores"]
    if quota is not None:
        candidates.append(max(1, math.floor(quota)))
    capacity = max(1, min(candidates))
    memory_limits = [value for value in (memory, constraints["memory_limit_bytes"]) if value is not None]
    effective_memory = min(memory_limits) if memory_limits else None
    remaining = [value for value in (available_memory, constraints["memory_remaining_bytes"]) if value is not None]
    if remaining and effective_memory is not None:
        remaining.append(effective_memory)
    return {
        "schema": SCHEMA,
        "platform": system,
        "machine": platform.machine(),
        "logical_cpus": logical,
        "physical_cpus": physical,
        "performance_cpus": performance,
        "efficiency_cpus": efficiency,
        "affinity_cpus": sorted(affinity) if affinity is not None else None,
        "cgroup_cpus": sorted(constraints["cpu_set"]) if constraints["cpu_set"] is not None else None,
        "cgroup_cpu_quota_cores": quota,
        "available_cpus": capacity,
        "physical_memory_bytes": memory,
        "effective_memory_limit_bytes": effective_memory,
        "available_memory_bytes": min(remaining) if remaining else None,
        "memory_availability_kind": "observed_available" if remaining else "not_measured",
    }


def make_resource_plan(cpu_cap=None, *, resources=None, pipeline_verification=False):
    """Budget sequential or pipelined phases without changing the four-game batch."""
    if type(pipeline_verification) is not bool:
        raise ValueError("Verification pipelining must be an explicit boolean")
    resources = discover_resources() if resources is None else dict(resources)
    if resources.get("schema") != SCHEMA or type(resources.get("available_cpus")) is not int or resources["available_cpus"] < 1:
        raise ValueError("Expected a valid observed training-resource snapshot")
    available = resources["available_cpus"]
    if cpu_cap is not None and (type(cpu_cap) is not int or not 1 <= cpu_cap <= available):
        raise ValueError(f"CPU cap must be a positive integer no greater than available capacity ({available})")
    budget = available if cpu_cap is None else cpu_cap
    background = max(0, budget - 1)
    reducers = int(background >= 2)
    plan = {
        "schema": PLAN_SCHEMA,
        "resources": resources,
        "requested_cpu_cap": cpu_cap,
        "budget_cpus": budget,
        "batch_size": 4,
        "collection_workers": 4 if budget >= 4 else 1,
        "verification_workers": 4 if budget >= 4 else 1,
        "native_prefix_threads": budget,
        "native_frame_threads": budget,
        "native_text_replay_workers": background - reducers,
        "native_text_reducer_workers": reducers,
        "torch_intraop_threads": 1,
        "phases_run_concurrently": False,
        "memory_budget_enforced": False,
        "utilization_limits": [
            "Four-game on-policy batches cap independent collection and verification jobs at four.",
            "The accepted execution contract selects four workers or a single worker when the CPU budget is below four.",
            "Native prefixes and differentiable frame features use at most the declared budget and available independent frames.",
            "Text backward reserves one CPU for the main graph and, when at least three CPUs are available, one for ordered reduction; replay uses the remaining budget.",
            "Recurrent transitions, minibatch order and gradient accumulation retain their original dependencies.",
            "Memory figures describe observed capacity; no unmeasured per-worker memory guarantee is inferred.",
        ],
    }
    if pipeline_verification:
        # Four collectors, four isolated replay processes and their coordinator
        # may compute together. Smaller budgets retain the accepted serial phases.
        overlap = budget >= 9
        plan.update(schema="heroes-training-resource-plan-v2",
                    pipeline_verification=overlap,
                    collection_verification_cpu_budget=9 if overlap else max(
                        plan["collection_workers"], plan["verification_workers"]),
                    phases_run_concurrently=overlap)
        plan["utilization_limits"].append(
            "Collection and verification overlap only with room for four collectors, four replay workers and their coordinator; PPO starts after all verification and cleanup.")
    return plan
