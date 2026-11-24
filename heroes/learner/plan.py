"""Batched execution plan for a compiled game history.

A history is a list of shared tree nodes plus, per frame, the roots the model
reads. ``plan`` groups those nodes by kind and depth so a backend can evaluate a
whole range of frames in a few batched operations.
"""

import math
from collections import defaultdict
import torch
from torch.nn import functional as F


def tensor(x, dtype=torch.long):
    return torch.tensor(x, dtype=dtype)


class History:
    """One game's weight-independent recipe: shared tree nodes and per-frame roots."""

    def __init__(self, nodes, frames):
        self.nodes = nodes
        self.frames = frames
        self.frame_count = len(frames)
        self.plans = {}

    def plan(self, start, end, decisions=()):
        key = (start, end, tuple(decisions))
        if key in self.plans:
            return self.plans[key]
        wanted = set(decisions)
        jobs = [i for i in range(start, end) if self.frames[i]["root"] >= 0]
        roots = []
        for job_index, i in enumerate(jobs):
            f = self.frames[i]
            roots.append((f["root"], job_index))
            if f["map"] is not None:
                roots.append((f["map"]["own"], job_index))
            if i in wanted:
                for c in f["conditionals"]:
                    roots.extend(
                        (root, job_index)
                        for root in (
                            c["roots"] if c["kind"] == "categorical" else [c["root"]]
                        )
                    )
        needed = set()
        depth = {}
        first_use = {}

        def visit(i, job_index):
            if i in needed:
                return depth[i]
            k, v, ch = self.nodes[i]
            dep = (
                0
                if k in ("text", "number")
                else 1 + max([visit(v, job_index)] + [visit(c, job_index) for c in ch])
            )
            needed.add(i)
            depth[i] = dep
            first_use[i] = job_index
            return dep

        # Roots are chronological, so the first visit is the earliest required job.
        for r, job_index in roots:
            visit(r, job_index)
        del visit
        ids = sorted(needed)
        local = {r: i for i, r in enumerate(ids)}
        zero = len(ids)
        groups = defaultdict(list)
        for r in ids:
            k, v, ch = self.nodes[r]
            size = len(v) if k == "text" else len(ch)
            bucket = -1 if size == 0 else (size - 1).bit_length()
            if k in ("text", "ordered") and size:
                bucket = 0
            if k not in ("text", "number", "ordered"):
                k = "pooled"
                bucket = 0
            groups[(depth[r], k, bucket)].append(r)
        compiled = []
        group_first_use = []
        for (_, k, _), rs in sorted(groups.items()):
            rs.sort(key=lambda r: (first_use[r], r))
            group_first_use.append([first_use[r] for r in rs])
            idx = tensor([local[r] for r in rs])
            data = {}
            if k == "text":
                length = max(len(self.nodes[r][1]) for r in rs)
                if length:
                    data = {
                        "ids": tensor(
                            [
                                [v + 1 for v in self.nodes[r][1]]
                                + [0] * (length - len(self.nodes[r][1]))
                                for r in rs
                            ]
                        ),
                        "lengths": tensor([len(self.nodes[r][1]) - 1 for r in rs]),
                        "rows": tensor(list(range(len(rs)))),
                    }
            elif k == "number":
                data = {"values": tensor([self.nodes[r][1] for r in rs], torch.float32)}
            else:
                length = max(len(self.nodes[r][2]) for r in rs)
                data = {"tags": tensor([local[self.nodes[r][1]] for r in rs])}
                if k == "pooled":
                    counts = [len(self.nodes[r][2]) for r in rs]
                    offsets = []
                    offset = 0
                    for count in counts:
                        offsets.append(offset)
                        offset += count
                    data.update(
                        flat=tensor([local[c] for r in rs for c in self.nodes[r][2]]),
                        offsets=tensor(offsets),
                        counts=tensor(counts),
                        scale=tensor([math.sqrt(n) for n in counts], torch.float32),
                    )
                elif length:
                    data.update(
                        children=tensor(
                            [
                                [local[c] for c in self.nodes[r][2]]
                                + [zero] * (length - len(self.nodes[r][2]))
                                for r in rs
                            ]
                        ),
                        lengths=tensor([len(self.nodes[r][2]) - 1 for r in rs]),
                        rows=tensor(list(range(len(rs)))),
                        scale=tensor(
                            [[math.sqrt(len(self.nodes[r][2]))] for r in rs],
                            torch.float32,
                        ),
                    )
            compiled.append((k, idx, data, len(rs)))
        maps = [
            self.frames[i]["map"] for i in jobs if self.frames[i]["map"] is not None
        ]
        map_positions = [
            j for j, i in enumerate(jobs) if self.frames[i]["map"] is not None
        ]
        map_plan = None
        map_reuse = []
        if maps:
            unique = []
            seen = {}
            for m in maps:
                map_key = tuple(
                    id(m[k])
                    for k in (
                        "terrain",
                        "road",
                        "flags",
                        "object_attributes",
                        "object_indices",
                    )
                )
                if map_key not in seen:
                    seen[map_key] = len(unique)
                    unique.append(m)
                map_reuse.append(seen[map_key])
            all_maps = maps
            maps = unique
            w, h = maps[0]["width"], maps[0]["height"]
            assert all((m["width"], m["height"]) == (w, h) for m in maps)
            size = w * h
            from itertools import accumulate

            delta = None
            if start == 0 and not decisions:
                from .map_plan import build_plan

                delta = build_plan(maps, w, h)
            map_plan = {
                "width": w,
                "height": h,
                "positions": tensor(map_positions),
                "own": tensor([local[m["own"]] for m in all_maps]),
                "reuse": tensor(map_reuse),
                "count": len(maps),
                "cpu_positions": map_positions,
                "cpu_reuse": map_reuse,
            }
            if delta is not None:
                # Prefix-only sparse CNNs do not consume dense per-map features.
                map_plan["delta"] = delta
            else:
                map_plan.update(
                    terrain=torch.stack([m["terrain"] for m in maps]),
                    road=torch.stack([m["road"] for m in maps]),
                    flags=torch.stack([m["flags"] for m in maps]),
                    attributes=torch.cat([m["object_attributes"] for m in maps]),
                    indices=torch.cat(
                        [m["object_indices"] + j * size for j, m in enumerate(maps)]
                    ),
                    attribute_ends=list(
                        accumulate(len(x["object_attributes"]) for x in maps)
                    ),
                )
        decision_plan = []
        jobindex = {i: j for j, i in enumerate(jobs)}
        mindex = {jobs[j]: map_reuse[i] for i, j in enumerate(map_positions)}
        for i in decisions:
            cs = []
            for c in self.frames[i]["conditionals"]:
                p = {**c}
                if c["kind"] == "categorical":
                    p["roots"] = tensor([local[r] for r in c["roots"]])
                else:
                    p["root"] = local[c["root"]]
                for name in ["positions", "indices", "numeric", "offsets"]:
                    if isinstance(p.get(name), torch.Tensor):
                        p[name] = p[name]
                cs.append(p)
            decision_plan.append((jobindex[i], mindex.get(i), cs))
        head_groups = defaultdict(list)
        for di, (j, mi, cs) in enumerate(decision_plan):
            for c in cs:
                size = (
                    len(c["roots"])
                    if c["kind"] == "categorical"
                    else (
                        2
                        if c["kind"] == "quantity"
                        else (
                            len(c["indices"])
                            if c["indices"] is not None
                            else len(c["positions"])
                        )
                    )
                )
                bucket = 0 if c["kind"] == "categorical" else (size - 1).bit_length()
                head_groups[(c["kind"], bucket)].append((di, j, mi, c, size))
        heads = []
        for (kind, bucket), items in sorted(head_groups.items()):
            width = (
                max(item[4] for item in items) if kind == "categorical" else 2**bucket
            )
            d = {
                "decision": tensor([i[0] for i in items]),
                "state": tensor([i[1] for i in items]),
                "selected": tensor([[i[3]["selected"]] for i in items]),
                "mask": tensor(
                    [
                        [0.0 if t < i[4] else -float("inf") for t in range(width)]
                        for i in items
                    ],
                    torch.float32,
                ),
            }
            if kind == "categorical":
                d["roots"] = torch.stack(
                    [F.pad(i[3]["roots"], (0, width - i[4]), value=zero) for i in items]
                )
                if any("window_memory" in conditional for _, _, _, conditional, _ in items):
                    memory = torch.zeros((len(items), width), dtype=torch.float32)
                    for row, (_, _, _, conditional, size) in enumerate(items):
                        if "window_memory" in conditional:
                            memory[row, :size] = tensor(conditional["window_memory"], torch.float32)
                    d["window_memory"] = memory
            else:
                d["roots"] = tensor([i[3]["root"] for i in items])
                if kind == "quantity":
                    d.update(
                        numeric=torch.stack([i[3]["numeric"] for i in items]),
                        offsets=torch.stack([i[3]["offsets"] for i in items]),
                    )
                else:
                    d.update(
                        cells=tensor([i[2] for i in items]),
                        positions=torch.stack([i[3]["positions"] for i in items]),
                        indices=torch.stack(
                            [
                                F.pad(
                                    (
                                        i[3]["indices"]
                                        if i[3]["indices"] is not None
                                        else torch.arange(i[4])
                                    ),
                                    (0, width - i[4]),
                                )
                                for i in items
                            ]
                        ),
                    )
            heads.append((kind, d))
        result = {
            "n": len(ids),
            "groups": compiled,
            "group_first_use": group_first_use,
            "jobs": jobs,
            "roots": tensor([local[self.frames[i]["root"]] for i in jobs]),
            "maps": map_plan,
            "decisions": decision_plan,
            "heads": heads,
            "decision_states": tensor([x[0] for x in decision_plan]),
        }
        # Pack constants of one dtype into one contiguous array each.
        found = []

        def collect(v):
            if isinstance(v, torch.Tensor):
                found.append(v)
            elif isinstance(v, dict):
                for x in v.values():
                    collect(x)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    collect(x)

        collect(result)
        del collect
        replacement = {}
        arenas = {}
        for dtype in (torch.int64, torch.int32, torch.float32):
            entries = [v for v in found if v.dtype == dtype]
            if not entries:
                continue
            packed = torch.cat([v.reshape(-1) for v in entries])
            arenas[dtype] = packed
            offset = 0
            for v in entries:
                replacement[id(v)] = packed[offset : offset + v.numel()].view(v.shape)
                offset += v.numel()

        def replace(v):
            if isinstance(v, torch.Tensor):
                return replacement[id(v)]
            if isinstance(v, dict):
                return {k: replace(x) for k, x in v.items()}
            if isinstance(v, list):
                return [replace(x) for x in v]
            if isinstance(v, tuple):
                return tuple(replace(x) for x in v)
            return v

        result = replace(result)
        del replace
        result["_arenas"] = arenas
        self.plans[key] = result
        return result
