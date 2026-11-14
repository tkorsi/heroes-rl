"""Experimental batched FP32 Metal backend; admission remains with the caller."""

import math
from collections import OrderedDict, defaultdict
import torch
from torch.nn import functional as F
from .metal_gru import gru

D = "mps"
FULL_PREFIX_PLAN = True
# Bound retained device constants; activations still use the MPS allocator limit.
PREFIX_PLAN_CACHE_BYTES = 4 * 1024**3


def tensor(x, dtype=torch.long):
    return torch.tensor(x, dtype=dtype)


def _upload_plan(plan):
    arenas = {dtype: values.to(D) for dtype, values in plan["_arenas"].items()}

    def move(value):
        if isinstance(value, torch.Tensor):
            start = value.storage_offset()
            return arenas[value.dtype][start : start + value.numel()].view(value.shape)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    result = move({key: value for key, value in plan.items() if key != "_arenas"})
    del move
    return result


class _History:
    def __init__(self, nodes, frames):
        self.nodes = nodes
        self.frames = frames
        self.frame_count = len(frames)
        self.plans = {}

    def plan(self, start, end, decisions=(), *, device=True):
        key = (start, end, tuple(decisions))
        if key in self.plans:
            result = self.plans[key]
            return _upload_plan(result) if device else result
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
                from .metal_map_delta import build_plan

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
        # Keep packed CPU constants so evicted device plans need no recompilation.
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
        return _upload_plan(result) if device else result


class NativeModel:
    def __init__(self, parameters, features, hidden):
        if (features, hidden) != (32, 64):
            raise ValueError("Metal learner currently requires feature32/hidden64")
        self.cpu = parameters
        self.f = features
        self.h = hidden
        self.p = {}
        self._prefix_plans = OrderedDict()
        self._prefix_plan_bytes = 0

    def begin_text_replay(self, *args):
        self.pending = []
        self.prefixes = {}
        self.evaluated = {}
        self.p = {k: v.to(D) for k, v in self.cpu.items()}
        # Preserve padding_idx=0 semantics despite this MPS embedding backward result.
        self.emb = {
            k: torch.cat([self.p[k][:1].detach(), self.p[k][1:]], 0)
            for k in ["byte.weight", "terrain.weight", "road.weight"]
        }

    def end_text_replay(self):
        self.p = {}
        self.emb = {}
        self.pending = []
        self.prefixes = {}
        self.evaluated = {}

    def close(self):
        self.end_text_replay()
        self._prefix_plans.clear()
        self._prefix_plan_bytes = 0

    def _prefix_plan(self, history, end):
        key = (history, end)
        if key in self._prefix_plans:
            self._prefix_plans.move_to_end(key)
            return self._prefix_plans[key][0]
        host = history.plan(0, end, device=False)
        size = sum(
            arena.numel() * arena.element_size() for arena in host["_arenas"].values()
        )
        while (
            self._prefix_plans
            and self._prefix_plan_bytes + size > PREFIX_PLAN_CACHE_BYTES
        ):
            entry = self._prefix_plans.popitem(last=False)
            self._prefix_plan_bytes -= entry[1][1]
            del entry
        plan = _upload_plan(host)
        if size <= PREFIX_PLAN_CACHE_BYTES:
            self._prefix_plans[key] = (plan, size)
            self._prefix_plan_bytes += size
        return plan

    def prepare_window_text(self, history, start, end, decisions, workers):
        self.pending.append((history, start, end, tuple(decisions)))

    def linear(self, x, n):
        return F.linear(x, self.p[n + ".weight"], self.p[n + ".bias"])

    def recurrent(self, x, name, initial=None):
        cell = name == "gru"
        h = self.h if cell else self.f
        if initial is None:
            initial = x.new_zeros((x.shape[0], h))
        suffix = "" if cell else "_l0"
        return gru(
            x,
            initial,
            *[
                self.p[name + "." + n + suffix]
                for n in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]
            ]
        )

    def encode(self, plan, frame_limit=None):
        from .metal_tree import encode
        from .metal_tree_kernels import write_rows

        if frame_limit is not None and torch.is_grad_enabled():
            raise ValueError("Prefix encodings require no-grad evaluation")
        tree_plan = plan
        if frame_limit is not None:
            import bisect

            groups = []
            for (kind, indices, data, _), first_use in zip(
                plan["groups"], plan["group_first_use"]
            ):
                count = bisect.bisect_left(first_use, frame_limit)
                if count:
                    groups.append(
                        (
                            kind,
                            indices[:count],
                            {
                                key: value if key == "flat" else value[:count]
                                for key, value in data.items()
                            },
                            count,
                        )
                    )
            tree_plan = {**plan, "groups": groups}
        table = encode(tree_plan, self.p)
        roots = table[
            plan["roots"] if frame_limit is None else plan["roots"][:frame_limit]
        ]
        glob = torch.zeros_like(roots)
        own = torch.zeros_like(roots)
        cells = None
        m = plan["maps"]
        if m and frame_limit is not None:
            import bisect

            nm = bisect.bisect_left(m["cpu_positions"], frame_limit)
            if nm:
                unique = max(m["cpu_reuse"][:nm]) + 1
                m = {
                    **m,
                    "count": unique,
                    **{k: m[k][:nm] for k in ("positions", "own", "reuse")},
                }
                if m.get("delta") is None:
                    na = m["attribute_ends"][unique - 1]
                    m.update(
                        **{k: m[k][:unique] for k in ("terrain", "road", "flags")},
                        **{k: m[k][:na] for k in ("attributes", "indices")},
                    )
            else:
                m = None
        if m and frame_limit is not None and m.get("delta") is not None:
            from .metal_map_delta import encode as encode_map_delta

            g = encode_map_delta(self, m)
            write_rows(glob, m["positions"], g[m["reuse"]])
            write_rows(own, m["positions"], table[m["own"]])
            return torch.cat([roots, glob, own], -1), table, None
        if m:
            size = m["width"] * m["height"]
            objects = roots.new_zeros((m["count"] * size, 16))
            objects.index_add_(
                0, m["indices"], self.linear(m["attributes"], "object_encoder.0").tanh()
            )
            grid = torch.cat(
                [
                    F.embedding(
                        m["terrain"], self.emb["terrain.weight"], padding_idx=0
                    ),
                    F.embedding(m["road"], self.emb["road.weight"], padding_idx=0),
                    objects.reshape(m["count"], size, 16),
                    m["flags"],
                ],
                -1,
            )
            inp = grid.reshape(m["count"], m["height"], m["width"], 33).permute(
                0, 3, 1, 2
            )
            if frame_limit is None:
                inp = inp.contiguous()
            a = F.conv2d(
                inp, self.p["map_cnn.0.weight"], self.p["map_cnn.0.bias"], padding=1
            ).tanh()
            a = F.conv2d(
                a, self.p["map_cnn.2.weight"], self.p["map_cnn.2.bias"], padding=1
            ).tanh()
            g = self.linear(
                F.adaptive_avg_pool2d(a, (4, 4)).flatten(1), "map_global.2"
            ).tanh()
            if frame_limit is not None:
                write_rows(glob, m["positions"], g[m["reuse"]])
                write_rows(own, m["positions"], table[m["own"]])
            else:
                glob.index_copy_(0, m["positions"], g[m["reuse"]])
                own.index_copy_(0, m["positions"], table[m["own"]])
            cells = a.permute(0, 2, 3, 1).reshape(m["count"], size, 16)
        return torch.cat([roots, glob, own], -1), table, cells

    def prefix(self, history, stops, workers):
        with torch.no_grad():
            plan_end = history.frame_count if FULL_PREFIX_PLAN else max(stops)
            plan = (
                self._prefix_plan(history, plan_end)
                if FULL_PREFIX_PLAN
                else history.plan(0, plan_end)
            )
            initial = torch.zeros((1, self.h), device=D)
            if not plan["jobs"]:
                result = {s: initial[0] for s in stops}
                self.prefixes[id(history)] = result
                return result
            import bisect

            count = bisect.bisect_left(plan["jobs"], max(stops))
            if count == 0:
                result = {s: initial[0] for s in stops}
                self.prefixes[id(history)] = result
                return result
            vectors, _, _ = self.encode(plan, count)
            states = self.recurrent(vectors[:count].unsqueeze(0), "gru", initial)[0]
            result = {
                s: (
                    states[bisect.bisect_left(plan["jobs"], s) - 1]
                    if bisect.bisect_left(plan["jobs"], s)
                    else initial[0]
                ).detach().clone()
                for s in stops
            }
            if not FULL_PREFIX_PLAN:
                history.plans.pop((0, plan_end, ()), None)
            self.prefixes[id(history)] = result
            return result

    def evaluate_window(self, history, start, end, decisions, initial, workers):
        key = (id(history), start, end, tuple(decisions))
        if not self.evaluated:
            self.evaluate_pending()
        return self.evaluated[key]

    def evaluate_pending(self):
        # Canonicalize only loss-window subgraphs across all selected histories.
        nodes = []
        frames = []
        shared = {}
        mapped = {}
        layout = []
        all_decisions = []

        def node(history, r):
            key = (id(history), r)
            if key in mapped:
                return mapped[key]
            k, v, ch = history.nodes[r]
            if k in ("text", "number"):
                value = v
                children = []
            else:
                value = node(history, v)
                children = [node(history, c) for c in ch]
            canonical = (
                k,
                tuple(float(x).hex() for x in value) if k == "number" else value,
                tuple(children),
            )
            found = shared.get(canonical)
            if found is None:
                found = len(nodes)
                nodes.append((k, value, children))
                shared[canonical] = found
            mapped[key] = found
            return found

        for history, start, end, decisions in self.pending:
            offset = len(frames)
            wanted = set(decisions)
            for i in range(start, end + 1):
                f = history.frames[i]
                root = -1 if f["root"] < 0 else node(history, f["root"])
                m = f["map"]
                if m is not None:
                    m = {**m, "own": node(history, m["own"])}
                cs = None
                if i in wanted:
                    cs = []
                    for c in f["conditionals"]:
                        c = {**c}
                        if c["kind"] == "categorical":
                            c["roots"] = [node(history, r) for r in c["roots"]]
                        else:
                            c["root"] = node(history, c["root"])
                        cs.append(c)
                frames.append({"root": root, "map": m, "conditionals": cs})
            positions = [offset + i - start for i in decisions]
            all_decisions.extend(positions)
            layout.append((history, start, end, decisions, offset, len(frames)))
        del node
        joined = _History(nodes, frames)
        plan = joined.plan(0, len(frames), all_decisions)
        vectors, table, cells = self.encode(plan)
        sequences = []
        last = []
        import bisect

        for history, start, end, decisions, left, right in layout:
            l = bisect.bisect_left(plan["jobs"], left)
            r = bisect.bisect_left(plan["jobs"], right)
            seq = self.recurrent(
                vectors[l:r].unsqueeze(0),
                "gru",
                self.prefixes[id(history)][start].detach().reshape(1, -1),
            )[0]
            sequences.append(seq)
            last.append(seq[-1])
        states = torch.cat(sequences, 0)
        logs, values = self.score(plan, states, table, cells)
        offset = 0
        for (history, start, end, decisions, _, _), final in zip(layout, last):
            count = len(decisions)
            self.evaluated[(id(history), start, end, decisions)] = (
                logs[offset : offset + count],
                values[offset : offset + count],
                final,
            )
            offset += count

    def score(self, plan, states, table, cells):
        decision_states = states[plan["decision_states"]]
        values = self.linear(decision_states, "value").squeeze(-1)
        query = (
            self.linear(decision_states, "policy_query").tanh()
            if any(kind == "categorical" for kind, _ in plan["heads"])
            else None
        )
        logs = states.new_zeros((len(plan["decisions"]),))
        for kind, d in plan["heads"]:
            if kind == "categorical":
                logits = (table[d["roots"]] * query[d["decision"]].unsqueeze(1)).sum(
                    -1
                ) / math.sqrt(self.f)
            elif kind == "quantity":
                ss = states[d["state"]]
                inp = torch.cat([ss, table[d["roots"]], d["numeric"]], -1)
                logits = (
                    self.linear(self.linear(inp, "quantity.0").tanh(), "quantity.2")
                    + d["offsets"]
                )
            else:
                ss = states[d["state"]]
                n = d["positions"].shape[1]
                inp = torch.cat(
                    [
                        ss.unsqueeze(1).expand(-1, n, -1),
                        table[d["roots"]].unsqueeze(1).expand(-1, n, -1),
                        cells[d["cells"]],
                        d["positions"],
                    ],
                    -1,
                )
                logits = (
                    self.linear(self.linear(inp, "coordinate.0").tanh(), "coordinate.2")
                    .squeeze(-1)
                    .gather(1, d["indices"])
                )
            lp = (logits + d["mask"]).log_softmax(1).gather(1, d["selected"]).squeeze(1)
            logs.index_add_(0, d["decision"], lp)
        return logs, values

    def backward_with_text_replay(self, loss):
        loss.backward()


def clipped_losses(
    log_probs,
    values,
    old_log_probs,
    old_values,
    advantages,
    returns,
    clip_ratio,
    value_clip,
    value_coefficient,
):
    from experiments.learner.ppo import clipped_losses, PPOConfig

    return clipped_losses(
        log_probs,
        values,
        old_log_probs,
        old_values,
        advantages,
        returns,
        PPOConfig(
            clip_ratio=clip_ratio,
            value_clip=value_clip,
            value_coefficient=value_coefficient,
        ),
    )


def checked_backend(reference):
    """Reuse the existing native recipe checks before owning immutable inputs."""
    from copy import deepcopy
    from types import SimpleNamespace

    if not torch.backends.mps.is_available() or not hasattr(
        torch.mps, "compile_shader"
    ):
        raise RuntimeError("Metal learner requires MPS and torch.mps.compile_shader")

    def history(nodes, frames):
        reference.NativeHistory(nodes, frames)
        nodes, frames = deepcopy((nodes, frames))
        return _History(nodes, frames)

    return SimpleNamespace(
        NativeHistory=history, NativeModel=NativeModel, clipped_losses=clipped_losses
    )


from contextlib import contextmanager


@contextmanager
def training_device(model, optimizer):
    """Keep the external checkpoint/Adam contract on CPU, including on failure."""
    from .native_recipes import require_native_model

    require_native_model(model)
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError("Metal learner requires the existing Adam optimizer")
    if not torch.backends.mps.is_available() or not hasattr(
        torch.mps, "compile_shader"
    ):
        raise RuntimeError("Metal learner requires MPS and torch.mps.compile_shader")
    original_foreach = [group["foreach"] for group in optimizer.param_groups]
    torch.mps.set_per_process_memory_fraction(0.25)
    try:
        model.to("mps")
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    values[key] = value.to("mps")
        for group in optimizer.param_groups:
            group["foreach"] = True
        yield
    finally:
        model.cpu()
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.cpu()
        for group, setting in zip(optimizer.param_groups, original_foreach):
            group["foreach"] = setting
