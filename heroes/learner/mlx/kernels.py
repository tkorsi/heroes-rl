"""Shared full-D Metal kernels with functional MLX first-order gradients.

The JSON bodies are adaptations of the production Metal shaders, shared with
Swift. MLX values stay immutable: each manual tree backward adds a zero-filled
scatter delta to its logical gradient arena rather than mutating an input buffer.
"""
from __future__ import annotations

from bisect import bisect_left
import hashlib
import json
from pathlib import Path

TREE_NAMES = (
    "byte.weight", "text_gru.weight_ih_l0", "text_gru.weight_hh_l0",
    "text_gru.bias_ih_l0", "text_gru.bias_hh_l0", "number.0.weight",
    "number.0.bias", "tree.0.weight", "tree.0.bias", "ordered.weight_ih_l0",
    "ordered.weight_hh_l0", "ordered.bias_ih_l0", "ordered.bias_hh_l0",
)


class KernelLibrary:
    def __init__(self, path):
        import mlx.core as mx
        self.mx = mx
        self.path = Path(path)
        self.definitions = json.loads(self.path.read_text())["kernels"]
        self.compiled = {}
        self.constants = {}

    def call(self, name, inputs, shapes, grid, threadgroup, template=(), init_value=None, dtypes=None):
        # These are runtime scalar arguments in the production Metal kernels.
        # MLX template arguments would compile another shader for each prefix
        # length/map count, so carry the same uint values in a tiny input buffer.
        scalars = tuple((key, int(value)) for key, value in template)
        if scalars:
            if scalars not in self.constants:
                self.constants[scalars] = self.mx.array([value for _, value in scalars], dtype=self.mx.uint32)
            inputs = [*inputs, self.constants[scalars]]
        # MLX 0.31 selects constant/device argument address space at eight
        # elements. Reusing one name across those signatures invalidates its
        # cached pipeline while unretained command buffers may still reference
        # it. Give each argument ABI a stable name, independent of exact sizes.
        output_dtypes = dtypes or [self.mx.float32] * len(shapes)
        abi = tuple((str(value.dtype).split(".")[-1], value.size < 8, value.ndim)
                    for value in inputs)
        out_abi = tuple(str(value).split(".")[-1] for value in output_dtypes)
        signature = (name, tuple(key for key, _ in scalars), abi, out_abi)
        if signature not in self.compiled:
            d = self.definitions[name]
            aliases = "\n".join(f"const uint {key} = runtime_parameters[{index}];"
                                  for index, (key, _) in enumerate(scalars))
            suffix = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
            self.compiled[signature] = self.mx.fast.metal_kernel(
                name="heroes_" + name + "_" + suffix,
                input_names=d["input_names"] + (["runtime_parameters"] if scalars else []),
                output_names=d["output_names"], source=aliases + "\n" + d["source"],
                header=d.get("header", ""), atomic_outputs=d.get("atomic_outputs", False),
                ensure_row_contiguous=True)
        arguments = dict(inputs=inputs, output_shapes=shapes,
                         output_dtypes=output_dtypes,
                         grid=grid, threadgroup=threadgroup)
        if init_value is not None:
            arguments["init_value"] = init_value
        return self.compiled[signature](**arguments)


class OptimizedOps:
    def __init__(self):
        self.library = KernelLibrary(Path(__file__).with_name("gru_tree.json"))
        self.mx = self.library.mx

    def gru_forward(self, x, h, wi, wh, bi, bh, lengths=None):
        b, t, _ = x.shape
        width = h.shape[-1]
        if width not in (32, 64) or t < 1:
            raise ValueError("Shared Metal GRU requires hidden32/64 and a nonempty sequence")
        projected = x @ wi.T + bi
        name = "forward_gru_r16" if b <= 2 and t > 256 else "forward_gru_simd"
        inputs = [projected, wh, bh, h]
        if lengths is not None:
            name += "_lengths"
            inputs.append(lengths)
        return self.library.call(name, inputs, [(b, t + 1, width), (b, t, 4 * width)],
                                 (b * width * 16, 1, 1), (width * 16, 1, 1),
                                 [("T", t), ("H", width)], init_value=0 if lengths is not None else None)

    def gru_backward(self, incoming, x, states, gates, wi, wh, lengths=None):
        b, t, features = x.shape
        width = wh.shape[-1]
        inputs = [incoming, states, gates, wh]
        name = "backward_gru"
        if lengths is not None:
            inputs.append(lengths)
            name += "_lengths"
        dgi, dgh, dh = self.library.call(name, inputs,
            [(b, t, 3 * width), (b, t, 3 * width), (b, width)],
            (b * width, 1, 1), (width, 1, 1), [("T", t), ("H", width)],
            init_value=0 if lengths is not None else None)
        a, c = dgi.reshape(-1, 3 * width), dgh.reshape(-1, 3 * width)
        return (dgi @ wi, dh, a.T @ x.reshape(-1, features),
                c.T @ states[:, :-1].reshape(-1, width), a.sum(0), c.sum(0))

    def gru_prefix(self, x, h, wi, wh, bi, bh):
        """Full detached main-memory recurrence, without unused backward gates."""
        b, t, _ = x.shape
        if b != 1 or h.shape != (1, 64) or t < 1:
            raise ValueError("Main prefix GRU requires batch1, hidden64 and a nonempty sequence")
        if t <= 256:
            return self.mx.stop_gradient(self.gru(x, h, wi, wh, bi, bh))
        projected = x @ wi.T + bi
        states = self.library.call("prefix_gru_r16_all", [projected, wh, bh, h],
            [(b, t, 64)], (b * 1024, 1, 1), (1024, 1, 1), [("T", t), ("H", 64)])[0]
        return self.mx.stop_gradient(states)

    def gru_prefixes(self, x, offsets, lengths, wi, wh, bi, bh):
        """Detached main-memory recurrences of several sequences laid end to end in ``x``, each from a zero state.

        Sequence ``k`` is the rows ``offsets[k]`` to ``offsets[k] + lengths[k]``
        of ``x``. One kernel call runs them side by side, a thread group each,
        so the time is that of the longest and not of their sum. Returns the
        state after every row.
        """
        mx = self.mx
        count = offsets.shape[0]
        projected = x @ wi.T + bi
        states = self.library.call("prefix_gru_r16_ragged",
                                   [projected, wh, bh, mx.zeros((count, 64), dtype=mx.float32), offsets, lengths],
                                   [(x.shape[0], 64)], (count * 1024, 1, 1), (1024, 1, 1), [("H", 64)])[0]
        return mx.stop_gradient(states)

    def gru(self, x, h, wi, wh, bi, bh, lengths=None):
        mx = self.mx
        @mx.custom_function
        def recurrent(*args):
            return tuple(self.gru_forward(*args, lengths=lengths))
        @recurrent.vjp
        def backward(primals, cotangents, outputs):
            x0, h0, wi0, wh0, _, _ = primals
            states, gates = outputs
            grads = self.gru_backward(cotangents[0][:, 1:], x0, states, gates, wi0, wh0, lengths)
            return (grads[0], grads[1] + cotangents[0][:, 0], *grads[2:])
        states, _ = recurrent(x, h, wi, wh, bi, bh)
        return states[:, 1:]

    def _groups(self, plan, frame_limit):
        if frame_limit is None:
            return plan["groups"]
        groups = []
        for index, (kind, rows, values, count) in enumerate(plan["groups"]):
            count = bisect_left(plan["group_first_use"][index], frame_limit)
            if count == 0:
                continue
            rows = rows[:count]
            values = {key: value if key == "flat" else value[:count]
                      for key, value in values.items()}
            if kind == "pooled":
                end = sum(plan["_pool_counts"][index][:count])
                values["flat"] = values["flat"][:end]
            groups.append((kind, rows, values, count))
        return groups

    def tree(self, parameters, plan, frame_limit=None):
        mx = self.mx
        groups = self._groups(plan, frame_limit)
        width = 32
        def gru_parameters(p, name):
            return [p[name + "." + n + "_l0"] for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
        @mx.custom_function
        def forward(*values):
            p = dict(zip(TREE_NAMES, values))
            table = mx.zeros((plan["n"] + 1, width))
            saved = []
            for kind, rows, d, count in groups:
                if kind == "text":
                    if not d:
                        result = mx.zeros((count, width))
                    else:
                        x = p["byte.weight"][d["ids"]]
                        states, gates = self.gru_forward(x, mx.zeros((count, width)), *gru_parameters(p, "text_gru"), lengths=d["lengths"] + 1)
                        result = states[mx.arange(count), d["lengths"] + 1]
                        saved.extend((x, states, gates))
                elif kind == "number":
                    result = mx.tanh(d["values"] @ p["number.0.weight"].T + p["number.0.bias"])
                elif kind != "ordered":
                    result, inp = self.library.call("pooled_forward",
                        [table, d["flat"], d["tags"], d["offsets"], d["counts"], d["scale"], p["tree.0.weight"], p["tree.0.bias"]],
                        [(count, width), (count, 2 * width)], (count * width, 1, 1), (width, 1, 1), [("H", width)])
                    saved.append(inp)
                else:
                    if "children" in d:
                        x = table[d["children"]]
                        states, gates = self.gru_forward(x, mx.zeros((count, width)), *gru_parameters(p, "ordered"), lengths=d["lengths"] + 1)
                        pooled = states[mx.arange(count), d["lengths"] + 1]
                        saved.extend((x, states, gates))
                    else:
                        pooled = mx.zeros((count, width))
                    inp = mx.concatenate((table[d["tags"]], pooled), axis=-1)
                    result = mx.tanh(inp @ p["tree.0.weight"].T + p["tree.0.bias"])
                    saved.append(inp)
                table = table.at[rows].add(result)
            return (table, *saved)
        @forward.vjp
        def backward(values, cotangents, outputs):
            p = dict(zip(TREE_NAMES, values))
            table, *saved = outputs
            arena = cotangents[0]
            grads = {}
            def add(name, value):
                grads[name] = grads[name] + value if name in grads else value
            def recurrent(name, d, gradient, aux):
                x, states, gates = aux
                incoming = mx.zeros((x.shape[0], x.shape[1], width))
                incoming = incoming.at[mx.arange(x.shape[0]), d["lengths"]].add(gradient)
                wi, wh, _, _ = gru_parameters(p, name)
                dx, _, dwi, dwh, dbi, dbh = self.gru_backward(incoming, x, states, gates, wi, wh, d["lengths"] + 1)
                for suffix, value in zip(("weight_ih", "weight_hh", "bias_ih", "bias_hh"), (dwi, dwh, dbi, dbh)):
                    add(name + "." + suffix + "_l0", value)
                return dx
            for kind, rows, d, count in reversed(groups):
                gradient, result = arena[rows], table[rows]
                if kind == "text":
                    if not d:
                        continue
                    aux = saved[-3:]; del saved[-3:]
                    dx = recurrent("text_gru", d, gradient, aux)
                    embedding = self.library.call("embedding_scatter", [d["ids"], dx], [p["byte.weight"].shape],
                        (dx.size, 1, 1), (min(256, dx.size), 1, 1), [("width", dx.shape[-1]), ("rows", d["ids"].size)], init_value=0)[0]
                    add("byte.weight", embedding)
                elif kind == "number":
                    delta = gradient * (1 - result * result)
                    add("number.0.weight", delta.T @ d["values"])
                    add("number.0.bias", delta.sum(0))
                elif kind != "ordered":
                    inp = saved.pop()
                    scatter, delta = self.library.call("pooled_backward",
                        [gradient, result, d["flat"], d["tags"], d["offsets"], d["counts"], d["scale"], p["tree.0.weight"]],
                        [arena.shape, (count, width)], (count * width, 1, 1), (width, 1, 1), [("H", width)], init_value=0)
                    arena = arena + scatter
                    add("tree.0.weight", delta.T @ inp)
                    add("tree.0.bias", delta.sum(0))
                else:
                    inp = saved.pop()
                    delta = gradient * (1 - result * result)
                    add("tree.0.weight", delta.T @ inp)
                    add("tree.0.bias", delta.sum(0))
                    dx = delta @ p["tree.0.weight"]
                    arena = arena.at[d["tags"]].add(dx[:, :width])
                    if "children" in d:
                        aux = saved[-3:]; del saved[-3:]
                        children = recurrent("ordered", d, dx[:, width:], aux)
                        arena = arena.at[d["children"].reshape(-1)].add(children.reshape(-1, width))
            if saved:
                raise ValueError("Unconsumed tree gradient residuals")
            return tuple(grads.get(name, mx.zeros_like(value)) for name, value in zip(TREE_NAMES, values))
        return forward(*[parameters[name] for name in TREE_NAMES])[0]
