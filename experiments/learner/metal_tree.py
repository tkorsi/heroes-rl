"""Batched semantic-tree gradients with one FP32 Metal accumulation arena."""

import torch
from torch.nn import functional as F
from .metal_gru import FusedGRU

NAMES = [
    "byte.weight",
    "text_gru.weight_ih_l0",
    "text_gru.weight_hh_l0",
    "text_gru.bias_ih_l0",
    "text_gru.bias_hh_l0",
    "number.0.weight",
    "number.0.bias",
    "tree.0.weight",
    "tree.0.bias",
    "ordered.weight_ih_l0",
    "ordered.weight_hh_l0",
    "ordered.bias_ih_l0",
    "ordered.bias_hh_l0",
]


class Context:
    def save_for_backward(self, *values):
        self.saved_tensors = values


class Tree(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan, *parameters):
        p = dict(zip(NAMES, parameters))
        f = p["tree.0.bias"].numel()
        table = parameters[0].new_zeros((plan["n"] + 1, f))
        saved = []
        for k, idx, d, count in plan["groups"]:
            aux = {}
            if k == "text":
                if not d:
                    result = table.new_zeros((count, f))
                else:
                    x = F.embedding(d["ids"], p["byte.weight"], padding_idx=0)
                    c = Context()
                    seq = FusedGRU.forward(
                        c,
                        x,
                        x.new_zeros((count, f)),
                        *[
                            p["text_gru." + n + "_l0"]
                            for n in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]
                        ]
                    )
                    result = seq[d["rows"], d["lengths"]]
                    aux = {"gru": c, "seqshape": seq.shape}
            elif k == "number":
                result = F.linear(
                    d["values"], p["number.0.weight"], p["number.0.bias"]
                ).tanh()
            elif k != "ordered":
                from .metal_tree_kernels import forward

                result, inp = forward(table, d, p["tree.0.weight"], p["tree.0.bias"])
                aux = {"input": inp, "fused": True}
            else:
                if "children" not in d:
                    pooled = table.new_zeros((count, f))
                else:
                    children = table[d["children"]]
                    if k == "ordered":
                        c = Context()
                        seq = FusedGRU.forward(
                            c,
                            children,
                            children.new_zeros((count, f)),
                            *[
                                p["ordered." + n + "_l0"]
                                for n in [
                                    "weight_ih",
                                    "weight_hh",
                                    "bias_ih",
                                    "bias_hh",
                                ]
                            ]
                        )
                        pooled = seq[d["rows"], d["lengths"]]
                        aux = {"gru": c, "seqshape": seq.shape}
                    else:
                        pooled = children.sum(1) / d["scale"]
                inp = torch.cat([table[d["tags"]], pooled], -1)
                result = F.linear(inp, p["tree.0.weight"], p["tree.0.bias"]).tanh()
                aux["input"] = inp
            table.index_copy_(0, idx, result)
            saved.append((k, idx, d, aux, result))
        ctx.plan = plan
        ctx.saved = saved
        ctx.save_for_backward(*parameters)
        return table

    @staticmethod
    def backward(ctx, incoming):
        if torch.is_grad_enabled():
            raise ValueError("Metal trees support first-order gradients only")
        p = dict(zip(NAMES, ctx.saved_tensors))
        f = p["tree.0.bias"].numel()
        grads = {}
        arena = incoming.contiguous().clone()

        def add(name, g):
            if name in grads:
                grads[name].add_(g)
            else:
                grads[name] = g

        def recurrent_back(name, d, aux, g):
            incoming = g.new_zeros(aux["seqshape"])
            incoming[d["rows"], d["lengths"]] = g
            dx, dh, dwi, dwh, dbi, dbh = FusedGRU.backward(aux["gru"], incoming)
            for n, v in zip(
                ["weight_ih", "weight_hh", "bias_ih", "bias_hh"], [dwi, dwh, dbi, dbh]
            ):
                add(name + "." + n + "_l0", v)
            return dx

        for k, idx, d, aux, result in reversed(ctx.saved):
            g = arena[idx]
            if k == "text":
                if not d:
                    continue
                dx = recurrent_back("text_gru", d, aux, g)
                embedding = p["byte.weight"].new_zeros(p["byte.weight"].shape)
                embedding.index_add_(
                    0, d["ids"].reshape(-1), dx.reshape(-1, dx.shape[-1])
                )
                embedding[0].zero_()
                add("byte.weight", embedding)
            elif k == "number":
                delta = g * (1 - result.square())
                add("number.0.weight", delta.T @ d["values"])
                add("number.0.bias", delta.sum(0))
            elif aux.get("fused"):
                from .metal_tree_kernels import backward

                delta = backward(arena, g, result, d, p["tree.0.weight"])
                add("tree.0.weight", delta.T @ aux["input"])
                add("tree.0.bias", delta.sum(0))
            else:
                delta = g * (1 - result.square())
                add("tree.0.weight", delta.T @ aux["input"])
                add("tree.0.bias", delta.sum(0))
                dx = delta @ p["tree.0.weight"]
                arena.index_add_(0, d["tags"], dx[:, :f])
                pooled = dx[:, f:]
                if "children" not in d:
                    continue
                if k == "ordered":
                    children = recurrent_back("ordered", d, aux, pooled)
                else:
                    children = (
                        (pooled / d["scale"])
                        .unsqueeze(1)
                        .expand(-1, d["children"].shape[1], -1)
                    )
                arena.index_add_(0, d["children"].reshape(-1), children.reshape(-1, f))
        return (None, *[grads.get(n) for n in NAMES])


def encode(plan, parameters):
    return Tree.apply(plan, *[parameters[n] for n in NAMES])
