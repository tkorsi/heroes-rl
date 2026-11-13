import torch
from pathlib import Path

SRC = (
    Path(__file__).resolve().parents[2] / "native/learner/metal/tree.metal"
).read_text()
_lib = None


def library():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(SRC)
    return _lib


def forward(table, d, w, bias):
    B = d["tags"].numel()
    H = bias.numel()
    out = table.new_empty((B, H))
    inp = table.new_empty((B, 2 * H))
    library().pooled_forward(
        table,
        d["flat"],
        d["tags"],
        d["offsets"],
        d["counts"],
        d["scale"],
        w,
        bias,
        out,
        inp,
        H,
        threads=B * H,
        group_size=H,
    )
    return out, inp


def backward(arena, g, out, d, w):
    B, H = g.shape
    delta = g.new_empty(g.shape)
    library().pooled_backward(
        arena,
        g.contiguous(),
        out,
        d["flat"],
        d["tags"],
        d["offsets"],
        d["counts"],
        d["scale"],
        w,
        delta,
        H,
        threads=B * H,
        group_size=H,
    )
    return delta
