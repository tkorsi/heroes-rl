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


def embedding_scatter(target, ids, values):
    """Accumulate text gradients while excluding the padding embedding row."""
    values = values.contiguous()
    width = values.shape[-1]
    rows = ids.numel()
    library().embedding_scatter(
        target,
        ids.contiguous(),
        values,
        width,
        rows,
        threads=rows * width,
        group_size=min(256, rows * width),
    )


def write_rows(table, rows, values):
    """Write unique rows in manual-autograd tree or no-grad prefix encodings.

    The recipe compiler assigns each node once per group; duplicate destinations
    would race and are not supported by this internal helper.
    """
    values = values.contiguous()
    count, width = values.shape
    if count == 0:
        return
    library().write_rows(
        table,
        rows,
        values,
        width,
        count,
        threads=count * width,
        group_size=min(256, count * width),
    )
