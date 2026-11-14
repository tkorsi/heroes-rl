"""No-grad sparse CNN patch kernels without materialized im2col matrices."""

from pathlib import Path
import torch

_source = (Path(__file__).resolve().parents[2] / "native/learner/metal/map.metal").read_text()
_library = None


def library():
    global _library
    if _library is None:
        _library = torch.mps.compile_shader(_source)
    return _library


def sparse_first(delta, neighbors, groups, cells, anchors, weight, size):
    count = cells.numel()
    output = delta.new_zeros((count + 1, 16))
    kernel = library().sparse_first_i32 if neighbors.dtype == torch.int32 else library().sparse_first
    kernel(delta, neighbors, groups, cells, anchors, weight,
        output, size, count, threads=count * 16, group_size=min(256, count * 16))
    return output


def sparse_second(delta, neighbors, groups, cells, anchors, weight, bins, pooled, size, area):
    count = cells.numel()
    if count:
        kernel = library().sparse_second_i32 if neighbors.dtype == torch.int32 else library().sparse_second
        kernel(delta, neighbors, groups, cells, anchors, weight,
            bins, pooled, size, count, area, threads=count * 16, group_size=min(256, count * 16))
