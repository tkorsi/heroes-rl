"""Current-weight sparse prefix CNNs using the shared full-D Metal kernels.

Only no-gradient recurrent prefixes use this path. Loss-window maps keep the
ordinary differentiable CNN, exactly as in the production Metal implementation.
No learned activation is cached across optimizer steps.
"""
from pathlib import Path


_library = None


def _kernels():
    global _library
    if _library is None:
        from .mlx_kernels import KernelLibrary
        _library = KernelLibrary(Path(__file__).resolve().parents[2] / "native/learner/mlx/map.json")
    return _library


def encode(backend, parameters, maps):
    """Return current-weight global features for the selected unique prefix maps."""
    mx = backend.mx
    d = maps["delta"]
    count = maps["count"]
    width, height = maps["width"], maps["height"]
    if not count or width < 4 or height < 4 or width % 4 or height % 4:
        raise ValueError("Sparse prefix CNN requires nonempty maps divisible into 4x4 bins")
    size = width * height
    group_count = d["group_ends"][count - 1]
    first_count = d["first_ends"][count - 1]
    second_count = d["second_ends"][count - 1]
    attribute_count = d["attribute_ends"][group_count - 1]
    descriptor_count = d["descriptor_ends"][count - 1]
    delta_count = d["delta_ends"][count - 1]
    object_count = d["object_ends"][count - 1]
    encoded = (mx.tanh(backend.linear(parameters, d["descriptors"][:descriptor_count], "object_encoder.0"))
               if descriptor_count else mx.zeros((0, 16), dtype=mx.float32))
    objects = mx.zeros((group_count * size, 16), dtype=mx.float32)
    if attribute_count:
        objects = objects.at[d["indices"][:attribute_count]].add(
            encoded[d["anchor_attribute_ids"][:attribute_count]])
    grid = mx.concatenate((
        backend.embedding(parameters, "terrain")[d["terrain"][:group_count]],
        backend.embedding(parameters, "road")[d["road"][:group_count]],
        objects.reshape(group_count, size, 16), d["flags"][:group_count]), axis=-1)
    first = (mx.conv2d(grid.reshape(group_count, height, width, 33),
                      mx.transpose(parameters["map_cnn.0.weight"], (0, 2, 3, 1)), padding=1)
             + parameters["map_cnn.0.bias"])
    second = (mx.conv2d(mx.tanh(first),
                       mx.transpose(parameters["map_cnn.2.weight"], (0, 2, 3, 1)), padding=1)
              + parameters["map_cnn.2.bias"])
    pooled = mx.mean(mx.tanh(second).reshape(group_count, 4, height // 4, 4, width // 4, 16), axis=(2, 4))
    pooled = pooled.reshape(group_count, 16, 16)[d["group_ids"][:count]].reshape(count * 16, 16)
    if first_count:
        object_delta = mx.zeros((delta_count, 16), dtype=mx.float32)
        if object_count:
            object_delta = object_delta.at[d["object_targets"][:object_count]].add(
                encoded[d["object_ids"][:object_count]] * d["object_scales"][:object_count, None])
        delta = mx.concatenate((object_delta, d["delta_flags"][:delta_count]), axis=-1)
        delta = mx.concatenate((mx.zeros((1, 21), dtype=mx.float32), delta), axis=0)
        library = _kernels()
        first_delta = library.call(
            "sparse_first_i32",
            [delta, d["first_neighbors"][:first_count], d["first_group"][:first_count],
             d["first_cell"][:first_count], first.reshape(group_count * size, 16), parameters["map_cnn.0.weight"]],
            [(first_count + 1, 16)], (first_count * 16, 1, 1), (min(256, first_count * 16), 1, 1),
            template=[("SIZE", size), ("COUNT", first_count)], init_value=0)[0]
        if second_count:
            # MLX outputs cannot alias and mutate an input. Accumulate the same
            # per-cell corrections into zero, then add current-weight anchors.
            pooled_delta = library.call(
                "sparse_second_i32",
                [first_delta, d["second_neighbors"][:second_count], d["second_group"][:second_count],
                 d["second_cell"][:second_count], second.reshape(group_count * size, 16),
                 parameters["map_cnn.2.weight"], d["second_pool"][:second_count]],
                [(count * 16, 16)], (second_count * 16, 1, 1), (min(256, second_count * 16), 1, 1),
                template=[("SIZE", size), ("COUNT", second_count), ("AREA", (width // 4) * (height // 4))],
                init_value=0)[0]
            pooled = pooled + pooled_delta
    flat = mx.transpose(pooled.reshape(count, 4, 4, 16), (0, 3, 1, 2)).reshape(count, 256)
    return mx.tanh(backend.linear(parameters, flat, "map_global.2"))
