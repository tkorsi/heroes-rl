"""Sparse object/flag CNN deltas for current-weight no-grad prefixes."""

from collections import Counter
from itertools import accumulate
import torch
from torch.nn import functional as F


def _long(values):
    return torch.tensor(values, dtype=torch.int64)


_KERNEL_INDEX_KEYS = (
    "first_group", "first_cell", "first_neighbors",
    "second_group", "second_cell", "second_neighbors", "second_pool",
)


def _compact_kernel_indices(plan):
    """Use exact int32 constants when all shader indices fit; retain int64 otherwise."""
    limit = torch.iinfo(torch.int32)
    if all(not plan[key].numel() or
           (int(plan[key].min()) >= limit.min and int(plan[key].max()) <= limit.max)
           for key in _KERNEL_INDEX_KEYS):
        for key in _KERNEL_INDEX_KEYS:
            plan[key] = plan[key].to(torch.int32)
    return plan


def build_plan(maps, width, height):
    """Compile sparse CPU metadata before the enclosing device-arena upload."""
    import numpy as np

    if width < 4 or height < 4 or width % 4 or height % 4:
        return None
    size = width * height
    cells = np.arange(size, dtype=np.int64)

    def neighbors(radius):
        offsets = np.arange(-radius, radius + 1, dtype=np.int64)
        dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
        yy = cells[:, None] // width + dy.reshape(1, -1)
        xx = cells[:, None] % width + dx.reshape(1, -1)
        valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
        return np.where(valid, yy * width + xx, size)

    neighbors9, neighbors25 = neighbors(1), neighbors(2)
    descriptors, descriptor_ids, descriptor_ends = [], {}, []
    attribute_cache, object_cache = {}, {}

    def object_recipe(m):
        array = m["object_attributes"].contiguous().numpy()
        attribute_key = array.tobytes()
        ids = attribute_cache.get(attribute_key)
        if ids is None:
            ids = []
            for row in array:
                row_key = row.tobytes()
                identity = descriptor_ids.get(row_key)
                if identity is None:
                    identity = len(descriptors)
                    descriptor_ids[row_key] = identity
                    descriptors.append(row.copy())
                ids.append(identity)
            ids = np.asarray(ids, dtype=np.int64)
            attribute_cache[attribute_key] = ids
        positions = m["object_indices"].contiguous().numpy()
        key = (attribute_key, positions.tobytes())
        counts = object_cache.get(key)
        if counts is None:
            counts = Counter(
                (int(position), int(identity))
                for position, identity in zip(positions, ids)
            )
            object_cache[key] = counts
        return ids, counts

    groups, group_lookup, group_ids, group_ends = [], {}, [], []
    anchor_counts, anchor_attribute_ids = [], []
    delta_flags, delta_ends = [], []
    object_targets, object_ids, object_scales, object_ends = [], [], [], []
    first_group, first_cell, first_neighbors, first_ends = [], [], [], []
    second_group, second_cell, second_neighbors, second_pool, second_ends = (
        [],
        [],
        [],
        [],
        [],
    )
    changed_count = first_count = second_count = 0
    for map_index, m in enumerate(maps):
        ids, counts = object_recipe(m)
        descriptor_ends.append(len(descriptors))
        key = (
            m["terrain"].contiguous().numpy().tobytes(),
            m["road"].contiguous().numpy().tobytes(),
        )
        group = group_lookup.get(key)
        if group is None:
            group = len(groups)
            group_lookup[key] = group
            groups.append(m)
            anchor_counts.append(counts)
            anchor_attribute_ids.append(ids)
        group_ids.append(group)
        group_ends.append(len(groups))
        flag_delta = np.subtract(m["flags"].numpy(), groups[group]["flags"].numpy())
        changed = np.any(flag_delta != 0, axis=1)
        # Compare multisets, not sets: several equal descriptors may occupy one
        # cell. Differences also cover moves, appearances, removals and owners.
        anchor = anchor_counts[group]
        differences = []
        if counts is not anchor:
            for pair in counts.keys() | anchor.keys():
                difference = counts.get(pair, 0) - anchor.get(pair, 0)
                if difference:
                    position, identity = pair
                    differences.append((position, identity, difference))
                    changed[position] = True
        positions = np.flatnonzero(changed)
        changed_lookup = np.zeros(size + 1, dtype=np.int64)
        changed_lookup[positions] = changed_count + np.arange(positions.size) + 1
        for position, identity, difference in differences:
            object_targets.append(int(changed_lookup[position]) - 1)
            object_ids.append(identity)
            object_scales.append(float(difference))
        delta_flags.append(flag_delta[positions])
        changed_count += positions.size
        delta_ends.append(int(changed_count))
        object_ends.append(len(object_targets))
        affected1 = np.unique(neighbors9[positions])
        affected1 = affected1[affected1 < size]
        affected2 = np.unique(neighbors25[positions])
        affected2 = affected2[affected2 < size]
        first_group.append(np.full(affected1.size, group, dtype=np.int64))
        first_cell.append(affected1)
        first_neighbors.append(changed_lookup[neighbors9[affected1]])
        first_lookup = np.zeros(size + 1, dtype=np.int64)
        first_lookup[affected1] = first_count + np.arange(affected1.size) + 1
        second_group.append(np.full(affected2.size, group, dtype=np.int64))
        second_cell.append(affected2)
        second_neighbors.append(first_lookup[neighbors9[affected2]])
        second_pool.append(
            map_index * 16
            + (affected2 // width // (height // 4)) * 4
            + affected2 % width // (width // 4)
        )
        first_count += affected1.size
        second_count += affected2.size
        first_ends.append(int(first_count))
        second_ends.append(int(second_count))

    if len(groups) >= len(maps) or second_count >= len(maps) * size:
        return None

    def packed(parts, dtype, *shape):
        data = np.concatenate(parts).astype(dtype, copy=False)
        result = torch.from_numpy(data)
        return result.reshape(*shape) if shape else result

    result = {
        "terrain": torch.stack([m["terrain"] for m in groups]),
        "road": torch.stack([m["road"] for m in groups]),
        "flags": torch.stack([m["flags"] for m in groups]).clone(),
        "descriptors": torch.from_numpy(
            np.stack(descriptors)
            if descriptors
            else np.empty((0, 67), dtype=np.float32)
        ),
        "descriptor_ends": descriptor_ends,
        "anchor_attribute_ids": packed(anchor_attribute_ids, np.int64),
        "indices": torch.cat(
            [m["object_indices"] + index * size for index, m in enumerate(groups)]
        ),
        "attribute_ends": list(accumulate(len(m["object_attributes"]) for m in groups)),
        "group_ids": _long(group_ids),
        "group_ends": group_ends,
        "delta_flags": packed(delta_flags, np.float32, -1, 5),
        "delta_ends": delta_ends,
        "object_targets": _long(object_targets),
        "object_ids": _long(object_ids),
        "object_scales": torch.tensor(object_scales, dtype=torch.float32),
        "object_ends": object_ends,
        "first_ends": first_ends,
        "second_ends": second_ends,
        "first_group": packed(first_group, np.int64),
        "first_cell": packed(first_cell, np.int64),
        "first_neighbors": packed(first_neighbors, np.int64, -1, 9),
        "second_group": packed(second_group, np.int64),
        "second_cell": packed(second_cell, np.int64),
        "second_neighbors": packed(second_neighbors, np.int64, -1, 9),
        "second_pool": packed(second_pool, np.int64),
    }
    return _compact_kernel_indices(result)


def encode(model, maps):
    """Recompute object embeddings, anchors and sparse deltas at current weights."""
    if torch.is_grad_enabled():
        raise ValueError("Prefix CNN deltas require no-grad evaluation")
    d = maps["delta"]
    count = maps["count"]
    width, height = maps["width"], maps["height"]
    size = width * height
    group_count = d["group_ends"][count - 1]
    first_count = d["first_ends"][count - 1]
    second_count = d["second_ends"][count - 1]
    attribute_count = d["attribute_ends"][group_count - 1]
    descriptor_count = d["descriptor_ends"][count - 1]
    delta_count = d["delta_ends"][count - 1]
    object_count = d["object_ends"][count - 1]
    weight = model.p["map_cnn.0.weight"]
    encoded = (
        model.linear(d["descriptors"][:descriptor_count], "object_encoder.0").tanh()
        if descriptor_count
        else weight.new_empty((0, 16))
    )
    objects = weight.new_zeros((group_count * size, 16))
    if attribute_count:
        objects.index_put_(
            (d["indices"][:attribute_count],),
            encoded[d["anchor_attribute_ids"][:attribute_count]],
            accumulate=True,
        )
    grid = torch.cat(
        [
            F.embedding(
                d["terrain"][:group_count], model.emb["terrain.weight"], padding_idx=0
            ),
            F.embedding(
                d["road"][:group_count], model.emb["road.weight"], padding_idx=0
            ),
            objects.reshape(group_count, size, 16),
            d["flags"][:group_count],
        ],
        -1,
    )
    inp = grid.reshape(group_count, height, width, 33).permute(0, 3, 1, 2)
    first = F.conv2d(inp, weight, model.p["map_cnn.0.bias"], padding=1)
    first_output = first.tanh()
    second = F.conv2d(
        first_output, model.p["map_cnn.2.weight"], model.p["map_cnn.2.bias"], padding=1
    )
    second_output = second.tanh()
    pooled = (
        F.adaptive_avg_pool2d(second_output, (4, 4))
        .permute(0, 2, 3, 1)
        .reshape(group_count, 16, 16)
    )
    pooled = pooled[d["group_ids"][:count]].clone().reshape(count * 16, 16)
    if first_count:
        # Input channels12..27 are encoded objects;28..32 are visible map flags.
        # Construct only cells that changed, and gather neighborhood indices.
        object_delta = weight.new_zeros((delta_count, 16))
        if object_count:
            object_delta.index_put_(
                (d["object_targets"][:object_count],),
                encoded[d["object_ids"][:object_count]]
                * d["object_scales"][:object_count, None],
                accumulate=True,
            )
        delta = torch.cat([object_delta, d["delta_flags"][:delta_count]], -1)
        delta = torch.cat([delta.new_zeros((1, 21)), delta], 0)
        from .metal_map_kernels import sparse_first, sparse_second

        first_cells = first.permute(0, 2, 3, 1).reshape(group_count * size, 16).contiguous()
        first_delta = sparse_first(delta, d["first_neighbors"][:first_count],
            d["first_group"][:first_count], d["first_cell"][:first_count],
            first_cells, weight, size)
        second_cells = second.permute(0, 2, 3, 1).reshape(group_count * size, 16).contiguous()
        sparse_second(first_delta, d["second_neighbors"][:second_count],
            d["second_group"][:second_count], d["second_cell"][:second_count],
            second_cells, model.p["map_cnn.2.weight"], d["second_pool"][:second_count],
            pooled, size, float((width // 4) * (height // 4)))
    flat = pooled.reshape(count, 4, 4, 16).permute(0, 3, 1, 2).reshape(count, 256)
    return model.linear(flat, "map_global.2").tanh()
