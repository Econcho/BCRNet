"""Frame-local address plans. No weights or feature values are cached across calls."""

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class TokenMap:
    """Bank rows describe source supports; inverse restores [B*K, tokens] ordering."""

    batch: torch.Tensor
    y: torch.Tensor
    x: torch.Tensor
    active: torch.Tensor
    inverse: torch.Tensor
    support: int

    @property
    def rows(self):
        return self.batch.numel()


@dataclass
class AccessPlan:
    indices: torch.Tensor
    core: TokenMap
    detail: TokenMap
    semantic: TokenMap | None
    core_offsets: torch.Tensor
    core_size: int
    halo: int
    e2_shape: tuple[int, int]
    shared: bool

    @property
    def slots(self):
        return self.indices.numel()

    def metadata(self):
        return {
            "slots": self.slots,
            "detail_references": self.detail.inverse.numel(),
            "detail_bank_rows": self.detail.rows,
            "semantic_references": 0 if self.semantic is None else self.semantic.inverse.numel(),
            "semantic_bank_rows": 0 if self.semantic is None else self.semantic.rows,
            "shared": self.shared,
            # Counts include padding/sentinel rows. Geometry statistics are reported separately.
            "count_scope": "allocated rows, including padding and possible sentinel",
        }


def _token_map(batch, y, x, active, *, side, support, shape, shared):
    """Integer key includes batch and full support origin, hence pooling phase."""
    count, tokens = y.shape
    batch = batch.expand_as(y)
    active = active.expand_as(y)
    height, width = shape
    # Entirely external supports are equivalent zero inputs; partially external ones are distinct.
    active = active & (y < height) & (x < width) & (y + support > 0) & (x + support > 0)
    if not shared:
        return TokenMap(
            batch.flatten(),
            y.flatten(),
            x.flatten(),
            active.flatten(),
            torch.arange(count * tokens, device=y.device).reshape(count, tokens),
            support,
        )
    margin = side * support + support
    encoded_h, encoded_w = height + 2 * margin, width + 2 * margin
    keys = (batch * encoded_h + y + margin) * encoded_w + x + margin
    keys = torch.where(active, keys, -1)
    unique, inverse = torch.unique(keys.flatten(), sorted=True, return_inverse=True)
    safe = unique.clamp_min(0)
    xx = safe.remainder(encoded_w) - margin
    yyb = torch.div(safe, encoded_w, rounding_mode="floor")
    yy = yyb.remainder(encoded_h) - margin
    bb = torch.div(yyb, encoded_h, rounding_mode="floor")
    return TokenMap(bb, yy, xx, unique >= 0, inverse.reshape(count, tokens), support)


class AccessPlanner:
    """Bounded cache of shape-only offsets, never indices, masks or feature values."""

    def __init__(self, capacity=8):
        self.capacity = capacity
        self.templates = OrderedDict()

    def build(self, indices, cfg, features, *, shared=False):
        key = (indices.device, tuple(indices.shape), cfg.core_size, cfg.halo)
        if key not in self.templates:
            self.templates[key] = _template(indices, cfg)
            if len(self.templates) > self.capacity:
                self.templates.popitem(last=False)
        self.templates.move_to_end(key)
        return build_access_plan(indices, cfg, features, shared=shared, template=self.templates[key])


def _template(indices, cfg):
    b, k = indices.shape
    template = {"batch": torch.arange(b, device=indices.device)[:, None].expand(b, k).reshape(-1, 1)}
    for side in {cfg.core_size, cfg.core_size + 2 * cfg.halo, (cfg.core_size + 2 * cfg.halo) // 4}:
        a = torch.arange(side, device=indices.device)
        template[side] = (a.repeat_interleave(side)[None], a.repeat(side)[None])
    return template


def build_access_plan(indices, cfg, features, *, shared=False, template=None):
    """indices must have passed the model's bounds/uniqueness validation or router."""
    if indices.ndim != 2 or indices.dtype != torch.long:
        raise ValueError("indices must be int64 [B,K]")
    b, k = indices.shape
    eh, ew = features["e2"].shape[-2:]
    if features["e2"].shape[0] != b or indices.device != features["e2"].device:
        raise ValueError("Feature and index batch/device mismatch")
    c, h = cfg.core_size, cfg.halo
    template = _template(indices, cfg) if template is None else template
    batch = template["batch"]
    safe = indices.clamp_min(0).reshape(-1, 1)
    row, col = torch.div(safe, ew // c, rounding_mode="floor"), safe.remainder(ew // c)
    active = indices.reshape(-1, 1) >= 0

    def grid(side, step, stride, origin):
        ay, ax = template[side]
        return row * stride + origin + ay * step, col * stride + origin + ax * step

    cy, cx = grid(c, 1, c, 0)
    core = _token_map(batch, cy, cx, active, side=c, support=1, shape=(eh, ew), shared=False)
    extent = c + 2 * h
    dy, dx = grid(extent, 1, c, -h)
    detail = _token_map(batch, dy, dx, active, side=extent, support=1, shape=(eh, ew), shared=shared)
    semantic = None
    if cfg.use_context:
        side = extent // 4
        sy, sx = grid(side, 2, c // 2, -h // 2)
        semantic = _token_map(
            batch,
            sy,
            sx,
            active,
            side=side,
            support=2,
            shape=features["e3"].shape[-2:],
            shared=shared,
        )
    offsets = (cy * ew + cx).reshape(b, k * c * c).clamp(0, eh * ew - 1)
    return AccessPlan(indices, core, detail, semantic, offsets, c, h, (eh, ew), shared)


def read_points(feature, batch, y, x, active):
    """Strided NHWC view + indexed read: never materializes a whole NHWC feature map."""
    height, width = feature.shape[-2:]
    inside = active & (y >= 0) & (x >= 0) & (y < height) & (x < width)
    values = feature.permute(0, 2, 3, 1)[batch, y.clamp(0, height - 1), x.clamp(0, width - 1)]
    return values * inside[..., None]


def read_support(feature, mapping, *, scale=1, support=1):
    offsets = torch.arange(support, device=feature.device)
    oy = offsets.repeat_interleave(support)[None]
    ox = offsets.repeat(support)[None]
    return read_points(
        feature,
        mapping.batch[:, None],
        mapping.y[:, None] * scale + oy,
        mapping.x[:, None] * scale + ox,
        mapping.active[:, None],
    )


def read_feature_mask(feature, mask, mapping, *, scale=1, support=1):
    """One address construction serves feature and mask consumers of the same support."""
    if support == 1:
        y, x = mapping.y[:, None] * scale, mapping.x[:, None] * scale
    elif support == 2:
        # Broadcast Python constants instead of launching arange/repeat for each consumer.
        y = torch.stack(
            (mapping.y * scale, mapping.y * scale, mapping.y * scale + 1, mapping.y * scale + 1), 1
        )
        x = torch.stack(
            (mapping.x * scale, mapping.x * scale + 1, mapping.x * scale, mapping.x * scale + 1), 1
        )
    else:
        raise ValueError("Supported preparation supports are 1 or 2")
    height, width = feature.shape[-2:]
    inside = mapping.active[:, None] & (y >= 0) & (x >= 0) & (y < height) & (x < width)
    y, x = y.clamp(0, height - 1), x.clamp(0, width - 1)
    batch = mapping.batch[:, None]
    values = feature.permute(0, 2, 3, 1)[batch, y, x]
    valid = mask.permute(0, 2, 3, 1)[batch, y, x] * inside[..., None]
    return values, valid.float()


def read_core(feature, plan):
    m = plan.core
    values = read_points(feature, m.batch, m.y, m.x, m.active)
    return (
        values.reshape(plan.slots, plan.core_size**2, feature.shape[1])
        .transpose(1, 2)
        .reshape(plan.slots, feature.shape[1], plan.core_size, plan.core_size)
    )


def write_cores(base, refined, plan, base_patches=None):
    """Delta scatter preserves reference rounding, nonselected cells and -1 slots."""
    b, channels, height, width = base.shape
    if plan.slots == 0:
        return base
    old = read_core(base, plan).reshape_as(refined) if base_patches is None else base_patches
    delta = (refined - old) * (plan.indices >= 0)[:, :, None, None, None]
    values = delta.permute(0, 2, 1, 3, 4).reshape(b, channels, -1)
    return (
        base.flatten(2)
        .scatter_add(2, plan.core_offsets[:, None].expand(-1, channels, -1), values)
        .reshape(b, channels, height, width)
    )
