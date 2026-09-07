"""Regular window geometry with gather-before-compute and nonoverlapping scatter."""

import torch
from torch.nn import functional as F


def window_view(x, core):
    b, c, h, w = x.shape
    if h % core or w % core:
        raise ValueError("Feature shape must divide the core size")
    return (
        x.reshape(b, c, h // core, core, w // core, core)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(b, (h // core) * (w // core), c, core, core)
    )


def pool_windows(x, mask, core, kind="mean"):
    patches = window_view(x, core)
    valid = window_view(mask, core).bool()
    if kind == "max":
        return patches.masked_fill(~valid, -torch.inf).flatten(-2).amax(-1).nan_to_num(neginf=0, posinf=0)
    if kind == "sum":
        return (patches * valid).sum((-2, -1))
    return (patches * valid).sum((-2, -1)) / valid.sum((-2, -1)).clamp_min(1)


def gather_regions(feature, indices, grid_width, core, halo=0):
    """Gather BK regions only. indices are B,K; -1 denotes a dummy slot."""
    b, channels, h, w = feature.shape
    count = indices.shape[1]
    extent = core + 2 * halo
    if count == 0:
        return feature.new_empty((b, 0, channels, extent, extent))
    start_y = indices.clamp_min(0).div(grid_width, rounding_mode="floor") * core - halo
    start_x = indices.clamp_min(0).remainder(grid_width) * core - halo
    offsets = torch.arange(extent, device=feature.device)
    yy = start_y[..., None, None] + offsets[None, None, :, None]
    xx = start_x[..., None, None] + offsets[None, None, None, :]
    yy, xx = torch.broadcast_tensors(yy, xx)
    inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w) & (indices[..., None, None] >= 0)
    locations = (yy.clamp(0, h - 1) * w + xx.clamp(0, w - 1)).flatten(1)
    gathered = feature.flatten(2).gather(2, locations[:, None].expand(-1, channels, -1))
    gathered = gathered.reshape(b, channels, count, extent, extent).permute(0, 2, 1, 3, 4)
    return gathered * inside[:, :, None].to(feature.dtype)


def scatter_cores(base, cores, indices, core):
    """B,C,H,W + B,K,C,m,m; dummy slots never overwrite position zero."""
    b, c, _h, w = base.shape
    k = indices.shape[1]
    if k == 0:
        return base
    gy, gx = indices.clamp_min(0) // (w // core), indices.clamp_min(0) % (w // core)
    y, x = torch.meshgrid(
        torch.arange(core, device=base.device), torch.arange(core, device=base.device), indexing="ij"
    )
    loc = ((gy[..., None, None] * core + y) * w + gx[..., None, None] * core + x).flatten(1)
    before = base.flatten(2).gather(2, loc[:, None].expand(-1, c, -1))
    values = cores.permute(0, 2, 1, 3, 4).reshape(b, c, -1)
    active = (indices >= 0)[..., None, None].expand(-1, -1, core, core).reshape(b, 1, -1)
    delta = (values - before) * active
    return base.flatten(2).scatter_add(2, loc[:, None].expand(-1, c, -1), delta).reshape_as(base)


def feature_masks(valid, features):
    return {
        name: F.adaptive_max_pool2d(valid.float(), value.shape[-2:]).bool()
        for name, value in features.items()
    }
