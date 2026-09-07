from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .windows import pool_windows


@dataclass
class RoutingOutput:
    coverage: torch.Tensor
    objectness: torch.Tensor
    utility: torch.Tensor | None
    valid_windows: torch.Tensor
    indices: torch.Tensor | None = None


class WindowRouter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.scene_projection = nn.Linear(cfg.width, cfg.scene_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.width + 5 + cfg.scene_dim, cfg.router_hidden),
            nn.SiLU(),
            nn.Linear(cfg.router_hidden, 1),
        )

    def forward(self, features, masks, predictions, tiny_logits, compute_utility=True):
        core = self.cfg.core_size
        mask = masks["e2"]
        scores = []
        for level in ("p2", "p3", "p4"):
            score = predictions[level][:, : self.cfg.num_classes].sigmoid().amax(1, keepdim=True)
            scores.append(F.interpolate(score, size=mask.shape[-2:], mode="nearest"))
        peaks = torch.cat([pool_windows(x, mask, core, "max") for x in scores], -1)
        objectness = peaks.amax(-1)
        tiny = (
            pool_windows(tiny_logits.sigmoid(), mask, core, "max").squeeze(-1)
            if tiny_logits is not None
            else torch.zeros_like(objectness)
        )
        coverage = torch.maximum(tiny, self.cfg.objectness_factor * objectness)
        valid_windows = pool_windows(mask.float(), mask, core, "sum").squeeze(-1) > 0
        utility = None
        if compute_utility:
            p = scores[0].float().clamp(1e-6, 1 - 1e-6)
            entropy = -p * p.log() - (1 - p) * (1 - p).log()
            local = torch.cat(
                (
                    pool_windows(features["e2"], mask, core),
                    peaks,
                    tiny.unsqueeze(-1),
                    pool_windows(entropy, mask, core),
                ),
                -1,
            ).detach()
            scene_mask = masks["t5"].to(features["t5"].dtype)
            scene = (features["t5"] * scene_mask).sum((-2, -1)) / scene_mask.sum((-2, -1)).clamp_min(1)
            # Detach the detector feature, NOT the trainable projection output.
            scene = self.scene_projection(scene.detach())
            desc = torch.cat((local, scene[:, None].expand(-1, local.shape[1], -1)), -1)
            utility = self.mlp(desc).squeeze(-1)
        return RoutingOutput(coverage, objectness, utility, valid_windows)


def ranked_indices(scores, valid, count):
    count = min(count, scores.shape[1])
    indices = scores.masked_fill(~valid, -torch.inf).argsort(dim=1, descending=True, stable=True)[:, :count]
    return indices.masked_fill(~valid.gather(1, indices), -1)


def select_windows(routing, budget, mode, coverage_fraction=0.5, generator=None):
    valid = routing.valid_windows
    b, total = valid.shape
    if mode == "dense":
        return torch.arange(total, device=valid.device)[None].expand(b, -1).masked_fill(~valid, -1)
    count = min(budget, total)
    if count == 0:
        return torch.empty(b, 0, dtype=torch.long, device=valid.device)
    if mode == "random":
        scores = torch.rand(valid.shape, generator=generator, device="cpu").to(valid.device)
        return ranked_indices(scores, valid, count)
    if mode in {"coverage", "objectness", "utility"}:
        scores = getattr(routing, mode)
        if scores is None:
            raise ValueError(f"{mode} scores were not computed")
        return ranked_indices(scores, valid, count)
    if mode != "full":
        raise ValueError(f"Unsupported selection mode {mode}")
    if routing.utility is None:
        raise ValueError("Full routing requires utility predictions")
    coverage_count = int(count * coverage_fraction)
    protected = ranked_indices(routing.coverage, valid, coverage_count)
    excluded = torch.zeros_like(valid, dtype=torch.long)
    excluded.scatter_add_(1, protected.clamp_min(0), (protected >= 0).long())
    gain = ranked_indices(routing.utility, valid & (excluded == 0), count - coverage_count)
    return torch.cat((protected, gain), 1)


def validate_custom_indices(indices, valid):
    if indices.ndim != 2 or indices.shape[0] != valid.shape[0] or indices.dtype != torch.long:
        raise ValueError("window_indices must be an int64 B,K tensor")
    if indices.device != valid.device or (indices < -1).any() or (indices >= valid.shape[1]).any():
        raise ValueError("Custom window indices have invalid bounds/device")
    for row in indices:
        used = row[row >= 0]
        if used.unique().numel() != used.numel():
            raise ValueError("Custom core indices must be unique per image")
    if ((indices >= 0) & ~valid.gather(1, indices.clamp_min(0))).any():
        raise ValueError("Custom windows cannot point to pure padding")
