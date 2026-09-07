"""Composition and explicit ablation modes. No labels enter ordinary inference."""

import math
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn

from ..config import ModelConfig
from .backbone import LightweightBackbone
from .common import DepthwiseSeparable
from .head import CenterHead
from .neck import LightweightFPN
from .refinement import ContextReader, ContextRefiner
from .router import RoutingOutput, WindowRouter, select_windows, validate_custom_indices
from .windows import feature_masks, gather_regions, scatter_cores


class ForwardMode(StrEnum):
    FEATURES = "features"
    BASE = "base"
    BASE_TINY = "base_tiny"
    FULL = "full"
    COVERAGE = "coverage"
    OBJECTNESS = "objectness"
    UTILITY = "utility"
    RANDOM = "random"
    DENSE = "dense"
    CUSTOM = "custom"


@dataclass
class RefinementOutput:
    indices: torch.Tensor
    base_logits: torch.Tensor
    refined_logits: torch.Tensor
    residual: torch.Tensor
    valid_mask: torch.Tensor


@dataclass
class ModelOutput:
    predictions: dict[str, torch.Tensor]
    base_predictions: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]
    mode: str
    tiny_logits: torch.Tensor | None = None
    routing: RoutingOutput | None = None
    refinement: RefinementOutput | None = None
    features: dict[str, torch.Tensor] | None = None


class BCRNet(nn.Module):
    def __init__(self, config=None, *, backbone=None, neck=None, head=None, reader=None, refiner=None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        ForwardMode(cfg.default_mode)
        self.backbone = backbone if backbone is not None else LightweightBackbone(cfg)
        self.neck = neck if neck is not None else LightweightFPN(cfg)
        self.adapters = nn.ModuleDict({f"e{i}": DepthwiseSeparable(cfg.width, cfg.width) for i in (2, 3, 4)})
        self.head = head if head is not None else CenterHead(cfg.width, cfg.num_classes)
        self.tiny_head = nn.Conv2d(cfg.width, 1, 1) if cfg.use_tiny else None
        if self.tiny_head is not None:
            nn.init.constant_(self.tiny_head.bias, math.log(0.01 / 0.99))
        self.router = WindowRouter(cfg)
        self.reader = reader if reader is not None else ContextReader(cfg)
        self.refiner = refiner if refiner is not None else ContextRefiner(cfg)

    def extract_features(self, images, valid_mask=None):
        if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError("images must be floating-point NCHW RGB")
        if any(n % self.config.input_divisor for n in images.shape[-2:]):
            raise ValueError(f"Image H/W must be multiples of {self.config.input_divisor}")
        if valid_mask is None:
            valid_mask = torch.ones_like(images[:, :1], dtype=torch.bool)
        if valid_mask.shape != images[:, :1].shape or valid_mask.device != images.device:
            raise ValueError("valid_mask must have shape B,1,H,W on the image device")
        features = self.backbone(images)
        pyramid = self.neck(features)
        features.update(pyramid)
        features.update({f"e{i}": self.adapters[f"e{i}"](pyramid[f"p{i}"]) for i in (2, 3, 4)})
        return features, feature_masks(valid_mask, features)

    def refine_windows(self, features, masks, base_p2, indices):
        """Public probe interface: reuse one dense pass during utility supervision."""
        b, k = indices.shape
        c = self.config.core_size
        grid_width = features["e2"].shape[-1] // c
        if k == 0:
            empty = base_p2.new_empty(b, 0, base_p2.shape[1], c, c)
            return RefinementOutput(
                indices,
                empty,
                empty,
                base_p2.new_empty(b, 0, self.config.width, c, c),
                torch.empty(b, 0, 1, c, c, dtype=torch.bool, device=base_p2.device),
            )
        query, memory, valid, core, core_mask = self.reader(features, masks, indices)
        residuals = []
        chunk = self.config.patch_chunk_size
        for start in range(0, b * k, chunk):
            stop = start + chunk
            residuals.append(
                self.refiner(
                    query[start:stop],
                    memory[start:stop],
                    valid[start:stop],
                    core[start:stop],
                    core_mask[start:stop],
                )
            )
        residual = torch.cat(residuals)
        refined = self.head(core + residual)
        base = gather_regions(base_p2, indices, grid_width, c)
        # The convolutional predictor bias must not change invalid core cells.
        refined = refined.reshape(b, k, -1, c, c)
        cmask = core_mask.reshape(b, k, 1, c, c)
        refined = torch.where(cmask, refined, base)
        return RefinementOutput(indices, base, refined, residual.reshape(b, k, -1, c, c), cmask)

    def forward(
        self,
        images,
        valid_mask=None,
        *,
        mode=None,
        budget=None,
        generator=None,
        window_indices=None,
        return_features=False,
    ):
        mode = ForwardMode(mode or self.config.default_mode)
        budget = self.config.budget if budget is None else budget
        if not isinstance(budget, int) or budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        features, masks = self.extract_features(images, valid_mask)
        if mode == ForwardMode.FEATURES:
            return ModelOutput({}, {}, masks, mode.value, features=features)
        base = {f"p{i}": self.head(features[f"e{i}"]) for i in (2, 3, 4)}
        out = ModelOutput(dict(base), base, masks, mode.value, features=features if return_features else None)
        if mode == ForwardMode.BASE:
            return out
        if self.tiny_head is not None:
            out.tiny_logits = self.tiny_head(features["e2"])
        if mode == ForwardMode.BASE_TINY:
            return out
        compute_utility = mode in {ForwardMode.FULL, ForwardMode.UTILITY, ForwardMode.CUSTOM}
        routing = self.router(features, masks, base, out.tiny_logits, compute_utility)
        if mode == ForwardMode.CUSTOM:
            if window_indices is None:
                raise ValueError(
                    "custom mode requires explicit window_indices; ordinary inference never uses GT"
                )
            validate_custom_indices(window_indices, routing.valid_windows)
            indices = window_indices
        else:
            if window_indices is not None:
                raise ValueError("Explicit windows are only allowed in custom mode")
            indices = select_windows(routing, budget, mode.value, self.config.coverage_fraction, generator)
        routing.indices = indices
        out.routing = routing
        if indices.shape[1]:
            out.refinement = self.refine_windows(features, masks, base["p2"], indices)
            out.predictions["p2"] = scatter_cores(
                base["p2"], out.refinement.refined_logits, indices, self.config.core_size
            )
        return out
