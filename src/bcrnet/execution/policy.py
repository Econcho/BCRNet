"""Small, inspectable execution policy. Geometry selection explicitly includes a host sync."""

import json
from itertools import pairwise
from pathlib import Path

import torch


def rectangle_union(rectangles):
    if not rectangles:
        return 0
    xs = sorted({v for x0, _, x1, _ in rectangles for v in (x0, x1)})
    total = 0
    for left, right in pairwise(xs):
        spans = sorted((y0, y1) for x0, y0, x1, y1 in rectangles if x0 < right and x1 > left)
        length, end = 0, None
        for y0, y1 in spans:
            if end is None or y0 > end:
                length += y1 - y0
            else:
                length += max(0, y1 - end)
            end = y1 if end is None else max(end, y1)
        total += (right - left) * length
    return total


def geometry_summary(indices, cfg, e2_shape):
    """Exact geometric union of valid supports, independent of feature/GT values.

    The CPU transfer synchronizes CUDA. Callers must include that cost in adaptive
    timing. It avoids constructing a GPU unique map merely to reject sharing.
    """
    eh, ew = e2_shape
    c, h = cfg.core_size, cfg.halo
    side = c + 2 * h
    totals = {"detail_references": 0, "detail_unique": 0, "semantic_references": 0, "semantic_unique": 0}
    for row in indices.detach().cpu().tolist():
        detail, semantic = [], []
        for index in row:
            if index < 0:
                continue
            yy, xx = divmod(index, ew // c)
            d = (
                max(0, xx * c - h),
                max(0, yy * c - h),
                min(ew, xx * c - h + side),
                min(eh, yy * c - h + side),
            )
            detail.append(d)
            totals["detail_references"] += (d[2] - d[0]) * (d[3] - d[1])
            if cfg.use_context:
                # Pool support origins occupy a stride-two lattice with phase (-h/2) mod 2.
                phase = (-h // 2) % 2
                sy, sx = (yy * c // 2 - h // 2 - phase) // 2, (xx * c // 2 - h // 2 - phase) // 2
                low = -1 if phase else 0
                ymax, xmax = (eh // 2 - 1 - phase) // 2 + 1, (ew // 2 - 1 - phase) // 2 + 1
                s = (max(low, sx), max(low, sy), min(xmax, sx + side // 4), min(ymax, sy + side // 4))
                semantic.append(s)
                totals["semantic_references"] += (s[2] - s[0]) * (s[3] - s[1])
        totals["detail_unique"] += rectangle_union(detail)
        totals["semantic_unique"] += rectangle_union(semantic)
    for kind in ("detail", "semantic"):
        refs, unique = totals[f"{kind}_references"], totals[f"{kind}_unique"]
        totals[f"{kind}_ratio"] = refs / unique if unique else 1.0
    refs = totals["detail_references"] + totals["semantic_references"]
    unique = totals["detail_unique"] + totals["semantic_unique"]
    totals["ratio"] = refs / unique if unique else 1.0
    totals["scope"] = "geometric source supports, excludes external/dummy supports; not content-mask sparsity"
    return totals


def workload_signature(cfg, features, indices):
    feature = features["e2"]
    return {
        "device_type": feature.device.type,
        "device_name": torch.cuda.get_device_name(feature.device) if feature.is_cuda else "cpu",
        "torch_version": str(torch.__version__),
        "dtype": str(feature.dtype),
        "batch": indices.shape[0],
        "budget": indices.shape[1],
        "e2_shape": list(feature.shape[-2:]),
        "width": cfg.width,
        "stem_channels": cfg.stem_channels,
        "heads": cfg.heads,
        "blocks": cfg.blocks,
        "mlp_ratio": cfg.mlp_ratio,
        "patch_chunk_size": cfg.patch_chunk_size,
        "core_size": cfg.core_size,
        "halo": cfg.halo,
        "use_detail": cfg.use_detail,
        "use_context": cfg.use_context,
    }


class AdaptivePolicy:
    def __init__(self, cfg):
        self.threshold = cfg.overlap_threshold
        self.minimum = cfg.minimum_references
        self.saved = None
        if cfg.policy_path:
            self.saved = json.loads(Path(cfg.policy_path).read_text(encoding="utf-8"))
            if self.saved.get("format_version") != 1:
                raise ValueError("Unsupported execution policy format")
            if self.saved.get("choice") not in {"threshold", "packed", "shared", "indexed"}:
                raise ValueError("Invalid execution policy choice")
            if self.saved.get("attention_backend", "auto") != cfg.attention_backend:
                raise ValueError("Policy attention backend does not match ExecutionConfig")
            if self.saved.get("reuse_feature_masks", False) != cfg.reuse_feature_masks:
                raise ValueError("Policy mask baseline does not match ExecutionConfig")
            if self.saved["choice"] == "threshold":
                self.threshold = float(self.saved["threshold"])
                if self.threshold < 1:
                    raise ValueError("Invalid calibrated overlap threshold")

    def choose(self, indices, model_cfg, features):
        signature = workload_signature(model_cfg, features, indices)
        if self.saved:
            if signature != self.saved["signature"]:
                return "packed", {"reason": "calibration_signature_mismatch"}
            if self.saved["choice"] != "threshold":
                return self.saved["choice"], {"reason": "calibrated_constant_policy"}
        if indices.numel() * (model_cfg.core_size + 2 * model_cfg.halo) ** 2 < self.minimum:
            return "packed", {"reason": "small_workload"}
        stats = geometry_summary(indices, model_cfg, features["e2"].shape[-2:])
        shared = stats["ratio"] >= self.threshold
        chosen = (
            (self.saved.get("shared_strategy", "indexed") if self.saved else "indexed")
            if shared
            else "packed"
        )
        return chosen, {
            "reason": "calibrated_threshold" if self.saved else "uncalibrated_geometry_heuristic",
            "threshold": self.threshold,
            "geometry": stats,
        }
