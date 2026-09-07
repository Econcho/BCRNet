"""Dense detection and counterfactual local-utility objectives, separate from model forward."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import aligned_giou, decode_regression
from .models.windows import gather_regions

DEFAULT_SCALE_SIGMA = math.log(2)


@dataclass
class LevelTargets:
    heatmap: torch.Tensor
    positive_weight: torch.Tensor
    regression: torch.Tensor
    regression_weight: torch.Tensor
    tiny_boost: torch.Tensor
    valid: torch.Tensor
    normalizer: torch.Tensor


def gaussian_max(heatmap, u, v, sigma):
    radius = math.ceil(3 * sigma)
    height, width = heatmap.shape
    left, right = max(0, u - radius), min(width, u + radius + 1)
    top, bottom = max(0, v - radius), min(height, v + radius + 1)
    y, x = torch.meshgrid(
        torch.arange(top, bottom, device=heatmap.device),
        torch.arange(left, right, device=heatmap.device),
        indexing="ij",
    )
    g = torch.exp(-((x - u).square() + (y - v).square()).float() / (2 * sigma * sigma))
    heatmap[top:bottom, left:right] = torch.maximum(heatmap[top:bottom, left:right], g)


class TargetBuilder:
    def __init__(
        self,
        num_classes=1,
        tiny_threshold=16,
        reference_scales=(16, 64, 160),
        scale_sigma=DEFAULT_SCALE_SIGMA,
    ):
        self.num_classes, self.tiny_threshold = num_classes, tiny_threshold
        self.reference_scales, self.scale_sigma = reference_scales, scale_sigma

    def build_level(self, targets, shape, stride, valid, level_index, tiny_only=False):
        b, _, h, w = shape
        classes = 1 if tiny_only else self.num_classes
        device = valid.device
        heat = torch.zeros(b, classes, h, w, device=device)
        positive = torch.zeros_like(heat)
        regression = torch.zeros(b, 4, h, w, device=device)
        rw = torch.zeros(b, 1, h, w, device=device)
        boost = torch.ones(b, 1, h, w, device=device)
        vm = valid.clone()
        normalizer = torch.ones(b, device=device)
        for n, target in enumerate(targets):
            boxes, labels = target["boxes"].to(device).float(), target["labels"].to(device).long()
            if boxes.ndim != 2 or boxes.shape[-1] != 4 or labels.shape != (len(boxes),):
                raise ValueError("Targets require boxes[N,4] xyxy and labels[N]")
            if not torch.isfinite(boxes).all() or ((boxes[:, 2:] - boxes[:, :2]) <= 0).any():
                raise ValueError("GT boxes must be finite with positive area")
            if ((labels < 0) | (labels >= self.num_classes)).any():
                raise ValueError("Target class index outside configured num_classes")
            for box in target.get("ignore_boxes", torch.empty(0, 4)).to(device):
                x1, y1, x2, y2 = box.tolist()
                vm[
                    n,
                    :,
                    max(0, math.floor(y1 / stride)) : min(h, math.ceil(y2 / stride)),
                    max(0, math.floor(x1 / stride)) : min(w, math.ceil(x2 / stride)),
                ] = False
            sizes = (boxes[:, 2:] - boxes[:, :2]).prod(-1).sqrt()
            keep = sizes < self.tiny_threshold if tiny_only else torch.ones_like(sizes, dtype=torch.bool)
            normalizer[n] = max(1, int(keep.sum()))
            # Large boxes first: on center collisions smaller objects own the shared regressor.
            for j in sizes.argsort(descending=True).tolist():
                if not keep[j]:
                    continue
                box, label, size = boxes[j], 0 if tiny_only else int(labels[j]), float(sizes[j])
                center = (box[:2] + box[2:]) / (2 * stride)
                u, v = int(torch.floor(center[0])), int(torch.floor(center[1]))
                if not 0 <= u < w or not 0 <= v < h:
                    raise ValueError("GT center outside the preprocessed image; fix the dataset transform")
                weight = 1.0
                if not tiny_only:
                    references = box.new_tensor(self.reference_scales)
                    weights = torch.softmax(
                        -((math.log(size) - references.log()) ** 2) / (2 * self.scale_sigma**2), 0
                    )
                    weight = weights[level_index]
                gaussian_max(heat[n, label], u, v, min(4, max(0.5, size / (6 * stride))))
                positive[n, label, v, u] = weight
                vm[n, :, v, u] = valid[n, :, v, u]
                if not bool(vm[n, 0, v, u]):
                    raise ValueError("GT center overlaps only image padding")
                regression[n, :2, v, u] = center - center.new_tensor((u, v))
                regression[n, 2:, v, u] = ((box[2:] - box[:2]) / stride).log()
                rw[n, :, v, u] = weight
                boost[n, :, v, u] = 2 if size < self.tiny_threshold else 1
        return LevelTargets(heat, positive, regression, rw, boost, vm, normalizer)

    def __call__(self, output, targets):
        if len(targets) != next(iter(output.base_predictions.values())).shape[0]:
            raise ValueError("Target batch size mismatch")
        result = {}
        for i, level in enumerate(("p2", "p3", "p4")):
            result[level] = self.build_level(
                targets, output.base_predictions[level].shape, 2 ** (i + 2), output.masks[f"e{i + 2}"], i
            )
        return result


def detection_loss_map(prediction, target, stride, num_classes, tiny_emphasis=False):
    # FP32 box exp/IoU and logarithms are deliberate under mixed precision.
    prediction = prediction.float()
    logits = prediction[:, :num_classes]
    prob = logits.sigmoid()
    is_positive = target.heatmap.eq(1)
    factor = target.tiny_boost if tiny_emphasis else 1
    positive_loss = -F.logsigmoid(logits) * (1 - prob).square() * target.positive_weight * factor
    negative_loss = -F.logsigmoid(-logits) * prob.square() * (1 - target.heatmap).pow(4)
    hm = torch.where(is_positive, positive_loss, negative_loss).sum(1, keepdim=True)
    reg = prediction[:, num_classes:]
    offset = F.smooth_l1_loss(reg[:, :2], target.regression[:, :2], reduction="none").sum(1, keepdim=True)
    size = F.smooth_l1_loss(reg[:, 2:], target.regression[:, 2:], reduction="none").sum(1, keepdim=True)
    giou = 1 - aligned_giou(decode_regression(reg, stride), decode_regression(target.regression, stride))
    total = hm + (offset + 0.5 * size + giou[:, None]) * target.regression_weight * factor
    return total * target.valid


def slice_targets(target, indices, grid_width, core):
    def take(value):
        return gather_regions(value.float(), indices, grid_width, core).flatten(0, 1)

    return LevelTargets(
        take(target.heatmap),
        take(target.positive_weight),
        take(target.regression),
        take(target.regression_weight),
        take(target.tiny_boost),
        take(target.valid).bool(),
        target.normalizer[:, None].expand(-1, indices.shape[1]).flatten(),
    )


class BCRCriterion(nn.Module):
    def __init__(self, cfg, *, base_weight=0.5, tiny_weight=0.2, utility_weight=0.1, residual_weight=1e-4):
        super().__init__()
        self.cfg = cfg
        self.builder = TargetBuilder(cfg.num_classes, cfg.tiny_threshold)
        self.base_weight, self.tiny_weight = base_weight, tiny_weight
        self.utility_weight, self.residual_weight = utility_weight, residual_weight
        self.register_buffer("utility_scale", torch.tensor(1e-3))
        self.register_buffer("utility_updates", torch.tensor(0, dtype=torch.long))

    def detection_loss(self, predictions, built):
        terms = []
        for level, stride, eta in (("p2", 4, 1), ("p3", 8, 0.5), ("p4", 16, 0.25)):
            lm = detection_loss_map(predictions[level], built[level], stride, self.cfg.num_classes)
            terms.append(eta * (lm.sum((1, 2, 3)) / built[level].normalizer).mean())
        return sum(terms)

    def utility_loss(self, output, built, refinement=None):
        patches = refinement if refinement is not None else output.refinement
        routing = output.routing
        zero = next(iter(output.predictions.values())).sum() * 0
        if patches is None or not patches.indices.shape[1] or routing is None or routing.utility is None:
            return zero, zero.detach()
        b, k = patches.indices.shape
        core = self.cfg.core_size
        grid_width = output.predictions["p2"].shape[-1] // core
        local = slice_targets(built["p2"], patches.indices, grid_width, core)
        before = detection_loss_map(
            patches.base_logits.flatten(0, 1), local, 4, self.cfg.num_classes, tiny_emphasis=True
        )
        after = detection_loss_map(
            patches.refined_logits.flatten(0, 1), local, 4, self.cfg.num_classes, tiny_emphasis=True
        )
        delta = ((before - after).sum((1, 2, 3)) / local.normalizer).reshape(b, k).detach()
        valid = patches.indices >= 0
        if self.training and valid.any():
            mean_abs = delta[valid].abs().mean().clamp_min(1e-6)
            with torch.no_grad():
                if self.utility_updates == 0:
                    self.utility_scale.copy_(mean_abs)
                else:
                    self.utility_scale.lerp_(mean_abs, 0.05)
                self.utility_updates.add_(1)
        expected = (delta / self.utility_scale.clamp_min(1e-6)).clamp(-5, 5)
        predicted = routing.utility.gather(1, patches.indices.clamp_min(0))
        loss = F.smooth_l1_loss(predicted.float(), expected, reduction="none")
        gt = local.regression_weight.flatten(1).sum(1).reshape(b, k) > 0
        score = routing.coverage.gather(1, patches.indices.clamp_min(0))
        groups = (valid & gt, valid & ~gt & (score >= 0.1), valid & ~gt & (score < 0.1))
        means = [loss[g].mean() for g in groups if g.any()]
        return (torch.stack(means).mean() if means else zero), (
            delta[valid].mean() if valid.any() else zero.detach()
        )

    def forward(self, output, targets, *, stage="D", probes=None):
        if not output.predictions:
            raise ValueError("features mode has no detection loss")
        built = self.builder(output, targets)
        base = self.detection_loss(output.base_predictions, built)
        final = self.detection_loss(output.predictions, built)
        zero = base * 0
        tiny = zero
        if output.tiny_logits is not None:
            tt = self.builder.build_level(
                targets, output.tiny_logits.shape, 4, output.masks["e2"], 0, tiny_only=True
            )
            logits = output.tiny_logits.float()
            p = logits.sigmoid()
            loss = torch.where(
                tt.heatmap.eq(1),
                -F.logsigmoid(logits) * (1 - p).square(),
                -F.logsigmoid(-logits) * p.square() * (1 - tt.heatmap).pow(4),
            )
            tiny = ((loss * tt.valid).sum((1, 2, 3)) / tt.normalizer).mean()
        utility, gain = self.utility_loss(output, built) if stage in {"C", "D"} else (zero, zero.detach())
        if probes is not None and stage in {"C", "D"}:
            probe_loss, _ = self.utility_loss(output, built, probes)
            utility = (utility + probe_loss) / 2
        residual = zero
        if output.refinement is not None:
            r = output.refinement
            denom = r.valid_mask.sum().clamp_min(1) * self.cfg.width
            residual = r.residual.float().square().sum() / denom
        if stage == "A":
            total = base + self.tiny_weight * tiny
        elif stage == "B":
            total = final + self.residual_weight * residual
        elif stage == "C":
            total = utility
        elif stage == "D":
            total = final + self.base_weight * base + self.tiny_weight * tiny + self.utility_weight * utility
            total = total + self.residual_weight * residual
        else:
            raise ValueError("stage must be A, B, C or D")
        return {
            "loss": total,
            "detection": final,
            "base": base,
            "tiny": tiny,
            "utility": utility,
            "residual": residual,
            "mean_local_gain": gain,
        }
