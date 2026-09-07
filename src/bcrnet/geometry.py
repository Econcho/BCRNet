"""Continuous xyxy geometry, shared by losses, inference and data transforms."""

import torch


def aligned_giou(boxes1, boxes2, eps=1e-7):
    lo = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    hi = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection = (hi - lo).clamp_min(0).prod(-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp_min(0).prod(-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp_min(0).prod(-1)
    union = (area1 + area2 - intersection).clamp_min(eps)
    enclosing = (
        (torch.maximum(boxes1[..., 2:], boxes2[..., 2:]) - torch.minimum(boxes1[..., :2], boxes2[..., :2]))
        .clamp_min(0)
        .prod(-1)
    )
    return intersection / union - (enclosing - union) / enclosing.clamp_min(eps)


def decode_regression(regression, stride):
    """N4HW -> NHW4 xyxy; offsets are relative to each cell's upper-left corner."""
    _, _, h, w = regression.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=regression.device), torch.arange(w, device=regression.device), indexing="ij"
    )
    centers = (regression[:, :2] + torch.stack((x, y)).unsqueeze(0)) * stride
    sizes = regression[:, 2:].clamp(-8, 8).exp() * stride
    return torch.cat((centers - sizes / 2, centers + sizes / 2), 1).permute(0, 2, 3, 1)


def inverse_letterbox(boxes, meta):
    result = boxes.clone()
    result[:, [0, 2]] = (result[:, [0, 2]] - meta["pad"][0]) / meta["scale"][0]
    result[:, [1, 3]] = (result[:, [1, 3]] - meta["pad"][1]) / meta["scale"][1]
    height, width = meta["original_size"]
    result[:, [0, 2]] = result[:, [0, 2]].clamp(0, width)
    result[:, [1, 3]] = result[:, [1, 3]].clamp(0, height)
    return result
