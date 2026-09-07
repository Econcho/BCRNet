import torch
from torch.nn import functional as F
from torchvision.ops import batched_nms

from .geometry import inverse_letterbox


@torch.no_grad()
def postprocess(output, metadata, num_classes=1, score_threshold=0.001, nms_iou=0.5, max_detections=100):
    """Decode raw predictions; low AP prefilter and operational threshold are separate CLI settings."""
    if not output.predictions:
        raise ValueError("Cannot decode features mode")
    results = []
    for batch_index, meta in enumerate(metadata):
        boxes_all, scores_all, labels_all = [], [], []
        x1, y1, x2, y2 = meta["valid_bounds"]
        for level, stride in (("p2", 4), ("p3", 8), ("p4", 16)):
            raw = output.predictions[level][batch_index].float()
            score = raw[:num_classes].sigmoid()
            peaks = score.eq(F.max_pool2d(score[None], 3, 1, 1)[0])
            mask = output.masks["e" + level[1]][batch_index]
            score = score.masked_fill(~peaks | ~mask, -1)
            h, w = score.shape[-2:]
            values, ids = score.flatten().topk(min(max_detections, score.numel()))
            good = values >= score_threshold
            values, ids = values[good], ids[good]
            labels, points = ids // (h * w), ids % (h * w)
            reg = raw[num_classes:].flatten(1)[:, points].T
            centers = (torch.stack((points % w, points // w), 1) + reg[:, :2]) * stride
            sizes = reg[:, 2:].clamp(-8, 8).exp() * stride
            boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), 1)
            inside = (
                (centers[:, 0] >= x1) & (centers[:, 0] < x2) & (centers[:, 1] >= y1) & (centers[:, 1] < y2)
            )
            boxes_all.append(boxes[inside])
            scores_all.append(values[inside])
            labels_all.append(labels[inside])
        boxes, scores, labels = torch.cat(boxes_all), torch.cat(scores_all), torch.cat(labels_all)
        if len(boxes):
            # Clip in input coordinates before NMS so padded box extents cannot alter overlap.
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(x1, x2)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(y1, y2)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & torch.isfinite(boxes).all(1)
            boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
            keep = batched_nms(boxes, scores, labels, nms_iou)[:max_detections]
            boxes, scores, labels = inverse_letterbox(boxes[keep], meta), scores[keep], labels[keep]
        results.append({"boxes": boxes, "scores": scores, "labels": labels})
    return results
