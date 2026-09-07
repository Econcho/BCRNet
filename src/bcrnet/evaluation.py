import contextlib
import io
import time

import numpy as np
import torch

from .inference import postprocess


@torch.no_grad()
def evaluate(model, loader, dataset, device, mode="full", max_batches=None, score_threshold=0.001):
    """COCO AP on original image coordinates, including empty frames."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    detections, image_ids, model_ms, post_ms = [], [], [], []
    empty_images, false_alarm_frames, false_alarm_boxes = 0, 0, 0
    gt_count = tiny_count = covered_gt = covered_tiny = selected_count = 0
    for batch_index, (images, masks, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, masks = images.to(device), masks.to(device)
        if images.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(images, masks, mode=mode)
        if images.is_cuda:
            torch.cuda.synchronize()
        after_model = time.perf_counter()
        decoded = postprocess(
            output, [t["meta"] for t in targets], model.config.num_classes, score_threshold=score_threshold
        )
        if images.is_cuda:
            torch.cuda.synchronize()
        after_post = time.perf_counter()
        model_ms.append(1000 * (after_model - start) / len(targets))
        post_ms.append(1000 * (after_post - after_model) / len(targets))
        for batch_item, (target, prediction) in enumerate(zip(targets, decoded)):
            if output.routing is not None:
                indices = output.routing.indices[batch_item].cpu()
                indices = indices[indices >= 0]
                selected_count += len(indices)
                gt = target["boxes"].cpu()
                centers = (gt[:, :2] + gt[:, 2:]) / 2
                core_pixels = model.config.core_size * 4
                grid_width = images.shape[-1] // core_pixels
                gt_indices = (centers[:, 1] // core_pixels).long() * grid_width + (
                    centers[:, 0] // core_pixels
                ).long()
                covered = torch.isin(gt_indices, indices)
                tiny = (gt[:, 2:] - gt[:, :2]).prod(-1).sqrt() < model.config.tiny_threshold
                gt_count += len(gt)
                tiny_count += int(tiny.sum())
                covered_gt += int(covered.sum())
                covered_tiny += int((covered & tiny).sum())
            image_id = target["meta"]["image_id"]
            image_ids.append(image_id)
            boxes = prediction["boxes"].cpu()
            scores, labels = prediction["scores"].cpu(), prediction["labels"].cpu()
            if not len(target["boxes"]) and not len(target.get("ignore_boxes", [])):
                empty_images += 1
                # Operational false-alarm metric is explicitly evaluated at 0.25.
                alarms = int((scores >= 0.25).sum())
                false_alarm_boxes += alarms
                false_alarm_frames += alarms > 0
            for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                x1, y1, x2, y2 = box
                detections.append(
                    {
                        "image_id": image_id,
                        "category_id": dataset.label_to_category[label],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )
    if not image_ids:
        raise ValueError("No images evaluated")
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(dataset.annotation_path))
        # pycocotools bbox evaluation honors iscrowd, not a standalone ignore flag.
        # Align evaluation with the dataset adapter's ignore-region training contract.
        for annotation in ground_truth.dataset.get("annotations", []):
            if annotation.get("ignore", False):
                annotation["iscrowd"] = 1
        if detections:
            predicted = ground_truth.loadRes(detections)
        else:
            predicted = COCO()
            predicted.dataset = {
                "images": ground_truth.dataset["images"],
                "categories": ground_truth.dataset["categories"],
                "annotations": [],
            }
            predicted.createIndex()
        evaluator = COCOeval(ground_truth, predicted, "bbox")
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    return {
        "images": len(image_ids),
        "AP50_95": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small_original_COCO": float(stats[3]),
        "AR100": float(stats[8]),
        "mode": mode,
        "model_ms_mean": float(np.mean(model_ms)),
        "postprocess_ms_mean": float(np.mean(post_ms)),
        "negative_frames": empty_images,
        "false_alarm_threshold": 0.25,
        "negative_frame_false_alarm_rate": false_alarm_frames / empty_images if empty_images else None,
        "false_boxes_per_negative_frame": false_alarm_boxes / empty_images if empty_images else None,
        "partial_evaluation": max_batches is not None,
        "gt_core_coverage": covered_gt / gt_count if gt_count else None,
        "tiny_gt_core_coverage": covered_tiny / tiny_count if tiny_count else None,
        "selected_windows_per_image": selected_count / len(image_ids),
        "tiny_definition": f"sqrt(input_box_area) < {model.config.tiny_threshold}; coverage diagnostic only",
        "ignore_policy": "ignore=true regions evaluated as COCO crowd regions",
    }, detections
