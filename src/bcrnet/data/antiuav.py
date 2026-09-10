"""Read-only Anti-UAV annotation and sampling policy. This module never decodes videos."""

import hashlib
import json
import math
import random
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path


@dataclass(frozen=True)
class FrameLabel:
    frame_index: int
    exists: bool
    box: tuple[float, float, float, float] | None
    valid: bool = True
    reason: str = ""


def load_visible_annotation(path, bbox_format="xywh"):
    if bbox_format not in {"xywh", "xyxy"}:
        raise ValueError("bbox_format must be xywh or xyxy")
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if "exist" not in data or "gt_rect" not in data:
        raise ValueError(f"{path}: expected exist and gt_rect arrays")
    if len(data["exist"]) != len(data["gt_rect"]):
        raise ValueError(f"{path}: existence/box length mismatch")
    rows = []
    for i, (exists, box) in enumerate(zip(data["exist"], data["gt_rect"])):
        if exists not in (0, 1, False, True):
            rows.append(FrameLabel(i, False, None, False, "invalid_exist"))
            continue
        if not exists:
            rows.append(FrameLabel(i, False, None))
            continue
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            rows.append(FrameLabel(i, True, None, False, "invalid_box_shape"))
            continue
        try:
            x, y, c, d = map(float, box)
        except (TypeError, ValueError):
            rows.append(FrameLabel(i, True, None, False, "nonnumeric_box"))
            continue
        xyxy = (x, y, x + c, y + d) if bbox_format == "xywh" else (x, y, c, d)
        valid = all(math.isfinite(v) for v in xyxy) and xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]
        rows.append(FrameLabel(i, True, xyxy if valid else None, valid, "" if valid else "invalid_box"))
    return rows


def recording_group(sequence):
    """Original naming convention: YYYYMMDD_HHMMSS_camera_clip; strip only clip suffix."""
    parts = sequence.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(f"Cannot infer recording group for {sequence!r}; provide a custom split map")
    return parts[0]


def assign_splits(sequences, policy="official", seed=42, val_fraction=0.2, custom_mapping=None):
    """sequences: (original_split, name). Original test is NEVER merged into training."""
    if policy not in {"official", "group", "custom"}:
        raise ValueError("split policy must be official, group or custom")
    pairs = list(sequences)
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate sequence entry")
    if any(split not in {"train", "val", "test"} for split, _ in pairs):
        raise ValueError("Unexpected source split")
    if policy == "official":
        return {(split, name): split for split, name in pairs}
    if policy == "custom":
        if not isinstance(custom_mapping, dict):
            raise ValueError("custom split policy requires a mapping")
        expected = {f"{split}/{name}" for split, name in pairs}
        missing, extra = expected - set(custom_mapping), set(custom_mapping) - expected
        if missing or extra:
            raise ValueError(f"Custom split map mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        values = set(custom_mapping.values())
        if not values <= {"train", "val", "test"}:
            raise ValueError(f"Invalid custom target splits: {sorted(values - {'train', 'val', 'test'})}")
        for source_split, name in pairs:
            if source_split == "test" and custom_mapping[f"{source_split}/{name}"] != "test":
                raise ValueError("Custom split map cannot move source test sequences into train/val")
        return {(split, name): custom_mapping[f"{split}/{name}"] for split, name in pairs}
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0,1)")
    groups = sorted({recording_group(name) for split, name in pairs if split != "test"})
    test_groups = {recording_group(name) for split, name in pairs if split == "test"}
    if set(groups) & test_groups:
        raise ValueError("Source test shares recording groups with train/val; review the source split")
    if len(groups) < 2:
        raise ValueError("At least two recording groups are needed for grouped train/val")
    random.Random(seed).shuffle(groups)
    n_val = min(len(groups) - 1, max(1, round(len(groups) * val_fraction)))
    val_groups = set(groups[:n_val])
    return {
        (split, name): (
            "test" if split == "test" else "val" if recording_group(name) in val_groups else "train"
        )
        for split, name in pairs
    }


def evenly_spaced(values, count):
    values = sorted(values)
    count = min(max(0, count), len(values))
    if count == 0:
        return []
    if count == 1:
        return [values[len(values) // 2]]
    return [values[round(i * (len(values) - 1) / (count - 1))] for i in range(count)]


@dataclass(frozen=True)
class SamplingConfig:
    train_stride: int = 5
    positive_stride: int | None = None
    negative_stride: int | None = None
    tiny_stride: int = 2
    eval_stride: int = 1
    val_stride: int | None = None
    test_stride: int | None = None
    tiny_threshold: float = 16
    reference_size: int = 640
    tiny_densification: bool = True
    transition_sampling: bool = True
    transition_radius: int = 2
    max_train_frames: int = 400
    negative_fraction_at_cap: float = 0.25
    phase_policy: str = "hashed"
    seed: int = 42

    def __post_init__(self):
        def inherited(value, fallback):
            return fallback if value is None else value

        strides = (
            self.train_stride,
            inherited(self.positive_stride, self.train_stride),
            inherited(self.negative_stride, self.train_stride),
            self.tiny_stride,
            self.eval_stride,
            inherited(self.val_stride, self.eval_stride),
            inherited(self.test_stride, self.eval_stride),
        )
        if min(*strides, self.reference_size) <= 0:
            raise ValueError("Sampling strides/reference_size must be positive")
        if self.transition_radius < 0 or self.max_train_frames < 0:
            raise ValueError("Invalid transition radius/frame cap")
        if not 0 <= self.negative_fraction_at_cap <= 1 or self.tiny_threshold <= 0:
            raise ValueError("Invalid negative fraction/tiny threshold")
        if self.phase_policy not in {"hashed", "zero"}:
            raise ValueError("phase_policy must be hashed or zero")


def sample_frame_indices(rows, width, height, split, sequence, cfg):
    if width <= 0 or height <= 0 or split not in {"train", "val", "test"}:
        raise ValueError("Invalid video size or split")
    valid = {r.frame_index for r in rows if r.valid}
    if split != "train":
        stride = (
            (cfg.val_stride if cfg.val_stride is not None else cfg.eval_stride)
            if split == "val"
            else (cfg.test_stride if cfg.test_stride is not None else cfg.eval_stride)
        )
        return sorted(i for i in valid if i % stride == 0)
    phase_seed = int.from_bytes(hashlib.sha256(f"{cfg.seed}:{sequence}".encode()).digest()[:8], "little")

    def phase_for(stride):
        return phase_seed % stride if cfg.phase_policy == "hashed" else 0

    positive_stride = cfg.train_stride if cfg.positive_stride is None else cfg.positive_stride
    negative_stride = cfg.train_stride if cfg.negative_stride is None else cfg.negative_stride
    selected = {
        r.frame_index
        for r in rows
        if r.valid
        and r.frame_index % (positive_stride if r.exists else negative_stride)
        == phase_for(positive_stride if r.exists else negative_stride)
    }
    scale = min(cfg.reference_size / width, cfg.reference_size / height)
    if cfg.tiny_densification:
        for r in rows:
            if not r.valid or r.box is None:
                continue
            size = math.sqrt((r.box[2] - r.box[0]) * (r.box[3] - r.box[1])) * scale
            if size < cfg.tiny_threshold and r.frame_index % cfg.tiny_stride == phase_for(cfg.tiny_stride):
                selected.add(r.frame_index)
    events = set()
    if cfg.transition_sampling:
        for previous, current in pairwise(rows):
            if previous.valid and current.valid and previous.exists != current.exists:
                events.update(
                    range(
                        max(0, current.frame_index - cfg.transition_radius),
                        min(len(rows), current.frame_index + cfg.transition_radius + 1),
                    )
                )
    events &= valid
    selected |= events
    cap = cfg.max_train_frames
    if cap == 0 or len(selected) <= cap:
        return sorted(selected)
    kept = set(evenly_spaced(events, cap))
    negative = {r.frame_index for r in rows if r.valid and not r.exists}
    desired_neg = max(0, round(cap * cfg.negative_fraction_at_cap) - len(kept & negative))
    kept.update(evenly_spaced((selected & negative) - kept, min(desired_neg, cap - len(kept))))
    kept.update(evenly_spaced((selected - negative) - kept, cap - len(kept)))
    kept.update(evenly_spaced(selected - kept, cap - len(kept)))
    return sorted(kept)
