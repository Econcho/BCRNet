"""Dataset plugin boundary: every adapter returns image, validity mask, target."""

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import Letterbox, flip_sample

DATASETS = {}


def register_dataset(name):
    def decorator(cls):
        if name in DATASETS:
            raise ValueError(f"Dataset already registered: {name}")
        DATASETS[name] = cls
        return cls

    return decorator


@register_dataset("coco")
class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        root,
        annotations,
        *,
        image_size=640,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        training=False,
        horizontal_flip=0.5,
    ):
        self.root = Path(root).resolve()
        self.annotation_path = Path(annotations)
        if not self.annotation_path.is_absolute():
            self.annotation_path = self.root / self.annotation_path
        self.document = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        self.images = self.document["images"]
        self.categories = sorted(self.document["categories"], key=lambda x: x["id"])
        self.category_to_label = {c["id"]: i for i, c in enumerate(self.categories)}
        self.label_to_category = {i: c["id"] for i, c in enumerate(self.categories)}
        self.num_classes = len(self.categories)
        self.by_image = defaultdict(list)
        seen = set()
        for row in self.images:
            if row["id"] in seen:
                raise ValueError("Duplicate image id")
            seen.add(row["id"])
        for row in self.document.get("annotations", []):
            if row["image_id"] not in seen or row["category_id"] not in self.category_to_label:
                raise ValueError("Annotation references unknown image/category")
            self.by_image[row["image_id"]].append(row)
        self.transform = Letterbox(image_size, mean, std)
        self.training, self.horizontal_flip = training, horizontal_flip

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        row = self.images[index]
        image_path = (self.root / row["file_name"]).resolve()
        if not image_path.is_relative_to(self.root):
            raise ValueError(f"Image escapes dataset root: {row['file_name']}")
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        if image.size != (row["width"], row["height"]):
            raise ValueError(f"Image dimensions disagree with annotation: {image_path}")
        boxes, labels, ignored = [], [], []
        for ann in self.by_image[row["id"]]:
            x, y, w, h = map(float, ann["bbox"])
            if w <= 0 or h <= 0:
                raise ValueError("Invalid COCO bbox")
            box = [max(0, x), max(0, y), min(row["width"], x + w), min(row["height"], y + h)]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("COCO bbox lies outside image")
            if ann.get("iscrowd", 0) or ann.get("ignore", False):
                ignored.append(box)
            else:
                boxes.append(box)
                labels.append(self.category_to_label[ann["category_id"]])
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        ignored = torch.tensor(ignored, dtype=torch.float32).reshape(-1, 4)
        tensor, mask, transformed, meta = self.transform(image, boxes)
        if len(ignored):
            ignored[:, [0, 2]] = ignored[:, [0, 2]] * meta["scale"][0] + meta["pad"][0]
            ignored[:, [1, 3]] = ignored[:, [1, 3]] * meta["scale"][1] + meta["pad"][1]
        meta.update(image_id=row["id"], file_name=row["file_name"], sequence_id=row.get("sequence_id", ""))
        target = {
            "boxes": transformed,
            "labels": torch.tensor(labels, dtype=torch.long),
            "ignore_boxes": ignored,
            "original_boxes": boxes,
            "meta": meta,
        }
        if self.training and random.random() < self.horizontal_flip:
            tensor, mask, target = flip_sample(tensor, mask, target)
        return tensor, mask, target


# Anti-UAV extraction exports COCO; no raw-video dependency in model/training.
DATASETS["antiuav_rgb"] = CocoDetectionDataset


def build_dataset(config, split, *, training=False, image_size=640, mean=None, std=None):
    name = config.get("type", "coco")
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; registered: {sorted(DATASETS)}")
    if split not in config["splits"]:
        raise ValueError(f"Dataset has no split {split!r}")
    options = dict(config.get("options", {}))
    options.update(image_size=image_size, training=training)
    if mean is not None:
        options["mean"] = mean
    if std is not None:
        options["std"] = std
    return DATASETS[name](config["root"], config["splits"][split], **options)


def collate_detection(batch):
    images, masks, targets = zip(*batch)
    return torch.stack(images), torch.stack(masks), list(targets)
