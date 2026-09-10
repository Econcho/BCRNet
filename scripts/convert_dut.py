"""Convert the DUT Pascal-VOC split into BCRNet's validated COCO contract."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from PIL import Image


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", type=Path, required=True, help="DUT root containing train/val/test")
    result.add_argument("--output-root", type=Path, required=True, help="New directory for COCO manifests")
    result.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"]
    )
    result.add_argument(
        "--image-mode",
        choices=("reference", "copy"),
        default="reference",
        help="reference keeps images in source-root; copy makes a self-contained output",
    )
    result.add_argument("--class-names", nargs="+", default=["UAV"])
    result.add_argument(
        "--coordinate-mode",
        choices=("inclusive", "half_open"),
        default="inclusive",
        help="VOC xmax/ymax convention; inclusive is the standard DUT setting",
    )
    result.add_argument("--unknown-class-policy", choices=("error", "skip"), default="error")
    result.add_argument("--difficult-policy", choices=("include", "ignore", "skip"), default="ignore")
    result.add_argument("--clip-boxes", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument(
        "--dry-run", action="store_true", help="Validate and print counts without writing output"
    )
    return result


def _number(node, tag):
    value = node.findtext(tag)
    if value is None:
        raise ValueError(f"Missing {tag}")
    return float(value)


def _image_path(image_dir, xml_path, filename):
    candidates = []
    if filename:
        candidate = Path(filename)
        if candidate.name != filename:
            raise ValueError(f"Unsafe image filename in {xml_path}: {filename!r}")
        candidates.append(image_dir / candidate)
    candidates.extend(image_dir / f"{xml_path.stem}{extension}" for extension in (".jpg", ".jpeg", ".png"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image found for {xml_path}; tried {[str(p) for p in candidates]}")


def _parse_objects(root, xml_path, image_id, width, height, class_to_id, args, start_id):
    annotations, skipped = [], []
    next_id = start_id
    for obj in root.findall("object"):
        try:
            name = (obj.findtext("name") or "").strip()
            if not name:
                raise ValueError("object has no class name")
            category_id = class_to_id.get(name.casefold())
            if category_id is None:
                if args.unknown_class_policy == "skip":
                    skipped.append({"xml": str(xml_path), "reason": f"unknown_class:{name}"})
                    continue
                raise ValueError(f"unknown class {name!r}")
            difficult = (obj.findtext("difficult") or "0").strip() in {"1", "true", "True"}
            if difficult and args.difficult_policy == "skip":
                skipped.append({"xml": str(xml_path), "reason": "difficult"})
                continue
            ignored = difficult and args.difficult_policy == "ignore"
            box = obj.find("bndbox")
            if box is None:
                raise ValueError(f"object {name!r} has no bndbox")
            x1, y1 = _number(box, "xmin"), _number(box, "ymin")
            x2, y2 = _number(box, "xmax"), _number(box, "ymax")
            if args.coordinate_mode == "inclusive":
                x2, y2 = x2 + 1.0, y2 + 1.0
            if args.clip_boxes:
                x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"non-positive box [{x1}, {y1}, {x2}, {y2}]")
            annotations.append(
                {
                    "id": next_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": int(ignored),
                    "ignore": bool(ignored),
                    "difficult": bool(difficult),
                }
            )
            next_id += 1
        except Exception as exc:
            if args.strict:
                raise ValueError(f"{xml_path}: {exc}") from exc
            skipped.append({"xml": str(xml_path), "reason": str(exc)})
    return annotations, skipped


def _parse_split(source, split, class_to_id, args):
    image_dir, xml_dir = source / split / "img", source / split / "xml"
    if not image_dir.is_dir() or not xml_dir.is_dir():
        raise FileNotFoundError(f"{split}: expected {image_dir} and {xml_dir}")
    images, annotations, skipped = [], [], []
    next_annotation = 1
    for image_id, xml_path in enumerate(sorted(xml_dir.glob("*.xml")), start=1):
        try:
            root = ET.parse(xml_path).getroot()
            image_path = _image_path(image_dir, xml_path, root.findtext("filename"))
            with Image.open(image_path) as handle:
                width, height = handle.size
            declared_width = root.findtext("size/width")
            declared_height = root.findtext("size/height")
            if (
                declared_width
                and declared_height
                and (int(float(declared_width)) != width or int(float(declared_height)) != height)
            ):
                raise ValueError(
                    f"{xml_path}: XML size {declared_width}x{declared_height} != image {width}x{height}"
                )
            image_name = image_path.name
            images.append(
                {
                    "id": image_id,
                    "file_name": f"{split}/img/{image_name}",
                    "width": width,
                    "height": height,
                    "sequence_id": split,
                    "source_split": split,
                    "source_annotation": str(xml_path),
                }
            )
            parsed, object_skipped = _parse_objects(
                root, xml_path, image_id, width, height, class_to_id, args, next_annotation
            )
            annotations.extend(parsed)
            skipped.extend(object_skipped)
            next_annotation += len(parsed)
        except Exception as exc:
            if args.strict:
                raise
            skipped.append({"xml": str(xml_path), "reason": str(exc)})
    return images, annotations, skipped


def _document(split, images, annotations, categories):
    return {
        "info": {"description": "DUT Anti-UAV converted for BCRNet", "source_split": split},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def main(argv=None):
    args = parser().parse_args(argv)
    source, output = args.source_root.resolve(), args.output_root.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be separate from, and not an ancestor/descendant of the DUT root")
    if len(set(args.class_names)) != len(args.class_names) or not all(args.class_names):
        raise ValueError("class-names must be nonempty and unique")
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory")
    class_to_id = {name.casefold(): index + 1 for index, name in enumerate(args.class_names)}
    categories = [
        {"id": index + 1, "name": name, "supercategory": "uav"} for index, name in enumerate(args.class_names)
    ]
    documents, skipped = {}, []
    started = time.perf_counter()
    for split in args.splits:
        images, annotations, split_skipped = _parse_split(source, split, class_to_id, args)
        documents[split] = _document(split, images, annotations, categories)
        skipped.extend(split_skipped)
    summary = {
        "status": "planned" if args.dry_run else "complete",
        "source_root": str(source),
        "output_root": str(output),
        "splits": args.splits,
        "image_mode": args.image_mode,
        "class_names": args.class_names,
        "coordinate_mode": args.coordinate_mode,
        "unknown_class_policy": args.unknown_class_policy,
        "difficult_policy": args.difficult_policy,
        "clip_boxes": args.clip_boxes,
        "strict": args.strict,
        "skipped": skipped,
        "counts": {
            split: {
                "images": len(document["images"]),
                "boxes": len(document["annotations"]),
                "negative_images": len(
                    {image["id"] for image in document["images"]}
                    - {annotation["image_id"] for annotation in document["annotations"]}
                ),
            }
            for split, document in documents.items()
        },
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    output.mkdir(parents=True)
    annotation_dir = output / "annotations"
    annotation_dir.mkdir()
    if args.image_mode == "copy":
        for split, document in documents.items():
            for row in document["images"]:
                source_image = source / row["file_name"]
                row["file_name"] = f"images/{row['file_name']}"
                target_image = output / row["file_name"]
                target_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_image, target_image)
        data_root = output
        split_paths = {split: f"annotations/{split}.json" for split in documents}
    else:
        data_root = source
        split_paths = {split: str((annotation_dir / f"{split}.json").resolve()) for split in documents}
    for split, document in documents.items():
        (annotation_dir / f"{split}.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    data = {"type": "coco", "root": str(data_root), "splits": split_paths}
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    summary["elapsed_seconds"] = time.perf_counter() - started
    (output / "conversion_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
