"""RGB-only extraction, opt-in execution. No overwrite, no random frame-level data split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import yaml

from bcrnet.data.antiuav import SamplingConfig, assign_splits, load_visible_annotation, sample_frame_indices


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional YAML defaults; explicit CLI values override it")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--source-splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"]
    )
    parser.add_argument("--video-name", default="visible.mp4")
    parser.add_argument("--annotation-name", default="visible.json")
    parser.add_argument("--attribute-dir-name", default="label_new")
    parser.add_argument("--include-attributes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-policy", choices=("official", "group", "custom"), default="official")
    parser.add_argument("--split-map", type=Path, help="JSON mapping source_split/sequence to train/val/test")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--bbox-format", choices=("xywh", "xyxy"), default="xywh")
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--positive-stride", type=int)
    parser.add_argument("--negative-stride", type=int)
    parser.add_argument("--tiny-stride", type=int, default=2)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--val-stride", type=int)
    parser.add_argument("--test-stride", type=int)
    parser.add_argument("--max-train-frames", type=int, default=400)
    parser.add_argument("--tiny-threshold", type=float, default=16)
    parser.add_argument("--reference-size", type=int, default=640)
    parser.add_argument("--tiny-densification", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transition-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transition-radius", type=int, default=2)
    parser.add_argument("--negative-fraction-at-cap", type=float, default=0.25)
    parser.add_argument("--phase-policy", choices=("hashed", "zero"), default="hashed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--png-compression", type=int, default=3)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--category-name", default="drone")
    parser.add_argument("--supercategory", default="uav")
    parser.add_argument("--invalid-box-policy", choices=("skip", "error"), default="skip")
    parser.add_argument("--clipped-box-policy", choices=("skip", "error"), default="skip")
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Read labels/video metadata only; write nothing",
    )
    return parser


def parse_args(argv=None):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    preliminary, _ = probe.parse_known_args(argv)
    parser = build_parser()
    if preliminary.config:
        loaded = yaml.safe_load(preliminary.config.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "extraction" in loaded:
            loaded = loaded["extraction"]
        if not isinstance(loaded, dict):
            parser.error("Extraction config must be a YAML mapping")
        allowed = {action.dest for action in parser._actions}
        unknown = set(loaded) - allowed
        if unknown:
            parser.error(f"Unknown extraction config keys: {sorted(unknown)}")
        parser.set_defaults(**loaded)
    args = parser.parse_args(argv)
    if args.source_root is None or args.output_root is None:
        parser.error("source-root and output-root are required via CLI or config")
    return args


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def summarize_document(document, reference_size, tiny_threshold):
    images = {row["id"]: row for row in document["images"]}
    positive_ids = {row["image_id"] for row in document["annotations"]}
    tiny_boxes = 0
    for annotation in document["annotations"]:
        image = images[annotation["image_id"]]
        scale = min(reference_size / image["width"], reference_size / image["height"])
        if (annotation["area"] ** 0.5) * scale < tiny_threshold:
            tiny_boxes += 1
    by_sequence = {}
    for image in document["images"]:
        row = by_sequence.setdefault(
            image["sequence_id"], {"images": 0, "positive_frames": 0, "negative_frames": 0}
        )
        row["images"] += 1
        row["positive_frames" if image["id"] in positive_ids else "negative_frames"] += 1
    return {
        "images": len(images),
        "boxes": len(document["annotations"]),
        "positive_frames": len(positive_ids),
        "negative_frames": len(images) - len(positive_ids),
        "tiny_boxes_at_reference_size": tiny_boxes,
        "sequences": len(by_sequence),
        "per_sequence": by_sequence,
    }


def inspect_sequences(source, args, sampling, custom_mapping=None):
    sequences = []
    attributes = {}
    for split in args.source_splits:
        folder = source / split
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        attribute_file = source / args.attribute_dir_name / f"{split}.json"
        if args.include_attributes and attribute_file.exists():
            attributes[split] = json.loads(attribute_file.read_text(encoding="utf-8-sig"))
        for sequence in sorted(folder.iterdir()):
            if sequence.is_dir():
                for name in (args.video_name, args.annotation_name):
                    if not (sequence / name).is_file():
                        raise FileNotFoundError(sequence / name)
                sequences.append((split, sequence.name))
    mapping = assign_splits(
        sequences, args.split_policy, args.seed, args.val_fraction, custom_mapping=custom_mapping
    )
    plan = []
    for source_split, name in sequences:
        directory = source / source_split / name
        rows = load_visible_annotation(directory / args.annotation_name, args.bbox_format)
        invalid = [row for row in rows if not row.valid]
        if invalid and args.invalid_box_policy == "error":
            raise ValueError(
                f"{name}: {len(invalid)} invalid annotation rows; first frame={invalid[0].frame_index}"
            )
        cap = cv2.VideoCapture(str(directory / args.video_name))
        try:
            if not cap.isOpened():
                raise OSError(f"Cannot open video {directory}")
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
        finally:
            cap.release()
        if frames != len(rows):
            raise ValueError(
                f"{name}: metadata has {frames} frames, annotation has {len(rows)}; audit alignment"
            )
        target_split = mapping[(source_split, name)]
        indices = sample_frame_indices(rows, width, height, target_split, name, sampling)
        plan.append(
            {
                "source_split": source_split,
                "sequence": name,
                "split": target_split,
                "width": width,
                "height": height,
                "frames": frames,
                "fps": fps,
                "rows": rows,
                "indices": indices,
                "attributes_unverified": attributes.get(source_split, {}).get(name, []),
            }
        )
    return plan


def main(argv=None):
    args = parse_args(argv)
    source, output = args.source_root.resolve(), args.output_root.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be separate from, and not an ancestor/descendant of, the raw data root")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("JPEG quality must be in [1,100]")
    if not 0 <= args.png_compression <= 9:
        raise ValueError("PNG compression must be in [0,9]")
    if args.category_id <= 0 or not args.category_name:
        raise ValueError("Category id must be positive and category name must be nonempty")
    if len(set(args.source_splits)) != len(args.source_splits):
        raise ValueError("source-splits cannot contain duplicates")
    custom_mapping = None
    if args.split_policy == "custom":
        if args.split_map is None:
            raise ValueError("--split-policy custom requires --split-map")
        custom_mapping = json.loads(args.split_map.read_text(encoding="utf-8-sig"))
    elif args.split_map is not None:
        raise ValueError("--split-map is only valid with --split-policy custom")
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite {output}; use a new output path")
    sampling = SamplingConfig(**{key: getattr(args, key) for key in SamplingConfig.__dataclass_fields__})
    plan = inspect_sequences(source, args, sampling, custom_mapping)
    report = {
        "status": "planned",
        "source_root": str(source),
        "output_root": str(output),
        "configuration_source": str(args.config.resolve()) if args.config else "CLI/defaults",
        "split_configuration": {
            "policy": args.split_policy,
            "map": str(args.split_map.resolve()) if args.split_map else None,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
        },
        "split_policy": args.split_policy,
        "bbox_format": args.bbox_format,
        "sampling": asdict(sampling),
        "source_splits": args.source_splits,
        "video_name": args.video_name,
        "annotation_name": args.annotation_name,
        "attribute_dir_name": args.attribute_dir_name,
        "include_attributes": args.include_attributes,
        "image_format": args.image_format,
        "encoding": {"jpeg_quality": args.jpeg_quality, "png_compression": args.png_compression},
        "category": {
            "id": args.category_id,
            "name": args.category_name,
            "supercategory": args.supercategory,
        },
        "annotation_policies": {
            "invalid_box": args.invalid_box_policy,
            "clipped_box": args.clipped_box_policy,
        },
        "sequences": [],
        "ignored_frames": [],
        "note": "label_new attributes are unverified for RGB; not bbox labels",
    }
    for entry in plan:
        report["sequences"].append(
            {key: value for key, value in entry.items() if key not in ("rows", "indices")}
            | {"selected_frames": len(entry["indices"])}
        )
        for row in entry["rows"]:
            if not row.valid:
                report["ignored_frames"].append(
                    {"sequence": entry["sequence"], "frame": row.frame_index, "reason": row.reason}
                )
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    output.mkdir(parents=True)
    (output / "annotations").mkdir()
    documents = {
        split: {
            "info": {"description": "Anti-UAV RGB frame detection", "split_policy": args.split_policy},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": args.category_id, "name": args.category_name, "supercategory": args.supercategory}
            ],
        }
        for split in ("train", "val", "test")
    }
    started = time.perf_counter()
    try:
        for entry in plan:
            split, name = entry["split"], entry["sequence"]
            destination = output / "images" / split / f"{entry['source_split']}_{name}"
            destination.mkdir(parents=True)
            chosen = set(entry["indices"])
            document = documents[split]
            cap = cv2.VideoCapture(str(source / entry["source_split"] / name / args.video_name))
            try:
                if not cap.isOpened():
                    raise OSError(f"Cannot open {name}")
                for i, row in enumerate(entry["rows"]):
                    if not cap.grab():
                        raise OSError(f"{name}: decode ended before annotation frame {i}")
                    if i not in chosen:
                        continue
                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        raise OSError(f"{name}: could not retrieve frame {i}")
                    height, width = frame.shape[:2]
                    if (width, height) != (entry["width"], entry["height"]):
                        raise ValueError(f"{name}: video resolution changed at frame {i}")
                    box = row.box
                    if box is not None:
                        x1, y1, x2, y2 = box
                        box = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
                        if box[2] <= box[0] or box[3] <= box[1]:
                            if args.clipped_box_policy == "error":
                                raise ValueError(f"{name}: positive box clipped out at frame {i}")
                            report["ignored_frames"].append(
                                {"sequence": name, "frame": i, "reason": "positive_box_clipped_out"}
                            )
                            continue
                    filename = destination / f"{i:06d}.{args.image_format}"
                    encode_parameters = (
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                        if args.image_format == "jpg"
                        else [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]
                    )
                    ok, encoded = cv2.imencode(f".{args.image_format}", frame, encode_parameters)
                    if not ok:
                        raise OSError(f"Image encoding failed: {name}/{i}")
                    encoded.tofile(str(filename))  # Supports Unicode Windows paths.
                    image_id = len(document["images"]) + 1
                    document["images"].append(
                        {
                            "id": image_id,
                            "file_name": filename.relative_to(output).as_posix(),
                            "width": width,
                            "height": height,
                            "sequence_id": name,
                            "source_split": entry["source_split"],
                            "frame_index": i,
                            "attributes_unverified": entry["attributes_unverified"],
                        }
                    )
                    if box is not None:
                        x1, y1, x2, y2 = box
                        document["annotations"].append(
                            {
                                "id": len(document["annotations"]) + 1,
                                "image_id": image_id,
                                "category_id": args.category_id,
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "area": (x2 - x1) * (y2 - y1),
                                "iscrowd": 0,
                            }
                        )
            finally:
                cap.release()
            print(f"{split}: {name}: {len(chosen)} planned frames", flush=True)
        for split, document in documents.items():
            atomic_json(output / "annotations" / f"{split}.json", document)
        report["status"] = "complete"
        report["elapsed_seconds"] = time.perf_counter() - started
        report["counts"] = {
            split: summarize_document(document, args.reference_size, args.tiny_threshold)
            for split, document in documents.items()
        }
        atomic_json(output / "conversion_report.json", report)
        (output / "data.yaml").write_text(
            "type: antiuav_rgb\nroot: '" + str(output).replace("'", "''") + "'\nsplits:\n"
            "  train: annotations/train.json\n  val: annotations/val.json\n  test: annotations/test.json\n",
            encoding="utf-8",
        )
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        atomic_json(output / "conversion_report.json", report)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
