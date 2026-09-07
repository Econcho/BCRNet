"""RGB-only extraction, opt-in execution. No overwrite, no random frame-level data split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from bcrnet.data.antiuav import SamplingConfig, assign_splits, load_visible_annotation, sample_frame_indices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-policy", choices=("official", "group"), default="official")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--bbox-format", choices=("xywh", "xyxy"), default="xywh")
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--tiny-stride", type=int, default=2)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--max-train-frames", type=int, default=400)
    parser.add_argument("--tiny-threshold", type=float, default=16)
    parser.add_argument("--reference-size", type=int, default=640)
    parser.add_argument("--transition-radius", type=int, default=2)
    parser.add_argument("--negative-fraction-at-cap", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--dry-run", action="store_true", help="Read labels/video metadata only; write nothing"
    )
    return parser.parse_args(argv)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def inspect_sequences(source, args, sampling):
    sequences = []
    attributes = {}
    for split in ("train", "val", "test"):
        folder = source / split
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        attribute_file = source / "label_new" / f"{split}.json"
        if attribute_file.exists():
            attributes[split] = json.loads(attribute_file.read_text(encoding="utf-8-sig"))
        for sequence in sorted(folder.iterdir()):
            if sequence.is_dir():
                for name in ("visible.mp4", "visible.json"):
                    if not (sequence / name).is_file():
                        raise FileNotFoundError(sequence / name)
                sequences.append((split, sequence.name))
    mapping = assign_splits(sequences, args.split_policy, args.seed, args.val_fraction)
    plan = []
    for source_split, name in sequences:
        directory = source / source_split / name
        rows = load_visible_annotation(directory / "visible.json", args.bbox_format)
        cap = cv2.VideoCapture(str(directory / "visible.mp4"))
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
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"Refusing to overwrite {output}; use a new output path")
    sampling = SamplingConfig(**{key: getattr(args, key) for key in SamplingConfig.__dataclass_fields__})
    plan = inspect_sequences(source, args, sampling)
    report = {
        "status": "planned",
        "source_root": str(source),
        "split_policy": args.split_policy,
        "bbox_format": args.bbox_format,
        "sampling": asdict(sampling),
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
            "categories": [{"id": 1, "name": "drone", "supercategory": "uav"}],
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
            cap = cv2.VideoCapture(str(source / entry["source_split"] / name / "visible.mp4"))
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
                            report["ignored_frames"].append(
                                {"sequence": name, "frame": i, "reason": "positive_box_clipped_out"}
                            )
                            continue
                    filename = destination / f"{i:06d}.jpg"
                    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                    if not ok:
                        raise OSError(f"JPEG encoding failed: {name}/{i}")
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
                                "category_id": 1,
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
            s: {"images": len(d["images"]), "boxes": len(d["annotations"])} for s, d in documents.items()
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
