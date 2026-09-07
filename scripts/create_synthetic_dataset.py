"""Create procedural engineering fixtures only; never reads Anti-UAV or extracts videos."""

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    for split in ("train", "val", "test"):
        images, annotations = [], []
        for index in range(4):
            image = Image.new("RGB", (96, 64), (90 + index * 10, 130, 170))
            filename = f"{split}_{index}.png"
            if index % 2 == 0:
                box = [12 + index * 4, 15, 8, 6]
                ImageDraw.Draw(image).rectangle([box[0], box[1], box[0] + 8, box[1] + 6], fill="black")
                annotations.append(
                    {
                        "id": index + 1,
                        "image_id": index + 1,
                        "category_id": 1,
                        "bbox": box,
                        "area": 48,
                        "iscrowd": 0,
                    }
                )
            image.save(root / filename)
            images.append({"id": index + 1, "file_name": filename, "width": 96, "height": 64})
        (root / f"{split}.json").write_text(
            json.dumps(
                {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "drone"}]}
            ),
            encoding="utf-8",
        )
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {"type": "coco", "root": str(root), "splits": {s: f"{s}.json" for s in ("train", "val", "test")}}
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
