import json

import torch
import yaml
from PIL import Image

from bcrnet.checkpoint import load_checkpoint
from bcrnet.cli import parser
from bcrnet.engine import train


def test_same_stage_optimizer_resume(tmp_path, cfg):
    Image.new("RGB", (96, 64), "gray").save(tmp_path / "image.png")
    doc = {
        "images": [{"id": 1, "file_name": "image.png", "width": 96, "height": 64}],
        "categories": [{"id": 1, "name": "drone"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 8, 8], "area": 64, "iscrowd": 0}
        ],
    }
    (tmp_path / "instances.json").write_text(json.dumps(doc))
    config = {
        "model": cfg.to_dict(),
        "preprocessing": {"image_size": [64, 96]},
        "training": {"epochs": 2, "batch_size": 1, "seed": 9, "amp": False},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "root": str(tmp_path),
                "type": "coco",
                "splits": {"train": "instances.json", "val": "instances.json"},
            }
        )
    )
    common = [
        "train",
        "--config",
        str(tmp_path / "config.yaml"),
        "--data",
        str(tmp_path / "data.yaml"),
        "--device",
        "cpu",
        "--stage",
        "A",
        "--mode",
        "base",
    ]
    train(parser().parse_args(common + ["--output", str(tmp_path / "continuous")]))
    train(parser().parse_args(common + ["--output", str(tmp_path / "resumed"), "--stop-after-epochs", "1"]))
    train(
        parser().parse_args(
            common + ["--output", str(tmp_path / "resumed"), "--resume", str(tmp_path / "resumed/last.pt")]
        )
    )
    a, b = [load_checkpoint(tmp_path / p / "last.pt") for p in ("continuous", "resumed")]
    assert a["epoch"] == b["epoch"] == 1
    assert all(torch.equal(t, b["model"][key]) for key, t in a["model"].items())
    assert a["scheduler"] == b["scheduler"]
    for index, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, b["optimizer"]["state"][index][name])
