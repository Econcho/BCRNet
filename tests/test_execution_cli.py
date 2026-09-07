import json
import sys

import torch
import yaml
from PIL import Image

from bcrnet import BCRNet
from bcrnet.execution import ExecutionBCRNet, ExecutionConfig
from bcrnet.execution.cli import main


def run_cli(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["bcrnet.execution", *map(str, arguments)])
    main()


def test_benchmark_calibration_and_coco_audit(tmp_path, cfg, monkeypatch):
    config = tmp_path / "model.yaml"
    config.write_text(yaml.safe_dump({"model": cfg.to_dict()}), encoding="utf-8")
    benchmark = tmp_path / "benchmark.json"
    run_cli(
        monkeypatch,
        [
            "benchmark",
            "--config",
            config,
            "--size",
            64,
            "--device",
            "cpu",
            "--patterns",
            "clustered",
            "dispersed",
            "--executors",
            "packed",
            "shared",
            "--warmup",
            0,
            "--iterations",
            2,
            "--output",
            benchmark,
            "--reuse-feature-masks",
        ],
    )
    data = json.loads(benchmark.read_text())
    assert data["passed"] and len(data["cases"]) == 2
    policy = tmp_path / "policy.json"
    run_cli(
        monkeypatch,
        [
            "calibrate",
            "--report",
            benchmark,
            "--budget",
            cfg.budget,
            "--halo",
            cfg.halo,
            "--shared-strategy",
            "shared",
            "--output",
            policy,
        ],
    )
    optimized = ExecutionBCRNet(
        BCRNet(cfg).eval(),
        ExecutionConfig(strategy="adaptive", policy_path=str(policy), reuse_feature_masks=True),
    )
    with torch.inference_mode():
        optimized(torch.randn(1, 3, 64, 64))
    assert optimized.last_execution["decision"]["reason"].startswith("calibrated")
    with torch.inference_mode():
        optimized(torch.randn(1, 3, 64, 96))
    assert optimized.last_execution["decision"]["reason"] == "calibration_signature_mismatch"

    root = tmp_path / "data"
    root.mkdir()
    for index in (1, 2):
        Image.new("RGB", (64, 64), (index * 40, 90, 30)).save(root / f"{index}.png")
    manifest = {
        "images": [
            {"id": i, "file_name": f"{i}.png", "width": 64, "height": 64, "sequence_id": f"synthetic_{i}"}
            for i in (1, 2)
        ],
        "categories": [{"id": 1, "name": "uav"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 12, 6, 7], "area": 42, "iscrowd": 0}
        ],
    }
    (root / "val.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        yaml.safe_dump({"type": "coco", "root": "data", "splits": {"val": "val.json"}}), encoding="utf-8"
    )
    checkpoint = tmp_path / "random.pt"
    torch.save(
        {
            "format_version": 1,
            "model_config": cfg.to_dict(),
            "model": BCRNet(cfg).state_dict(),
            "preprocessing": {"image_size": 64},
            "category_to_label": {1: 0},
            "categories": manifest["categories"],
        },
        checkpoint,
    )
    audit = tmp_path / "audit.json"
    run_cli(
        monkeypatch,
        [
            "audit",
            "--checkpoint",
            checkpoint,
            "--dataset",
            dataset,
            "--size",
            64,
            "--batch-size",
            2,
            "--device",
            "cpu",
            "--executor",
            "indexed",
            "--output",
            audit,
        ],
    )
    result = json.loads(audit.read_text())
    assert result["tensor_checks_passed"]
    assert len(result["rows"]) == 2
    assert result["rows"][0]["positive"] and not result["rows"][1]["positive"]
    from bcrnet.cli import parser, predict

    execution_config = tmp_path / "execution.yaml"
    execution_config.write_text("strategy: packed\n", encoding="utf-8")
    prediction_dir = tmp_path / "prediction"
    args = parser().parse_args(
        [
            "predict",
            "--checkpoint",
            str(checkpoint),
            "--image",
            str(root / "1.png"),
            "--output",
            str(prediction_dir),
            "--device",
            "cpu",
            "--execution-config",
            str(execution_config),
        ]
    )
    predict(args)
    prediction = json.loads((prediction_dir / "prediction.json").read_text())
    assert prediction["execution"]["strategy"] == "packed"
