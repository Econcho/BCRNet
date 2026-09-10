import json

import yaml
from PIL import Image

from bcrnet import engine
from bcrnet.checkpoint import load_checkpoint
from bcrnet.cli import parser


def make_training_case(tmp_path, cfg, *, epochs=5, val_interval=2, patience=1):
    Image.new("RGB", (96, 64), "gray").save(tmp_path / "image.png")
    document = {
        "images": [{"id": 1, "file_name": "image.png", "width": 96, "height": 64}],
        "categories": [{"id": 1, "name": "drone"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 8, 8], "area": 64, "iscrowd": 0}
        ],
    }
    (tmp_path / "instances.json").write_text(json.dumps(document), encoding="utf-8")
    config = {
        "model": cfg.to_dict(),
        "preprocessing": {"image_size": [64, 96]},
        "training": {
            "epochs": epochs,
            "batch_size": 1,
            "val_interval": val_interval,
            "patience": patience,
            "progress": False,
            "seed": 9,
            "amp": False,
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "root": str(tmp_path),
                "type": "coco",
                "splits": {"train": "instances.json", "val": "instances.json"},
            }
        ),
        encoding="utf-8",
    )
    args = parser().parse_args(
        [
            "train",
            "--config",
            str(tmp_path / "config.yaml"),
            "--data",
            str(tmp_path / "data.yaml"),
            "--output",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--stage",
            "A",
            "--mode",
            "base",
        ]
    )
    return args


def test_training_controls_parse_and_validation_schedule():
    args = parser().parse_args(
        [
            "train",
            "--config",
            "config.yaml",
            "--data",
            "data.yaml",
            "--output",
            "run",
            "--val-interval",
            "3",
            "--patience",
            "4",
            "--min-delta",
            "0.01",
            "--monitor",
            "AP50",
            "--no-progress",
        ]
    )
    controls = engine._resolve_training_controls(
        args,
        {"val_interval": 1, "patience": 0, "min_delta": 0.0, "monitor": "AP50_95", "progress": True},
    )
    assert controls == {
        "val_interval": 3,
        "patience": 4,
        "min_delta": 0.01,
        "monitor": "AP50",
        "early_stop_stage": "D",
        "progress": False,
    }
    assert engine._should_validate(1, 5, 3, False) is False
    assert engine._should_validate(3, 5, 3, False) is True
    assert engine._should_validate(4, 5, 3, True) is True
    assert engine._should_validate(5, 5, 3, False) is True
    single_stage = parser().parse_args(
        [
            "train",
            "--config",
            "config.yaml",
            "--data",
            "data.yaml",
            "--output",
            "run",
            "--stage",
            "A",
        ]
    )
    assert (
        engine._resolve_training_controls(single_stage, {"early_stop_stage": "D"})["early_stop_stage"] == "A"
    )


def test_validation_interval_and_early_stopping_checkpoint_state(tmp_path, cfg, monkeypatch):
    args = make_training_case(tmp_path, cfg, epochs=5, val_interval=2, patience=1)
    values = iter([0.50, 0.40])
    calls = []

    def fake_evaluate(*_args, **_kwargs):
        calls.append(True)
        return {"AP50_95": next(values)}, []

    monkeypatch.setattr(engine, "evaluate", fake_evaluate)
    engine.train(args)

    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 2
    assert [row["validated"] for row in records] == [False, True, False, True]
    assert records[-1]["early_stopping"] is True
    assert records[-1]["bad_validations"] == 1
    assert records[-1]["best_metric"] == 0.5
    assert load_checkpoint(tmp_path / "run" / "last.pt")["epoch"] == 3
    best = load_checkpoint(tmp_path / "run" / "best.pt")
    assert best["epoch"] == 1
    assert best["monitor"] == "AP50_95"
    assert best["val_interval"] == 2
    assert best["patience"] == 1


def test_resume_preserves_early_stopping_counter(tmp_path, cfg, monkeypatch):
    args = make_training_case(tmp_path, cfg, epochs=5, val_interval=2, patience=2)
    args.stop_after_epochs = 2
    values = iter([0.50, 0.40, 0.30])

    def fake_evaluate(*_args, **_kwargs):
        return {"AP50_95": next(values)}, []

    monkeypatch.setattr(engine, "evaluate", fake_evaluate)
    engine.train(args)
    paused = load_checkpoint(tmp_path / "run" / "last.pt")
    assert paused["epoch"] == 1
    assert paused["bad_validations"] == 0

    resumed = make_training_case(tmp_path, cfg, epochs=5, val_interval=2, patience=2)
    resumed.resume = str(tmp_path / "run" / "last.pt")
    engine.train(resumed)
    final = load_checkpoint(tmp_path / "run" / "last.pt")
    assert final["epoch"] == 4
    assert final["early_stopped"] is True
    assert final["bad_validations"] == 2
    assert final["best_metric"] == 0.5
