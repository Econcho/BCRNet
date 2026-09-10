import json
import random

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from bcrnet import BCRNet
from bcrnet.checkpoint import atomic_save, load_checkpoint, restore_rng, rng_state
from bcrnet.data.antiuav import (
    FrameLabel,
    SamplingConfig,
    assign_splits,
    load_visible_annotation,
    sample_frame_indices,
)
from bcrnet.data.datasets import CocoDetectionDataset, collate_detection
from bcrnet.evaluation import evaluate
from bcrnet.geometry import inverse_letterbox


def test_annotation_schema(tmp_path):
    path = tmp_path / "visible.json"
    path.write_text(
        json.dumps({"exist": [1, 0, 1, 1], "gt_rect": [[2, 3, 4, 5], [8, 8, 0, 0], [0, 0, -1, 2], []]})
    )
    rows = load_visible_annotation(path)
    assert rows[0].box == (2.0, 3.0, 6.0, 8.0)
    assert rows[1].valid and rows[1].box is None and not rows[1].exists
    assert not rows[2].valid and rows[2].exists
    assert not rows[3].valid


def test_sampling_and_recording_split():
    rows = [FrameLabel(i, i % 100 < 75, (0, 0, 8, 8) if i % 100 < 75 else None) for i in range(1000)]
    cfg = SamplingConfig(max_train_frames=100)
    indices = sample_frame_indices(rows, 1920, 1080, "train", "video", cfg)
    assert indices == sample_frame_indices(rows, 1920, 1080, "train", "video", cfg)
    assert len(indices) == 100 and len(set(indices)) == 100
    assert any(not rows[i].exists for i in indices)
    assert len(sample_frame_indices(rows, 1920, 1080, "test", "video", cfg)) == 1000
    pairs = [("train", "a_1_1"), ("val", "a_1_2"), ("train", "b_1_1"), ("test", "c_1_1")]
    grouped = assign_splits(pairs, "group")
    assert grouped[pairs[0]] == grouped[pairs[1]]
    assert grouped[pairs[-1]] == "test"
    assert assign_splits(pairs)[pairs[1]] == "val"
    with pytest.raises(ValueError, match="test shares"):
        assign_splits(pairs + [("test", "a_1_3")], "group")


def test_sampling_ablation_controls_and_custom_split():
    rows = [FrameLabel(i, i < 10, (0, 0, 4, 4) if i < 10 else None) for i in range(20)]
    cfg = SamplingConfig(
        positive_stride=2,
        negative_stride=5,
        val_stride=3,
        test_stride=4,
        tiny_densification=False,
        transition_sampling=False,
        max_train_frames=0,
        phase_policy="zero",
    )
    assert sample_frame_indices(rows, 64, 64, "train", "sequence", cfg) == [0, 2, 4, 6, 8, 10, 15]
    assert sample_frame_indices(rows, 64, 64, "val", "sequence", cfg) == [0, 3, 6, 9, 12, 15, 18]
    assert sample_frame_indices(rows, 64, 64, "test", "sequence", cfg) == [0, 4, 8, 12, 16]

    pairs = [("train", "a"), ("val", "b"), ("test", "c")]
    mapping = {"train/a": "val", "val/b": "train", "test/c": "test"}
    assert assign_splits(pairs, "custom", custom_mapping=mapping) == {
        ("train", "a"): "val",
        ("val", "b"): "train",
        ("test", "c"): "test",
    }
    with pytest.raises(ValueError, match="cannot move source test"):
        assign_splits(pairs, "custom", custom_mapping=mapping | {"test/c": "train"})
    with pytest.raises(ValueError, match="positive"):
        SamplingConfig(positive_stride=0)

    transition_cfg = SamplingConfig(
        positive_stride=100,
        negative_stride=100,
        tiny_densification=False,
        transition_sampling=True,
        transition_radius=1,
        max_train_frames=0,
        phase_policy="zero",
    )
    assert sample_frame_indices(rows, 64, 64, "train", "sequence", transition_cfg) == [0, 9, 10, 11]


def test_rng_checkpoint_safe_roundtrip(tmp_path):
    state = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(2))
    atomic_save(tmp_path / "test.pt", {"format_version": 1, "rng": state})
    restored = load_checkpoint(tmp_path / "test.pt")
    restore_rng(restored["rng"])
    assert expected[0] == random.random() and expected[1] == np.random.rand()
    assert torch.equal(expected[2], torch.rand(2))


def make_coco(tmp_path):
    Image.new("RGB", (100, 50), "gray").save(tmp_path / "image.jpg")
    Image.new("RGB", (100, 50), "white").save(tmp_path / "negative.jpg")
    document = {
        "images": [
            {"id": i + 1, "file_name": name, "height": 50, "width": 100}
            for i, name in enumerate(("image.jpg", "negative.jpg"))
        ],
        "categories": [{"id": 7, "name": "drone"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 10, 8, 8], "area": 64, "iscrowd": 0}
        ],
    }
    (tmp_path / "instances.json").write_text(json.dumps(document))
    return CocoDetectionDataset(tmp_path, "instances.json", image_size=(64, 96))


def test_dataset_and_inverse_geometry(tmp_path):
    dataset = make_coco(tmp_path)
    image, mask, target = dataset[0]
    assert image.shape == (3, 64, 96)
    assert mask.sum() == 48 * 96
    assert torch.allclose(inverse_letterbox(target["boxes"], target["meta"]), target["original_boxes"])
    assert dataset[1][2]["boxes"].shape == (0, 4)
    assert dataset.label_to_category == {0: 7}


def test_empty_prediction_coco_eval(tmp_path, cfg):
    dataset = make_coco(tmp_path)
    model = BCRNet(cfg).eval()
    metrics, predictions = evaluate(
        model,
        DataLoader(dataset, batch_size=2, collate_fn=collate_detection),
        dataset,
        torch.device("cpu"),
        mode="base",
        score_threshold=1,
    )
    assert predictions == [] and metrics["AP50_95"] == 0
    assert metrics["negative_frames"] == 1 and metrics["negative_frame_false_alarm_rate"] == 0
