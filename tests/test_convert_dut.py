import importlib.util
import json
from pathlib import Path

import yaml
from PIL import Image

from bcrnet.data.datasets import build_dataset


def load_converter():
    path = Path(__file__).parents[1] / "scripts" / "convert_dut.py"
    spec = importlib.util.spec_from_file_location("convert_dut_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_xml(path, filename, width, height, objects):
    rows = [
        "<annotation>",
        "<folder>train</folder>",
        f"<filename>{filename}</filename>",
        f"<size><width>{width}</width><height>{height}</height><depth>3</depth></size>",
    ]
    for name, box in objects:
        x1, y1, x2, y2 = box
        rows += [
            "<object>",
            f"<name>{name}</name>",
            "<difficult>0</difficult>",
            f"<bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox>",
            "</object>",
        ]
    rows.append("</annotation>")
    path.write_text("".join(rows), encoding="utf-8")


def test_dut_voc_conversion_keeps_empty_images_and_writes_data_yaml(tmp_path):
    script = load_converter()
    source = tmp_path / "DUT"
    (source / "train" / "img").mkdir(parents=True)
    (source / "train" / "xml").mkdir(parents=True)
    Image.new("RGB", (100, 60), "black").save(source / "train" / "img" / "00001.jpg")
    Image.new("RGB", (100, 60), "white").save(source / "train" / "img" / "00002.jpg")
    write_xml(source / "train" / "xml" / "00001.xml", "00001.jpg", 100, 60, [("UAV", (10, 12, 40, 30))])
    write_xml(source / "train" / "xml" / "00002.xml", "00002.jpg", 100, 60, [])
    output = tmp_path / "DUT_coco"

    assert script.main(["--source-root", str(source), "--output-root", str(output), "--splits", "train"]) == 0
    document = json.loads((output / "annotations" / "train.json").read_text(encoding="utf-8"))
    assert len(document["images"]) == 2
    assert len(document["annotations"]) == 1
    assert document["annotations"][0]["bbox"] == [10.0, 12.0, 31.0, 19.0]
    data = (output / "data.yaml").read_text(encoding="utf-8")
    assert "type: coco" in data and "train.json" in data
    dataset_config = yaml.safe_load(data)
    dataset = build_dataset(dataset_config, "train", image_size=64)
    assert len(dataset) == 2
    assert dataset[0][2]["boxes"].shape == (1, 4)
    assert dataset[1][2]["boxes"].shape == (0, 4)

    copied = tmp_path / "DUT_coco_copy"
    assert (
        script.main(
            [
                "--source-root",
                str(source),
                "--output-root",
                str(copied),
                "--splits",
                "train",
                "--image-mode",
                "copy",
            ]
        )
        == 0
    )
    assert (copied / "images" / "train" / "img" / "00001.jpg").is_file()
    copied_config = yaml.safe_load((copied / "data.yaml").read_text(encoding="utf-8"))
    copied_dataset = build_dataset(copied_config, "train", image_size=64)
    assert copied_dataset[0][2]["boxes"].shape == (1, 4)


def test_dut_voc_converter_strictly_rejects_unknown_class(tmp_path):
    script = load_converter()
    source = tmp_path / "DUT"
    (source / "train" / "img").mkdir(parents=True)
    (source / "train" / "xml").mkdir(parents=True)
    Image.new("RGB", (20, 20), "black").save(source / "train" / "img" / "00001.jpg")
    write_xml(source / "train" / "xml" / "00001.xml", "00001.jpg", 20, 20, [("car", (1, 1, 10, 10))])
    try:
        script.main(
            ["--source-root", str(source), "--output-root", str(tmp_path / "out"), "--splits", "train"]
        )
    except ValueError as exc:
        assert "unknown class" in str(exc)
    else:
        raise AssertionError("unknown class should fail in strict mode")
