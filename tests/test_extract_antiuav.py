import importlib.util
from pathlib import Path

import pytest


def load_extraction_script():
    path = Path(__file__).parents[1] / "scripts" / "extract_antiuav.py"
    spec = importlib.util.spec_from_file_location("extract_antiuav_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extraction_yaml_defaults_and_cli_override(tmp_path):
    script = load_extraction_script()
    config = tmp_path / "extraction.yaml"
    config.write_text(
        """extraction:
  source_root: raw
  output_root: output
  positive_stride: 7
  negative_stride: 11
  tiny_densification: false
  image_format: png
  dry_run: true
""",
        encoding="utf-8",
    )
    args = script.parse_args(["--config", str(config), "--positive-stride", "3", "--dry-run"])
    assert args.source_root == Path("raw") and args.output_root == Path("output")
    assert args.positive_stride == 3 and args.negative_stride == 11
    assert not args.tiny_densification and args.image_format == "png" and args.dry_run
    assert not script.parse_args(["--config", str(config), "--no-dry-run"]).dry_run


def test_extraction_config_rejects_unknown_keys(tmp_path):
    script = load_extraction_script()
    config = tmp_path / "invalid.yaml"
    config.write_text("extraction:\n  mystery_option: 1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        script.parse_args(["--config", str(config)])


def test_extraction_summary_uses_reference_input_tiny_definition():
    script = load_extraction_script()
    document = {
        "images": [
            {"id": 1, "width": 1280, "height": 720, "sequence_id": "a"},
            {"id": 2, "width": 1280, "height": 720, "sequence_id": "a"},
        ],
        "annotations": [{"image_id": 1, "area": 400}],
    }
    summary = script.summarize_document(document, reference_size=640, tiny_threshold=16)
    assert summary == {
        "images": 2,
        "boxes": 1,
        "positive_frames": 1,
        "negative_frames": 1,
        "tiny_boxes_at_reference_size": 1,
        "sequences": 1,
        "per_sequence": {"a": {"images": 2, "positive_frames": 1, "negative_frames": 1}},
    }
