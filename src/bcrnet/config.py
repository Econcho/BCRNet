"""Typed model configuration; unknown keys fail instead of silently being ignored."""

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    num_classes: int = 1
    stem_channels: int = 24
    backbone_channels: tuple[int, ...] = (48, 96, 160, 256)
    backbone_depths: tuple[int, ...] = (1, 2, 2, 2)
    width: int = 64
    core_size: int = 8
    halo: int = 4
    budget: int = 16
    coverage_fraction: float = 0.5
    objectness_factor: float = 0.5
    scene_dim: int = 16
    router_hidden: int = 32
    heads: int = 4
    blocks: int = 2
    mlp_ratio: int = 2
    tiny_threshold: float = 16.0
    use_detail: bool = True
    use_context: bool = True
    use_tiny: bool = True
    refiner_type: str = "attention"
    patch_chunk_size: int = 32
    default_mode: str = "full"

    def __post_init__(self):
        self.backbone_channels = tuple(self.backbone_channels)
        self.backbone_depths = tuple(self.backbone_depths)
        if len(self.backbone_channels) != 4 or len(self.backbone_depths) != 4:
            raise ValueError("Backbone needs four stages C2..C5")
        sizes = (
            self.num_classes,
            self.stem_channels,
            self.width,
            self.core_size,
            self.scene_dim,
            self.router_hidden,
            self.heads,
            self.blocks,
            self.mlp_ratio,
            self.patch_chunk_size,
            *self.backbone_channels,
        )
        if any(not isinstance(x, int) or x <= 0 for x in sizes):
            raise ValueError("Channel counts, block dimensions and chunk size must be positive integers")
        if any(not isinstance(x, int) or x < 0 for x in self.backbone_depths):
            raise ValueError("Backbone depths must be nonnegative integers")
        if self.core_size % 4 or self.halo < 0 or self.halo % 2:
            raise ValueError("core_size must be a multiple of 4; halo must be nonnegative and even")
        if self.width % self.heads or self.budget < 0:
            raise ValueError("width must divide heads; budget must be nonnegative")
        if not 0 <= self.coverage_fraction <= 1 or not 0 <= self.objectness_factor <= 1:
            raise ValueError("Routing fractions must be in [0,1]")
        if self.tiny_threshold <= 0 or self.refiner_type not in {"attention", "conv"}:
            raise ValueError("Invalid tiny threshold or refiner_type")

    @property
    def input_divisor(self):
        return math.lcm(32, self.core_size * 4)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model keys: {sorted(unknown)}")
        return cls(**value)


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a YAML mapping")
    return value
