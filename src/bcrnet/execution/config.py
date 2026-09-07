"""Execution choices are separate from learned ModelConfig and checkpoint weights."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionConfig:
    strategy: str = "packed"
    attention_backend: str = "auto"
    key_tile: int = 64
    overlap_threshold: float = 1.4
    minimum_references: int = 512
    policy_path: str | None = None
    trace: bool = False
    reuse_feature_masks: bool = False

    def __post_init__(self):
        if self.strategy not in {"reference", "packed", "shared", "indexed", "adaptive"}:
            raise ValueError(f"Unknown execution strategy: {self.strategy}")
        if self.attention_backend not in {"auto", "sdpa", "torch_indexed", "cuda_indexed"}:
            # Additional registered backends may be selected through the public registry.
            from .attention import ATTENTION_BACKENDS

            if self.attention_backend not in ATTENTION_BACKENDS:
                raise ValueError(f"Unknown attention backend: {self.attention_backend}")
        if self.key_tile < 1 or self.overlap_threshold < 1 or self.minimum_references < 0:
            raise ValueError("Invalid tile or adaptive policy parameters")

    @classmethod
    def from_dict(cls, value):
        return cls(**value)
