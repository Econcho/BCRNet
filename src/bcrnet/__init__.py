"""BCRNet: Budgeted Contextual Refinement Network."""

from .config import ModelConfig
from .models.detector import BCRNet, ForwardMode

__version__ = "0.1.0"
__all__ = ["BCRNet", "ForwardMode", "ModelConfig"]
