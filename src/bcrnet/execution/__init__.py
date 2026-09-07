"""Opt-in, replaceable execution backends for the unchanged BCRNet model."""

from .attention import register_attention
from .config import ExecutionConfig
from .model import ExecutionBCRNet
from .plan import AccessPlan, build_access_plan

__all__ = ["AccessPlan", "ExecutionBCRNet", "ExecutionConfig", "build_access_plan", "register_attention"]
