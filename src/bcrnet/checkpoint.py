"""Versioned epoch-boundary checkpoints with safe tensor/primitive deserialization."""

import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .models.detector import BCRNet


def rng_state():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, map_location="cpu"):
    value = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(value, dict) or value.get("format_version") != 1:
        raise ValueError("Expected a BCRNet format_version=1 checkpoint")
    return value


def load_model(path, device="cpu"):
    checkpoint = load_checkpoint(path)
    model = BCRNet(ModelConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device), checkpoint


def load_compatible_weights(model, path):
    checkpoint = load_checkpoint(path)
    own = model.state_dict()
    compatible = {
        key: value
        for key, value in checkpoint["model"].items()
        if key in own and own[key].shape == value.shape
    }
    if not compatible:
        raise ValueError("No compatible checkpoint tensors")
    result = model.load_state_dict(compatible, strict=False)
    return {
        "loaded_tensors": len(compatible),
        "missing": result.missing_keys,
        "skipped": sorted(set(checkpoint["model"]) - set(compatible)),
    }
