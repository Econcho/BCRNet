"""Count executed Conv/Linear and attention matrix MACs, with explicit exclusions."""

import torch
from torch import nn

from .models.refinement import CrossAttentionBlock


@torch.no_grad()
def count_macs(model, images, mode="full"):
    totals = {"conv": 0, "linear": 0, "attention_matmul": 0}
    hooks = []

    def collect(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            totals["conv"] += (
                output.numel()
                * (module.in_channels // module.groups)
                * module.kernel_size[0]
                * module.kernel_size[1]
            )
        elif isinstance(module, nn.Linear):
            totals["linear"] += output.numel() * module.in_features
        elif isinstance(module, CrossAttentionBlock):
            query, memory = inputs[:2]
            totals["attention_matmul"] += (
                2 * query.shape[0] * query.shape[1] * memory.shape[1] * query.shape[2]
            )

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, CrossAttentionBlock)):
            hooks.append(module.register_forward_hook(collect))
    training_states = {module: module.training for module in model.modules()}
    model.eval()
    try:
        model(images, mode=mode)
    finally:
        for hook in hooks:
            hook.remove()
        for module, state in training_states.items():
            module.training = state
    return {
        "per_batch": totals,
        "GMAC_per_image": sum(totals.values()) / images.shape[0] / 1e9,
        "convention": "one multiply-accumulate = one MAC; Conv, Linear, QK and AV only",
        "excluded": "bias, normalization, activations, softmax, pooling, sorting, gather/scatter and decode",
    }
