"""Attention backend registry. All backends implement the same indexed KV contract."""

import math

import torch
from torch.nn import functional as F

ATTENTION_BACKENDS = {}


def register_attention(name):
    def decorate(fn):
        if name in ATTENTION_BACKENDS:
            raise ValueError(f"Attention backend already registered: {name}")
        ATTENTION_BACKENDS[name] = fn
        return fn

    return decorate


@register_attention("sdpa")
def gathered_sdpa(query, kv_bank, inverse, valid, bias, *, key_tile=64):
    kv = kv_bank[inverse]
    key, value = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
    mask = bias[None].expand(query.shape[0], -1, -1, -1).masked_fill(~valid[:, None, None], -torch.inf)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0)


@register_attention("torch_indexed")
def tiled_indexed(query, kv_bank, inverse, valid, bias, *, key_tile=64):
    """Portable online-softmax oracle; only one key tile is gathered at a time."""
    n, heads, nq, dim = query.shape
    q = query.float()
    maximum = q.new_full((n, heads, nq, 1), -torch.inf)
    denominator = q.new_zeros(n, heads, nq, 1)
    accumulator = q.new_zeros(n, heads, nq, dim)
    # Do not let outer AMP downcast the FP32 streaming accumulations.
    with torch.autocast(device_type=query.device.type, enabled=False):
        for start in range(0, inverse.shape[1], key_tile):
            stop = start + key_tile
            kv = kv_bank[inverse[:, start:stop]].float()
            key, value = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
            score = q @ key.transpose(-2, -1) / math.sqrt(dim)
            score = score + bias[None, :, :, start:stop].float()
            score = score.masked_fill(~valid[:, None, None, start:stop], -torch.inf)
            next_max = torch.maximum(maximum, score.amax(-1, keepdim=True))
            safe_max = torch.where(torch.isfinite(next_max), next_max, 0)
            rescale = torch.exp(maximum - safe_max)
            probability = torch.exp(score - safe_max)
            accumulator = accumulator * rescale + probability @ value
            denominator = denominator * rescale + probability.sum(-1, keepdim=True)
            maximum = next_max
        return (accumulator / denominator.clamp_min(1e-30)).to(query.dtype)


@register_attention("cuda_indexed")
def native_indexed(query, kv_bank, inverse, valid, bias, *, key_tile=64):
    from .cuda_backend import indexed_attention

    return indexed_attention(query, kv_bank, inverse, valid, bias)


def run_shared_refiner(refiner, context, backend, *, chunk_size, key_tile, scope):
    """K/V computed once per bank per block; query work remains window-specific."""
    query = context.query
    for index, block in enumerate(refiner.blocks):
        with scope(f"block_{index}_kv"):
            width = query.shape[-1]
            kv = block.kv(block.z_norm(context.memory)).reshape(-1, 2, block.heads, block.head_dim)
        outputs = []
        with scope(f"block_{index}_attention_ffn"):
            for start in range(0, query.shape[0], chunk_size):
                stop = start + chunk_size
                local = query[start:stop]
                n, nq, _ = local.shape
                q = block.q(block.q_norm(local)).reshape(n, nq, block.heads, block.head_dim).transpose(1, 2)
                bias = block.relative_bias[block.relative_index].permute(2, 0, 1).to(q.dtype)
                attended = ATTENTION_BACKENDS[backend](
                    q,
                    kv,
                    context.inverse[start:stop],
                    context.valid[start:stop],
                    bias,
                    key_tile=key_tile,
                )
                update = attended.transpose(1, 2).reshape(n, nq, width)
                local = local + block.projection(update)
                outputs.append(local + block.ffn(block.ff_norm(local)))
            query = torch.cat(outputs)
    with scope("residual"):
        initial = context.core.flatten(2).transpose(1, 2)
        residual = refiner.residual_scale * refiner.output(query - initial)
        return residual.transpose(1, 2).reshape_as(context.core) * context.core_valid
