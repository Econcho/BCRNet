"""Materialize packed or unique context prefixes under the same address contract."""

from dataclasses import dataclass

import torch

from .plan import read_core, read_feature_mask


@dataclass
class ContextBank:
    query: torch.Tensor
    memory: torch.Tensor
    inverse: torch.Tensor
    valid: torch.Tensor
    core: torch.Tensor
    core_valid: torch.Tensor

    def materialize(self):
        return self.memory[self.inverse]


def prepare_context(reader, features, masks, plan, *, scope):
    with scope("core_read"):
        core = read_core(features["e2"], plan)
        core_valid = read_core(masks["e2"], plan).bool()
    with scope("detail_prepare"):
        dm = plan.detail
        if reader.cfg.use_detail:
            raw, mask = read_feature_mask(features["d1"], masks["d1"], dm, scale=2, support=2)
            # Pixel-unshuffle channel order is [channel, dy, dx], not [dy, dx, channel].
            raw = (raw * mask).transpose(1, 2).flatten(1)
            dvalid = mask.any(1).squeeze(-1)
        else:
            raw, mask = read_feature_mask(features["e2"], masks["e2"], dm)
            raw, mask = raw.squeeze(1), mask.squeeze(1)
            raw = raw * mask
            dvalid = mask.squeeze(-1).bool()
        detail = reader.detail_projection(reader.detail_norm(raw))
        side, c, h = plan.core_size + 2 * plan.halo, plan.core_size, plan.halo
        core_inverse = dm.inverse.reshape(plan.slots, side, side)[:, h : h + c, h : h + c].reshape(
            plan.slots, c * c
        )
        query = core.flatten(2).transpose(1, 2) + reader.detail_scale * detail[core_inverse]
        memories = [detail + reader.types[0]]
        inverses, validity = [dm.inverse], [dvalid[dm.inverse]]
    if plan.semantic is not None:
        with scope("semantic_prepare"):
            sm = plan.semantic
            raw, mask = read_feature_mask(features["e3"], masks["e3"], sm, support=2)
            # Original reader: avg(masked values) / avg(mask). Accumulation stays FP32 under AMP.
            pooled = (raw * mask).sum(1) / mask.sum(1).clamp_min(1)
            semantic = reader.semantic_norm(pooled)
            memories.append(semantic + reader.types[1])
            inverses.append(sm.inverse + detail.shape[0])
            validity.append(mask.any(1).squeeze(-1)[sm.inverse])
    with scope("bank_assemble"):
        bank = torch.cat(memories)
        inverse = torch.cat(inverses, dim=1)
        valid = torch.cat(validity, dim=1).bool()
        empty = ~valid.any(1)
        dummy_id = bank.shape[0]
        # Empty reference windows have zero memory BEFORE per-block z_norm/kv (including affine bias).
        bank = torch.cat((bank, bank.new_zeros(1, bank.shape[-1])))
        inverse = torch.where(empty[:, None], dummy_id, inverse)
        first = torch.arange(valid.shape[1], device=valid.device)[None] == 0
        valid = valid | (empty[:, None] & first)
    return ContextBank(query, bank, inverse, valid, core, core_valid)
