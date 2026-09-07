import torch
from torch import nn
from torch.nn import functional as F

from .windows import gather_regions


class ContextReader(nn.Module):
    """Read only selected regions; no global unfolding or full-image detail projection."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        detail_dim = cfg.stem_channels * 4 if cfg.use_detail else cfg.width
        self.detail_norm = nn.LayerNorm(detail_dim)
        self.detail_projection = nn.Linear(detail_dim, cfg.width)
        self.semantic_norm = nn.LayerNorm(cfg.width)
        self.types = nn.Parameter(torch.zeros(2, cfg.width))
        self.detail_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features, masks, indices):
        c, h = self.cfg.core_size, self.cfg.halo
        grid_width = features["e2"].shape[-1] // c
        b, k = indices.shape
        core = gather_regions(features["e2"], indices, grid_width, c).flatten(0, 1)
        core_mask = gather_regions(masks["e2"].float(), indices, grid_width, c).flatten(0, 1).bool()
        if self.cfg.use_detail:
            raw = gather_regions(features["d1"], indices, grid_width, c * 2, h * 2).flatten(0, 1)
            raw_mask = gather_regions(masks["d1"].float(), indices, grid_width, c * 2, h * 2).flatten(0, 1)
            detail = F.pixel_unshuffle(raw * raw_mask, 2)
            detail_mask = F.max_pool2d(raw_mask, 2).flatten(1).bool()
        else:
            detail = gather_regions(features["e2"], indices, grid_width, c, h).flatten(0, 1)
            raw_mask = gather_regions(masks["e2"].float(), indices, grid_width, c, h).flatten(0, 1)
            detail = detail * raw_mask
            detail_mask = raw_mask.flatten(1).bool()
        extent = c + 2 * h
        zd = self.detail_projection(self.detail_norm(detail.flatten(2).transpose(1, 2)))
        detail_core = zd.reshape(b * k, extent, extent, -1)[:, h : h + c, h : h + c].reshape(b * k, c * c, -1)
        query = core.flatten(2).transpose(1, 2) + self.detail_scale * detail_core
        memories, valid_memories = [zd + self.types[0]], [detail_mask]
        if self.cfg.use_context:
            sem = gather_regions(features["e3"], indices, grid_width, c // 2, h // 2).flatten(0, 1)
            smask = gather_regions(masks["e3"].float(), indices, grid_width, c // 2, h // 2).flatten(0, 1)
            sem = F.avg_pool2d(sem * smask, 2) / F.avg_pool2d(smask, 2).clamp_min(1e-6)
            sem_mask = F.max_pool2d(smask, 2).flatten(1).bool()
            zs = self.semantic_norm(sem.flatten(2).transpose(1, 2))
            memories.append(zs + self.types[1])
            valid_memories.append(sem_mask)
        memory = torch.cat(memories, 1)
        valid = torch.cat(valid_memories, 1)
        # A harmless dummy key avoids all-masked rows; final query residuals are masked to zero.
        empty = ~valid.any(1)
        valid = valid.clone()
        valid[empty, 0] = True
        memory = torch.where(empty[:, None, None], torch.zeros_like(memory), memory)
        return query, memory, valid, core, core_mask


def relative_index(cfg):
    c, h = cfg.core_size, cfg.halo

    def grid(size, scale=1, origin=0):
        a = (torch.arange(size, dtype=torch.float32) + 0.5) * scale + origin
        y, x = torch.meshgrid(a, a, indexing="ij")
        return torch.stack((y, x), -1).reshape(-1, 2)

    query = grid(c)
    memory = [grid(c + 2 * h, origin=-h)]
    if cfg.use_context:
        memory.append(grid((c + 2 * h) // 4, scale=4, origin=-h))
    delta = ((query[:, None] - torch.cat(memory)[None]) * 2).round().long()
    unique, index = torch.unique(delta.reshape(-1, 2), dim=0, return_inverse=True)
    return index.reshape(c * c, -1), len(unique)


class CrossAttentionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.width // cfg.heads
        self.q_norm, self.z_norm, self.ff_norm = (nn.LayerNorm(cfg.width) for _ in range(3))
        self.q = nn.Linear(cfg.width, cfg.width)
        self.kv = nn.Linear(cfg.width, cfg.width * 2)
        self.projection = nn.Linear(cfg.width, cfg.width)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.width, cfg.width * cfg.mlp_ratio),
            nn.GELU(),
            nn.Linear(cfg.width * cfg.mlp_ratio, cfg.width),
        )
        index, count = relative_index(cfg)
        self.register_buffer("relative_index", index)
        self.relative_bias = nn.Parameter(torch.zeros(count, cfg.heads))

    def forward(self, query, memory, valid):
        n, nq, width = query.shape
        nm = memory.shape[1]
        q = self.q(self.q_norm(query)).reshape(n, nq, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(self.z_norm(memory)).reshape(n, nm, 2, self.heads, self.head_dim)
        key, value = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        bias = self.relative_bias[self.relative_index].permute(2, 0, 1).to(q.dtype)[None]
        mask = bias.expand(n, -1, -1, -1).masked_fill(~valid[:, None, None], -torch.inf)
        attention = F.scaled_dot_product_attention(q, key, value, attn_mask=mask, dropout_p=0.0)
        update = attention.transpose(1, 2).reshape(n, nq, width)
        query = query + self.projection(update)
        return query + self.ffn(self.ff_norm(query))


class ContextRefiner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.refiner_type == "attention":
            self.blocks = nn.ModuleList(CrossAttentionBlock(cfg) for _ in range(cfg.blocks))
        else:
            # GroupNorm avoids sparse-patch batch-statistic dependence in the conv ablation.
            self.blocks = nn.ModuleList(
                nn.Sequential(
                    nn.Conv2d(cfg.width, cfg.width, 3, padding=1, groups=cfg.width),
                    nn.GroupNorm(1, cfg.width),
                    nn.SiLU(),
                    nn.Conv2d(cfg.width, cfg.width * cfg.mlp_ratio, 1),
                    nn.GELU(),
                    nn.Conv2d(cfg.width * cfg.mlp_ratio, cfg.width, 1),
                )
                for _ in range(cfg.blocks)
            )
        self.output = nn.Linear(cfg.width, cfg.width)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, query, memory, memory_valid, core, core_valid):
        initial = core.flatten(2).transpose(1, 2)
        if self.cfg.refiner_type == "attention":
            for block in self.blocks:
                query = block(query, memory, memory_valid)
        else:
            context = (memory * memory_valid[..., None]).sum(1) / memory_valid.sum(1, keepdim=True).clamp_min(
                1
            )
            x = (query + context[:, None]).transpose(1, 2).reshape_as(core)
            for block in self.blocks:
                x = x + block(x)
            query = x.flatten(2).transpose(1, 2)
        residual = self.residual_scale * self.output(query - initial)
        return residual.transpose(1, 2).reshape_as(core) * core_valid
