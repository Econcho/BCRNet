from torch import nn
from torch.nn import functional as F

from .common import DepthwiseSeparable, conv_bn_act


class LightweightFPN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lateral = nn.ModuleDict(
            {
                f"c{i}": conv_bn_act(c, cfg.width, activate=False)
                for i, c in enumerate(cfg.backbone_channels, 2)
            }
        )
        self.smooth = nn.ModuleDict({f"p{i}": DepthwiseSeparable(cfg.width, cfg.width) for i in (2, 3, 4)})

    def forward(self, features):
        x = self.lateral["c5"](features["c5"])
        out = {"t5": x}
        for i in (4, 3, 2):
            lateral = self.lateral[f"c{i}"](features[f"c{i}"])
            x = self.smooth[f"p{i}"](lateral + F.interpolate(x, size=lateral.shape[-2:], mode="nearest"))
            out[f"p{i}"] = x
        return out
