import math

import torch
from torch import nn


class CenterHead(nn.Module):
    """Shared pointwise head: C center logits, 2 offsets, 2 log sizes."""

    def __init__(self, width, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.center = nn.Conv2d(width, num_classes, 1)
        self.offset = nn.Conv2d(width, 2, 1)
        self.size = nn.Conv2d(width, 2, 1)
        nn.init.constant_(self.center.bias, math.log(0.01 / 0.99))
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.size.bias)

    def forward(self, features):
        return torch.cat((self.center(features), self.offset(features), self.size(features)), dim=1)
