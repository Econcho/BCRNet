from torch import nn

from .common import DepthwiseSeparable, InvertedResidual, conv_bn_act


class LightweightBackbone(nn.Module):
    """Contract: input NCHW RGB, return d1/c2/c3/c4/c5 at strides 2/4/8/16/32."""

    def __init__(self, cfg):
        super().__init__()
        self.stem = conv_bn_act(3, cfg.stem_channels, 3, 2)
        previous = cfg.stem_channels
        stages = []
        for channels, depth in zip(cfg.backbone_channels, cfg.backbone_depths):
            stages.append(
                nn.Sequential(
                    DepthwiseSeparable(previous, channels, 2),
                    *[InvertedResidual(channels) for _ in range(depth)],
                )
            )
            previous = channels
        self.stages = nn.ModuleList(stages)

    def forward(self, images):
        x = self.stem(images)
        outputs = {"d1": x}
        for i, stage in enumerate(self.stages, 2):
            x = stage(x)
            outputs[f"c{i}"] = x
        return outputs
