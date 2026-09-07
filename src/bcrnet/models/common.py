from torch import nn


def conv_bn_act(ci, co, kernel=1, stride=1, groups=1, activate=True):
    layers = [nn.Conv2d(ci, co, kernel, stride, kernel // 2, groups=groups, bias=False), nn.BatchNorm2d(co)]
    if activate:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class DepthwiseSeparable(nn.Sequential):
    def __init__(self, ci, co, stride=1):
        super().__init__(conv_bn_act(ci, ci, 3, stride, groups=ci), conv_bn_act(ci, co))


class InvertedResidual(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.body = nn.Sequential(
            conv_bn_act(channels, hidden),
            conv_bn_act(hidden, hidden, 3, groups=hidden),
            conv_bn_act(hidden, channels, activate=False),
        )

    def forward(self, x):
        return x + self.body(x)
