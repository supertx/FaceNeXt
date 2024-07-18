"""
@author supermantx
@date 2024/7/17 9:55
"""

import torch
from torch import nn
import MinkowskiEngine
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU
)
from MinkowskiOps import (
    to_sparse
)

from models.utils import (
    LayerNorm,
    MinkowskiLayerNorm,
    MinkowskiGRN,
    MinkowskiDropPath
)
from models import ConvBLock, get_activation


class DownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, D=3):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(
            MinkowskiLayerNorm(in_channels, 1e-6)
        )
        self.layer.append(
            MinkowskiConvolution(in_channels, out_channels, kernel_size, stride, bias=True, dimension=D)
        )

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


class DepthWise(nn.Module):
    """
    the block here is based on the mobilenet not the ConvNeXtv2,
    ConvNeXtv2 could take into account afterward
    """

    def __init__(self, in_channels, out_channels, residual=True, kernel=3, groups=1, stride=2, padding=1, D=3):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList()
        self.layers.extend(
            [MinkowskiConvolution(in_channels, groups, kernel_size=1, stride=1, bias=True, dimension=D),
             MinkowskiLayerNorm(groups, 1e-6),
             MinkowskiDepthwiseConvolution(groups, kernel_size=3, bias=True, dimension=D),
             MinkowskiConvolution(groups, out_channels, kernel_size=1, stride=1, bias=True, dimension=D),
             MinkowskiGELU(),
             MinkowskiGRN(out_channels)]
        )

    def forward(self, x):
        y = self.layers(x)
        if self.residual:
            return y + x
        else:
            return x


class StageBlock(nn.Module):

    def __init__(self, channels, num_block, groups, kernel=3, stride=1, padding=1, D=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_block):
            self.layers.append(DepthWise(channels, channels, True, kernel, groups, stride, padding, D))

    def forward(self, x):
        return self.layers(x)


class MBFSparse(nn.Module):
    def __init__(self, fp16=False, num_feature=512, stage=(3, 9, 3, 3), scale=2):
        super().__init__()
        self.scale = scale
        self.fp16 = fp16
        self.stem = ConvBLock(in_channels=3, out_channels=64 * scale, kernel_size=3, stride=2, padding=1)
        self.stage_channel = [64, 64, 128, 128]
        self.down_sample = nn.ModuleList()
        self.stages = nn.ModuleList()

        for i in range(len(stage)):
            self.stages.append(StageBlock(self.stage_channel[i] * scale, stage[i], self.stage_channel[i] * scale, D=3))

        for i in range(len(stage) - 1):
            self.down_sample.append(
                DownSampleLayer(self.stage_channel[i] * scale, self.stage_channel[i + 1] * scale, 3, 2))

        self.conv_sep = MinkowskiConvolution(self.stage_channel[-1] * scale,
                                             512, kernel_size=1, stride=1, bias=True, dimension=3)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            nn.init.normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiDepthwiseConvolution):
            nn.init.normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            nn.init.normal_(m.linear.weight, std=.02)
            nn.init.constant_(m.linear.bias, 0)
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=.01)
            nn.init.constant_(m.bias, 0)
    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p). \
            repeat_interleave(scale, axis=1). \
            repeat_interleave(scale, axis=2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        num_stages = len(self.stages)
        mask = self.upsample_mask(mask, 2 ** (num_stages - 1))
        mask = mask.unsqueeze(1).type_as(x)

        x = self.stem(x)
        x *= (1 - mask)
        x = to_sparse(x)
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            if i < len(self.down_sample):
                x = self.down_sample[i](x)
        x = self.conv_sep(x)

        x = x.dense()[0]
        return x


if __name__ == '__main__':
    model = MBFSparse()
    input = torch.randn(1, 3, 112, 112)
    y = model(input, torch.randn(1, 49))
    print(y.shape)
    print(y)