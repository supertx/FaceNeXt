"""
@author supermantx
@date 2024/7/17 9:55
"""

import torch
from torch import nn
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU
)

from .utils import (
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
            MinkowskiConvolution(in_channels, out_channels, kernel_size, stride, bias=True, D=D)
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
        self.layer = nn.ModuleList()
        self.layer.append(
            MinkowskiConvolution(in_channels, groups, kernel_size=1, stride=1, D=D)
        )

class MBFSparse(nn.Module):
    def __init__(self, fp16=False, num_feature=512, stage=(3, 9, 3, 3), scale=2):
        super().__init__()
        self.scale = scale
        self.fp16 = fp16
        self.stem = ConvBLock(in_channels=3, out_channels=64 * scale, kernel_size=3, stride=2, padding=1)

        self.down_sample = nn.ModuleList()
        self.stages = nn.ModuleList()
        for i in range(len(stage)):
            pass
