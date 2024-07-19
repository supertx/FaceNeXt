"""
@author supermantx
@date 2024/7/17 11:05
基础模块,先模仿mbf制作模块
"""
from torch import nn
from models.utils import LayerNorm, GRN


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, activation='prelu'):
        super(ConvBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(get_activation(activation))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(0.1)
    elif activation == 'silu':
        return nn.SiLU()
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"activation {activation} is not supported.")


class DepthWise(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False, groups=1):
        super(DepthWise, self).__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, groups, kernel_size=1, stride=1, groups=1),
            LayerNorm(groups, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(groups, groups, kernel_size=3, stride=1, padding=1, groups=groups),
            nn.Conv2d(groups, out_channels, kernel_size=1, stride=1, groups=1),
            get_activation("gelu"),
            GRN(out_channels)
        )

    def forward(self, x):
        short_cut = None
        if self.residual:
            short_cut = x
        x = self.layers(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output
