"""
@author supermantx
@date 2024/7/17 11:05
基础模块,先模仿mbf制作模块
"""
from torch import nn


class ConvBLock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, activation='prelu'):
        super(ConvBLock, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(get_activation(activation))

    def forward(self, x):
        return self.layers(x)



def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(0.1)
    elif activation == 'silu':
        return nn.SiLU()
    elif activation == 'prelu':
        return nn.PReLU()
    else:
        raise ValueError(f"activation {activation} is not supported.")
