"""
@author supermantx
@date 2024/7/17 11:05
基础模块,先模仿mbf制作模块
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy.random as random

import torch
import torch.nn as nn
import torch.nn.functional as F
from MinkowskiEngine import SparseTensor


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
    def __init__(self, in_channels, out_channels, residual=False, groups=1, inner_scale=1):
        super(DepthWise, self).__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, inner_scale * groups, kernel_size=1, stride=1, groups=1),
            LayerNorm(inner_scale * groups, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(inner_scale * groups, inner_scale * groups, kernel_size=3, stride=1, padding=1, groups=groups),
            nn.Conv2d(inner_scale * groups, out_channels, kernel_size=1, stride=1, groups=1),
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



class MinkowskiGRN(nn.Module):
    """ GRN layer for sparse tensors.
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key

        Gx = torch.norm(x.F, p=2, dim=0, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return SparseTensor(
            self.gamma * (x.F * Nx) + self.beta + x.F,
            coordinate_map_key=in_key,
            coordinate_manager=cm)


class MinkowskiDropPath(nn.Module):
    """ Drop Path for sparse tensors.
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(MinkowskiDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key
        keep_prob = 1 - self.drop_prob
        mask = torch.cat([
            torch.ones(len(_)) if random.uniform(0, 1) > self.drop_prob
            else torch.zeros(len(_)) for _ in x.decomposed_coordinates
        ]).view(-1, 1).to(x.device)
        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)
        return SparseTensor(
            x.F * mask,
            coordinate_map_key=in_key,
            coordinate_manager=cm)


class MinkowskiLayerNorm(nn.Module):
    """ Channel-wise layer normalization for sparse tensors.
    """

    def __init__(
            self,
            normalized_shape,
            eps=1e-6,
    ):
        super(MinkowskiLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, input):
        output = self.ln(input.F)
        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))
        # self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        # self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    def forward(self, x):
        shape = x.shape
        x = x.flatten(-2)
        Gx = torch.norm(x, p=2, dim=(-1), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        x = self.gamma * (x * Nx) + self.beta + x
        return x.reshape(*shape)
        # Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        # Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        # return self.gamma * (x * Nx) + self.beta + x