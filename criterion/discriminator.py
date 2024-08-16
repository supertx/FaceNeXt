"""
@author supermantx
@date 2024/8/12 13:55
使用gan的判别器,判断生成的人脸喝真实人脸的差别
"""
from torch import nn
from torch.nn import functional as F


class DownBlock2d(nn.Module):

    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        if norm:
            self.norm = nn.InstanceNorm2d(out_features)
        else:
            self.norm = None
        self.pool = nn.AvgPool2d(2) if pool else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = F.leaky_relu(x, 0.2)
        if self.pool is not None:
            x = self.pool(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, channel_basic=64, num_blocks=4, max_feature=512, sn=False):
        super().__init__()

        down_blocks = []
        i = 0
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(3 if i == 0 else min(max_feature, channel_basic * (2 ** i)),
                            min(max_feature, channel_basic * (2 ** (i + 1))),
                            norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn)
            )

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(min(max_feature, channel_basic * (2 ** i)), out_channels=1, kernel_size=1)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        for block in self.down_blocks:
            x = block(x)
        x = self.conv(x)
        return x
