"""
@author supermantx
@date 2024/7/29 14:43
mobileFacenet的非稀疏版本,和原版的mobileFacenet有所区别
"""
import torch
from torch import nn

from models import DepthWise, LayerNorm, ConvBlock


class DownSampleLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(
            LayerNorm(in_channels, 1e-6, data_format="channels_first")
        )
        self.layer.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=True)
        )

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


class StageBlock(nn.Module):

    def __init__(self, channels, num_block, group, inner_scale=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_block):
            self.layers.append(DepthWise(channels, channels, True, groups=group, inner_scale=inner_scale))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GDC(nn.Module):

    def __init__(self, embedding_size=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=7,
                      stride=1, padding=0, groups=512, bias=False),
            LayerNorm(512, 1e-6, data_format="channels_first"),
            nn.Flatten(),
            nn.Linear(512, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        return self.layers(x)


class MBF(nn.Module):
    """
    model structure mostly like the mbf_sparse, but different at the last layer
    MBF is for face recognition, so the last layer is average pooling
    """

    def __init__(self, fp16=False, num_features=512,
                 stages=(3, 3, 9, 3), stages_channel=(128, 128, 256, 256),
                 inner_scale=1):
        super().__init__()
        self.fp16 = fp16
        self.stem = ConvBlock(in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.stages_channel = stages_channel
        self.stages = stages
        self.inner_scale = inner_scale
        self.down_sample = nn.ModuleList()
        self.stage_blocks = nn.ModuleList()

        for i in range(len(stages) - 1):
            self.down_sample.append(
                DownSampleLayer(stages_channel[i], stages_channel[i + 1], 3, 2))

        for i in range(len(stages)):
            self.stage_blocks.append(StageBlock(stages_channel[i], stages[i], stages_channel[i], inner_scale))

        self.conv_sep = ConvBlock(stages_channel[-1], 512, kernel_size=1, stride=1, padding=0)
        self.features = GDC(num_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, LayerNorm):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                m.bias.data.zero_()

    def frozen(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfrozen(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        # with torch.cuda.amp.autocast(self.fp16):
        x = self.stem(x)
        for i in range(len(self.stages)):
            x = self.stage_blocks[i](x)
            if i < len(self.stages) - 1:
                x = self.down_sample[i](x)
        x = self.conv_sep(x)
        x = self.features(x)
        return x


if __name__ == '__main__':
    model = MBF()
    x = torch.randn(2, 3, 112, 112)
    y = model(x)
    print(y.shape)
    # print(y)
    print("model parameters: ", sum(p.numel() for p in model.parameters()))
    print("model size: ", sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024, "MB")
