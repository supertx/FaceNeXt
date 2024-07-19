import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF


class ExampleNetwork(ME.MinkowskiNetwork):
    def __init__(self, in_feat, out_feat, D=3):
        super(ExampleNetwork, self).__init__(D)
        self.conv =  ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=D)
        self.bn = ME.MinkowskiBatchNorm(64)
        self.conv_tr = ME.MinkowskiConvolutionTranspose(
                in_channels=64,
                out_channels=4,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=D)
    def forward(self, x):
        print('input: ', x.coordinates.size(), x.features.size())
        out = self.conv(x)
        print('conv: ', out.coordinates.size(), out.features.size())
        out = self.bn(out)
        print('bn: ', out.coordinates.size(), out.features.size())
        out = MEF.relu(out)
        print('relu: ', out.coordinates.size(), out.features.size())
        out = self.conv_tr(out)
        print('conv_tr', out.coordinates.size(), out.features.size())
        return out


if __name__ == '__main__':
    origin_pc1 = 5 * np.random.uniform(0, 1, (100, 3))
    feat1 = np.ones((100, 3), dtype=np.float32)
    origin_pc2 = 100 * np.random.uniform(0, 1, (6, 3))
    feat2 = np.ones((6, 3), dtype=np.float32)

    coords, feats = ME.utils.sparse_collate([origin_pc1, origin_pc2], [feat1, feat2])
    input = ME.SparseTensor(feats, coordinates=coords)

    net = ExampleNetwork(in_feat=3, out_feat=32)
    output = net(input)

    print(torch.equal(input.coordinates, output.coordinates))
    print(torch.equal(input.features, output.features))
