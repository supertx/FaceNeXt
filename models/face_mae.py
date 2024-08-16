"""
@author supermantx
@date 2024/7/19 14:24
模仿ConNeXtv2的MAE应用到mobilenet上
"""
import torch
from torch import nn

from models import DepthWise, MBFSparse
from utils import generate_landmark_mask


class Decoder(nn.Module):
    def __init__(self, in_channels, decoder_embed_dim, decoder_depth, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=decoder_embed_dim,
            kernel_size=1
        )

        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [DepthWise(decoder_embed_dim, decoder_embed_dim, residual=True, groups=decoder_embed_dim, inner_scale=4) for _ in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)
        self.pred = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=patch_size ** 2 * 3,
            kernel_size=1
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=.01)
            nn.init.constant_(m.bias, 0)
        if hasattr(self, 'mask_token'):
            torch.nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x, mask):
        x = self.proj(x)
        mask = mask.reshape(-1, *x.shape[2:]).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1 - mask) + mask_token * mask
        x = self.decoder(x)
        pred = self.pred(x)
        return pred


class FaceMAE(nn.Module):
    def __init__(self,
                 depth=(9, 9, 27, 9),
                 dims=(128, 256, 512, 512),
                 decoder_depth=1,
                 decoder_embed_dim=512,
                 patch_size=16,
                 mask_landmark_num=4,
                 mask_ratio=0.6,
                 norm_pix_loss=True,
                 inner_scale=1
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_landmark_num = mask_landmark_num
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.inner_scale = inner_scale
        self.encoder = MBFSparse(stages=depth, stages_channel=dims, inner_scale=inner_scale)

        self.decoder = Decoder(in_channels=decoder_embed_dim,
                               decoder_embed_dim=decoder_embed_dim,
                               decoder_depth=decoder_depth,
                               patch_size=patch_size)

    def get_random_mask(self, x_shape, mask_ratio, device):
        patch_num = (x_shape[2] // self.patch_size) ** 2
        noise = torch.randn(x_shape[0], patch_num, device=device)
        len_keep = int(patch_num * (1 - mask_ratio))

        # get shuffle id
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.ones(x_shape[0], patch_num, device=device)
        mask[:, :len_keep] = 0
        # shuffle mask
        mask = torch.gather(mask, dim=1, index=ids_shuffle)
        return mask

    def patchify(self, img):
        """
        img: (N, C, H, W)
        return: (N, L, patch_size**2 * C)
        """
        assert img.shape[2] == img.shape[3] and img.shape[2] % self.patch_size == 0

        h = w = img.shape[2] // self.patch_size
        x = img.reshape(img.shape[0], img.shape[1], h, self.patch_size, w, self.patch_size)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(img.shape[0], h * w, self.patch_size ** 2 * img.shape[1])
        return x

    def unpatchify(self, x):
        """
        x: (N, patchsize ** 2 * C, h, w)
        return: (N, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[2])
        # assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], p, p, 3, h, w))
        x = torch.einsum('npqchw->nchpwq', x)
        img = x.reshape(x.shape[0], 3, h * p, w * p)
        return img

    def compute_loss(self, imgs, pred, mask):
        """
        img: (N, C, H, W)
        pred: (N, C * `patch_size **2, L ** .5, L ** .5)
        mask: (N, L) 1 is masked
        """
        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum('ncl->nlc', pred)
        target = self.patchify(imgs)
        # 根据每个patch进行归一化
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, anno):
        mask = generate_landmark_mask(imgs.shape[2], self.patch_size, anno,
                                      mask_landmark_num=self.mask_landmark_num, mask_ratio=self.mask_ratio)
        x = self.encoder(imgs, mask)
        pred = self.decoder(x, mask)
        # loss = self.compute_loss(imgs, pred, mask)
        # return loss, pred, mask
        return self.unpatchify(pred), mask



