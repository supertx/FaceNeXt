"""
@author supermantx
@date 2024/8/16 13:00
"""
import os

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F

from criterion.iresnet import iresnet100
from criterion.discriminator import Discriminator


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def fcd_loss(raw_logits, pred_logit):
    # 将特征归一化
    raw_logits = normalize(raw_logits)
    pred_logit = normalize(pred_logit)
    # 计算l2距离
    loss_fcd = F.mse_loss(pred_logit, raw_logits, reduction="mean")
    return loss_fcd


def compute_pixel_loss(imgs, pred, mask, norm_pix_loss=True):
    """
    img: (N, C, H, W)
    pred: (N, C * patch_size **2, h * w) -> (N, C, p * h, p * w)
    mask: (N, L) 1 is masked
    """
    patch_size = int((pred.shape[1] / 3) ** 0.5)
    if len(pred.shape) == 3:
        pred = rearrange(pred, 'n (c p q) (h w) -> n c (p h) (q w)', p=patch_size, q=patch_size,
                         h=imgs.shape[2] // patch_size, w=imgs.shape[3] // patch_size)
    # 根据每个patch进行归一化
    if norm_pix_loss:
        mean = imgs.mean(dim=(2, 3), keepdim=True)
        var = imgs.var(dim=(2, 3), keepdim=True)
        imgs = (imgs - mean) / (var + 1e-6) ** .5
    loss = (pred - imgs) ** 2
    loss = loss.mean(dim=-1)
    mask = rearrange(mask, 'n (h w) -> n 1 h w', h=imgs.shape[2] // patch_size, w=imgs.shape[3] // patch_size)
    mask = F.interpolate(mask, size=(imgs.shape[2], imgs.shape[3]), mode='nearest')
    loss = (loss * mask).sum() / mask.sum()
    return loss


class PretrainLoss(nn.Module):

    def __init__(self, cfg, discriminator, weight_dict=None):
        super().__init__()
        self.arcface = iresnet100(pretrained=True)
        assert os.path.isfile(cfg.model.arcface_pth), f"{cfg.model.arcface_pth} not found"
        self.arcface.load_state_dict(torch.load(cfg.model.arcface_pth))
        self.arcface.eval()
        self.arcface.cuda()
        self.discriminator = discriminator
        self.gan_loss = nn.MSELoss()
        if not weight_dict:
            self.weight_dict = {'arcface': 1, 'gan': 1, 'pixel': 1}
        else:
            self.weight_dict = weight_dict

    def forward(self, raw_img, pred_img, mask):
        raw_logits = self.arcface(raw_img)
        pred_logits = self.arcface(pred_img)
        torch.cuda.empty_cache()
        arcface_loss = fcd_loss(raw_logits, pred_logits)
        gan_loss = self.gan_loss(self.discriminator(pred_img), torch.zeros(pred_img.size(0), 1).to(pred_img.device))
        pixel_loss = compute_pixel_loss(raw_img, pred_img, mask)
        return (self.weight_dict['arcface'] * arcface_loss
                + self.weight_dict['gan'] * gan_loss
                + self.weight_dict['pixel'] * pixel_loss)


class DiscriminatorLoss(nn.Module):

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator
        self.gan_loss = nn.MSELoss()

    def forward(self, raw_img, pred_img):
        real_loss = self.gan_loss(self.discriminator(raw_img), torch.ones(raw_img.size(0), 1).to(raw_img.device))
        fake_loss = self.gan_loss(self.discriminator(pred_img), torch.zeros(pred_img.size(0), 1).to(pred_img.device))
        return real_loss + fake_loss