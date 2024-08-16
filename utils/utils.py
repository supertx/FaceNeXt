import math
import random
from collections import OrderedDict

import torch
from torch.nn import functional as F


def unpatchify(x, p):
    """
    x: (patchsize ** 2 * C, h, w)
    return: (C, H, W)
    """
    h = w = int(x.shape[1])

    x = x.reshape(shape=(p, p, 3, h, w))
    x = torch.einsum('pqchw->chpwq', x)
    img = x.reshape(3, h * p, w * p)
    return img


def patchify(img, p):
    """
    img: (C, H, W)
    return: (L, patch_size**2 * C)
    """

    h = w = img.shape[2] // p
    x = img.reshape(3, h, p, w, p)
    x = torch.einsum('chpwq->hwpqc', x)
    x = x.reshape(h * w, p ** 2 * 3)
    return x


def organize_model_weights(state_dict):
    """
    input state_dict is a sparse_model params, out put a non-sparse model params
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("encoder"):
            key_new = key.replace("encoder.", "")
            key_new = key_new.replace("conv_sep.", "conv_sep.layers.0.")
            if key_new.endswith("kernel"):
                key_new = key_new.replace(".kernel", ".weight")
                if len(value.shape) == 3:
                    kv, in_dim, out_dim = value.shape
                    ks = int(math.sqrt(kv))
                    new_state_dict[key_new] = value.permute(2, 1, 0). \
                        reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
                elif len(value.shape) == 2 and value.shape[0] != value.shape[1] and "conv_sep" not in key_new:
                    kv, dim = value.shape
                    ks = int(math.sqrt(kv))
                    new_state_dict[key_new] = value.permute(1, 0). \
                        reshape(dim, ks, ks).unsqueeze(1).transpose(3, 2)
                else:
                    new_state_dict[key_new] = value.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
            elif 'ln' in key_new or 'linear' in key_new:
                key_new = key_new.split('.')
                key_new.pop(-2)
                key_new = '.'.join(key_new)
                new_state_dict[key_new] = value
            else:
                new_state_dict[key_new] = value
    for key, value in new_state_dict.items():
        if key.endswith('bias') and len(value.shape) != 1:
            new_state_dict[key] = value.reshape(-1)
        if key.endswith('gamma') or key.endswith('beta'):
            new_state_dict[key] = value.unsqueeze(-1)

    return new_state_dict


def __generate_mask(img_size, patch_size, anno, mask_landmark_num=4):
    """
        generate mask for landmark points
        randomly mask 4 points in 5 points
        input anno: (14)
        """
    if isinstance(img_size, tuple):
        img_size = img_size[0]
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    patch_num = img_size // patch_size
    masked_x = torch.empty((5,), dtype=torch.int8)
    masked_y = torch.empty((5,), dtype=torch.int8)
    for i in range(5):
        masked_y[i] = anno[2 * i + 4] * 7 / img_size
        masked_x[i] = anno[2 * i + 5] * 7 / img_size
    # random sample 4 points
    indices = random.sample(range(5), mask_landmark_num)
    masked_x = masked_x[indices]
    masked_y = masked_y[indices]
    mask = torch.full((patch_num, patch_num), 1, dtype=torch.float32)
    for x, y in zip(masked_x, masked_y):
        mask[int(x), int(y)] = 0
    return mask


def generate_landmark_mask(img_size, patch_size, anno, mask_landmark_num=4, mask_ratio=0.6):
    """
    generate mask for landmark points
    randomly mask 4 points in 5 points
    anno:(bs, 14)
    """
    bs = anno.shape[0]
    if isinstance(img_size, tuple):
        img_size = img_size[0]
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    patch_num = img_size // patch_size
    masked_xy = torch.empty((bs, 5), dtype=torch.long)
    for i in range(5):
        masked_xy[:, i] = (anno[:, 2 * i + 5] * 7 // img_size) * patch_num + anno[:, 2 * i + 4] * 7 // img_size
    # random sample 4 points
    _, indices = torch.randn((bs, 5)).topk(mask_landmark_num, dim=-1)
    masked_xy_ = masked_xy.masked_select(F.one_hot(indices).sum(-2).bool()).reshape(bs, 4)
    left_mask_num = int(patch_num * patch_num * mask_ratio) - mask_landmark_num
    random_n = torch.randn((bs, patch_num ** 2))
    random_n[torch.arange(bs)[:, None], masked_xy] = -10
    _, indices = random_n.topk(left_mask_num, dim=-1)
    indices = torch.cat([indices, masked_xy_], dim=1)
    mask = torch.full((bs, patch_num * patch_num), 1, dtype=torch.float32)
    mask[torch.arange(bs)[:, None], indices] = 0
    return mask


def weights_to_sparse(state_dict):
    """
    input state_dict is non-sparse model params, out put a sparse model params

    """
