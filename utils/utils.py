import math
from collections import OrderedDict

import torch


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


def weights_to_sparse(state_dict):
    """
    input state_dict is non-sparse model params, out put a sparse model params

    """