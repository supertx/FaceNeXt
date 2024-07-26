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
