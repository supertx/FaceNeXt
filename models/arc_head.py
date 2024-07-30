"""
@author supermantx
@date 2024/7/30 10:43
arcface loss
"""

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

def margin_logit(logits, s, m1, m2, m3):
    # TODO 这里报错，梯度被截断了，研究这段代码的作用
    if m3 == 0:
        logits_c = logits.detach()

        logits_c = torch.acos(logits_c)
        logits_c = logits_c + m2
        logits_c.cos_()

        return logits * s
    elif m3 > 0:
        logits = logits - m3
        logits = logits * s
    else:
        raise Exception("error argument")
    return logits


class ArcHead(nn.Module):

    def __init__(self, embedding_size, num_classes, s=64, m=(1, 0.5, 0), training=True):
        super().__init__()
        self.s = s
        self.m = eval(m)
        self.layer = nn.Linear(embedding_size, num_classes, bias=False)
        self.training = training

    @staticmethod
    def forward_loss(feature, label):
        return F.cross_entropy(feature, label)

    def forward(self, logit, label=None):
        feature = self.layer(F.normalize(logit))
        feature = feature.clamp(-1, 1)
        feature = margin_logit(feature, self.s, *self.m)
        if self.training:
            return self.forward_loss(feature, label)
        else:
            return feature
