"""
@author supermantx
@date 2024/7/30 10:43
arcface loss
"""
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F


def margin_logit(logits, s, m1, m2, m3, label):
    """
    base on official arcface-torch code
    FIXME debug
    """
    # select the target logit
    target_logit = logits[:, label.view(-1)]
    if m2 > 0:
        # arcFace margin
        # logits_ = logits.detach()
        # target_logit_ = target_logit.detach()
        # logits_ = torch.acos(logits_)
        # target_logit_ = torch.acos(target_logit_)
        # target_logit_ = target_logit + m2
        # logits_[:, label.view(-1)] = target_logit_
        # logits_.cos_()
        with torch.no_grad():
            target_logit.arcos_()
            logits.arccos_()
            final_target_logit = target_logit + m2
            logits[:, label.view(-1)] = final_target_logit
            logits.cos_()
        return logits * s
    if m3 > 0:
        # cosFace margin
        target_logit = target_logit - m3
        logits[:, label.view(-1)] = target_logit
        logits = logits * s
    else:
        raise Exception("error argument")
    return logits


def margin_logit_v2(logits, s, m1, m2, m3, label):
    """
    based on ronghuaiyang/arcface-pytorch code
    """
    cos_m = math.cos(m2)
    sin_m = math.sin(m2)
    th = math.cos(math.pi - m2)
    mm = math.sin(math.pi - m2) * m2
    sine = torch.sqrt((1.0 - torch.pow(logits, 2)).clamp(0, 1)).half()
    phi = logits * cos_m - sine * sin_m
    phi = torch.where(logits > th, phi, logits - mm)
    one_hot = torch.zeros_like(logits).to(logits.device)
    one_hot.scatter_(1, label.view(-1, 1), 1)
    output = (one_hot * phi) + ((1.0 - one_hot) * logits)
    output *= s
    return output


class ArcHead(nn.Module):

    def __init__(self, embedding_size, num_classes, s=64, m=(1, 0.5, 0), training=True):
        super().__init__()
        self.s = s
        self.m = eval(m)
        # self.layer = nn.Linear(embedding_size, num_classes, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.training = training

    @staticmethod
    def forward_loss(feature, label):
        return F.cross_entropy(feature, label)

    def forward(self, logit, label=None):
        feature = F.linear(F.normalize(logit), F.normalize(self.weight))
        feature = feature.clamp(-1, 1)
        feature = margin_logit_v2(feature, self.s, *self.m, label)
        if self.training:
            return self.forward_loss(feature, label)
        else:
            return feature
