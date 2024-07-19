"""
@author supermantx
@date 2024/7/19 16:52
"""
import os

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
from matplotlib import pyplot as plt

from models import FaceMAE
from dataset import getDataloader


def train_one_epoch(model, dataloader, optimizer, epoch):
    model.train()
    t = torch.tqdm(dataloader, desc=f"Epoch {epoch}/400 loss: 0.0", ncols=120)
    for i, (img, _) in enumerate(t):
        img = img.cuda()
        optimizer.zero_grad()
        loss, pred, mask = model(img)
        t.desc = f"Epoch {epoch}/100 loss: {loss.item()}"
        loss.backward()
        optimizer.step()
        if i % 400 == 0:
            subplot = plt.subplot(1, 2, 1)
            subplot.imshow(img[0].detach.cpu().permute(1, 2, 0).numpy())
            subplot.set_title("original img")
            subplot = plt.subplot(1, 2, 2)
            img_pred = model.unpatchify(pred.detach().flatten(2).permute(0, 2, 1))[0]
            subplot.imshow(img_pred.cpu().permute(1, 2, 0).numpy())
            subplot.set_title("pred masked img")
    t.close()


def train():
    dataloader = getDataloader("/data/tx/MS1Mv3", 512, num_workers=16, is_train=True)
    model = FaceMAE()
    model.cuda()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    schedule_lr = StepLR(optimizer, 10, 0.1)

    for epoch in range(1, 401):
        train_one_epoch(model, dataloader, optimizer, epoch)
        schedule_lr.step()
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(".logs/", "model_{epoch}.pth"))
            check_point = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "schedule_lr": schedule_lr.state_dict(),
                "epoch": epoch
            }
            torch.save(check_point, os.path.join(".logs/", "check_point_{epoch}.pth"))


def main():
    train()


if __name__ == '__main__':
    main()
