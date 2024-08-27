"""
@author supermantx
@date 2024/7/19 16:52
"""
import os
from datetime import datetime

import torch
import timm.optim.optim_factory as optim_factory
from torch.optim import AdamW, SGD

from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2 as cv

from dataset import getDataloader
from utils import adjust_learning_rate, get_config, print_config, unpatchify, patchify, TensorboardLogger
from models import FaceMAE
from criterion import PretrainLoss, DiscriminatorLoss, Discriminator


def train_one_epoch(model,
                    dataloader,
                    optimizer,
                    gan_optimizer,
                    pretrain_loss,
                    discriminator_loss,
                    epoch,
                    logger,
                    cfg):
    model.train()
    t = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.solver.epochs} loss: 0.0", ncols=120)
    log_dict = {}
    for step_in_epoch, (img, anno, _) in enumerate(t):
        adjust_learning_rate(optimizer, step_in_epoch / len(dataloader) + epoch, cfg)
        img = img.cuda()
        anno.cuda()
        optimizer.zero_grad()
        pred, mask = model(img, anno)
        f_loss, p_loss = pretrain_loss(img, pred, mask, step_in_epoch >= cfg.train.start_face_loss_step)
        loss = f_loss + p_loss
        loss.backward()
        optimizer.step()
        if step_in_epoch <= cfg.train.train_discriminator_step:
            gan_optimizer.zero_grad()
            g_loss = discriminator_loss(img, pred.detach())
            g_loss.backward()
            gan_optimizer.step()
        torch.cuda.empty_cache()
        if step_in_epoch <= cfg.train.train_discriminator_step:
            t.desc = (f"Epoch {epoch}/{cfg.solver.epochs} lr: {optimizer.param_groups[0]['lr']:.2}" +
                      f" p_loss: {p_loss.item():.2f} g_loss: {g_loss.item():.2f}")
        else:
            t.desc = (f"Epoch {epoch}/{cfg.solver.epochs} lr: {optimizer.param_groups[0]['lr']:.2}" +
                      f" f_loss: {f_loss.item():.2f} p_loss: {p_loss.item():.2f}")
        log_dict['f_loss'] = f_loss.item()
        log_dict['p_loss'] = p_loss.item()
        log_dict['lr'] = optimizer.param_groups[0]['lr']
        logger.log_everything(log_dict, (epoch - 1) * len(dataloader) + step_in_epoch)
        if step_in_epoch % cfg.train.show_interval == 0:
            show(mask[0], img[0], pred[0], (epoch - 1) + step_in_epoch / len(dataloader), logger, cfg)
        logger.flush()
    t.close()


def train(args):
    # load config file
    cfg = get_config(args.config, is_pretrain=True)
    print_config(cfg)
    # make log dir
    log_dir = os.path.join(cfg.train.log_dir,
                           datetime.strftime(datetime.now(), "%m%d_%H%M_") + cfg.model.name + "_pretrain").__str__()
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    logger = TensorboardLogger(log_dir=log_dir)
    # get dataloader
    dataloader = getDataloader(cfg.dataset.data_dir, cfg.dataset.batch_size,
                               num_workers=cfg.dataset.num_workers, is_train=True)
    cfg.epoch_step = len(dataloader)
    cfg.global_step = len(dataloader) * cfg.solver.epochs

    model = FaceMAE(depth=cfg.model.depth,
                    dims=cfg.model.dims,
                    decoder_depth=cfg.model.decoder_depth,
                    patch_size=cfg.dataset.patch_size,
                    mask_ratio=cfg.model.mask_ratio,
                    inner_scale=cfg.model.inner_scale)
    discriminator = Discriminator(sn=True)
    pretrain_loss = PretrainLoss(cfg, discriminator)
    discriminator_loss = DiscriminatorLoss(discriminator)
    if cfg.print_model:
        print(model)
    model.cuda()
    # model.load_state_dict(torch.load("/home/power/tx/FaceNeXt/logs/0730_1709_faceNeXt_tiny_pretrain/model_10.pth"))
    discriminator.cuda()
    # compute params num
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    param_groups = optim_factory.add_weight_decay(model, cfg.solver.weight_decay)
    optimizer = AdamW(param_groups, lr=cfg.solver.base_lr)
    gan_optimizer = SGD(discriminator.parameters(), lr=cfg.solver.base_lr)
    # AdamW()
    start_epoch = 1
    if cfg.resume.is_resume:
        check_point = torch.load(cfg.resume.resume_path)
        model.load_state_dict(check_point["model"])
        optimizer.load_state_dict(check_point["optimizer"])
        start_epoch = check_point["epoch"]
    for epoch in range(start_epoch, cfg.solver.epochs + 1):
        train_one_epoch(model,
                        dataloader,
                        optimizer,
                        gan_optimizer,
                        pretrain_loss,
                        discriminator_loss,
                        epoch,
                        logger,
                        cfg)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, f"model_{epoch}.pth"))
            check_point = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(check_point, os.path.join(log_dir, f"check_point_{epoch}.pth"))
    logger.close()
    print("training finished")


def show(mask, img, img_pred, epoch, logger, cfg):
    # img_pred = unpatchify(img_pred, cfg.dataset.patch_size)
    img_mask_infer, masked_img = mask_draw_pre(mask, img, img_pred)
    plt.figure(figsize=(12, 4))
    subplot = plt.subplot(1, 3, 1)
    subplot.imshow(normalize_img(img.detach().cpu().permute(1, 2, 0).numpy()))
    subplot.set_title("original img")
    subplot = plt.subplot(1, 3, 2)
    subplot.set_title("mask img")
    subplot.imshow(normalize_img(masked_img.detach().cpu().permute(1, 2, 0).numpy()))
    subplot = plt.subplot(1, 3, 3)
    subplot.imshow(normalize_img(img_mask_infer.detach().cpu().permute(1, 2, 0).numpy()))
    subplot.set_title("pred masked img")
    plt.savefig(os.path.join(cfg.log_dir, f"{epoch}.png"))
    imshow = cv.imread(os.path.join(cfg.log_dir, f"{epoch}.png"))
    imshow = cv.cvtColor(imshow, cv.COLOR_BGR2RGB)
    logger.log_image("FaceNeXt", imshow, epoch, dataformats='HWC')
    # plt.savefig(os.path.join(cfg.log_dir, f"{epoch}.png"))
    # plt.show()


def mask_draw_pre(mask, img, pred):
    """
    mask: (bs, h(w) / patch_size ** 2)
    return a img contain pred img in masked place and original img in other place
    """
    patch_num = int(mask.shape[0] ** .5)
    patch_size = int(img.shape[1] // patch_num)
    mask = mask.reshape(-1, patch_num, patch_num)
    mask = torch.repeat_interleave(mask, patch_size, dim=1)
    mask = torch.repeat_interleave(mask, patch_size, dim=2)
    mask = mask.repeat(3, 1, 1)
    """
    forward process normalize img in each patch,so for a better visualization,
    we need to normalize img in each patch for a better visualization
    img: (c, h, w)
    """
    img_patchify = patchify(img, patch_size)
    mean = img_patchify.mean(dim=-1, keepdim=True)
    var = img_patchify.var(dim=-1, keepdim=True)
    img_patchify = (img_patchify - mean) / (var + 1.e-6) ** .5
    img_patchify = img_patchify.permute(1, 0).reshape(-1, patch_num, patch_num)
    img = unpatchify(img_patchify, patch_size)
    img_mask = img * (1 - mask)
    pred = pred * mask
    return img_mask + pred, img_mask


def normalize_img(img):
    img = img - img.min()
    img = img / img.max()
    return img


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="pretrain_FaceNeXt_tiny.yml", type=str, help="config file")
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
