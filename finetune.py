"""
@author supermantx
@date 2024/7/29 14:29
finetune the pretrain model(self-supervised)
"""
import os
from collections import OrderedDict
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import MBF, ArcHead
from dataset import getDataloader
from utils import (organize_model_weights,
                   adjust_learning_rate,
                   get_config,
                   print_config,
                   TensorboardLogger,
                   EvaluateLogger)


def train_one_epoch(model, head, optimizer, dataloader, epoch, logger, cfg):
    model.train()
    t = tqdm(dataloader, desc=f"Epoch: {epoch}/{cfg.solver.epochs} lr: 0.0 loss: 0.0", ncols=120)
    log_dict = {}
    for step_in_epoch, (img, label) in enumerate(t):
        adjust_learning_rate(optimizer, step_in_epoch / len(dataloader) + epoch, cfg)
        img = img.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        logit = model(img)
        loss = head(logit, label)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        t.desc = f"Epoch: {epoch}/{cfg.solver.epochs} lr: {optimizer.param_groups[0]['lr']:.2} loss: {loss.item()}"
        log_dict['loss'] = loss.item()
        log_dict['lr'] = optimizer.param_groups[0]['lr']
        logger.log_everything(log_dict, epoch * len(dataloader) + step_in_epoch)
    logger.flush()
    t.close()


def train(args):
    # load config file
    cfg = get_config(args.config)
    print_config(cfg)
    # make log dir
    log_dir = os.path.join(cfg.train.log_dir,
                           datetime.strftime(datetime.now(), "%m%d_%H%M_") + cfg.model.name + "_finetune").__str__()
    logger = TensorboardLogger(log_dir=log_dir)
    evaluate_logger = EvaluateLogger(log_dir=log_dir,
                                     eval_bin_names=cfg.eval.eval_datasets,
                                     bin_root=cfg.eval.bin_root,
                                     cfg=cfg)

    # get dataloader
    dataloader = getDataloader(cfg.dataset.data_dir,
                               cfg.dataset.batch_size,
                               num_workers=cfg.dataset.num_workers,
                               is_train=True)

    pretrain_state_dict = torch.load(cfg.model.pretrain_path)
    state_dict = organize_model_weights(pretrain_state_dict)
    model = MBF(num_features=512,
                stages=(3, 3, 9, 3),
                stages_channel=(128, 128, 256, 256),
                inner_scale=1)
    model.load_state_dict(state_dict, strict=False)
    head = ArcHead(embedding_size=512,
                   num_classes=cfg.dataset.num_classes,
                   s=cfg.model.s, m=cfg.model.m)

    if cfg.print_model:
        print(model)
    params_num = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())
    print("number of trainable params:", params_num)

    optimizer = torch.optim.AdamW(
        [{'params': model.parameters()}, {'params': head.parameters()}],
        lr=cfg.solver.base_lr, weight_decay=cfg.solver.weight_decay)

    model.cuda()
    head.cuda()

    state_epoch = 1
    if cfg.resume.is_resume:
        assert os.path.isfile(cfg.resume.resume_path), "No checkpoint found!"
        check_point = torch.load(cfg.resume.resume_path)
        model.load_state_dict(check_point["model"])
        head.load_state_dict(check_point["head"])
        optimizer.load_state_dict(check_point["optimizer"])
        state_epoch = check_point["epoch"]

    for epoch in range(state_epoch, cfg.solver.epochs):
        train_one_epoch(model, head, optimizer, dataloader, epoch, logger, cfg)
        evaluate_logger.evaluate(model, logger, epoch)

        if epoch % 5 == 0:
            torch.save({
                "model": model.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, os.path.join(log_dir, f"model_{epoch}.pth"))
    logger.close()
    print("training finished")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="pretrain_FaceNeXt_tiny.yml", type=str, help="config file")
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
