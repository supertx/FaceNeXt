"""
@author supermantx
@date 2024/7/26 17:00
"""
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from eval.verification import load_bin, test


class TensorboardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def log_scalar(self, tag, scalar_value, global_step):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def log_everything(self, log_dict: dict, global_step):
        for key, value in log_dict.items():
            self.log_scalar(key, value, global_step)

    def log_image(self, tag, image_tensor, global_step, **kwargs):
        self.writer.add_image(tag, image_tensor, global_step, **kwargs)

    def log_histogram(self, tag, values, global_step):
        self.writer.add_histogram(tag, values, global_step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


class EvaluateLogger:

    def __init__(self, log_dir, bin_root, eval_bin_names, cfg):
        self.log_dir = log_dir
        self.eval_bins = self.__load_bins(bin_root, eval_bin_names)
        self.log_file = open(os.path.join(log_dir, cfg.model.name + "_eval.txt"), "w")

    @staticmethod
    def __load_bins(bin_root, eval_bin_names):
        eval_bins = {}
        for name in eval_bin_names:
            eval_bins[name] = load_bin(os.path.join(bin_root, name + ".bin"), image_size=(112, 112))
        return eval_bins

    @torch.no_grad()
    def evaluate(self, model, logger, epoch):
        model.eval()
        for name, bin in self.eval_bins.items():
            acc, std, xnorm = test(bin, model)[2: 5]
            self.log_file.write(f"epoch : {epoch} \t {name} : acc {acc * 100:.2f}% std: {std:.2f} xnorm: {xnorm:.2f}\n")
            self.log_file.write("\n")
            logger.log_scalar(name, acc, epoch)
        logger.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
