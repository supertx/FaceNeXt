"""
@author supermantx
@date 2024/7/26 10:28
学习率工具
"""
import math


def adjust_learning_rate(optimizer, epoch, cfg):
    """
    epoch = step_in_epoch / step_one_epoch + epoch
    math: $lr=\begin{cases}l_b*e_c/e_w,& e_c \le e_w \\l_m+\frac{1}{2}*(l_b-l_m)*(1+\cos(\frac{(e_c-e_w)}{e_t-e_w}*\pi),&e_c>e_w \end{cases}$
    """
    cfg_solver = cfg.solver
    if epoch < cfg_solver.warmup_epoch:
        lr = cfg_solver.base_lr * epoch / cfg_solver.warmup_epoch
    else:
        lr = cfg_solver.min_lr + 0.5 * (cfg_solver.base_lr - cfg_solver.min_lr) * (
                    1 + math.cos((epoch - cfg_solver.warmup_epoch) / (cfg_solver.epochs - cfg_solver.warmup_epoch) * math.pi))
    for param_group in optimizer.param_groups:
        if 'lr_scale' in param_group:
            param_group['lr'] = lr * param_group['lr_scale']
        else:
            param_group['lr'] = lr
