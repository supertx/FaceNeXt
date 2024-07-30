"""
@author supermantx
@date 2024/7/19 16:54
"""
import yaml
import logging
import os


def get_config(config_path, is_pretrain=False):
    config = Dict()
    if is_pretrain:
        with open("./config/pretrain_base.yml", "r") as f:
            base = yaml.load(f, Loader=yaml.FullLoader)
            config.update(base)
    else:
        with open("./config/finetune_base.yml", "r") as f:
            base = yaml.load(f, Loader=yaml.FullLoader)
            config.update(base)
    assert os.path.isfile(os.path.join("./config", config_path)), f"{config_path} not found"
    with open(os.path.join("./config", config_path), "r") as f:
        load = yaml.load(f, Loader=yaml.FullLoader)
        config.update(load)
        return config


def print_config(v, s=""):
    for k, v in v.items():
        if isinstance(v, dict):
            print_config(v, s + k + ".")
        else:
            num_space = 30 - len(s + k)
            print(f"{s + k}: " + " " * num_space + f"{v}")


class Dict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and k not in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = type(value)(self.__class__(x)
                                if isinstance(x, dict) else x for x in value)
        elif isinstance(value, dict) and not isinstance(value, Dict):
            value = Dict(value)
        super(Dict, self).__setattr__(name, value)
        super(Dict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            if isinstance(d[k], dict):
                dic = Dict()
                if hasattr(self, k):
                    dic.update(getattr(self, k))
                dic.update(d[k])
                setattr(self, k, dic)
            else:
                setattr(self, k, d[k])

    def pop(self, k, *args):
        if hasattr(self, k):
            delattr(self, k)
        return super(Dict, self).pop(k, *args)
