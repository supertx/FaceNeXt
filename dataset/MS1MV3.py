"""
@author supermantx
@date 2024/7/16 9:27
"""
import os
import numbers

import mxnet as mx
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(MXFaceDataset, self).__init__()
        if not transform:
            # masked decoder期间不用发杂的数据增强
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform
        self.imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(root_dir, 'train.idx'),
                                                    os.path.join(root_dir, 'train.rec'),
                                                    'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return 10000
        # return len(self.imgidx)
