"""
@author supermantx
@date 2024/7/16 9:27
"""
import os
import numbers

import mxnet as mx
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2 as cv
from matplotlib import pyplot as plt

from utils.utils import generate_landmark_mask


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
        self.imgidx = np.array(list(self.imgrec.keys))
        self.colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]

    def __get_raw_item(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        img_raw = mx.image.imdecode(img).asnumpy()
        img_raw = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)
        return img_raw, header.label

    def get_raw_item(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        img_raw = mx.image.imdecode(img).asnumpy()
        img_raw = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)
        return img_raw, header.label

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.id2
        label = torch.tensor(label, dtype=torch.long)
        anno = torch.tensor(header.label, dtype=torch.float32)
        sample = mx.image.imdecode(img).asnumpy()
        sample = cv.cvtColor(sample, cv.COLOR_BGR2RGB)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, anno, label

    def show(self, index):
        img_raw, header = self.__get_raw_item(index)
        img_raw = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)
        anno = list(map(int, header))
        cv.rectangle(img_raw, (anno[0], anno[1]), (anno[2], anno[3]), (0, 255, 0), 1)
        for i in range(5):
            cv.circle(img_raw, (anno[2 * i + 4], anno[2 * i + 5]), 1, self.colors[i], 1)
        plt.imshow(img_raw)
        plt.show()

    def __len__(self):
        # return 100000
        return len(self.imgidx)


if __name__ == '__main__':

    dataset = MXFaceDataset("/data/tx/MS1MV4", transform=None)

    import random
    import torch.nn.functional as F

    annos = torch.stack([dataset[i][1] for i in range(10)])
    img = torch.stack([torch.tensor(dataset.get_raw_item(i)[0]).permute(2, 0, 1) for i in range(10)])
    mask = generate_landmark_mask(112, 16, annos)
    mask = F.interpolate(mask.unsqueeze(1), size=(112, 112), mode='nearest')
    img_mask = img / 255 * mask
    for i in range(10):
        plt.imshow(img_mask[i].permute(1, 2, 0))
        plt.show()


