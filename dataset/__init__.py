"""
@author supermantx
@date 2024/7/19 16:58
"""
import os

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset.MS1MV3 import MXFaceDataset


def getDataloader(data_dir, batch_size, num_workers=4, is_train=True):
    if not os.path.isfile(os.path.join(data_dir, 'train.rec')) \
            or not os.path.isfile(os.path.join(data_dir, 'train.idx')):
        raise FileNotFoundError(f"not dataset found in {data_dir}")
    return DataLoader(MXFaceDataset(data_dir, transform=getTransform()), batch_size=batch_size, shuffle=is_train,
                      num_workers=num_workers, pin_memory=True, drop_last=True)


def getTransform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
