# encoding: utf-8
"""
@author: liadbiz
@email: zhuhh2@shanghaitech.edu.cn
"""
import os

import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from utils.augmentor.randomerasing import RandomErasing


def get_dataloader(config):
    base_aug = [tf.ToTensor(), tf.Normalize(mean=[0.4924, 0.4822, 0.4465], std=[0.202, 0.199, 0.201])]
    train_aug = list()
    train_aug.append(tf.RandomResizedCrop(config.augmentation.img_size))

    if config.augmentation.random_mirror:
        train_aug.append(tf.RandomHorizontalFlip())
        train_aug.append(tf.RandomVerticalFlip())

    train_aug.extend(base_aug)

    if config.augmentation.random_erasing:
        train_aug.append(RandomErasing())

    train_aug = tf.Compose(train_aug)

    test_aug = list()
    test_aug.append((tf.Resize(config.augmentation.img_size)))
    test_aug.extend((base_aug))
    test_aug = tf.Compose(test_aug)

    train_dataset = ImageFolder(config.dataset.train_img_path, train_aug)
    valid_dataset = ImageFolder(config.dataset.valid_img_path, train_aug)
    test_dataset = TestDataset(config.dataset.test_img_path, test_aug)
    train_loader = DataLoader(train_dataset, config.train.batch_size, shuffle=True, num_workers=config.train.num_workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset, config.train.batch_size, shuffle=False, num_workers=config.train.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, config.train.batch_size, shuffle=False, num_workers=config.train.num_workers,
                              pin_memory=True)

    return train_loader, valid_loader, test_loader


class TestDataset(Dataset):
    def __init__(self, path, transforms):
        self.path = path
        self.img_list = os.listdir(path)
        self.transforms = transforms

    def __getitem__(self, item):
        fname = self.img_list[item]
        ima_path = os.path.join(self.path, fname)
        img = Image.open(ima_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        return img, fname

    def __len__(self):
        return len(self.img_list)