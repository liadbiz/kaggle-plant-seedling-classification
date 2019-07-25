# encoding: utf-8
"""
@author: liadbiz
@email: zhuhh2@shanghaitech.edu.cn
"""
import os
from config import Config
import numpy as np
import shutil
import math

def mkdir_if_not_exists(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))


labels = os.listdir(Config.dataset.train_img_path)

for l in labels:
    images_per_class = os.listdir(os.path.join(Config.dataset.train_img_path, l))
    valid_images = np.random.choice(images_per_class, size=int(math.floor(len(images_per_class) *
                                                                          Config.dataset.valid_ratio)), replace=False)
    print(len(images_per_class), len(valid_images))
    mkdir_if_not_exists(['data', 'valid', l])
    for img in valid_images:
        shutil.move(os.path.join('data/train', l, img),  os.path.join('data/valid', l))