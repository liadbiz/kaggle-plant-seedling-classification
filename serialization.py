# encoding: utf-8
"""
@author: liadbiz
@email: zhuhh2@shanghaitech.edu.cn
"""
import errno
import os
import shutil

import os.path as osp
import torch


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


