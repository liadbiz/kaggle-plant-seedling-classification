# encoding: utf-8
"""
@author: liadbiz
@email: zhuhh2@shanghaitech.edu.cn
"""

from easydict import EasyDict as ed

Config = ed()
Config.seed = 0

Config.dataset = ed()
Config.dataset.train_img_path = 'data/train/'
Config.dataset.valid_img_path = 'data/valid/'
Config.dataset.test_img_path = 'data/test/'
Config.dataset.valid_ratio = 0.2

Config.augmentation = ed()
Config.augmentation.img_size = (224, 224)
Config.augmentation.random_erasing = True
Config.augmentation.random_mirror = True

Config.train = ed()
Config.train.optimizer = 'SGD'
Config.train.lr = 0.01
Config.train.momentum = 0.9
Config.train.mode = 'min'
Config.train.factor = 0.1
Config.train.patience = 10
Config.train.verbose = True
Config.train.threshold = 0.0001
Config.train.threshold_mode = 'rel'
Config.train.num_epoches = 50
Config.train.batch_size = 128
Config.train.device_name = '0'
Config.train.num_workers = 4

Config.model = ed()
# Config.models = ['resnet50', 'InceptionV3', 'DenseNet201', 'Xception']
Config.models = ['resnet50']

Config.misc = ed()
Config.misc.log_inteval = 50
Config.misc.eval_step = 5
Config.misc.save_step = 5
Config.misc.save_path = 'checkpoints/'

