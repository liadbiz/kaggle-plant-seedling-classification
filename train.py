import logging
import sys
import pprint
from config import Config
from datasets import get_dataloader
import models
import os
from torch.backends import cudnn
from solver import Solver

import torch
from torch import nn
import torch.optim as optim


def init():
    FORMAT = '[%(levelname)s]: %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
        stream=sys.stdout
    )

    logging.info("==========config==========")
    logging.info(pprint(Config))
    logging.info("==========end==========")

    os.environ['CUDA_VISIBLE_DEVICES'] = Config.train.device_name
    cudnn.benchmark = True


def train():
    logging.info("==========loading data==========")
    train_data, valid_data, test_data = get_dataloader(Config)
    logging.info("==========end==========")

    logging.info("==========loading model==========")
    model = getattr(models, Config.model.name)(Config.model.num_class)
    logging.info("==========end==========")
    optimizer = getattr(optim, Config.train.optimizer)(model.parameters(), lr=Config.train.lr,
                                                       weight_decay=Config.train.wd, momentum=Config.train.momentum)
    ce_loss = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau()
    model = model.cuda()
    model_solver = Solver(Config, model)
    model_solver.fit(train_data=train_data, valid_data=valid_data, optimizer=optimizer, criterion=ce_loss,
                     lr_schduler=lr_scheduler)
    # model_solver.evaluate(test_data)

if __name__ == '__main__':
    train()