import logging
import time
import numpy as np
import torch

from metric import AverageValueMetric
from serialization import save_checkpoint
from config import Config

class Solver(object):
    def __init__(self, model):
        self.model = model
        self.loss = AverageValueMetric()
        self.acc = AverageValueMetric()

    def fit(self, train_data, valid_data, optimizer, criterion, lr_scheduler):
        best_valid_acc = -float('inf')
        logging.info("==========Start training==========")
        for epoch in range(Config.train.num_epoches):
            self.loss.reset()
            self.acc.reset()
            self.model.train()

            for i, data in enumerate(train_data):
                imgs, label = data
                labels = labels.cuda()
                scores = self.model(imgs)
                loss = criterion(scores, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.loss.add(loss.item())
                acc = (scores.max(1)[1] == labels.long()).float().mean()
                self.acc.add(acc.item())

            loss_mean = self.loss.value()[0]
            acc_mean = self.acc.value()[0]



        logging.info("==========End==========")