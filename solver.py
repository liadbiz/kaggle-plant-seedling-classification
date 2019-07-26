import logging
import time
import numpy as np
import torch

from metric import AverageValueMetric
from serialization import *
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
            logging.info("[Epoch {} start]:\n".format(epoch))
            self.loss.reset()
            self.acc.reset()
            self.model.train()

            batch_tic = time.time()
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

                # log info every [Config.misc.log_interval] batch
                if Config.misc.log_interval and not (i + 1) % Config.misc.log_interval:
                    loss_mean = self.loss.value()[0]
                    acc_mean = self.acc.value()[0]
                    logging.info(
                        '[Epoch {0}]: Batch {1}, training speed: {2} samples/second, loss: {3}, acc: {4}'.format(
                            epoch, i, train_data.batchsize * Config.misc.log_interval / (time.time() - batch_tic),
                            loss_mean, acc_mean
                        ))
                    batch_tic = time.time()

            # get mean loss and mean acc after one epoch
            loss_mean = self.loss.value()[0]
            acc_mean = self.acc.value()[0]
            logging.info("[Epoch {} end], loss: {}, acc: {}".format(epoch, loss_mean, acc_mean))

            # evaluate model every epoch
            is_best = False
            if valid_data and Config.misc.eval_step and not (epoch + 1) % Config.misc.eval_step:
                valid_acc = self.evaluate(valid_data)
                logging.info("[Epoch {}]: valid acc: {}".format(epoch, valid_acc))
                if valid_acc > best_valid_acc:
                    best_test_acc = valid_acc
                    mkdir_if_missing(Config.misc.save_dir)
                    torch.save(self.model.module.state_dict(), os.path.join(Config.misc.save_dir, 'model_best.pth'))

        logging.info("==========End==========")

    def evaluate(self, data):
        self.model.eval()
        num_correct, num_images = 0, 0
        for d in data:
            images, labels = d
            labels.cuda()
            with torch.no_grad():
                pred = self.model(imgs)
            num_correct += (pred.max(1)[1] == labels).float().sum().item()
            num_images += images.shape[0]
        return num_correct / num_images