import math
import numpy as np


class AverageValueMetric(object):
    def __init__(self):
        super(AverageValueMetric, self).__init__()
        self.reset()

    def add(self, value):
        self.value = value
        self.sum += value
        self.var += value * value
        self.number += 1

        if self.number == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.number == 1:
            self.mean, self.std = np.nan, np.inf
        else:
            self.mean = self.sum / self.number
            self.std = math.sqrt(self.var - self.number * self.mean * self.mean) / (self.number - 1.0)

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.value = 0.0
        self.sum = 0.0
        self.number = 0
        self.var = 0.0
        self.mean = np.nan
        self.std = np.nan