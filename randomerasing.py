# encoding: utf-8
"""
@author: liadbiz
@email: zhuhh2@shanghaitech.edu.cn
"""
import random
import math

class RandomErasing(object):
    """Random Erasing Data Augmentation. Random Erasing randomly selects a rectangle region in an image and erases its
        pixels  with  random  values.
        See original paper for more details.
        https://arxiv.org/pdf/1708.04896.pdf

    Args:
        epsilon: erasing probability.
        sl: Erasing area ratio range(minimum)
        sh: Erasing area ratio range(maximum)
        rl: Erasing aspect.(for height)
        rh: Erasing aspect.(for weight)
        mean: Erasing values
    """
    def __init__(self, epsilon=0.5, sl=0.02, sh=0.4, rl=0.3, rh=3.33, mean=[0.4924, 0.4822, 0.4465]):
        self.EPSIlON = epsilon
        self.sl = sl
        self.sh = sh
        self.rl = rl
        self.rh = rh
        self.mean = mean

    def __call__(self, img):
        if random.uniform(0, 1) > self.EPSIlON:
            return img

        for _ in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.rl, self.rh)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[1]
                return img

