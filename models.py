from torch import nn
from torchvision import models

def resnet50(classes, pretrain=True):
    if pretrain:
        net = models.resnet50(pretrained=True)
    else:
        net = models.resnet50(pretrained=False)

    net.avgpool = nn.AdaptiveAvgPool2d(1)
    net.fc = nn.Linear(net.fc.in_features, classes)
    return net