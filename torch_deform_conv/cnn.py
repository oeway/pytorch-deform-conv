from __future__ import absolute_import, division

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_deform_conv.layers import ConvOffset2D

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # conv11
        self.conv11 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(32)

        # conv12
        self.conv12 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn12 = nn.BatchNorm2d(64)

        # conv21
        self.conv21 = nn.Conv2d(64, 128, 3, padding= 1)
        self.bn21 = nn.BatchNorm2d(128)

        # conv22
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.bn22 = nn.BatchNorm2d(128)

        # out
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn11(x)

        x = F.relu(self.conv12(x))
        x = self.bn12(x)

        x = F.relu(self.conv21(x))
        x = self.bn21(x)

        x = F.relu(self.conv22(x))
        x = self.bn22(x)

        x = F.avg_pool2d(x, kernel_size=[x.size(2), x.size(3)])
        x = self.fc(x.view(x.size()[:2]))#
        x = F.softmax(x)
        return x

class DeformConvNet(nn.Module):
    def __init__(self):
        super(DeformConvNet, self).__init__()

        # conv11
        self.conv11 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(32)

        # conv12
        self.offset12 = ConvOffset2D(32)
        self.conv12 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn12 = nn.BatchNorm2d(64)

        # conv21
        self.offset21 = ConvOffset2D(64)
        self.conv21 = nn.Conv2d(64, 128, 3, padding= 1)
        self.bn21 = nn.BatchNorm2d(128)

        # conv22
        self.offset22 = ConvOffset2D(128)
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.bn22 = nn.BatchNorm2d(128)

        # out
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn11(x)

        x = self.offset12(x)
        x = F.relu(self.conv12(x))
        x = self.bn12(x)

        x = self.offset21(x)
        x = F.relu(self.conv21(x))
        x = self.bn21(x)

        x = self.offset22(x)
        x = F.relu(self.conv22(x))
        x = self.bn22(x)

        x = F.avg_pool2d(x, kernel_size=[x.size(2), x.size(3)])
        x = self.fc(x.view(x.size()[:2]))#
        x = F.softmax(x)
        return x

class DeformConvNet(nn.Module):
    def __init__(self):
        super(DeformConvNet, self).__init__()

        # conv11
        self.conv11 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(32)

        # conv12
        self.offset12 = ConvOffset2D(32)
        self.conv12 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn12 = nn.BatchNorm2d(64)

        # conv21
        self.offset21 = ConvOffset2D(64)
        self.conv21 = nn.Conv2d(64, 128, 3, padding= 1)
        self.bn21 = nn.BatchNorm2d(128)

        # conv22
        self.offset22 = ConvOffset2D(128)
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.bn22 = nn.BatchNorm2d(128)

        # out
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn11(x)

        x = self.offset12(x)
        x = F.relu(self.conv12(x))
        x = self.bn12(x)

        x = self.offset21(x)
        x = F.relu(self.conv21(x))
        x = self.bn21(x)

        x = self.offset22(x)
        x = F.relu(self.conv22(x))
        x = self.bn22(x)

        x = F.avg_pool2d(x, kernel_size=[x.size(2), x.size(3)])
        x = self.fc(x.view(x.size()[:2]))#
        x = F.softmax(x)
        return x

def get_cnn():
    return ConvNet()

def get_deform_cnn(trainable=True):
    if trainable:
        return DeformConvNet()
    else:
        raise Exception('not implemented yet')
