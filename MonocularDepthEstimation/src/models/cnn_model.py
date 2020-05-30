import torch.nn as nn
import torch.nn.functional as F
from .depthwise_seperable_conv2d import DepthwiseSeparableConv2d


class CNN_Model(nn.Module):

    def __init__(self):
        super(CNN_Model, self).__init__()

        self.inputblock = nn.Sequential(
            # Defining a 2D convolution layer                                               RF = 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, bias=False, dilation=1, padding=1),
            # RF = 3
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.convblock1 = nn.Sequential(
            # Defining a 2D convolution layer
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=False, dilation=1, padding=1),  # RF = 5
            DepthwiseSeparableConv2d(32, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=2, bias=False, padding=1),  # RF = 9
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.convblock2 = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(64, 32, kernel_size=3, stride=1, bias=False, dilation=1, padding=1),  # RF = 14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=1, bias=False, padding=1),  # RF = 18
            DepthwiseSeparableConv2d(32, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=2, bias=False, padding=1),
            # RF = 26
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.convblock3 = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(64, 32, kernel_size=3, stride=1, bias=False, dilation=1, padding=1),  # RF = 36
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=1, bias=False, padding=1),  # RF = 44
            DepthwiseSeparableConv2d(32, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=2, bias=False, padding=0),  # RF = 60
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # RF = 10 - 28

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=2))  # RF = 60

        self.linear = nn.Linear(64, 10, bias=False)  # RF = 60

    def forward(self, x):
        x = self.inputblock(x)
        x = self.convblock1(x)
        x = self.maxpool(x)
        x = self.convblock2(x)
        x = self.maxpool(x)
        x = self.convblock3(x)
        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)
