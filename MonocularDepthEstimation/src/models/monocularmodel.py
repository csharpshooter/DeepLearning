import torch
import torch.nn as nn
import torch.nn.functional as F

from .depthwise_seperable_conv2d import DepthwiseSeparableConv2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv2d(input=in_planes, output=planes, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DepthwiseSeparableConv2d(input=planes, output=planes, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False,
                          groups=in_planes),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu((self.conv1(out)))
        out = (self.conv2(out))
        out += self.shortcut(x)
        # out = self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 128

        self.prep = self.make_convblock_depthwise_conv(kernel_size=3, in_channels=3, out_channels=64, stride=1,
                                                       padding=1,
                                                       doMaxPool=False)
        self.cb1 = self.make_convblock_depthwise_conv(kernel_size=3, in_channels=64, out_channels=128, stride=1,
                                                      padding=1)
        self.res1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.cb2 = self.make_convblock_depthwise_conv(kernel_size=3, in_channels=128, out_channels=64, stride=1,
                                                      padding=1)
        self.cb3 = self.make_convblock_depthwise_conv(kernel_size=3, in_channels=64, out_channels=32, stride=1,
                                                      padding=1)
        self.in_planes = 32
        self.res2 = self._make_layer(block, 32, num_blocks[1], stride=1)
        self.convblockfinal = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(64, 3, 3, stride=1, bias=False, padding=1),
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # DepthwiseSeparableConv2d(32, 32, 1)

    def make_convblock_depthwise_conv(self, kernel_size, in_channels, out_channels, stride, padding, doMaxPool=True):
        seq = nn.Sequential()
        seq.add_module("DepthwiseConv2d",
                       DepthwiseSeparableConv2d(input=in_channels, output=out_channels, padding=padding, stride=stride))

        # if doMaxPool:
        #     seq.add_module("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2))

        # seq.add_module("BatchNorm", nn.BatchNorm2d(out_channels))
        seq.add_module("Relu", nn.ReLU())
        return seq

    def make_convblock(self, kernel_size, in_channels, out_channels, stride, padding, doMaxPool=True):
        seq = nn.Sequential()
        seq.add_module("Conv2d", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding, bias=False))

        # if doMaxPool:
        #     seq.add_module("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2))

        # seq.add_module("BatchNorm", nn.BatchNorm2d(out_channels))
        seq.add_module("Relu", nn.ReLU())
        return seq

    def doforward(self, x):
        x = self.prep(x)  # input layer 3-> 64
        l1 = self.cb1(x)  # Layer 1 X 64 -> 128
        r1 = self.res1(l1)  # Resblock 1 128 -> 128
        x = l1 + r1
        l2 = self.cb2(x)  # Layer 2 128 -> 64
        l3 = self.cb3(l2)  # Layer 3 X 64 -> 32
        r2 = self.res2(l3)  # Resblock 2 32 -> 32
        x = l3 + r2
        return x

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]

        x1 = self.doforward(x1)
        x2 = self.doforward(x2)

        x = torch.cat([x1, x2], 1)

        x = self.convblockfinal(x)

        return x


def MonocularModel():
    return ResNet(BasicBlock, [1, 1])
