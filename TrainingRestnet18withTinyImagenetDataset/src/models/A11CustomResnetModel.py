import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 128

        self.prep = self.make_convblock(kernel_size=3, in_channels=3, out_channels=64, stride=1, padding=1,
                                        doMaxPool=False)
        self.cb1 = self.make_convblock(kernel_size=3, in_channels=64, out_channels=128, stride=1, padding=1)
        self.res1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.cb2 = self.make_convblock(kernel_size=3, in_channels=128, out_channels=256, stride=1, padding=1)
        self.cb3 = self.make_convblock(kernel_size=3, in_channels=256, out_channels=512, stride=1, padding=1)
        self.in_planes = 512
        self.res2 = self._make_layer(block, 512, num_blocks[1], stride=1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.maxpool4 = nn.MaxPool2d(kernel_size=4, stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def make_convblock(self, kernel_size, in_channels, out_channels, stride, padding, doMaxPool=True):
        seq = nn.Sequential()
        seq.add_module("Conv2d", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding, bias=False))

        if doMaxPool:
            seq.add_module("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2))

        seq.add_module("BatchNorm", nn.BatchNorm2d(out_channels))
        seq.add_module("Relu", nn.ReLU())
        return seq

    def forward(self, x):
        x = self.prep(x)  # input layer 3-> 64
        l1 = self.cb1(x)  # Layer 1 X 64 -> 128
        r1 = self.res1(l1)  # Resblock 1 128 -> 128
        x = l1 + r1
        l2 = self.cb2(x)  # Layer 2 128 -> 256
        l3 = self.cb3(l2)  # Layer 3 X 256 -> 512
        r2 = self.res2(l3)  # Resblock 2 512 -> 512
        x = l3 + r2
        out = self.maxpool4(x)  # MaxPooling with Kernel Size 4
        out = out.view(out.size(0), -1)
        # out = out.view(-1, 512)
        out = self.linear(out)  # FC Layer
        return F.log_softmax(out, dim=-1)  # Softmax


def A11CustomResnetModel():
    return ResNet(BasicBlock, [1, 1])
