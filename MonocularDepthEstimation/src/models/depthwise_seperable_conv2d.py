import torch.nn as nn


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, input, output, padding=0, bias=False, stride=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(input, input, kernel_size=3, padding=padding, groups=input, bias=bias, stride=stride)
        self.pointwise = nn.Conv2d(input, output, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
