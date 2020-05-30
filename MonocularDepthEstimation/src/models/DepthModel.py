import torch
import torch.nn as nn

from src.models import DepthwiseSeparableConv2d


class DepthModel(nn.Module):

    def __init__(self):
        super(DepthModel, self).__init__()

        self.d1 = self.downsample_conv(16, 16)

        self.d2 = self.downsample_conv(16, 32)

        self.d3 = self.downsample_conv(32, 64)

        self.d4 = self.downsample_conv(64, 128)

        # self.u1 = self.upconv(128, 128)

        self.u2 = self.upconv(128, 64)

        self.u3 = self.upconv(64, 32)

        self.u4 = self.upconv(32, 16)

        self.final = DoubleConv(16, 3)

        self.prep = DoubleConv(6, 16)

    def downsample_conv(self, in_planes, out_planes, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            # DepthwiseSeparableConv2d(input=in_planes, output=out_planes, padding=1, bias=False, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            # DepthwiseSeparableConv2d(input=out_planes, output=out_planes, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_planes, out_planes):
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(inplace=True)
        )

    def crop_like(self, input, ref):
        assert (input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
        return input[:, :, :ref.size(2), :ref.size(3)]

    def forward(self, data):
        x = torch.cat((data[0], data[1]), 1)
        x = self.prep(x)
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        # up_input0 = self.crop_like(self.u1(d4), d3)
        # up_input0 = torch.cat((up_input0, d3), 1)
        # u0 = self.u1(up_input0)

        up_input1 = self.crop_like(self.u2(d4), d3)
        up_input1 = torch.cat((up_input1, d3), 1)
        u1 = self.u2(up_input1)

        up_input2 = self.crop_like(self.u3(u1), d2)
        up_input2 = torch.cat((up_input2, d2), 1)
        u2 = self.u3(up_input2)

        up_input2 = self.crop_like(self.u4(u2), d1)
        up_input3 = torch.cat((up_input2, d1), 1)
        u3 = self.u4(up_input3)

        final_x = self.final(u3)
        return final_x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
