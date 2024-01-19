import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnRelu3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class DeconvBnRelu3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, **kwargs):
        super(DeconvBnRelu3d, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                         bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.deconv(x)), inplace=True)


# for an aggregated module that based on nn.sequential
def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True)
    )


# a u-net for regularization
class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnRelu3d(in_channels, base_channels)

        self.conv1 = ConvBnRelu3d(base_channels, 2 * base_channels, stride=2)
        self.conv2 = ConvBnRelu3d(2 * base_channels, 2 * base_channels)

        self.conv3 = ConvBnRelu3d(2 * base_channels, 4 * base_channels, stride=2)
        self.conv4 = ConvBnRelu3d(4 * base_channels, 4 * base_channels)

        self.conv5 = ConvBnRelu3d(4 * base_channels, 8 * base_channels, stride=2)
        self.conv6 = ConvBnRelu3d(8 * base_channels, 8 * base_channels)

        self.conv7 = DeconvBnRelu3d(8 * base_channels, 4 * base_channels, output_padding=1)
        self.conv8 = DeconvBnRelu3d(4 * base_channels, 2 * base_channels)
        self.conv9 = DeconvBnRelu3d(2 * base_channels, base_channels, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))  # xyz res halved
        conv4 = self.conv4(self.conv3(conv2))  # xyz res halved
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv8(x)
        x = conv0 + self.conv9(x)
        x = self.prob(x)

        return x
