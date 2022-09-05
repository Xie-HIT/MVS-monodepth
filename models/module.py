import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class FeatureNet(nn.Module):
    def __init__(self, base_channels):
        super(FeatureNet, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            ConvBnRelu(in_channels=3, out_channels=self.base_channels, kernel_size=3, stride=1,
                       padding=1, bias=False),
            ConvBnRelu(in_channels=self.base_channels, out_channels=self.base_channels, kernel_size=3, stride=1,
                       padding=1, bias=False)
        )

        self.conv1 = nn.Sequential(
            ConvBnRelu(in_channels=self.base_channels, out_channels=self.base_channels * 2, kernel_size=5, stride=2,
                       padding=2, bias=False),
            ConvBnRelu(in_channels=self.base_channels * 2, out_channels=self.base_channels * 2, kernel_size=3, stride=1,
                       padding=1, bias=False),
            ConvBnRelu(in_channels=self.base_channels * 2, out_channels=self.base_channels * 2, kernel_size=3, stride=1,
                       padding=1, bias=False),
        )

        self.conv2 = nn.Sequential(
            ConvBnRelu(in_channels=self.base_channels * 2, out_channels=self.base_channels * 4, kernel_size=5, stride=2,
                       padding=2, bias=False),
            ConvBnRelu(in_channels=self.base_channels * 4, out_channels=self.base_channels * 4, kernel_size=3, stride=1,
                       padding=1, bias=False),
            ConvBnRelu(in_channels=self.base_channels * 4, out_channels=self.base_channels * 4, kernel_size=3, stride=1,
                       padding=1, bias=False),
        )

        self.inner1 = ConvBnRelu(in_channels=self.base_channels, out_channels=self.base_channels * 4, kernel_size=1,
                                 stride=1, padding=1, bias=True)

        self.inner2 = ConvBnRelu(in_channels=self.base_channels * 2, out_channels=self.base_channels * 4, kernel_size=1,
                                 stride=1, padding=1, bias=True)

        self.out1 = ConvBnRelu(in_channels=self.base_channels * 4, out_channels=self.base_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.out2 = ConvBnRelu(in_channels=self.base_channels * 4, out_channels=self.base_channels * 2, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.out3 = ConvBnRelu(in_channels=self.base_channels * 4, out_channels=self.base_channels * 4, kernel_size=1,
                               stride=1, padding=1, bias=False)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x)
        x3 = self.conv2(x)

        intra_feat3 = x3
        intra_feat2 = F.interpolate(intra_feat3, scale_factor=2, mode='nearest') + self.inner2(x2)
        intra_feat1 = F.interpolate(intra_feat2, scale_factor=2, mode='nearest') + self.inner1(x1)

        outputs = dict()
        outputs['stage1'] = self.out3(intra_feat3)
        outputs['stage2'] = self.out2(intra_feat2)
        outputs['stage3'] = self.out1(intra_feat1)

        return outputs
