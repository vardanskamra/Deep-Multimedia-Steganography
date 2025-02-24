import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_channels // 4, kernel_size=1, padding=0)
        self.branch2 = ConvBlock(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.branch3 = ConvBlock(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_channels // 4, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual

class PrepNetwork(nn.Module):
    def __init__(self):
        super(PrepNetwork, self).__init__()
        self.layers = nn.Sequential(
            InceptionModule(3, 128),
            ResidualBlock(128, 128),
            DepthwiseSeparableConv(128, 128),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, secret):
        return self.layers(secret)

class HidingNetwork(nn.Module):
    def __init__(self):
        super(HidingNetwork, self).__init__()
        self.layers = nn.Sequential(
            InceptionModule(6, 128),
            ResidualBlock(128, 128),
            DepthwiseSeparableConv(128, 128),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, cover, secret_prepared):
        x = torch.cat([cover, secret_prepared], dim=1)
        return self.layers(x)

class RevealNetwork(nn.Module):
    def __init__(self):
        super(RevealNetwork, self).__init__()
        self.layers = nn.Sequential(
            InceptionModule(3, 128),
            ResidualBlock(128, 128),
            DepthwiseSeparableConv(128, 128),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, stego):
        return self.layers(stego)
