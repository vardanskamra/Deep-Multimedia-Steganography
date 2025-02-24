import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class PrepNetwork(nn.Module):
    def __init__(self):
        super(PrepNetwork, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, secret):
        return self.layers(secret)
    
class HidingNetwork(nn.Module):
    def __init__(self):
        super(HidingNetwork, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(6, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, cover, secret_prepared):
        x = torch.cat([cover, secret_prepared], dim=1)
        return self.layers(x)
    
class RevealNetwork(nn.Module):
    def __init__(self):
        super(RevealNetwork, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, stego):
        return self.layers(stego)