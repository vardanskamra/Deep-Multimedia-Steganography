import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm([out_channels, 128, 128])  # LayerNorm instead of BatchNorm
        self.silu = nn.SiLU()  # Swish activation

    def forward(self, x):
        return self.silu(self.norm(self.conv(x)))
    
class ConvBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(ConvBranch, self).__init__()
        layers = []
        for k in kernel_sizes:
            layers.append(ConvBlock(in_channels, out_channels, k, padding=k//2))
            in_channels = out_channels  # Maintain consistent channels through layers
        self.branch = nn.Sequential(*layers)

    def forward(self, x):
        return self.branch(x)
    
class PrepNetwork(nn.Module):
    def __init__(self):
        super(PrepNetwork, self).__init__()
        self.branch1 = ConvBranch(3, 64, [3, 3, 3])
        self.branch2 = ConvBranch(3, 64, [5, 5, 5])
        self.branch3 = ConvBranch(3, 64, [7, 7, 7])
        self.final_conv = ConvBlock(192, 3, 3, padding=1)

    def forward(self, secret):
        b1 = self.branch1(secret)
        b2 = self.branch2(secret)
        b3 = self.branch3(secret)
        combined = torch.cat([b1, b2, b3], dim=1)
        return self.final_conv(combined)

class HidingNetwork(nn.Module):
    def __init__(self):
        super(HidingNetwork, self).__init__()
        self.conv1 = ConvBlock(6, 128, 3, padding=1)
        self.conv2 = ConvBlock(128, 128, 3, padding=1)
        self.conv3 = ConvBlock(128, 3, 3, padding=1)

    def forward(self, cover, secret_prepared):
        x = torch.cat([cover, secret_prepared], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)
    
class RevealNetwork(nn.Module):
    def __init__(self):
        super(RevealNetwork, self).__init__()
        self.conv1 = ConvBlock(3, 128, 3, padding=1)
        self.conv2 = ConvBlock(128, 128, 3, padding=1)
        self.conv3 = ConvBlock(128, 3, 3, padding=1)

    def forward(self, stego):
        x = self.conv1(stego)
        x = self.conv2(x)
        return self.conv3(x)
