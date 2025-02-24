import torch
from torch import nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.bn = nn.BatchNorm2d(3 * out_channels)
        self.skip = nn.Conv2d(in_channels, 3 * out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.branch1(x)
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        out = torch.cat([x1, x3, x5], dim=1)
        return nn.functional.silu(self.bn(out + self.skip(x)))  # SiLU activation + Skip connection
    
class PrepNetwork(nn.Module):
    def __init__(self):
        super(PrepNetwork, self).__init__()

        # 3x3 convolutions
        self.conv3_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # 5x5 convolutions
        self.conv5_1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.conv5_2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv5_3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        
        # 4x4 convolutions
        self.conv7_1 = nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3)
        self.conv7_2 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3)
        self.conv7_3 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)

        # BatchNorm layers
        self.bn3 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(64)

        self.final_conv = nn.Conv2d(192, 3, kernel_size=3, stride=1, padding=1)
        self.final_bn = nn.BatchNorm2d(3)

    def forward(self, x):
        x3 = self.bn3(self.conv3_3(nn.functional.silu(self.conv3_2(nn.functional.silu(self.conv3_1(x))))))
        x5 = self.bn5(self.conv5_3(nn.functional.silu(self.conv5_2(nn.functional.silu(self.conv5_1(x))))))
        x7 = self.bn7(self.conv7_3(nn.functional.silu(self.conv7_2(nn.functional.silu(self.conv7_1(x))))))
        x = torch.cat([x3, x5, x7], dim=1)  # Concatenation
        x = self.final_bn(self.final_conv(x))
        return torch.sigmoid(x)  # Normalized output
    
class HidingNetwork(nn.Module):
    def __init__(self):
        super(HidingNetwork, self).__init__()
        self.inc1 = InceptionBlock(6, 32)   
        self.inc2 = InceptionBlock(96, 64)  
        self.inc3 = InceptionBlock(192, 128)  

        self.conv = nn.Conv2d(384, 3, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, cover, secret_prepared):
        x = torch.cat([cover, secret_prepared], dim=1)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.inc3(x)
        x = self.bn(self.conv(x))
        return torch.sigmoid(x)  # Stego image
    
class RevealNetwork(nn.Module):
    def __init__(self):
        super(RevealNetwork, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, stego):
        return self.decoder(stego)  # Recovered secret image
