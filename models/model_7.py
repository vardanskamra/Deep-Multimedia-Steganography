import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.silu  = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.silu(out)
        return out
    
class PrepNetwork(nn.Module):
    """
    A full-scale autoencoder to process (prepare) the secret image.
    Downsamples 256x256 images to a low-dimensional representation and upsamples back.
    """
    def __init__(self):
        super(PrepNetwork, self).__init__()
        # Encoder: 256 -> 128 -> 64 -> 32
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        # Decoder: 32 -> 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32 -> 64
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64 -> 128
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128 -> 256
            nn.Sigmoid()  # ensure output in [0, 1]
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class HidingNetwork(nn.Module):
    """
    Hides the processed secret in the cover image.
    Uses a deeper network with residual blocks to capture fine details.
    """
    def __init__(self):
        super(HidingNetwork, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.resblock1 = ResidualBlock(64)
        self.resblock2 = ResidualBlock(64)
        self.resblock3 = ResidualBlock(64)
        self.resblock4 = ResidualBlock(64)
        self.resblock5 = ResidualBlock(64)
        self.resblock6 = ResidualBlock(64)
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # output stego image in [0, 1]
        )
    
    def forward(self, cover, secret_prepared):
        x = torch.cat([cover, secret_prepared], dim=1)
        x = self.initial(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.final(x)
        return x
    
class RevealNetwork(nn.Module):
    """
    Reveals the secret from the stego image.
    Uses a similar residual-based architecture to extract hidden signals.
    """
    def __init__(self):
        super(RevealNetwork, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.resblock1 = ResidualBlock(64)
        self.resblock2 = ResidualBlock(64)
        self.resblock3 = ResidualBlock(64)
        self.resblock4 = ResidualBlock(64)
        self.resblock5 = ResidualBlock(64)
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # output revealed secret in [0, 1]
        )
    
    def forward(self, stego):
        x = self.initial(stego)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.final(x)
        return x
