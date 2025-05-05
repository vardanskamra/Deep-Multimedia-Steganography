import torch
from torch import nn

class PrepNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial multi-scale processing
        self.initialP3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU())
        
        self.initialP5 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU())
        
        self.initialP7 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU())

        # Upsampling and final processing
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(192, 150, kernel_size=4, stride=2, padding=1),  # 128->256
            nn.SiLU(),
            nn.Conv2d(150, 150, 3, padding=1),
            nn.SiLU()
        )

    def forward(self, p):
        p1 = self.initialP3(p)
        p2 = self.initialP5(p)
        p3 = self.initialP7(p)
        combined = torch.cat((p1, p2, p3), 1)
        return self.upscale(combined)

class HidingNetwork(nn.Module):
    def __init__(self):
        super(HidingNetwork, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(153, 32)   # output: 32 channels, same spatial dims
        self.pool1 = nn.MaxPool2d(2)            # downsample by factor of 2

        self.enc2 = self.conv_block(32, 64)     # output: 64 channels
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self.conv_block(64, 128)    # output: 128 channels
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = self.conv_block(128, 256)   # output: 256 channels
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 256)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(256 + 256, 128)  # concatenate with enc4

        self.upconv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128 + 128, 64)

        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64 + 64, 32)

        self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(32 + 32, 16)

        self.final_conv = nn.Conv2d(16, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        return block

    def forward(self, cover, secret_prepared):
        # Concatenate cover and secret_prepared along channel dimension.
        x = torch.cat([cover, secret_prepared], dim=1)

        # Encoder
        e1 = self.enc1(x)   # shape: [B, 32, H, W]
        p1 = self.pool1(e1) # shape: [B, 32, H/2, W/2]

        e2 = self.enc2(p1)  # shape: [B, 64, H/2, W/2]
        p2 = self.pool2(e2) # shape: [B, 64, H/4, W/4]

        e3 = self.enc3(p2)  # shape: [B, 128, H/4, W/4]
        p3 = self.pool3(e3) # shape: [B, 128, H/8, W/8]

        e4 = self.enc4(p3)  # shape: [B, 256, H/8, W/8]
        p4 = self.pool4(e4) # shape: [B, 256, H/16, W/16]

        # Bottleneck
        b = self.bottleneck(p4)  # shape: [B, 256, H/16, W/16]

        # Decoder
        d4 = self.upconv4(b)     # shape: [B, 256, H/8, W/8]
        d4 = torch.cat([d4, e4], dim=1)  # Skip connection from encoder stage 4
        d4 = self.dec4(d4)       # shape: [B, 128, H/8, W/8]

        d3 = self.upconv3(d4)    # shape: [B, 128, H/4, W/4]
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)       # shape: [B, 64, H/4, W/4]

        d2 = self.upconv2(d3)    # shape: [B, 64, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)       # shape: [B, 32, H/2, W/2]

        d1 = self.upconv1(d2)    # shape: [B, 32, H, W]
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)       # shape: [B, 16, H, W]

        out = self.final_conv(d1)
        out = self.sigmoid(out)
        return out
    
class RevealNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Downsampling layers with stride=2
        self.initialR3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 256->128
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU())
        
        self.initialR5 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),  # 256->128
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU())
        
        self.initialR7 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 256->128
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU())

        # Final processing at 128x128 resolution
        self.finalR = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(64 ,3, kernel_size=1),
            nn.Sigmoid())

    def forward(self, r):
        r1 = self.initialR3(r)
        r2 = self.initialR5(r)
        r3 = self.initialR7(r)
        combined = torch.cat((r1, r2, r3), 1)
        return self.finalR(combined)