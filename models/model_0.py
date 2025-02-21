import torch
from torch import nn

class PrepNetwork(nn.Module):
  def __init__(self):
    super(PrepNetwork, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    self.relu = nn.ReLU()

  def forward(self, secret):
    x = self.relu(self.conv1(secret))
    x = self.relu(self.conv2(x))
    x = self.conv3(x)
    return x

class HidingNetwork(nn.Module):
  def __init__(self):
    super(HidingNetwork, self).__init__()
    self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)  # Cover + Secret
    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    self.relu = nn.ReLU()

  def forward(self, cover, secret_prepared):
    x = torch.cat([cover, secret_prepared], dim=1)
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.conv3(x)
    return x  # Stego image

class RevealNetwork(nn.Module):
  def __init__(self):
    super(RevealNetwork, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    self.relu = nn.ReLU()

  def forward(self, stego):
    x = self.relu(self.conv1(stego))
    x = self.relu(self.conv2(x))
    x = self.conv3(x)
    return x  # Reconstructed secret image