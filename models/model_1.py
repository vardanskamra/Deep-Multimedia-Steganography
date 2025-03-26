import torch
from torch import nn

class PrepNetwork(nn.Module):
  def __init__(self):
    super(PrepNetwork, self).__init__()
    self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
    self.relu = nn.ReLU()

  def forward(self, secret):
    x = self.relu(self.conv1(secret))
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x))
    x = self.conv4(x)
    return x

class HidingNetwork(nn.Module):
  def __init__(self):
    super(HidingNetwork, self).__init__()
    self.conv1 = nn.Conv2d(6, 128, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
    self.relu = nn.ReLU()

  def forward(self, cover, secret_prepared):
    x = torch.cat([cover, secret_prepared], dim=1)
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x))
    x = self.conv4(x)
    return x

class RevealNetwork(nn.Module):
  def __init__(self):
    super(RevealNetwork, self).__init__()
    self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(128, 3, kernel_size=3, padding=1)
    self.relu = nn.ReLU()

  def forward(self, stego):
    x = self.relu(self.conv1(stego))
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x))
    x = self.conv4(x)
    return x