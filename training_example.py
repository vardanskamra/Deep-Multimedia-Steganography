import torch
import torchvision

from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader

from utils.transforms import train_test_transform
from utils.metrics import loss_function
from utils.train import train
from utils.test import test

from utils.visualizations import plot_metrics
from utils.visualizations import visualize_images

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_test_transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=train_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train Dataset Length: {len(train_dataset)}")
print(f"Test Dataset Length: {len(test_dataset)}")
print(f"Train DataLoader Length: {len(train_loader)}")
print(f"Test DataLoader Length: {len(test_loader)}")

sample, label = next(iter(train_loader))
print(f"Sample Shape: {sample.shape}")
print(f"Label Shape: {label.shape}")

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

print(f"Device: {device}")
prep_net = PrepNetwork().to(device)
hide_net = HidingNetwork().to(device)
reveal_net = RevealNetwork().to(device)

optimizer = torch.optim.Adam(list(prep_net.parameters()) +
                       list(hide_net.parameters()) +
                       list(reveal_net.parameters()), lr=0.001)

metrics = train(dataloader=train_loader,
                prep_net=prep_net,
                hide_net=hide_net,
                reveal_net=reveal_net,
                optimizer=optimizer,
                loss_fn=loss_function,
                beta=0.75,
                epochs=1,
                device=device)

plot_metrics(metrics)

metrics = test(prep_net=prep_net,
               hide_net=hide_net,
               reveal_net=reveal_net,
               dataloader=test_loader,
               loss_fn = loss_function,
               beta = 0.75,
               visualize = True,
               device=device)

torch.save(prep_net.state_dict(), "prep_net.pth")
torch.save(hide_net.state_dict(), "hide_net.pth")
torch.save(reveal_net.state_dict(), "reveal_net.pth")

