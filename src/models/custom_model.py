import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
  def __init__(self):
    super().__init__()

    # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

    # self.relu = nn.ReLU()

    # self.pool = nn.MaxPool2d(kernel_size=2)

    # self.flatten = nn.Flatten()

    # self.fc = nn.LazyLinear(2)

    self.features = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, padding=1),
      nn.Relu,
      nn.Conv2d(32, 64, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.LazyLinear(2)
    )

  def forward(self, x):
    # x = self.conv1(x)
    # x = self.relu(x)
    # x = self.pool(x)
    # x = self.flatten(x)
    # x = self.fc(x)

    x = self.features(x)
    x = self.classifier(x)
    return x 
  
