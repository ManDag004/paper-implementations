import torch
import torch.nn as nn

EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-3
NUM_CLASSES = 10
IMG_DIM = 28
class LeNet5(nn.Module):
  def __init__(self, in_ch=1, classes=NUM_CLASSES):
    super().__init__()

    self.net = nn.Sequential(
      nn.Conv2d(in_ch, 6, 5, padding=2), # 28x28x1 -> 28x28x6
      nn.ReLU(),
      nn.MaxPool2d(2,2),
      nn.Conv2d(6, 16, 5),
      nn.ReLU(),
      nn.MaxPool2d(2,2),
      nn.Conv2d(16, 120, 5),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(120, 84),
      nn.ReLU(),
      nn.Linear(84, classes)
    )

  def forward(self, x):
    return self.net(x)
