import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()

        self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.base.conv1 = nn.Conv2d(1, 64, (64, 64), (2, 2), (3, 3), bias=False)

        self.base.fc = nn.Sequential(
            nn.Linear(self.base.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        )

    # x represents our data
    def forward(self, x):
        return self.base(x)
