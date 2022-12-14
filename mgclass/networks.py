import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional


class ResNet(nn.Module):
    def __init__(self, num_classes, first_layer_kernel=(7, 7)):
        super(ResNet, self).__init__()

        self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.base.conv1 = nn.Conv2d(1, 64, first_layer_kernel, (2, 2), (3, 3), bias=False)

        self.base.fc = nn.Sequential(
            nn.Linear(self.base.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        )

    # x represents our data
    def forward(self, x):
        return self.base(x)


# noinspection DuplicatedCode
class MusicRecNet(nn.Module):
    def __init__(self, num_classes):
        super(MusicRecNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.25)
        self.dropout4 = nn.Dropout(0.25)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(25088, 128)
        self.fc2 = nn.Linear(128, num_classes)

    # x represents our data
    def forward(self, x):
        x = self.conv1(x)
        x = functional.relu(x)
        x = self.bn1(x)
        x = functional.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = functional.relu(x)
        x = self.bn2(x)
        x = functional.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = functional.relu(x)
        x = self.bn3(x)
        x = functional.max_pool2d(x, 2)
        x = self.dropout3(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = functional.relu(x)
        x = self.bn4(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        x = functional.relu(x)

        output = functional.softmax(x, dim=1)
        return output


# noinspection DuplicatedCode
class MgcNet(nn.Module):
    def __init__(self, num_classes):
        super(MgcNet, self).__init__()
        self.conv1a = nn.Conv2d(1, 32, 3, 1)
        self.conv1b = nn.Conv2d(32, 32, 3, 1)
        self.conv2a = nn.Conv2d(32, 64, 3, 1)
        self.conv2b = nn.Conv2d(64, 64, 3, 1)
        self.conv3a = nn.Conv2d(64, 128, 3, 1)
        self.conv3b = nn.Conv2d(128, 128, 3, 1)
        self.conv4a = nn.Conv2d(128, 256, 3, 1)
        self.conv4b = nn.Conv2d(256, 256, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.25)
        self.dropout4 = nn.Dropout2d(0.25)
        self.dropout5 = nn.Dropout(0.25)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(4096, 128)
        self.fc2 = nn.Linear(128, num_classes)

    # x represents our data
    def forward(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = functional.relu(x)
        x = self.bn1(x)
        x = functional.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = functional.relu(x)
        x = self.bn2(x)
        x = functional.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = functional.relu(x)
        x = self.bn3(x)
        x = functional.max_pool2d(x, 2)
        x = self.dropout3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = functional.relu(x)
        x = self.bn4(x)
        x = functional.max_pool2d(x, 2)
        x = self.dropout4(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = functional.relu(x)
        x = self.bn5(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        x = functional.relu(x)

        output = functional.softmax(x, dim=1)
        return output
