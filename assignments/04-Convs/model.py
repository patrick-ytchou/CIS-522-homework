import torch
import torch.nn as nn


class Model(nn.Module):
    """
    A simple CNN with 2 convolutional layers and 2 fully-connected layers
    """
    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(12)
        # self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(12 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define training sequence for the CNN.

        Args:
            x (torch.Tensor): input sequence of features

        Returns:
            torch.Tensor: output after neural network
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        # print(x.shape)

        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = nn.functional.relu(x)
        # x = self.pool(x)

        x = x.view(-1, 12 * 8 * 8)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)

        return x
