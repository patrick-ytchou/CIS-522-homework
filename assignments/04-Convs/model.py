import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    A simple CNN with 2 convolutional layers and 2 fully-connected layers
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.out_channels1 = 36
        # self.out_channels2 = 20
        # self.fc_hidden = 64

        self.conv1 = nn.Conv2d(
            num_channels,
            self.out_channels1,
            kernel_size=3,
            stride=2,
            padding=2,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(self.out_channels1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(self.out_channels1 * 8 * 8, num_classes)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define training sequence for the CNN.

        Args:
            x (torch.Tensor): input sequence of features

        Returns:
            torch.Tensor: output after neural network
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, self.out_channels1 * 8 * 8)
        x = self.fc1(x)

        return x
