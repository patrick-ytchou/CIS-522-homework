import torch
import torch.nn as nn
from typing import Callable


class MLP(torch.nn.Module):
    """Neural Network to classify MNIST dataset."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            hidden_count: The number of hidden layers.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.hidden_count = hidden_count
        self.actv = activation
        self.initializer = initializer

        self.dropout = nn.Dropout(0.2)
        self.layers = nn.ModuleList()

        self.layers += [nn.Linear(self.input_size, self.hidden_size)]
        for i in range(self.hidden_count - 1):
            self.layers += [nn.Linear(self.hidden_size, self.hidden_size)]

        self.out = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        # flattening
        x = x.reshape(x.shape[0], -1)

        # train the model. Note that there can be multplie "layer" in each "layers"
        for layer in self.layers:
            x = self.actv(layer(x))
            x = self.dropout(x)
        output = self.out(x)

        return output
