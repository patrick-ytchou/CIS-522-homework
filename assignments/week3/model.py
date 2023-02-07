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
        activation: Callable = torch.nn.ReLU(),
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

        # Define dropout layer
        self.dropout = nn.Dropout(0.2)

        # Define feedforward neural network
        self.layers = nn.ModuleList()

        # Define number of neurons in each layer, using auto-encoder like structure
        # From Ed, we can change number of neurons in each layer as lon as the API remains unchanged
        self.n_neurons = (
            [input_size]
            + [hidden_size // 2**i for i in range((hidden_count + 1) // 2)]
            + [hidden_size // 2**i for i in reversed(range(hidden_count // 2))]
        )

        # Define Feedforward neural network and init weights and bias
        for i in range(self.hidden_count):
            # Define Feedforward neural network and init weights and bias
            self.layers += [nn.Linear(self.n_neurons[i], self.n_neurons[i + 1])]
            self.initializer(self.layers[-1].weight)
            self.layers[-1].bias.data.uniform_

        # Define output layer and initialize weight and bias
        self.out = nn.Linear(self.n_neurons[-1], self.num_classes)
        self.initializer(self.out.weight)
        self.out.bias.data.uniform_

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """

        # Train the model.
        # Order for layers: Linear -> BatchNorm -> Activation -> Dropout
        for enum, layer in enumerate(self.layers):
            # Prediction
            x = layer(x)

            # Batchnorm
            x = nn.BatchNorm1d(layer.out_features)(x)

            # Activation
            x = self.actv(x)

            # Dropout every two layers
            if enum % 2 == 0:
                x = self.dropout(x)

        output = self.out(x)

        return output
