from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    """Config file for the scheduler"""

    batch_size = 128
    num_epochs = 30
    initial_learning_rate = 0.00125
    initial_weight_decay = 0.0001

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "last_epoch": -1,
        "gamma": 0.1,
        "c": 0.8,
        "step_size": 3,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
