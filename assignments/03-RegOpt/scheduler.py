from typing import List

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """Custom learning rate scheduler"""

    def __init__(self, optimizer, last_epoch=-1, gamma=0.5, c=1.5):
        """
        Create a new scheduler.

        Arguments:
            optimizer: (torch.optim.Optimizer)
                Optimizer object

            last_epoch: (int)
                The index of last epoch

            gamma : (float)
                The gamma parameter for the exponential distribution

            c : (float)
                The c parameter for the weibull distribution used on gamma
        """
        # ... Your Code Here ...
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        self.gamma = gamma
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.c = c

    def get_lr(self) -> List[float]:
        """Function to get learning rate

        Returns:
            List[float]: list of learning rates by the definition of the scheduler
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            group["lr"] * np.exp(self.gamma**self.c)
            for group in self.optimizer.param_groups
        ]
