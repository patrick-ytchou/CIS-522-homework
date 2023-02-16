from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """Custom learning rate scheduler"""

    def __init__(self, optimizer, last_epoch=-1, gamma=0.5, c=1.5):
        """
        Create a new scheduler.

        Arguments:
            optimizer: (torch.optim.Optimizer)

            last_epoch: (int)

            gamma : (float)

            c : (float)


        Returns:
            None


        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        TODO: Weibull
        """
        # ... Your Code Here ...
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        self.gamma = gamma
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.c = c

    def get_lr(self) -> List[float]:
        """_summary_

        Returns:
            List[float]: _description_
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            group["lr"] * self.gamma**self.c for group in self.optimizer.param_groups
        ]
