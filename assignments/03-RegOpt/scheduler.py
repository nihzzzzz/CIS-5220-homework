from typing import List
import numpy as np

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Exponential LR
    Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, num_epochs, initial_learning_rate, last_epoch=-1):
        """
        Create a new scheduler.
        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.
        """
        # ... Your Code Here ...
        self.num_epochs = num_epochs
        self.initial_learning_rate = initial_learning_rate
        self.total_iters = self.num_epochs * 782
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Note to students: You CANNOT change the arguments or return type of
        this function (because it is called internally by Torch)

        Arguments:
        None
        Returns:
        List[float] the learning rate for this epoch
        """
        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        if self.last_epoch == 0:
            return [
                0.0002 + (i - 0.0001) * (1 + np.cos(np.pi)) / 2 for i in self.base_lrs
            ]

        return [
            0.0002
            + (i - 0.0001)
            * (1 + np.cos(np.pi * self.last_epoch / self.total_iters))
            / 2
            for i in self.base_lrs
        ]
