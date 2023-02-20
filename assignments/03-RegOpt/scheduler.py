from typing import List
import numpy as np

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    CosineAnnealingLR
    with reduce learning rate.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_epochs (int): The total number of epochs.
        initial_learning_rate (float): The initial learning rate.
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
        self.min_lrs = [0] * len(optimizer.param_groups)
        self.num_bad_epochs = 0
        self.best = None
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def _reduce_lr(self) -> None:
        """
        Arguments:None
        Returns: None
        """
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * 0.01, self.min_lrs[i])
            if old_lr - new_lr > 1e-8:
                param_group["lr"] = new_lr

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
                0.00001 + (i - 0.00001) * (1 + np.cos(np.pi)) / 2 for i in self.base_lrs
            ]

        if self.last_epoch >= 5000:
            self._reduce_lr()
            return [
                0.00001
                + (i - 0.00001)
                * 1
                * (1 + np.cos(np.pi * self.last_epoch / self.total_iters))
                / 2
                for i in self.base_lrs
            ]
        return [
            0.00001
            + (i - 0.00001)
            * (1 + np.cos(np.pi * self.last_epoch / self.total_iters))
            / 2
            for i in self.base_lrs
        ]
