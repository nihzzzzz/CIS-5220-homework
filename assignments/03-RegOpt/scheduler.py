from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Decays the learning rate of each parameter group by a small constant factor until the
    number of epoch reaches a pre-defined milestone: total_iters. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.

    """

    def __init__(self, optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.factor = factor
        self.total_iters = total_iters
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Note to students: You CANNOT change the arguments or return type of
        this function (because it is called internally by Torch)

        """
        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        return [
            i
            * (self.factor + (self.last_epoch >= self.total_iters) * (1 - self.factor))
            for i in self.base_lrs
        ]
