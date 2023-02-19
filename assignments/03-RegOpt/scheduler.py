from typing import List
import numpy as np
import math

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    """

    def __init__(self, optimizer, gamma=0.1, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        # count_ms = []
        # for i in milestones:
        #     temp = np.ones(len(milestones)) * i
        #     count = np.sum(temp == milestones)
        #     count_ms.append((str(i), count))
        # self.milestones = dict(count_ms)
        # self.gamma = gamma
        self.gamma = gamma
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Note to students: You CANNOT change the arguments or return type of
        this function (because it is called internally by Torch)

        """
        # ... Your Code Here ...

        return [base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs]

        # Here's our dumb baseline implementation:
        # return [i for i in self.base_lrs]
