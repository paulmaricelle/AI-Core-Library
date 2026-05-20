import numpy as np
from abc import ABC, abstractmethod

class LRScheduler(ABC):
    @abstractmethod
    def get_lr(self, current_step: int) -> float:
        pass

class CosineWarmupScheduler(LRScheduler):
    def __init__(self, lr_max: float, lr_min: float, warmup_steps: int, total_steps: int):
        """A step is a whole batch"""
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, current_step):
        """ Current step is the number of batchs already seen + 1"""
        if current_step < self.warmup_steps:
            return self.lr_max * (current_step / self.warmup_steps)

        if current_step >= self.total_steps:
            return self.lr_min
        
        p = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * p))
        return self.lr_min + (self.lr_max - self.lr_min) * cosine_decay