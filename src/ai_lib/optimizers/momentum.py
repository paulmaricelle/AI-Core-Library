from .opti import Optimizer
import numpy as np
from typing import List

class Momentum(Optimizer):
    def __init__(self, learning_rate: float =1e-2, momentum: float =0.9, weight_decay: float =0):
        super().__init__()
        self.lr = learning_rate
        self.beta = momentum
        self.weight_decay = weight_decay
        self.momentums: List[np.ndarray] = []

    def _init_state(self):
        self.momentums = [np.zeros_like(p) for p in self.params]
        
    def step(self, accumulation_steps: int) -> None:
        self.get_grads()
        if len(self.params) != len(self.grads):
            raise ValueError(f"Mismatch: Optimizer has {len(self.params)} params "
                     f"but received {len(self.grads)} gradients.")
        
        for i in range(len(self.params)):
            self.momentums[i] = self.beta * self.momentums[i] + ((1-self.beta)/accumulation_steps) * self.grads[i]
            self.params[i] -= self.lr * (self.momentums[i] + self.weight_decay * (self.params[i] if self.to_reg[i] == True else 0))

