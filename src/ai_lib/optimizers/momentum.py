from .opti import Optimizer
import numpy as np

class Momentum(Optimizer):
    def __init__(self, learning_rate=10e-2, momentum = 0.9):
        super().__init__()
        self.lr = learning_rate
        self.beta = momentum
        self.momentums = []

    def _init_state(self):
        self.momentums = [np.zeros_like(p) for p in self.params]
        
    def step(self, accumulation_steps):
        self.get_grads()
        if len(self.params) != len(self.grads):
            raise ValueError(f"Mismatch: Optimizer has {len(self.params)} params "
                     f"but received {len(self.grads)} gradients.")
        
        for i in range(len(self.params)):
            self.momentums[i] = self.beta * self.momentums[i] + ((1-self.beta)/accumulation_steps) * self.grads[i]
            self.params[i] -= self.lr * self.momentums[i]

