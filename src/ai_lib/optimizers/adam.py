from .opti import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self, learning_rate= 10e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 10e-8):
        super().__init__()
        self.lr = learning_rate
        self.b1 = beta_1
        self.b2 = beta_2 
        self.epsilon = epsilon
        self.m = []
        self.v = []
        self.t = 1

    def _init_state(self):
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        
    def step(self, accumulation_steps):
        self.get_grads()
        if len(self.params) != len(self.grads):
            raise ValueError(f"Mismatch: Optimizer has {len(self.params)} params "
                     f"but received {len(self.grads)} gradients.")
        
        for i in range(len(self.params)):
            grads_average = self.grads[i] / accumulation_steps
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grads_average
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (grads_average ** 2)
            
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            
            self.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        self.t += 1