from .opti import Optimizer
import numpy as np

class Momentum(Optimizer):
    def __init__(self, learning_rate=10e-2, momentum = 0.9):
        super().__init__()
        self.lr = learning_rate
        self.mu = momentum

        self.momentums = {}
        for layer in self.layers:
            if hasattr(layer, 'W') and layer.W is not None:
                self.momentums[layer] = {'W': np.zeros(layer.W.shape), 'b': np.zeros(layer.b.shape)}


    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'W') and layer.W is not None:
                m = self.momentums[layer]

                m['W'] = self.mu * m['W'] + (1-self.mu) * layer.grad_W
                m['b'] = self.mu * m['b'] + (1-self.mu) * layer.grad_b

                layer.W -= self.lr * m['W']
                layer.b -= self.lr * m['b']