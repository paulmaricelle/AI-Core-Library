from .opti import Optimizer

class Sgd(Optimizer):
    def __init__(self, learning_rate=10e-2):
        super().__init__()
        self.lr = learning_rate

    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'W') and layer.W is not None:
                layer.W -= self.lr * layer.grad_W
                layer.b -= self.lr * layer.grad_b