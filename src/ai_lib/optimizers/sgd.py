from .opti import Optimizer

class Sgd(Optimizer):
    def __init__(self, learning_rate=10e-2):
        super().__init__()
        self.lr = learning_rate

    def setup(self, layers):
        self.layers = layers
        params = []
        for layer in self.layers:
            params += layer.get_params()
        self.params = params

    def step(self):
        if len(self.params) != len(self.grads):
            raise ValueError(f"Mismatch: Optimizer has {len(self.params)} params "
                     f"but received {len(self.grads)} gradients.")
        
        self.get_grads()

        for i in range(len(self.params)):
            self.params[i] -= self.lr * self.grads[i]
