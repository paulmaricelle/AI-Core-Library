from .opti import Optimizer

class Sgd(Optimizer):
    def __init__(self, learning_rate=1e-2, weight_decay=0):
        super().__init__()
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def _init_state(self):
        pass

    def step(self, accumulation_steps):
        self.get_grads()
        if len(self.params) != len(self.grads):
            raise ValueError(f"Mismatch: Optimizer has {len(self.params)} params "
                     f"but received {len(self.grads)} gradients.")

        for i in range(len(self.params)):
            self.params[i] -= self.lr * (self.grads[i] / accumulation_steps + self.weight_decay * (self.params[i] if self.to_reg[i] == True else 0))
