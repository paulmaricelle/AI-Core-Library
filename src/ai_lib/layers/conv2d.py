import numpy as np

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        he_factor = np.sqrt(2 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * he_factor
        self.b = np.zeros((out_channels, 1))
        self.grad_W = None
        self.grad_b = None

    def get_params(self):
        return [self.W, self.b]
    
    def get_reg_info(self):
        return [True, False]
    
    def get_grads(self):
        return [self.grad_W, self.grad_b]