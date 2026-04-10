import numpy as np
from .layer import Layer

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        he_factor = np.sqrt(2 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * he_factor
        self.b = np.zeros((out_channels,))
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        self.input_shape = X.shape
        self.X_col = self.im2col(X)

        W_flat = self.W.reshape(self.out_channels, -1)

        out = np.tensordot(W_flat, self.X_col, axes=([1], [1]))
        out = out.transpose([1, 0, 2])

        out = out.reshape(X.shape[0], self.out_channels, self.input_shape[2], self.input_shape[3])

        return out + self.b.reshape(1, self.out_channels, 1, 1)

    def get_params(self):
        return [self.W, self.b]
    
    def get_reg_info(self):
        return [True, False]
    
    def get_grads(self):
        return [self.grad_W, self.grad_b]