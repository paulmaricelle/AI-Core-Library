import numpy as np
from .layer import Layer
from ..im2col import im2col, col2im

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        he_factor = np.sqrt(2 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * he_factor
        self.b = self.b = np.zeros((1, out_channels, 1, 1))
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        self.input = X
        self.input_shape = X.shape
        self.out_H = (self.input_shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.out_W = (self.input_shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.X_col = im2col(X, self.kernel_size, self.stride, self.padding)
        W_flat = self.W.reshape(self.out_channels, -1)

        out = np.tensordot(W_flat, self.X_col, axes=([1], [1]))
        out = out.transpose([1, 0, 2])

        out = out.reshape(self.input_shape[0], self.out_channels, self.out_H, self.out_W)

        return out + self.b
    
    def backward(self, grad_wrt_output):
        dY_cols = grad_wrt_output.reshape(grad_wrt_output.shape[0], self.out_channels, -1)
        dW_flat = np.tensordot(dY_cols, self.X_col, axes=([0, 2], [0, 2]))

        grad_W = dW_flat.reshape(self.W.shape)
        grad_b = np.sum(grad_wrt_output, axis=(0, 2, 3), keepdims=True)
        
        W_flat = self.W.reshape(self.out_channels, -1)
        dX_col = np.tensordot(W_flat, dY_cols, axes=([0], [1])).transpose([1, 0, 2])
        grad_wrt_input = col2im(dX_col, self.input_shape, self.kernel_size, self.stride, self.padding)

        if self.grad_W is None:
            self.grad_W = grad_W
        else:
            self.grad_W += grad_W

        if self.grad_b is None:
            self.grad_b = grad_b
        else:
            self.grad_b += grad_b

        return grad_wrt_input
        
    def get_params(self):
        return [self.W, self.b]
    
    def get_reg_info(self):
        return [True, False]
    
    def get_grads(self):
        return [self.grad_W, self.grad_b]
    
    def zero_grad(self):
        self.grad_W = None
        self.grad_b = None