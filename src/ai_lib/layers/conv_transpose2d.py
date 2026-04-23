import numpy as np
from .layer import Layer
from ..im2col import im2col, col2im
from typing import Optional

class ConvTranspose2d(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int =1, padding: int =0):
        super().__init__()
        he_factor = np.sqrt(2 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(in_channels, out_channels, kernel_size, kernel_size) * he_factor
        self.b = self.b = np.zeros((1, out_channels, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.grad_W: Optional[np.ndarray] = None
        self.grad_b: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray):
        self.input_shape = X.shape
        B, C, H, W = X.shape

        X_flat = X.reshape(B, C, H * W)
        self.X_flat = X_flat

        W_flat = self.W.reshape(self.in_channels, self.out_channels * self.kernel_size * self.kernel_size)

        Y_flat = np.tensordot(X_flat, W_flat, axes=[[1], [0]])
        Y_flat_transposed = np.transpose(Y_flat, axes=[0, 2, 1])

        raw_H_out = (H - 1) * self.stride + self.kernel_size
        raw_W_out = (W - 1) * self.stride + self.kernel_size
        Y_col = col2im(Y_flat_transposed, (B, self.out_channels, raw_H_out, raw_W_out), self.kernel_size, self.stride, 0)

        if self.padding > 0:
            return Y_col[:, :, self.padding:-self.padding, self.padding:-self.padding] + self.b
        else:
            return Y_col + self.b
        
    def backward(self, grad_wrt_output: np.ndarray):
        B, C, H_in, W_in = self.input_shape
        grad_b = np.sum(grad_wrt_output, axis=(0, 2, 3), keepdims=True)

        if self.padding > 0:
            p = self.padding
            grad_padded = np.pad(grad_wrt_output, pad_width=[(0, 0), (0, 0), (p, p), (p, p)])
        else:
            grad_padded = grad_wrt_output

        grad_col = im2col(grad_padded, self.kernel_size, self.stride, padding=0)
        # grad_col has shape (B, out_channels * kernel_size * kernel_size, H_in * W_in)
        dY_flat = np.transpose(grad_col, axes=[0, 2, 1])

        dW_flat = np.tensordot(self.X_flat, dY_flat, axes=[[0, 2], [0, 1]])
        grad_W = dW_flat.reshape(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size)

        W_flat = self.W.reshape(self.in_channels, self.out_channels * self.kernel_size * self.kernel_size)
        dX = np.tensordot(grad_col, W_flat, axes=[[1], [1]])
        dX_transposed = np.transpose(dX, axes=[0, 2, 1])

        if self.grad_W is None:
            self.grad_W = grad_W
        else:
            self.grad_W += grad_W
        
        if self.grad_b is None:
            self.grad_b = grad_b
        else:
            self.grad_b += grad_b
        
        return dX_transposed.reshape(B, C, H_in, W_in)
    
    def get_params(self):
        return [self.W, self.b]
    
    def get_reg_info(self):
        return [True, False]
    
    def get_grads(self):
        return [self.grad_W, self.grad_b]
    
    def zero_grad(self):
        self.grad_W = None
        self.grad_b = None