import numpy as np
from .layer import Layer
from ..im2col import im2col, col2im

class MaxPooling2D(Layer):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.activation_mask = None
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        B, C, H, W = X.shape

        X_col = im2col(X, self.kernel_size, self.stride, self.padding)

        X_col_reshaped = X_col.reshape(B * C, self.kernel_size * self.kernel_size, -1)
        out_col = np.max(X_col_reshaped, axis=1, keepdims=True)

        self.activation_mask = (X_col == out_col)

        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        return out_col.reshape(B, C, out_h, out_w)

    def backward(self, grad_wrt_output):
        B, C, out_h, out_w = grad_wrt_output.shape
        
        dY_col = grad_wrt_output.reshape(B * C, 1, -1)
        dX_col = dY_col * self.activation_mask

        dX_col_reshaped = dX_col.reshape(B, C * self.kernel_size * self.kernel_size, -1)

        return col2im(dX_col_reshaped, self.input_shape, self.kernel_size, self.stride, self.padding)
    
    def get_params(self):
        return []
        
    def get_reg_info(self):
        return []
        
    def get_grads(self):
        return []