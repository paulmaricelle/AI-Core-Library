from .layer import Layer
import numpy as np
from typing import Optional

class Dropout(Layer):
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()
        self.rate = dropout_rate
        self.mask: Optional[np.ndarray] = None

    def forward(self, X):
        if self.training:
            mask = np.random.rand(*X.shape) > self.rate
            self.mask = mask
            return (X * mask) / (1 - self.rate)
        else:
            return X
        
    def backward(self, grad_wrt_output):
        return grad_wrt_output * self.mask / (1 - self.rate)
    
    def get_params(self):
        return []
    
    def get_reg_info(self):
        return []
    
    def get_grads(self):
        return []
    
    def zero_grad(self):
        pass