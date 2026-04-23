import numpy as np
from .layer import Layer
from typing import Optional, Tuple

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape: Optional[Tuple[int]] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, grad_wrt_input: np.ndarray) -> np.ndarray:
        return grad_wrt_input.reshape(self.input_shape)
    
    def get_params(self):
        return []
    
    def get_reg_info(self):
        return []
    
    def get_grads(self):
        return []