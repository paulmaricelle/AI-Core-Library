import numpy as np
from .layer import Layer
from typing import Tuple, Optional

class Reshape(Layer):
    def __init__(self, output_shape: Tuple[int, ...]):
        super().__init__()
        self.output_shape = output_shape
        self.input_shape: Optional[Tuple[int, ...]] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input_shape = X.input_shape
        return X.reshape((X.shape[0],) + self.output_shape)
    
    def backward(self, grad_wrt_input: np.ndarray) -> np.ndarray:
        return grad_wrt_input.reshape(self.input_shape)
    
    def get_params(self):
        return []
    
    def get_reg_info(self):
        return []
    
    def get_grads(self):
        return []