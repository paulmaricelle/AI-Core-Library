from .layer import Layer
import numpy as np
from typing import Optional

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.mask: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.mask = X > 0
        return np.where(self.mask, X, 0.0) 
    
    def backward(self, grad_wrt_output: np.ndarray) -> np.ndarray:
        return np.where(self.mask, grad_wrt_output, 0.0) 
    
    def get_params(self):
        return []
    
    def get_reg_info(self):
        return []
    
    def get_grads(self):
        return []
    
    def get_state(self):
        return {}
    
    def set_state(self, state):
        pass


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.output: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.output = 1 / (1+np.exp(-X))
        return self.output
    
    def backward(self, grad_wrt_output: np.ndarray) -> np.ndarray:
        return grad_wrt_output * self.output * (1-self.output)
    
    def get_params(self):
        return []
    
    def get_reg_info(self):
        return []

    def get_grads(self):
        return []

    def get_state(self):
        return {}
    
    def set_state(self, state):
        pass

class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.output: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.output = np.tanh(X)
        return self.output
    
    def backward(self, grad_wrt_output: np.ndarray) -> np.ndarray:
        return grad_wrt_output * (1 - self.output ** 2)
    
    def get_params(self):
        return []

    def get_reg_info(self):
        return []
        
    def get_grads(self):
        return []
    
    def get_state(self):
        return {}
    
    def set_state(self, state):
        pass