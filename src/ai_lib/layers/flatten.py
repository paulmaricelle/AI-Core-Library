import numpy as np
from .layer import Layer

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, grad_wrt_input):
        return grad_wrt_input.reshape(self.input_shape)
    
    def get_params(self):
        return []
    
    def get_reg_info(self):
        return []
    
    def get_grads(self):
        return []