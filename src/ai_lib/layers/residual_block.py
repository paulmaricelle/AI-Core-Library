import numpy as np
from .layer import Layer

class ResidualBlock(Layer):
    def __init__(self, inner_layers):
        super().__init__()
        self.layers = inner_layers

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out + X
    
    def backward(self, grad_wrt_output):
        whole_grad = grad_wrt_output
        for layer in reversed(self.layers):
            whole_grad = layer.backward(whole_grad)
        return whole_grad + grad_wrt_output

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
    
    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params
    
    def get_reg_info(self):
        reg_info = []
        for layer in self.layers:
            reg_info += layer.get_reg_info()
        return reg_info
    
    def get_grads(self):
        grads = []
        for layer in self.layers:
            grads += layer.get_grads()
        return grads