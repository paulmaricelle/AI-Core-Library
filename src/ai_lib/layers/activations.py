from .layer import Layer
import numpy as np

class ReLU(Layer):
    def __init__(self):
        super().__init___()
        self.mask = None

    def forward(self, X):
        self.mask = X > 0
        return np.where(self.mask, X, 0.0) 
    
    def backward(self, grad_wrt_output):
        return np.where(self.mask, grad_wrt_output, 0.0) 
    
    def get_params(self):
        return []
    
    def get_grads(self):
        return []


class Sigmoid(Layer):
    def __init__(self):
        super().__init___()
        self.output = None

    def forward(self, X):
        self.output = 1 / (1+np.exp(-X))
        return self.output
    
    def backward(self, grad_wrt_output):
        return grad_wrt_output * self.output * (1-self.output)
    
    def get_params(self):
        return []
    
    def get_grads(self):
        return []


class Tanh(Layer):
    def __init__(self):
        super().__init___()
        self.output = None

    def forward(self, X):
        self.output = np.tanh(X)
        return self.output
    
    def backward(self, grad_wrt_output):
        return grad_wrt_output * (1 - self.input ** 2)
    
    def get_params(self):
        return []
    
    def get_grads(self):
        return []