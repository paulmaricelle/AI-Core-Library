from .layer import Layer
import numpy as np

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, X):
        self.mask = X > 0
        return np.where(self.mask, X, 0.0) 
    
    def backward(self, grad_wrt_output):
        return np.where(self.mask, grad_wrt_output, 0.0) 
    
    def get_params(self):
        return []
    
    def get_reg_info(self):
        return []
    
    def get_grads(self):
        return []


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, X):
        self.output = 1 / (1+np.exp(-X))
        return self.output
    
    def backward(self, grad_wrt_output):
        return grad_wrt_output * self.output * (1-self.output)
    
    def get_params(self):
        return []
    
    def get_reg_info(self):
        return []

    def get_grads(self):
        return []


class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, X):
        self.output = np.tanh(X)
        return self.output
    
    def backward(self, grad_wrt_output):
        return grad_wrt_output * (1 - self.output ** 2)
    
    def get_params(self):
        return []

    def get_reg_info(self):
        return []
        
    def get_grads(self):
        return []
    

class InferenceOnlySoftmax(Layer):
    # I already created a SoftmaCrossEntropy loss which does
    # not have a "is_training" attribute, therefore, I had a problem
    # when calulating metrics and performing early_stopping
    # Therefore I though of adding this layer being the identity during 
    # training and a softmax during inference
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        if self.training:
            return X
        else:
            exp = np.exp(X - np.max(X, axis=0, keepdims=True))
            return exp/np.sum(exp, axis=0, keepdims=True)

    def backward(self, grad_wrt_output):
        # Backward n'est appelé que durant l'entraînement quand la loss est là
        return grad_wrt_output
    
    def get_params(self):
        return []

    def get_reg_info(self):
        return []
        
    def get_grads(self):
        return []