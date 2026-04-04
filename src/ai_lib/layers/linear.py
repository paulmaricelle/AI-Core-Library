from .layer import Layer
import numpy as np

class Linear(Layer):
    def __init__(self, n_input, n_out, init_method = "xavier"):
        super().__init__()
        if init_method == "xavier":
            limit = np.sqrt(6.0 / (n_out + n_input))
            self.W = np.random.uniform(-limit, limit, (n_out, n_input))
        elif init_method == "he":
            std = np.sqrt(2 / n_input)
            self.W = np.random.randn(n_out, n_input) * std
        else:
            self.W = np.random.randn(n_out, n_input) * 10e-2
        self.b = np.zeros((1, n_out))

        self.grad_W = None
        self.grad_b = None
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X
        self.output = np.dot(self.input, self.W.T) + self.b
        return self.output
    
    def backward(self, grad_wrt_output):
        #Gradient par rapport aux poids
        grad_W_current = np.dot(grad_wrt_output.T, self.input)
        
        #Gradient par rapport au biais
        grad_b_current = np.sum(grad_wrt_output, axis=0, keepdims=True)
        
        grad_wrt_input = np.dot(grad_wrt_output, self.W)

        if self.grad_W is None:
            self.grad_W = grad_W_current
        else:
            self.grad_W += grad_W_current

        if self.grad_b is None:
            self.grad_b = grad_b_current
        else:
            self.grad_b += grad_b_current

        return grad_wrt_input
    
    def get_params(self):
        return [self.W, self.b]
    
    def get_grads(self):
        return [self.grad_W, self.grad_b]
    
    def get_reg_info(self):
        return [True, False]
    
    def zero_grad(self):
        self.grad_W = None
        self.grad_b = None