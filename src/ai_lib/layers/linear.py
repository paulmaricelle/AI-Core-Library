from .layer import Layer
import numpy as np

class Linear(Layer):
    def __init__(self, n_input, n_out):
        super().__init__()
        #Initialisation uniforme Xavier
        xavier = np.sqrt(6.0/(n_out+n_input))

        self.W = np.random.uniform(-xavier, xavier, (n_out, n_input))
        self.b = np.zeros((n_out, 1))

        self.grad_W = None
        self.grad_b = None
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X
        self.output = np.dot(self.W, self.input) + self.b
        return self.output
    
    def backward(self, grad_wrt_output):
        batch_size = self.input.shape[1]

        #Gradient par rapport aux poids
        self.grad_W = np.dot(grad_wrt_output, self.input.T)/batch_size

        #Gradient par rapport au biais
        self.grad_b = np.sum(grad_wrt_output, axis=1, keepdims=True)/batch_size

        grad_wrt_input = np.dot(self.W.T, grad_wrt_output)/batch_size

        return grad_wrt_input