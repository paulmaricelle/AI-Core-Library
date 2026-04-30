from .layer import Layer
import numpy as np

class LayerNormalization(Layer):
    def __init__(self, n_features: int, epsilon : float = 1e-10):
        super().__init__()

        self.n_features = n_features
        self.gamma = np.ones((1, self.n_features))
        self.beta = np.zeros((1, self.n_features))
        self.grad_gamma = np.zeros((1, self.n_features))
        self.grad_beta = np.zeros((1, self.n_features))
        self.epsilon = epsilon
        

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input = X
        self.mean = np.mean(X, axis=1, keepdims=True)
        self.var = np.var(X, axis=1, mean=self.mean, keepdims=True)
        self.X_centered = X - self.mean
        self.std_inv = 1 / (np.sqrt(self.var + self.epsilon))
        self.X_hat = self.X_centered * self.std_inv
        return self.gamma * self.X_hat + self.beta
    
    def backward(self, grad_wrt_output: np.ndarray) -> np.ndarray:
        self.grad_gamma += np.sum(grad_wrt_output * self.X_hat, axis=0, keepdims=True)
        self.grad_beta += np.sum(grad_wrt_output, axis=0, keepdims=True)

        n_features = self.n_features

        dx_hat = grad_wrt_output * self.gamma
        da = (1.0 / n_features) * (
            n_features * dx_hat - 
            np.sum(dx_hat, axis=1, keepdims=True) - 
            self.X_hat * np.sum(dx_hat * self.X_hat, axis=1, keepdims=True)
        )
        return self.std_inv * da
    
    def get_params(self):
        return [self.gamma, self.beta]
    
    def get_reg_info(self):
        return [False, False]
    
    def get_grads(self):
        return [self.grad_gamma, self.grad_beta]
    
    def zero_grad(self):
        self.grad_gamma = np.zeros((1, self.n_features))
        self.grad_beta = np.zeros((1, self.n_features))

    def get_state(self):
        return {'gamma' : self.gamma, 'beta' : self.beta}
    
    def set_state(self, state):
        self.gamma = state["gamma"]
        self.beta = state["beta"]