from .layer import Layer
import numpy as np
from typing import Optional

class LayerNormalization(Layer):
    def __init__(self, n_features: int, epsilon : float = 1e-5):
        super().__init__()

        self.n_features = n_features
        self.epsilon = epsilon
        
        self.gamma = np.ones(self.n_features).astype(np.float32)
        self.beta = np.zeros(self.n_features).astype(np.float32)

        self.grad_gamma: Optional[np.ndarray] = None
        self.grad_beta: Optional[np.ndarray] = None
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input = X
        self.mean = np.mean(X, axis=-1, keepdims=True)
        self.var = np.var(X, axis=-1, mean=self.mean, keepdims=True)

        self.X_centered = X - self.mean
        self.std_inv = 1 / (np.sqrt(self.var + self.epsilon))
        self.X_hat = self.X_centered * self.std_inv

        return self.gamma * self.X_hat + self.beta
    
    def backward(self, grad_wrt_output: np.ndarray) -> np.ndarray:
        # We want to sum along all axes except for the last one
        sum_axes = tuple(range(grad_wrt_output.ndim - 1))

        grad_gamma_current = np.sum(grad_wrt_output * self.X_hat, axis=sum_axes)
        grad_beta_current = np.sum(grad_wrt_output, axis=sum_axes)

        if self.grad_gamma is None:
            self.grad_gamma = grad_gamma_current
            self.grad_beta = grad_beta_current
        else:
            self.grad_gamma += grad_gamma_current
            self.grad_beta += grad_beta_current

        D = self.n_features

        dx_hat = grad_wrt_output * self.gamma
        da = (1.0 / D) * (
            D * dx_hat - 
            np.sum(dx_hat, axis=-1, keepdims=True) - 
            self.X_hat * np.sum(dx_hat * self.X_hat, axis=-1, keepdims=True)
        )
        return self.std_inv * da
    
    def get_params(self):
        return [self.gamma, self.beta]
    
    def get_reg_info(self):
        return [False, False]
    
    def get_grads(self):
        return [self.grad_gamma, self.grad_beta]
    
    def zero_grad(self):
        self.grad_gamma = None
        self.grad_beta = None

    def get_state(self):
        return {'gamma' : self.gamma, 'beta' : self.beta}
    
    def set_state(self, state):
        self.gamma = state["gamma"].copy()
        self.beta = state["beta"].copy()


class RMSNorm(Layer):
    def __init__(self, n_features: int, epsilon: float = 1e-5):
        super().__init__()

        self.n_features = n_features
        self.epsilon = epsilon

        self.gamma = np.ones(self.n_features)
        self.grad_gamma: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input = X

        self.mean_square = np.mean(X**2, axis=-1, keepdims=True)
        
        self.rms_inv = 1.0 / np.sqrt(self.mean_square + self.epsilon)
        self.X_hat = X * self.rms_inv

        return self.gamma * self.X_hat

    def backward(self, grad_wrt_output: np.ndarray) -> np.ndarray:
        # We want to sum along all axes except for the last one
        sum_axes = tuple(range(grad_wrt_output.ndim - 1))

        grad_gamma_current = np.sum(grad_wrt_output * self.X_hat, axis=sum_axes)

        if self.grad_gamma is None:
            self.grad_gamma = grad_gamma_current
        else:
            self.grad_gamma += grad_gamma_current

        # Gradient with respect to x hat
        dx_hat = grad_wrt_output * self.gamma

        dx = self.rms_inv * (
            dx_hat - 
            self.X_hat * np.mean(dx_hat * self.X_hat, axis=-1, keepdims=True)
        )

        return dx
    
    def get_params(self):
        return [self.gamma]
    
    def get_reg_info(self):
        return [False]
    
    def get_grads(self):
        return [self.grad_gamma]
    
    def zero_grad(self):
        self.grad_gamma = None

    def get_state(self):
        return {'gamma': self.gamma}
    
    def set_state(self, state):
        self.gamma = state["gamma"].copy()