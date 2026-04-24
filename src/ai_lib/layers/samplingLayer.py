import numpy as np
from .layer import Layer
from typing import Optional

class SamplingLayer(Layer):
    def __init__(self, kl_weight: float = 1.0) -> None:
        super().__init__()
        self.kl_weight = kl_weight

        self.mu: Optional[np.ndarray] = None
        self.log_var: Optional[np.ndarray] = None
        self.epsilon: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """ X must have shape (Batch, 2*latent_space_dimension)"""
        half = X.shape[1] // 2
        self.mu = X[:, :half]
        self.log_var = X[:, half:]

        self.sigma = np.exp(0.5 * self.log_var)
        if self.training:
            self.epsilon = np.random.randn(*self.mu.shape)
        else:
            self.epsilon = np.zeros_like(self.mu)

        z = self.mu + self.sigma * self.epsilon
        self.kl_loss = -0.5 * np.sum(1 + self.log_var - self.mu**2 - np.exp(self.log_var)) / X.shape[0]

        return z
    
    def backward(self, grad_wrt_output: np.ndarray) -> np.ndarray:
        """ The KL divergence gradient is computed and added with this backward """
        grad_mu_loss = grad_wrt_output
        grad_log_var_loss = grad_wrt_output * self.epsilon * 0.5 * self.sigma

        grad_mu_kl = self.mu
        grad_log_var_kl = 0.5 * (np.exp(self.log_var) - 1.0)

        grad_mu = grad_mu_loss + self.kl_weight * grad_mu_kl
        grad_log_var = grad_log_var_loss + self.kl_weight * grad_log_var_kl

        grad_X = np.concatenate([grad_mu, grad_log_var], axis=1)

        return grad_X
    
    def get_params(self):
        return []
    
    def get_reg_info(self):
        return []
    
    def get_grads(self):
        return []