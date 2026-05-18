from .layer import Layer
import numpy as np
from typing import Optional

class SwiGLU(Layer):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        
        # Merging W1 and W3 in one matrix for faster forward
        # Shape: (in_features, 2 * hidden_features)
        self.W13 = np.random.randn(in_features, 2 * hidden_features) * np.sqrt(2.0 / in_features)
        self.W2 = np.random.randn(hidden_features, in_features) * np.sqrt(2.0 / hidden_features)
        
        self.grad_W13: Optional[np.ndarray] = None
        self.grad_W2: Optional[np.ndarray] = None
        
        # Backward
        self.input: Optional[np.ndarray] = None
        self.gate_sigmoid: Optional[np.ndarray] = None
        self.gate_silu: Optional[np.ndarray] = None
        self.value: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input = X
        
        proj = X @ self.W13 
        
        gate, self.value = np.split(proj, 2, axis=-1)
    
        self.gate_sigmoid = 1.0 / (1.0 + np.exp(-gate))
        self.gate_silu = gate * self.gate_sigmoid
        
        # Gating
        h = self.gate_silu * self.value
        
        # Final projection
        return h @ self.W2

    def backward(self, grad_wrt_output: np.ndarray) -> np.ndarray:
        self.grad_W2 = (self.gate_silu * self.value).T @ grad_wrt_output
        dh = grad_wrt_output @ self.W2.T

        dvalue = dh * self.gate_silu
        dgate_silu = dh * self.value
        
        dgate = dgate_silu * (self.gate_sigmoid + self.gate_silu * (1.0 - self.gate_sigmoid))
        
        dproj = np.concatenate([dgate, dvalue], axis=-1)
        self.grad_W13 = self.input.T @ dproj
        
        return dproj @ self.W13.T