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
        self.h: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input = X
        
        proj = X @ self.W13 
        
        gate, self.value = np.split(proj, 2, axis=-1)
    
        self.gate_sigmoid = 1.0 / (1.0 + np.exp(-gate))
        self.gate_silu = gate * self.gate_sigmoid
        
        # Gating
        self.h = self.gate_silu * self.value
        
        # Final projection
        return self.h @ self.W2

    def backward(self, grad_wrt_output: np.ndarray) -> np.ndarray:
        h_flat = self.h.reshape(-1, self.h.shape[-1])
        grad_out_flat = grad_wrt_output.reshape(-1, grad_wrt_output.shape[-1])
        
        grad_W2_current = h_flat.T @ grad_out_flat
        dh = grad_wrt_output @ self.W2.T

        dvalue = dh * self.gate_silu
        dgate_silu = dh * self.value
        
        dgate = dgate_silu * (self.gate_sigmoid + self.gate_silu * (1.0 - self.gate_sigmoid))
        
        dproj = np.concatenate([dgate, dvalue], axis=-1)

        dproj_flat = dproj.reshape(-1, dproj.shape[-1])
        input_flat = self.input.reshape(-1, self.input.shape[-1])
        
        grad_W13_current = input_flat.T @ dproj_flat

        if self.grad_W2 is None:
            self.grad_W2 = grad_W2_current
            self.grad_W13 = grad_W13_current
        else:
            self.grad_W2 += grad_W2_current
            self.grad_W13 += grad_W13_current
        
        return dproj @ self.W13.T
    
    def get_params(self):
        return [self.W13, self.W2]
    
    def get_grads(self):
        return [self.grad_W13, self.grad_W2]
    
    def get_reg_info(self):
        return [True, True]
    
    def zero_grad(self):
        self.grad_W13 = None
        self.grad_W2 = None

    def get_state(self):
        return {'W13' : self.W13, 'W2' : self.W2}
    
    def set_state(self, state):
        self.W13 = state["W13"].copy()
        self.W2 = state["W2"].copy()