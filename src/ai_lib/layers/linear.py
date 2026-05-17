from .layer import Layer
import numpy as np
from typing import Optional

class Linear(Layer):
    def __init__(self, n_input: int, n_out: int, init_method: str = "xavier"):
        super().__init__()
        if init_method == "xavier":
            limit = np.sqrt(6.0 / (n_out + n_input))
            self.W = np.random.uniform(-limit, limit, (n_out, n_input)).astype(np.float32)
        elif init_method == "he":
            std = np.sqrt(2 / n_input)
            self.W = np.random.randn(n_out, n_input).astype(np.float32) * std
        else:
            self.W = np.random.randn(n_out, n_input).astype(np.float32) * 10e-2
        self.b = np.zeros((1, n_out))

        self.grad_W: Optional[np.ndarray] = None
        self.grad_b: Optional[np.ndarray] = None
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.original_input_shape = X.shape
        
        # Flatten and do the maths in 2d in case data is of larger dimension
        self.input_flat = X.reshape(-1, X.shape[-1])
        
        output_flat = np.dot(self.input_flat, self.W.T) + self.b
        
        # Back to initial shape
        new_shape = list(self.original_input_shape[:-1]) + [self.W.shape[0]]
        self.output = output_flat.reshape(*new_shape)
        
        return self.output
    
    def backward(self, grad_wrt_output: np.ndarray) -> np.ndarray:
        grad_out_flat = grad_wrt_output.reshape(-1, grad_wrt_output.shape[-1])
        
        grad_W_current = np.dot(grad_out_flat.T, self.input_flat)
        
        grad_b_current = np.sum(grad_out_flat, axis=0, keepdims=True)

        grad_in_flat = np.dot(grad_out_flat, self.W)

        # Gradient accumulation
        if self.grad_W is None:
            self.grad_W = grad_W_current
            self.grad_b = grad_b_current
        else:
            self.grad_W += grad_W_current
            self.grad_b += grad_b_current

        return grad_in_flat.reshape(self.original_input_shape)
    
    def get_params(self):
        return [self.W, self.b]
    
    def get_grads(self):
        return [self.grad_W, self.grad_b]
    
    def get_reg_info(self):
        return [True, False]
    
    def zero_grad(self):
        self.grad_W = None
        self.grad_b = None

    def get_state(self):
        return {'W' : self.W, 'b' : self.b}
    
    def set_state(self, state):
        self.W = state["W"].copy()
        self.b = state["b"].copy()