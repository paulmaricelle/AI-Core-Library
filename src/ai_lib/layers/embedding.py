import numpy as np
from .layer import Layer

class Embedding(Layer):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.W = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.01
        self.dW = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Takes an array of token ids X (Batch, Seq Length) and converts it to an
        array of embeddings (Batch, Seq Length, d_model)
        """

        self.X = X
        return self.W[X]
    
    def backward(self, dOutput: np.ndarray) -> None:
        if self.dW is None:
            self.dW = np.zeros_like(self.W, dtype=np.float32)
        np.add.at(self.dW, self.X, dOutput)

    def get_params(self):
        return [self.W]

    def get_grads(self):
        return [self.dW]
    
    def zero_grad(self):
        self.dW = None

    def get_reg_info(self):
        return [False]
    
    def get_state(self):
        return {"W": self.W.copy()}

    def set_state(self, state):
        self.W = state["W"].copy()

class TiedOutputProjection(Layer):
    def __init__(self, embedding_layer: Layer):
        self.embedding_layer = embedding_layer

    def forward(self, X:np.ndarray) -> np.ndarray:
        self.input = X
        return X @ self.embedding_layer.W.T
    
    def backward(self, grad_wrt_output: np.ndarray) -> np.ndarray:
        dX = grad_wrt_output @ self.embedding_layer.W
        
        X_flat = self.input.reshape(-1, self.input.shape[-1])
        grad_flat = grad_wrt_output.reshape(-1, grad_wrt_output.shape[-1])
        
        dW_out = (X_flat.T @ grad_flat).T
        
        if getattr(self.embedding_layer, 'dW', None) is None:
            self.embedding_layer.dW = dW_out
        else:
            self.embedding_layer.dW += dW_out
            
        return dX
    
    # This layer does not handle parameters for the optimizer
    def get_params(self): return []
    def get_grads(self): return []
    def get_reg_info(self): return []
    
    def get_state(self): return {}
    def set_state(self, state): pass