import numpy as np
from .layer import Layer

class Embedding(Layer):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.W = np.random.randn(vocab_size, d_model) * 0.01

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Takes an array of token ids X (Batch, Seq Length) and converts it to an
        array of embeddings (Batch, Seq Length, d_model)
        """

        self.X = X
        return self.W[X]
    
    def backward(self, dOutput: np.ndarray) -> None:
        self.dW = np.zeros_like(self.W)
        np.add.at(self.dW, self.X, dOutput)

    def get_params(self):
        return [self.W]

    def get_grads(self):
        return [self.dW]
    
    def zero_grad(self):
        self.dW = None

    def get_reg_info(self):
        return [True]
    
    def get_state(self):
        return {"W": self.W.copy()}

    def set_state(self, state):
        self.W = state["W"].copy()