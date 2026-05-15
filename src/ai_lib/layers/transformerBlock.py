from .layer import Layer
from .layer_normalization import LayerNormalization
from .linear import Linear
from .residual_block import ResidualBlock
from .multiSelfAttentionHead import MultiSelfAttentionHead
from .activations import ReLU
from .sequential import Sequential

import numpy as np


class TransformerBlock(Layer):
    def __init__(self, n_heads: int, d_model: int, d_ff: int, is_causal: bool) -> None:
        """
        Initializes a TransformerBlock with pre-normalization :
        ResidualBlock( _, MultiSelfHeadAttention(LayerNorm())) -> LayerNorm -> ResidualBlock(FFN with d_ff-dimension hidden layer)
        """
        super().__init__()

        ln1 = LayerNormalization(d_model)
        mha = MultiSelfAttentionHead(n_heads=n_heads, d_model=d_model, is_causal=is_causal)

        ln2 = LayerNormalization(d_model)
        
        res_mha = ResidualBlock([ln1, mha])
        res_ffn = ResidualBlock([ln2, Linear(d_model, d_ff), ReLU(), Linear(d_ff, d_model)])

        self.seq = Sequential([res_mha, res_ffn])

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.seq.forward(X)
    
    def backward(self, dOutput: np.ndarray) -> np.ndarray:
        return self.seq.backward(dOutput)
    
    def get_params(self) -> list:
        return self.seq.get_params()

    def get_grads(self) -> list:
        return self.seq.get_grads()
    
    def zero_grad(self):
        return self.seq.zero_grad()
    
    def get_reg_info(self):
        return self.seq.get_reg_info()
    
    def set_training(self, training_mode):
        return self.seq.set_training(training_mode)

    def get_state(self) -> dict:
        return self.seq.get_state()

    def set_state(self, state: dict) -> None:
        self.seq.set_state(state)

    