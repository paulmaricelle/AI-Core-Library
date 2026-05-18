from .layer import Layer
from .normalization import LayerNormalization
from .linear import Linear
from .residual_block import ResidualBlock
from .multiSelfAttentionHead import MultiSelfAttentionHead, RoPEMultiAttentionHead
from .activations import ReLU
from .dropout import Dropout
from .sequential import Sequential

import numpy as np


class TransformerBlock(Layer):
    def __init__(self, n_heads: int, d_model: int, d_ff: int, is_causal: bool, context_window: int, dropout_rate: float =0, RoPE = False) -> None:
        """
        Initializes a TransformerBlock with pre-normalization :
        ResidualBlock( _, MultiSelfHeadAttention(LayerNorm())) -> LayerNorm -> ResidualBlock(FFN with d_ff-dimension hidden layer)
        context_window : For RoPE only; otherwise simply ignored
        """
        super().__init__()
        if RoPE:
            mha = RoPEMultiAttentionHead(n_heads=n_heads, d_model=d_model, is_causal=is_causal, block_size=context_window)
        else:
            mha = MultiSelfAttentionHead(n_heads=n_heads, d_model=d_model, is_causal=is_causal, )
        
        res_mha = ResidualBlock([
            LayerNormalization(d_model),
            mha,
            Dropout(dropout_rate=dropout_rate)
        ])
        
        res_ffn = ResidualBlock([
            LayerNormalization(d_model), 
            Linear(d_model, d_ff), 
            ReLU(), 
            Linear(d_ff, d_model),
            Dropout(dropout_rate=dropout_rate)
        ])

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

    def set_use_cache(self, use_cache: bool) -> None:
        self.seq.set_use_cache(use_cache)

    def reset_cache(self) -> None:
        self.seq.reset_cache()


    