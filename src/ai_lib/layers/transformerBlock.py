from .layer import Layer
from .residual_block import ResidualBlock
from .dropout import Dropout
from .sequential import Sequential

import numpy as np

class TransformerBlock(Layer):
    def __init__(self, attention_builder, norm_builder, ffn_builder, dropout_rate: float = 0.0) -> None:
        """
        Initializes a TransformerBlock with pre-normalization using builders :
        ResidualBlock(norm_builder() -> attention_builder() -> Dropout) 
        -> ResidualBlock(norm_builder() -> ffn_builder() -> Dropout)
        """
        super().__init__()
        
        res_mha = ResidualBlock([
            norm_builder(),
            attention_builder(),
            Dropout(dropout_rate=dropout_rate)
        ])
        
        res_ffn = ResidualBlock([
            norm_builder(), 
            ffn_builder(),
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