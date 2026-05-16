from .layer import Layer
import numpy as np

class SineCosEncoder(Layer):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.use_cache = False
        self.current_pos = 0
        
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]

        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div)
        pe[:, 1::2] = np.cos(position * div)

        self.pe = pe[np.newaxis, :, :]

    def forward(self, X: np.ndarray) -> np.ndarray:
        T = X.shape[1]

        if self.use_cache:
            pos_start = self.current_pos
            pos_end = self.current_pos + T
            if pos_end > self.max_len:
                raise ValueError(f"Generated text exceeds max length for positional encoder :({self.max_len}).")


            out = X + self.pe[:, pos_start:pos_end, :]
            self.current_pos += T

            return out 
        else:
            return X + self.pe[:, :T, :]

    def backward(self, dX: np.ndarray) -> np.ndarray:
        return dX
    
    def get_params(self): return []
    def get_grads(self): return []
    def get_reg_info(self): return []
    def zero_grad(self): pass
    def get_state(self): return {}
    def set_state(self, state): pass

    def set_use_cache(self, use_cache: bool) -> None:
        self.use_cache = use_cache
        
    def reset_cache(self) -> None:
        self.current_pos = 0