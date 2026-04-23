import numpy as np
from typing import Optional

class Loss:
    def __init__(self):
        self.y_pred: Optional[np.ndarray] = None
        self.y_true: Optional[np.ndarray] = None

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.forward(y_pred, y_true)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError("Forward n'est pas implémenté")

    def backward(self) -> np.ndarray:
        raise NotImplementedError("Backward n'est pas implémenté")