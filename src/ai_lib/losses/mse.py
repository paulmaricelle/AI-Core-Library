from .loss import Loss
import numpy as np

class MSE(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        self.y_size = y_pred.size
        return float(np.mean((y_true-y_pred)**2))
    
    def backward(self) -> np.ndarray:
        return 2.0 * (self.y_pred - self.y_true) / self.y_size
    

