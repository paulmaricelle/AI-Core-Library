from .loss import Loss
import numpy as np

class BinaryCrossEntropy(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_size = y_pred.size
        self.epsilon = 1e-15

        loss = -np.mean(
            y_true * np.log(y_pred + self.epsilon) + 
            (1 - y_true) * np.log(1 - y_pred + self.epsilon)
        )
        return float(loss)

    def backward(self) -> np.ndarray:
            grad = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred) + self.epsilon)
            return grad / self.y_size

