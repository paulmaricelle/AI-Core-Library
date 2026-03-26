from .loss import Loss
import numpy as np

class MSE(Loss):
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        batch_size = y_pred.shape[1]
        return np.sum((y_true-y_pred)**2)/batch_size
    
    def backward(self):
        batch_size = self.y_pred.shape[1]
        return 2.0 * (self.y_pred - self.y_true)/batch_size
    

