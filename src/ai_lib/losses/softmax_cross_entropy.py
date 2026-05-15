from .loss import Loss
import numpy as np

#Class which acts as a composition of the CrossEntropy loss and of the Softmax activation
#It eases the backpropagation
class SoftmaxCrossEntropy(Loss):
    def forward(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Supports classic format (Batch, Classes) as well as NLP format (Batch, Time, Vocab)
        """
        self.original_X_shape = X.shape
        
        X_flat = X.reshape(-1, X.shape[-1])
        
        
        if y_true.shape == X.shape:
            # If class is one hot encoded
            self.y_flat = np.argmax(y_true.reshape(-1, X.shape[-1]), axis=1)
        else:
            # If class is directly predicted with its index
            self.y_flat = y_true.reshape(-1)
            
        self.N = X_flat.shape[0]
        
        exp = np.exp(X_flat - np.max(X_flat, axis=1, keepdims=True))
        self.softmax = exp / np.sum(exp, axis=1, keepdims=True)
        
        correct_class_probs = self.softmax[np.arange(self.N), self.y_flat]
        
        log_prob = -np.log(correct_class_probs + 1e-15)
        loss = np.sum(log_prob) / self.N
        
        return loss
    
    def backward(self) -> np.ndarray:
        dX_flat = self.softmax.copy()
        
        # Gradient of the SoftmaxCrossEntropy is predictioin - truth
        dX_flat[np.arange(self.N), self.y_flat] -= 1

        dX_flat = dX_flat / self.N
        return dX_flat.reshape(self.original_X_shape)

