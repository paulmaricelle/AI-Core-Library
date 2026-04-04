from .loss import Loss
import numpy as np

#Class which acts as a composition of the CrossEntropy loss and of the Softmax activation
#It eases the backpropagation
class SoftmaxCrossEntropy(Loss):
    def forward(self, X, y_true):
        self.y_true = y_true

        exp = np.exp(X - np.max(X, axis=1, keepdims=True))
        softmax = exp/np.sum(exp, axis=1, keepdims=True )
        self.y_pred = softmax

        batch_size = X.shape[0]
        log_prob = -np.log(softmax + 1e-15)

        loss = np.sum(log_prob * y_true) / batch_size
        return loss
    
    def backward(self):
        batchsize = self.y_true.shape[0]
        return (self.y_pred - self.y_true) / batchsize

