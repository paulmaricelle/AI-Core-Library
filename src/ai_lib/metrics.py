import numpy as np

def accuracy(y_pred, y_true):
    preds = np.argmax(y_pred, axis=0)
    truth = np.argmax(y_true, axis=0)
    return np.mean(preds==truth)

def mae(y_pred, y_true):
    batch_size = y_true.shape[1]
    return np.sum(np.abs(y_pred-y_true)) / batch_size