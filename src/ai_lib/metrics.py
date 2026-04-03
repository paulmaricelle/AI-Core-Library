import numpy as np

def accuracy(y_pred, y_true):
    preds = np.argmax(y_pred, axis=0)
    truth = np.argmax(y_true, axis=0)
    return np.mean(preds==truth)

def binary_metrics(y_pred, y_true, threshold = 0.5):
    tp, fp, tn, fn = confusion_matrix(y_pred, y_true, threshold=threshold)
    epsilon = 10e-15
    precision = tp / (tp + fp + epsilon)
    recall =  tp / (tp + fn + epsilon)
    return {'precision' : precision, 'recall' : recall,
            'accuracy' : (tp + tn) / (tp + tn + fp + fn + epsilon),
            'f1' : 2 * precision * recall / (precision + recall + epsilon)}

def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred-y_true))

def mse(y_pred, y_true):
    return np.mean((y_pred-y_true)**2)

def confusion_matrix(y_pred, y_true, threshold = 0.5):
    mask = y_pred > threshold
    truth = y_true.astype(int)
    tp = np.sum((mask == True ) & (truth == 1))
    fp = np.sum((mask == True ) & (truth == 0))
    tn = np.sum((mask == False ) & (truth == 0))
    fn = np.sum((mask == False ) & (truth == 1))
    return tp, fp, tn, fn