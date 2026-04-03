import numpy as np

def accuracy(y_pred, y_true):
    preds = np.argmax(y_pred, axis=0)
    truth = np.argmax(y_true, axis=0)
    return round(np.mean(preds==truth), 5)

def binary_metrics(y_pred, y_true, threshold = 0.5):
    tp, fp, tn, fn = confusion_matrix(y_pred, y_true, threshold=threshold)
    epsilon = 10e-15
    precision = tp / (tp + fp + epsilon)
    recall =  tp / (tp + fn + epsilon)
    return {'precision' : round(precision, 5), 'recall' : round(recall, 5),
            'accuracy' : round((tp + tn) / (tp + tn + fp + fn + epsilon), 5),
            'f1' : round(2 * precision * recall / (precision + recall + epsilon), 5)}

def mae(y_pred, y_true):
    return round(np.mean(np.abs(y_pred-y_true)), 5)

def mse(y_pred, y_true):
    return round(np.mean((y_pred-y_true)**2), 5)

def confusion_matrix(y_pred, y_true, threshold = 0.5):
    mask = y_pred > threshold
    truth = y_true.astype(int)
    tp = np.sum((mask == True ) & (truth == 1))
    fp = np.sum((mask == True ) & (truth == 0))
    tn = np.sum((mask == False ) & (truth == 0))
    fn = np.sum((mask == False ) & (truth == 1))
    return tp, fp, tn, fn