class Loss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):
        raise NotImplementedError("Forward n'est pas implémenté")

    def backward(self):
        raise NotImplementedError("Backward n'est pas implémenté")