class Layer:
    def __init__(self):
        self.input = None
        
    def forward(self):
        raise NotImplementedError("Le forward n'est pas implémenté")

    def backward(self):
        raise NotImplementedError("Le backward n'est pas implémenté")