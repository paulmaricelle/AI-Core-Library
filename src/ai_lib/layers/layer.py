class Layer:
    def __init__(self):
        pass

    def zero_grad(self):
        pass
    
    def forward(self):
        raise NotImplementedError("Le forward n'est pas implémenté")

    def backward(self):
        raise NotImplementedError("Le backward n'est pas implémenté")
    
    def get_params(self):
        raise NotImplementedError("L'obtention des paramètres n'est pas implémentée")
    
    def get_grads(self):
        raise NotImplementedError("L'obtention des gradients n'est pas implémentée")