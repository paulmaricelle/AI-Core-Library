class Layer:
    def __init__(self):
        self.training: bool = True

    def zero_grad(self):
        pass
    
    def forward(self):
        raise NotImplementedError("Le forward n'est pas implémenté")

    def backward(self):
        raise NotImplementedError("Le backward n'est pas implémenté")
    
    def get_params(self):
        raise NotImplementedError("L'obtention des paramètres n'est pas implémentée")
    
    def get_reg_info(self):
        raise NotImplementedError("L'obtention des paramètres à régulariser n'est pas implémentée")
    
    def get_grads(self):
        raise NotImplementedError("L'obtention des gradients n'est pas implémentée")
    
    def get_state(self) -> dict:
        """ Returns a dictionnary with the layer's parameters"""
        raise NotImplementedError("L'obtention du dictionnaire avec les paramètres n'est pas implémenté ")
    
    def set_state(self) -> None:
        """ Sets parameters using a dictionnary"""
        raise NotImplementedError("The set_state method is not implemented")