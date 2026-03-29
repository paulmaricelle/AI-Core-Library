class Optimizer:
    def __init__(self, layers):
        self.layers = layers

    def step(self):
        NotImplementedError("Méthode step non implémentée")

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'grad_W'):
                layer.grad_W = None 
                layer.grad_b = None