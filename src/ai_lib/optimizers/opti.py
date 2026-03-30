class Optimizer:
    def __init__(self):
        self.params = []
        self.grads = []

    def zero_grad(self):   
        for layer in self.layer:
            layer.zero_grad()

    def get_grads(self):
        grads = []
        for layer in self.layers:
            grads += layer.get_grads()
        self.grads = grads

    def step(self):
        NotImplementedError("Méthode step non implémentée")

    def setup(self, layers):
        NotImplementedError("Méthode setup non implémentée")

        