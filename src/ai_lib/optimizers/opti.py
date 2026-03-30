class Optimizer:
    def __init__(self):
        self.params = []
        self.grads = []

    def zero_grad(self):   
        for layer in self.layers:
            layer.zero_grad()

    def get_grads(self):
        grads = []
        for layer in self.layers:
            grads += layer.get_grads()
        self.grads = grads

    def setup(self, layers):
        self.layers = layers
        self.params = []
        for layer in self.layers:
            self.params += layer.get_params()
        self._init_state()

    def step(self):
        raise NotImplementedError("Méthode step non implémentée")

    def _init_state(self):
        raise NotImplementedError("Méthode _init_state non implémentée")

        