from .layer import Layer

class Sequential(Layer):
    def __init__(self, layers: list):
        super().__init__()
        self.layers = layers
    
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def set_training(self, mode):
        for layer in self.layers:
            layer.training = mode

    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params

    def get_grads(self):
        grads = []
        for layer in self.layers:
            grads.extend(layer.get_grads())
        return grads

    def get_state(self):
        state = {}
        for i, layer in enumerate(self.layers):
            layer_state = layer.get_state()
            for key, val in layer_state.items():
                state[f"{i}_{key}"] = val
        return state

    def set_state(self, state):
        for i, layer in enumerate(self.layers):
            layer_state = {k.split('_', 1)[1]: v for k, v in state.items() if k.startswith(f"{i}_")}
            if layer_state:
                layer.set_state(layer_state)