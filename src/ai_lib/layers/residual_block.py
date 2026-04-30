import numpy as np
from .layer import Layer
from typing import List, Optional

class ResidualBlock(Layer):
    def __init__(self, inner_layers: List[Layer], shortcut: Optional[Layer] =None):
        super().__init__()
        self.layers = inner_layers
        self.shortcut = shortcut

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        if self.shortcut == None:
            return out + X
        else:
            return self.shortcut.forward(X) + out
    
    def backward(self, grad_wrt_output: np.ndarray) -> np.ndarray:
        whole_grad = grad_wrt_output
        for layer in reversed(self.layers):
            whole_grad = layer.backward(whole_grad)

        if self.shortcut == None:
            return whole_grad + grad_wrt_output
        else:
            return whole_grad + self.shortcut.backward(grad_wrt_output)
        

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
        if self.shortcut != None:
            self.shortcut.zero_grad()
    
    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        if self.shortcut != None:
            params += self.shortcut.get_params()
        return params
    
    def get_reg_info(self):
        reg_info = []
        for layer in self.layers:
            reg_info += layer.get_reg_info()
        if self.shortcut != None:
            reg_info += self.shortcut.get_reg_info()
        return reg_info
    
    def get_grads(self):
        grads = []
        for layer in self.layers:
            grads += layer.get_grads()
        if self.shortcut != None:
            grads += self.shortcut.get_grads()
        return grads
    
    def get_state(self) -> dict:
        state = {}
        
        for i, layer in enumerate(self.layers):
            layer_state = layer.get_state()
            for key, value in layer_state.items():
                state[f"inner_{i}_{key}"] = value
                
        if self.shortcut is not None:
            shortcut_state = self.shortcut.get_state()
            for key, value in shortcut_state.items():
                state[f"shortcut_{key}"] = value              
        return state
    
    def set_state(self, state: dict) -> None:
        for i, layer in enumerate(self.layers):
            layer_state = {}
            prefix = f"inner_{i}_"
            
            for key, value in state.items():
                if key.startswith(prefix):
                    real_key = key[len(prefix):] 
                    layer_state[real_key] = value
                    
            if layer_state:
                layer.set_state(layer_state)
                
        if self.shortcut is not None:
            shortcut_state = {}
            prefix = "shortcut_"
            
            for key, value in state.items():
                if key.startswith(prefix):
                    real_key = key[len(prefix):]
                    shortcut_state[real_key] = value
                    
            if shortcut_state:
                self.shortcut.set_state(shortcut_state)