from .layer import Layer
import numpy as np

class MoEBlock(Layer):
    def __init__(self, expert_builder, d_model:int, n_experts=8, top_k=2, alpha=1e-2):
        """
        Mixture of experts Layer. Remplaces the FFN layer
        expert_builder : function that returns an instance of a Layer
        alpha : Load balancing loss parameter
        """
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.alpha = alpha

        self.W_router = np.random.randn(d_model, n_experts) * np.sqrt(2.0 / d_model)
        self.grad_W_router = None

        self.experts = [expert_builder() for _ in range(n_experts)]

        self.aux_loss = 0.0

    def forward(self, X:np.ndarray) -> np.ndarray:
        B, T, D = X.shape

        N = B * T
        X_flat = X.reshape(B * T, D)

        logits = X_flat @ self.W_router # (N, n_experts)

        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        sorted_indices = np.argsort(logits, axis=-1)
        top_k_indices = sorted_indices[:, -self.top_k:][:, ::-1] # (N, top_k)

        N_range = np.arange(N)[:, None]
        top_k_probs = probs[N_range, top_k_indices]

        top_k_weights = top_k_probs / np.sum(top_k_probs, axis=-1, keepdims=True)

        # Computation of auxiliary loss (Load balancing loss)
        tokens_per_expert = np.bincount(top_k_indices.flatten(), minlength=self.n_experts)
        f_i = tokens_per_expert / (N * self.top_k)

        P_i = np.mean(probs, axis=0) # Mean probability of a token being assigned to i-th expert

        self.aux_loss = self.alpha * self.n_experts * np.sum(f_i * P_i)

        self.cache = {
        'X_flat': X_flat,
        'probs': probs,
        'top_k_indices': top_k_indices,
        'top_k_weights': top_k_weights,
        'top_k_probs': top_k_probs,
        'f_i': f_i,
        'expert_results': {}
        }

        # Computation of output
        out_flat = np.zeros_like(X_flat)
        for i, expert in enumerate(self.experts):
            is_selected = (top_k_indices == i) #(N, top_k)
            token_mask = np.any(is_selected, axis=-1) #(N,)

            batch_idx = np.where(token_mask)[0]

            if len(batch_idx) > 0:
                expert_weights = top_k_weights[is_selected] 
                expert_weights = expert_weights[:, None]

                expert_out = expert.forward(X_flat[batch_idx])
                self.cache['expert_results'][i] = (batch_idx, expert_out, is_selected)
                out_flat[batch_idx] += expert_weights * expert_out

        return out_flat.reshape(B, T, D)
    
    def backward(self, grad_wrt_output):
        B, T, D = grad_wrt_output.shape
        grad_flat = grad_wrt_output.reshape(B * T, D)

        X_flat = self.cache['X_flat']
        probs = self.cache['probs']
        top_k_indices = self.cache['top_k_indices']
        top_k_weights = self.cache['top_k_weights']
        top_k_probs = self.cache['top_k_probs']
        f_i = self.cache['f_i']

        N, E = probs.shape
        K = self.top_k

        dX_flat = np.zeros_like(X_flat)
        d_top_k_weights = np.zeros((N, K))

        for i, expert in enumerate(self.experts):
            if i not in self.cache['expert_results']:
                continue

            batch_idx, expert_out, is_selected = self.cache['expert_results'][i]

            weights_i = top_k_weights[is_selected][:, None]
            grad_expert_in = grad_flat[batch_idx] * weights_i

            dX_expert = expert.backward(grad_expert_in)
            dX_flat[batch_idx] += dX_expert

            # Backprop of the routing process
            d_weight_i = np.sum(grad_flat[batch_idx] * expert_out, axis=-1)

            d_top_k_weights[is_selected] = d_weight_i

        sum_probs = np.sum(top_k_probs, axis=-1, keepdims=True)
        sum_dw_w = np.sum(d_top_k_weights * top_k_weights, axis=-1, keepdims=True)
        d_top_k_probs = (d_top_k_weights - sum_dw_w) / sum_probs

        d_probs = np.zeros((N, E))
        np.put_along_axis(d_probs, top_k_indices, d_top_k_probs, axis=1)

        # Adding auxiliary loss gradient 
        d_probs_aux = (self.alpha * E * f_i) / N
        d_probs += d_probs_aux

        # Backprop through the Softmax
        sum_p_dp = np.sum(probs * d_probs, axis=-1, keepdims=True)
        d_logits = probs * (d_probs - sum_p_dp)

        dW_router_current = X_flat.T @ d_logits
        
        # Gradient accumulation
        if getattr(self, 'grad_W_router', None) is None:
            self.grad_W_router = dW_router_current
        else:
            self.grad_W_router += dW_router_current
        
        dX_router = d_logits @ self.W_router.T
        dX_flat += dX_router
        
        return dX_flat.reshape(B, T, D)
    

    def get_params(self):
        params = [self.W_router]
        for expert in self.experts:
            params.extend(expert.get_params())
        return params

    def get_grads(self):
        grads = [self.grad_W_router]
        for expert in self.experts:
            grads.extend(expert.get_grads())
        return grads

    def get_reg_info(self):
        # No weight decay on the router
        reg = [False] 
        for expert in self.experts:
            reg.extend(expert.get_reg_info())
        return reg

    def zero_grad(self):
        self.grad_W_router = None
        for expert in self.experts:
            expert.zero_grad()

    def get_state(self):
        state = {'W_router': self.W_router}
        for i, expert in enumerate(self.experts):
            state[f'expert_{i}'] = expert.get_state()
        return state

    def set_state(self, state):
        self.W_router = state['W_router']
        for i, expert in enumerate(self.experts):
            expert.set_state(state[f'expert_{i}'])