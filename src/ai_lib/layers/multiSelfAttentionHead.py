from .layer import Layer
import numpy as np
from typing import Optional

class MultiSelfAttentionHead(Layer):
    def __init__(self, n_heads: int, d_model: int, is_causal=False) -> None:
        """
        Initializes query, key and value matrices for self-attention mechanism 
        with n_heads heads each using d_model / n_heads -dimension queries, keys and values
        """
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads

        self.use_cache = False
        self.kv_cache: Optional[np.ndarray] = None
        self.is_causal = is_causal

        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)

        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)

        self.dW_q = None
        self.dW_k = None
        self.dW_v = None
        self.dW_o = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the scaled dot-product attention for an array X (Batch, Sequence length, model dimension (d_model))
        No implementation of KV cache for now, as it would require me to modify Model, Sequential.
        """
        self.X = X
        B, T, D = X.shape

        Q = np.einsum('bid,dj->bij', X, self.W_q)
        K = np.einsum('bid,dj->bij', X, self.W_k)
        V = np.einsum('bid,dj->bij', X, self.W_v)

        # Reshaping for multi-head format (d_model -> (n_heads, d_k))

        Q = Q.reshape(B, T, self.n_heads, self.d_k)
        K = K.reshape(B, T, self.n_heads, self.d_k)
        V = V.reshape(B, T, self.n_heads, self.d_k)

        # If kv_cache is used :
        if self.use_cache:
            if self.kv_cache is not None:
                past_K, past_V = self.kv_cache
                # Concatenate past and present along "time" axis
                K = np.concatenate([past_K, K], axis=1)
                V = np.concatenate([past_V, V], axis=1)
            
            self.kv_cache = (K, V)

        # Shape (B, n_head, T, T)
        S = np.einsum('btnd,bsnd->bnts', Q, K) / np.sqrt(self.d_k)

        # Causal mask: When is_causal, tokens only attend to previous ones.
        if self.is_causal:
            T_q = T              # Temporal length of queries
            T_k = S.shape[3]     # Temporal length of keys (entire cache)

            offset = T_k - T_q  # Offset for the diagonal of the mask since the newly computed attention will be (Tq, Tk)

            mask = np.triu(np.ones((T_q, T_k)), k=offset + 1).astype(bool)
            S[:, :, mask] = -np.inf
            

        S_max = np.max(S, axis=-1, keepdims=True)
        exp_S = np.exp(S - S_max) 
        A = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
        self.A = A

        Z = np.einsum('bnts,bsnd->btnd', A, V)
        Z = Z.reshape(B, T, D) # Concatenation of heads

        Out = np.einsum('bid,dj->bij', Z, self.W_o)

        self.Q, self.K, self.V, self.Z = Q,K, V, Z
        return Out
    
    def backward(self, d_output: np.ndarray) -> np.ndarray:
        """
        Computes the gradients of the loss with respect to the parameters and to X
        """

        if self.use_cache:
            raise RuntimeError("KV cache currently is active. Make sure it is not during training")


        B, T, D = self.X.shape
        dW_o = np.einsum('bid,bij->dj', self.Z, d_output)

        dZ_flat = np.einsum('bij,dj->bid', d_output, self.W_o)
        dZ = dZ_flat.reshape(B, T, self.n_heads, self.d_k)

        # gradients with respect to V and A
        dV = np.einsum('bhts,bthd->bshd', self.A, dZ)
        dA = np.einsum('bthd,bshd->bhts', dZ, self.V)

        # Softmax
        sum_dA_A = np.sum(dA * self.A, axis=-1, keepdims=True)
        dS = self.A * (dA - sum_dA_A) / np.sqrt(self.d_k)

        # Gradients with respect to Q and K
        dQ = np.einsum('bhts,bshd->bthd', dS, self.K)
        dK = np.einsum('bhts,bthd->bshd', dS, self.Q)

        # Back to (B, T, model_d) shape
        dQ_flat = dQ.reshape(B, T, D)
        dK_flat = dK.reshape(B, T, D)
        dV_flat = dV.reshape(B, T, D)
        
        # Forward : Q = X * W_q etc
        dW_q = np.einsum('bid,bij->dj', self.X, dQ_flat)
        dW_k = np.einsum('bid,bij->dj', self.X, dK_flat)
        dW_v = np.einsum('bid,bij->dj', self.X, dV_flat)

        if self.dW_q is None:
            self.dW_q = dW_q
            self.dW_k = dW_k
            self.dW_v = dW_v
            self.dW_o = dW_o
        else:
            self.dW_q += dW_q
            self.dW_k += dW_k
            self.dW_v += dW_v
            self.dW_o += dW_o

        # Get gradients with respect to different X and sum them
        dX_q = np.einsum('bij,dj->bid', dQ_flat, self.W_q)
        dX_k = np.einsum('bij,dj->bid', dK_flat, self.W_k)
        dX_v = np.einsum('bij,dj->bid', dV_flat, self.W_v)
        
        dX = dX_q + dX_k + dX_v
        return dX

    def get_params(self):
        return [self.W_q, self.W_k, self.W_v, self.W_o]
    
    def get_grads(self):
        return [self.dW_q, self.dW_k, self.dW_v, self.dW_o]
    
    def get_reg_info(self):
        return [True, True, True, True]
    
    def zero_grad(self):
        self.dW_q = None
        self.dW_k = None
        self.dW_v = None
        self.dW_o = None

    def get_state(self):
        return {'W_q' : self.W_q, 'W_k' : self.W_k, 'W_v' : self.W_v, 'W_o' : self.W_o}
    
    def set_state(self, state):
        self.W_q = state["W_q"].copy()
        self.W_k = state["W_k"].copy()
        self.W_v = state["W_v"].copy()
        self.W_o = state["W_o"].copy()

    def set_use_cache(self, use_cache: bool) -> None:
        self.use_cache = use_cache
        
    def reset_cache(self) -> None:
        self.kv_cache = None



class RoPEMultiAttentionHead(Layer):
    def __init__(self, n_heads: int, d_model: int, block_size: int, is_causal=False) -> None:
        """
        Initializes query, key and value matrices for self-attention mechanism 
        with n_heads heads each using d_model / n_heads -dimension queries, keys and values
        With ATTENTION SINK
        block_size : Context window
        """
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.block_size = block_size

        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads

        self.use_cache = False
        self.kv_cache: Optional[np.ndarray] = None
        self.is_causal = is_causal

        self.current_pos = 0
        max_len = 2048 # Arbitrary lenght for encoding. If the generation finds itself to be longer,
        # then cos and sin would have to be lengthened on the fly

        theta = 10000.0 ** (-np.arange(0, self.d_k, 2) / self.d_k)
        m = np.arange(max_len)
        freqs = np.outer(m, theta)
        # Duplicate frequences to compute rotation
        freqs_cis = np.concatenate([freqs, freqs], axis=-1)
        self.cos = np.cos(freqs_cis)
        self.sin = np.sin(freqs_cis)

        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)

        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)

        self.dW_q = None
        self.dW_k = None
        self.dW_v = None
        self.dW_o = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the scaled dot-product attention for an array X (Batch, Sequence length, model dimension (d_model))
        KV Cache is implemented
        """
        self.X = X
        B, T, D = X.shape

        Q = np.einsum('bid,dj->bij', X, self.W_q)
        K = np.einsum('bid,dj->bij', X, self.W_k)
        V = np.einsum('bid,dj->bij', X, self.W_v)

        # Reshaping for multi-head format (d_model -> (n_heads, d_k))

        Q = Q.reshape(B, T, self.n_heads, self.d_k)
        K = K.reshape(B, T, self.n_heads, self.d_k)
        V = V.reshape(B, T, self.n_heads, self.d_k)

        # RoPE
        if self.use_cache:
            pos_start = self.current_pos
            pos_end = self.current_pos + T
            self.current_pos += T
        else:
            pos_start = 0
            pos_end = T

        # Use the slices needed for embedding
        cos_sliced = self.cos[pos_start:pos_end, :]
        sin_sliced = self.sin[pos_start:pos_end, :]

        Q = apply_rotary_embedding(Q, cos_sliced, sin_sliced)
        K = apply_rotary_embedding(K, cos_sliced, sin_sliced)
        
        self.cos_sliced = cos_sliced # Saved for backward
        self.sin_sliced = sin_sliced

        # If kv_cache is used
        if self.use_cache and self.kv_cache is not None:
            past_K, past_V = self.kv_cache
            K = np.concatenate([past_K, K], axis=1)
            V = np.concatenate([past_V, V], axis=1)
            
        # Sliding context window + ATTENTION SINK
        if self.use_cache and K.shape[1] > self.block_size:
            n_sinks = 4  # Industry standard
            n_recent = self.block_size - n_sinks
            
            # Only keeps the block_size last tokens (including the sinks now so not exactly),
            # removes the others from RAM with .copy()
            K = np.concatenate([K[:, :n_sinks, :, :], K[:, -n_recent:, :, :]], axis=1).copy()
            V = np.concatenate([V[:, :n_sinks, :, :], V[:, -n_recent:, :, :]], axis=1).copy()
            
        if self.use_cache:
            self.kv_cache = (K, V)

        # Shape (B, n_head, T, T)
        S = np.einsum('btnd,bsnd->bnts', Q, K) / np.sqrt(self.d_k)

        # Causal mask: When is_causal, tokens only attend to previous ones.
        if self.is_causal:
            T_q = T              # Temporal length of queries
            T_k = S.shape[3]     # Temporal length of keys (entire cache)

            offset = T_k - T_q  # Offset for the diagonal of the mask since the newly computed attention will be (Tq, Tk)

            mask = np.triu(np.ones((T_q, T_k)), k=offset + 1).astype(bool)
            S[:, :, mask] = -np.inf
            

        S_max = np.max(S, axis=-1, keepdims=True)
        exp_S = np.exp(S - S_max) 
        A = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
        self.A = A

        Z = np.einsum('bnts,bsnd->btnd', A, V)
        Z = Z.reshape(B, T, D) # Concatenation of heads

        Out = np.einsum('bid,dj->bij', Z, self.W_o)

        self.Q, self.K, self.V, self.Z = Q,K, V, Z
        return Out
    
    def backward(self, d_output: np.ndarray) -> np.ndarray:
        """
        Computes the gradients of the loss with respect to the parameters and to X
        """

        if self.use_cache:
            raise RuntimeError("KV cache currently is active. Make sure it is not during training")


        B, T, D = self.X.shape
        dW_o = np.einsum('bid,bij->dj', self.Z, d_output)

        dZ_flat = np.einsum('bij,dj->bid', d_output, self.W_o)
        dZ = dZ_flat.reshape(B, T, self.n_heads, self.d_k)

        # gradients with respect to V and A
        dV = np.einsum('bhts,bthd->bshd', self.A, dZ)
        dA = np.einsum('bthd,bshd->bhts', dZ, self.V)

        # Softmax
        sum_dA_A = np.sum(dA * self.A, axis=-1, keepdims=True)
        dS = self.A * (dA - sum_dA_A) / np.sqrt(self.d_k)

        # Gradients with respect to Q and K
        dQ = np.einsum('bhts,bshd->bthd', dS, self.K)
        dK = np.einsum('bhts,bthd->bshd', dS, self.Q)

        # Rotate back (cos is "even", sine is "uneven", therefore the matrix of -theta is pretty straightforward)
        dQ = apply_rotary_embedding(dQ, self.cos_sliced, -self.sin_sliced)
        dK = apply_rotary_embedding(dK, self.cos_sliced, -self.sin_sliced)

        # Back to (B, T, model_d) shape
        dQ_flat = dQ.reshape(B, T, D)
        dK_flat = dK.reshape(B, T, D)
        dV_flat = dV.reshape(B, T, D)
        
        # Forward : Q = X * W_q etc
        dW_q = np.einsum('bid,bij->dj', self.X, dQ_flat)
        dW_k = np.einsum('bid,bij->dj', self.X, dK_flat)
        dW_v = np.einsum('bid,bij->dj', self.X, dV_flat)

        if self.dW_q is None:
            self.dW_q = dW_q
            self.dW_k = dW_k
            self.dW_v = dW_v
            self.dW_o = dW_o
        else:
            self.dW_q += dW_q
            self.dW_k += dW_k
            self.dW_v += dW_v
            self.dW_o += dW_o

        # Get gradients with respect to different X and sum them
        dX_q = np.einsum('bij,dj->bid', dQ_flat, self.W_q)
        dX_k = np.einsum('bij,dj->bid', dK_flat, self.W_k)
        dX_v = np.einsum('bij,dj->bid', dV_flat, self.W_v)
        
        dX = dX_q + dX_k + dX_v
        return dX

    def get_params(self):
        return [self.W_q, self.W_k, self.W_v, self.W_o]
    
    def get_grads(self):
        return [self.dW_q, self.dW_k, self.dW_v, self.dW_o]
    
    def get_reg_info(self):
        return [True, True, True, True]
    
    def zero_grad(self):
        self.dW_q = None
        self.dW_k = None
        self.dW_v = None
        self.dW_o = None

    def get_state(self):
        return {'W_q' : self.W_q, 'W_k' : self.W_k, 'W_v' : self.W_v, 'W_o' : self.W_o}
    
    def set_state(self, state):
        self.W_q = state["W_q"].copy()
        self.W_k = state["W_k"].copy()
        self.W_v = state["W_v"].copy()
        self.W_o = state["W_o"].copy()

    def set_use_cache(self, use_cache: bool) -> None:
        self.use_cache = use_cache
        
    def reset_cache(self) -> None:
        self.kv_cache = None
        self.current_pos = 0




def apply_rotary_embedding(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Applies rotation to x
    x shape: (Batch, Time, n_heads, d_k)
    cos, sin shape: (Time, d_k)
    """
    # Adding dimensions for broadcast : (1, Time, 1, d_k)
    cos = cos[np.newaxis, :, np.newaxis, :]
    sin = sin[np.newaxis, :, np.newaxis, :]
    
    d_k = x.shape[-1]
    x1 = x[..., :d_k//2]
    x2 = x[..., d_k//2:]
    
    x_rotated = np.concatenate([-x2, x1], axis=-1)
    # Apply rotation matrix in 2d
    return x * cos + x_rotated * sin


class GQARoPEAttentionHead(Layer):
    def __init__(self, n_heads: int, n_kv_heads: int, d_model: int, block_size: int, is_causal=False) -> None:

        """
        Initializes query, key and value matrices for GQA mechanism 
        With ATTENTION SINK and RoPE
        block_size : Context window
        """
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.block_size = block_size

        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads

        self.use_cache = False
        self.kv_cache: Optional[np.ndarray] = None
        self.is_causal = is_causal

        self.current_pos = 0
        max_len = 2048 # Arbitrary lenght for encoding. If the generation finds itself to be longer,
        # then cos and sin would have to be lengthened on the fly

        theta = 10000.0 ** (-np.arange(0, self.d_k, 2) / self.d_k)
        m = np.arange(max_len)
        freqs = np.outer(m, theta)
        # Duplicate frequences to compute rotation
        freqs_cis = np.concatenate([freqs, freqs], axis=-1)
        self.cos = np.cos(freqs_cis)
        self.sin = np.sin(freqs_cis)

        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)
        self.W_k = np.random.randn(d_model, self.n_kv_heads * self.d_k) * np.sqrt(2 / d_model)
        self.W_v = np.random.randn(d_model, self.n_kv_heads * self.d_k) * np.sqrt(2 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)

        self.dW_q = None
        self.dW_k = None
        self.dW_v = None
        self.dW_o = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the scaled dot-product attention for an array X (Batch, Sequence length, model dimension (d_model))
        KV Cache is implemented
        """
        self.X = X
        B, T, D = X.shape

        Q = np.einsum('bid,dj->bij', X, self.W_q)
        K = np.einsum('bid,dj->bij', X, self.W_k)
        V = np.einsum('bid,dj->bij', X, self.W_v)

        # Reshaping for multi-head format (d_model -> (n_heads, d_k))

        Q = Q.reshape(B, T, self.n_heads, self.d_k)
        K = K.reshape(B, T, self.n_heads, self.d_k)
        V = V.reshape(B, T, self.n_heads, self.d_k)

        # RoPE
        if self.use_cache:
            pos_start = self.current_pos
            pos_end = self.current_pos + T
            self.current_pos += T
        else:
            pos_start = 0
            pos_end = T

        # Use the slices needed for embedding
        cos_sliced = self.cos[pos_start:pos_end, :]
        sin_sliced = self.sin[pos_start:pos_end, :]

        Q = apply_rotary_embedding(Q, cos_sliced, sin_sliced)
        K = apply_rotary_embedding(K, cos_sliced, sin_sliced)
        
        self.cos_sliced = cos_sliced # Saved for backward
        self.sin_sliced = sin_sliced

        # If kv_cache is used
        if self.use_cache and self.kv_cache is not None:
            past_K, past_V = self.kv_cache
            K = np.concatenate([past_K, K], axis=1)
            V = np.concatenate([past_V, V], axis=1)
            
        # Sliding context window + ATTENTION SINK
        if self.use_cache and K.shape[1] > self.block_size:
            n_sinks = 4  # Industry standard
            n_recent = self.block_size - n_sinks
            
            # Only keeps the block_size last tokens (including the sinks now so not exactly),
            # removes the others from RAM with .copy()
            K = np.concatenate([K[:, :n_sinks, :, :], K[:, -n_recent:, :, :]], axis=1).copy()
            V = np.concatenate([V[:, :n_sinks, :, :], V[:, -n_recent:, :, :]], axis=1).copy()
            
        if self.use_cache:
            self.kv_cache = (K, V)

        # Grouped Query part starting here
        K_rep = np.repeat(K, self.n_rep, axis=2)
        V_rep = np.repeat(V, self.n_rep, axis=2)

        # Shape (B, n_head, T, T)
        S = np.einsum('btnd,bsnd->bnts', Q, K_rep) / np.sqrt(self.d_k)

        # Causal mask: When is_causal, tokens only attend to previous ones.
        if self.is_causal:
            T_q = T              # Temporal length of queries
            T_k = S.shape[3]     # Temporal length of keys (entire cache)

            offset = T_k - T_q  # Offset for the diagonal of the mask since the newly computed attention will be (Tq, Tk)

            mask = np.triu(np.ones((T_q, T_k)), k=offset + 1).astype(bool)
            S[:, :, mask] = -np.inf
            

        S_max = np.max(S, axis=-1, keepdims=True)
        exp_S = np.exp(S - S_max) 
        A = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
        self.A = A

        Z = np.einsum('bnts,bsnd->btnd', A, V_rep)
        Z = Z.reshape(B, T, D) # Concatenation of heads

        Out = np.einsum('bid,dj->bij', Z, self.W_o)

        self.Q, self.K, self.V, self.Z = Q,K, V, Z
        return Out
    
    def backward(self, d_output: np.ndarray) -> np.ndarray:
        """
        Computes the gradients of the loss with respect to the parameters and to X
        """

        if self.use_cache:
            raise RuntimeError("KV cache currently is active. Make sure it is not during training")

        B, T, D = self.X.shape
        dW_o = np.einsum('bid,bij->dj', self.Z, d_output)

        dZ_flat = np.einsum('bij,dj->bid', d_output, self.W_o)
        dZ = dZ_flat.reshape(B, T, self.n_heads, self.d_k)

        # gradients with respect to V and A
        dV_rep = np.einsum('bhts,bthd->bshd', self.A, dZ)
        dA = np.einsum('bthd,bshd->bhts', dZ, np.repeat(self.V, self.n_rep, axis=2))

        # Softmax
        sum_dA_A = np.sum(dA * self.A, axis=-1, keepdims=True)
        dS = self.A * (dA - sum_dA_A) / np.sqrt(self.d_k)

        # Gradients with respect to Q and K
        dQ = np.einsum('bhts,bshd->bthd', dS, np.repeat(self.K, self.n_rep, axis=2))
        dK_rep = np.einsum('bhts,bthd->bshd', dS, self.Q)

        # Backprop through the repeat therefore we just need to sum
        dV = dV_rep.reshape(B, T, self.n_kv_heads, self.n_rep, self.d_k).sum(axis=3)
        dK = dK_rep.reshape(B, T, self.n_kv_heads, self.n_rep, self.d_k).sum(axis=3)

        # Rotate back (cos is "even", sine is "uneven", therefore the matrix of -theta is pretty straightforward)
        dQ = apply_rotary_embedding(dQ, self.cos_sliced, -self.sin_sliced)
        dK = apply_rotary_embedding(dK, self.cos_sliced, -self.sin_sliced)

        # Back to (B, T, model_d) shape
        dQ_flat = dQ.reshape(B, T, D)
        dK_flat = dK.reshape(B, T, self.n_kv_heads * self.d_k)
        dV_flat = dV.reshape(B, T, self.n_kv_heads * self.d_k)
        
        # Forward : Q = X * W_q etc
        dW_q = np.einsum('bid,bij->dj', self.X, dQ_flat)
        dW_k = np.einsum('bid,bij->dj', self.X, dK_flat)
        dW_v = np.einsum('bid,bij->dj', self.X, dV_flat)

        if self.dW_q is None:
            self.dW_q = dW_q
            self.dW_k = dW_k
            self.dW_v = dW_v
            self.dW_o = dW_o
        else:
            self.dW_q += dW_q
            self.dW_k += dW_k
            self.dW_v += dW_v
            self.dW_o += dW_o

        # Get gradients with respect to different X and sum them
        dX_q = np.einsum('bij,dj->bid', dQ_flat, self.W_q)
        dX_k = np.einsum('bij,dj->bid', dK_flat, self.W_k)
        dX_v = np.einsum('bij,dj->bid', dV_flat, self.W_v)
        
        dX = dX_q + dX_k + dX_v
        return dX

    def get_params(self):
        return [self.W_q, self.W_k, self.W_v, self.W_o]
    
    def get_grads(self):
        return [self.dW_q, self.dW_k, self.dW_v, self.dW_o]
    
    def get_reg_info(self):
        return [True, True, True, True]
    
    def zero_grad(self):
        self.dW_q = None
        self.dW_k = None
        self.dW_v = None
        self.dW_o = None

    def get_state(self):
        return {'W_q': self.W_q, 'W_k': self.W_k, 'W_v': self.W_v, 'W_o': self.W_o}
    
    def set_state(self, state):
        self.W_q = state["W_q"].copy()
        self.W_k = state["W_k"].copy()
        self.W_v = state["W_v"].copy()
        self.W_o = state["W_o"].copy()

    def set_use_cache(self, use_cache: bool) -> None:
        self.use_cache = use_cache
        
    def reset_cache(self) -> None:
        self.kv_cache = None
        self.current_pos = 0