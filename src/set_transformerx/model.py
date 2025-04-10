"""
Referece: https://github.com/stevengogogo/set_transformer/blob/master/modules.py
"""
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random as jr
from types import Callable

class MAB(eqx.Module):
    fc_q: eqx.nn.Linear 
    fc_k: eqx.nn.Linear
    fc_qv: eqx.nn.Linear
    fc_o: eqx.nn.Linear
    res1: Callable # residual connection 1
    res2: Callable # residual connection 2
    dim_V: int 
    num_heads: int

    def __init__(self, dim_Q, dim_K, dim_V, num_heads:int, ln=False, key=jr.PRNGKey(0)):
        """
        ln: layernorm
        """
        k = jr.split(key, 4)
        self.dim_V = dim_V 
        self.num_heads = num_heads 
        self.fc_q = eqx.nn.Linear(dim_Q, dim_V, key=k[0])
        self.fc_k = eqx.nn.Linear(dim_K, dim_V, key=k[1])
        self.fc_qv = eqx.nn.Linear(dim_K, dim_V, key=k[2])

        if ln: 
            self.res1 = eqx.nn.LayerNorm(dim_V)
            self.res2 = eqx.nn.LayerNorm(dim_V)
        else:
            self.res1 = lambda x: x
            self.res2 = lambda x: x

        self.fc_o = eqx.nn.Linear(dim_V, dim_V, key=k[3])

    def __call__(self, Q, K):
        Q = jax.vmap(self.fc_q)(Q)
        K, V = jax.vmap(self.fc_k)(K), jax.vmap(self.fc_qv)(K)
        Q = jnp.expand_dims(Q, axis=0)
        K = jnp.expand_dims(K, axis=0)
        V = jnp.expand_dims(V, axis=0)


        dim_split = self.dim_V // self.num_heads
        Q_ = jnp.concatenate(jnp.split(Q, dim_split, axis=2), axis=0)
        K_ = jnp.concatenate(jnp.split(K, dim_split, axis=2), axis=0)
        V_ = jnp.concatenate(jnp.split(V, dim_split, axis=2), axis=0)


        A = jax.nn.softmax(jax.lax.batch_matmul(Q_, K_.transpose(0,2,1))) / jnp.sqrt(self.dim_V)
        
        O = (Q_ + jax.lax.batch_matmul(A, V_)).reshape(-1,self.dim_V)
        
        O = self.res1(O)
        O = O + jax.nn.relu(jax.vmap(self.fc_o)(O))
        O = self.res2(O)
        return O

class SAB(eqx.Module):
    mab: MAB
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
    def __call__(self, X):
        return self.mab(X,X)

class ISAB(eqx.Module):
    I: jnp.ndarray
    mab0: MAB
    mab1: MAB
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, key=jr.PRNGKey(0)):
        init = jax.nn.initializers.glorot_uniform()
        self.I = init(key, (num_inds, dim_out)) 
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln) 
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
    def __call__(self, X):
        H = self.mab0(self.I,X)
        return self.mab1(X,H)